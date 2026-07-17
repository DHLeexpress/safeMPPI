"""AFE2: corrected two-arm 10-round study (user spec 2026-07-16b).

Differences from grid_expand_afe.py (v1), all per spec:
  * EVOLVING representation phi_s^(n): sigma features come from the CURRENT policy (initialized at
    the pretrained phi_s^(0)); encoder+trunk+head all trainable. No permanently frozen phi0.
  * Cumulative raw query archive D_n; at the START of every round, re-embed EVERY stored query with
    phi_s^(n) and REBUILD A = I + lam^-1 sum z z^T from scratch; theta and phi are held fixed during
    the round's gathering while A updates sequentially after each successful full-verifier query.
    A is never carried across a representation update. socp_error queries update NOTHING.
  * EXPERT-FREE: no SafeMPPI, no fallback action, ever (expansion AND evaluation). Execution accepts
    a full-H positive or a certified prefix whose endpoint first enters the absorbing goal set. The
    latter never enters D+ unless its full H-window is independently positive. If neither exists,
    the rollout TERMINATES with NO_VERIFIED_POSITIVE.
  * Execution among admissible queries: fixed J_exec = maximum progress, truncated at the first
    goal hit when the absorbing terminal rule applies.
  * Complete fixed gamma sweep every round: one episode per gamma, all seven gammas, fixed order.
  * Two matched arms sharing code, configuration, initial checkpoint, and keyed RNG streams;
    their learned representations, candidate plans, archives, and A matrices diverge by design:
      --arm prox : corrected proximal control (batch 128, lr 2e-5, eta 0.01, stop fstep>=0.03 or 40)
      --arm afe  : uniform cumulative D+ replay, batch 128, lr 1e-4, 250 steps, NO proximal term.
    No curriculum, expert replay, anchors, easy/frontier, or automatic collapse rollback.
  * beta fixed from a pre-training ESS calibration (--calibrate): the beta in {0.01,0.02,0.05}
    whose median acquisition ESS/K over representative round-0 all-K pools lies in [0.25,0.5].
  * Diagnostics per round: all-K and selected-B sigma quantiles, ESS, acquisition entropy,
    selected-vs-pool sigma uplift, eigen spectrum + effective rank of A, total CFM loss, per-module
    gradient norms (encoder/trunk/head), relative per-module parameter change, fixed-probe
    representation cosine drift, per-gamma query/positive/distinct-trained counts, untilted raw
    validity (audit), and the expert-free verified-controller evaluation (fixed-index equal-count
    rollouts; SR / CR / NO_VERIFIED_POSITIVE rate / true min clearance / time), round 0 included.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
sys.path.insert(0, _WORK)
sys.path.insert(0, _REV)
sys.path.insert(0, _HERE)                                   # local copies ALWAYS win (grid_metrics2!)

import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_hp_expt as HP
import grid_expand_hardtail as HT              # reuse: _apply_wall_plugs, _save_hp_atomic
from di_grid_viz import di_step

import afe_core as AC
import afe2_calibration as BC
from afe2_scene_profiles import (
    SCENE_PROFILES,
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)


@dataclass
class AFE2Config:
    rounds: int = 10
    T: int = 300
    reach: float = 0.15
    K: int = 64
    B: int = 8
    beta: float = 0.02                 # SET FROM CALIBRATION (--calibrate); fixed for both arms
    s: float = 0.9
    lam: float = 10.0                  # measured live-sigma choice (analysis/afe_lam_study.py)
    nfe: int = 8
    temp: float = 1.0
    n_theta: int = 180
    gammas: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    # arms
    arm: str = "prox"                  # prox | afe
    batch: int = 128
    prox_lr: float = 2e-5
    prox_eta: float = 0.01
    prox_max_inner: int = 40
    prox_fstep_stop: float = 0.03
    afe_lr: float = 1e-4
    afe_steps: int = 250
    grad_clip: float = 1.0
    # tracking
    audit_pos: int = 12
    audit_plans: int = 4
    M_eval: int = 8                    # fixed-index equal-count controller rollouts per gamma
    prog_eps: float = 1e-9             # J_exec = max progress (any positive executes; ties by r)
    dither_bar: float = 0.05
    terminal_mode: str = "absorbing_goal_prefix"
    taskspace_epsilon: float = float(GM.EPS_TASK)
    # environment
    wall_plugs: int = 8
    start_eps: float = 0.3
    goal_xy: tuple = (4.7, 4.7)
    scene_profile: str = "claude_grid_v1"
    seed: int = 910


REFERENCE_RECIPE = {
    "rounds": 10,
    "T": 300,
    "reach": 0.15,
    "K": 64,
    "B": 8,
    "s": 0.9,
    "lam": 10.0,
    "nfe": 8,
    "temp": 1.0,
    "n_theta": 180,
    "gammas": (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
    "batch": 128,
    "prox_lr": 2e-5,
    "prox_eta": 0.01,
    "prox_max_inner": 40,
    "prox_fstep_stop": 0.03,
    "afe_lr": 1e-4,
    "afe_steps": 250,
    "grad_clip": 1.0,
    "audit_pos": 12,
    "audit_plans": 4,
    "M_eval": 8,
    "prog_eps": 1e-9,
    "dither_bar": 0.05,
    "terminal_mode": "absorbing_goal_prefix",
    "taskspace_epsilon": float(GM.EPS_TASK),
    "seed": 910,
}
REFERENCE_BEHAVIOR_COMMIT = "e97eeadeabffc93775ea96332dbf3b56210442a7"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_provenance(module):
    path = os.path.abspath(module.__file__)
    return {"path": path, "sha256": _sha256_file(path)}


def _json_safe(value):
    """Convert nested diagnostics to strict JSON without changing calculations."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _git_state():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=_HERE,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        worktree_dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=root,
            check=False,
        ).returncode != 0
        index_dirty = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=root,
            check=False,
        ).returncode != 0
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        untracked_runtime_sources = sorted(
            path for path in untracked
            if path.startswith("overnight_run_07_06/")
            and path.endswith((".py", ".sh"))
        )
        return {
            "commit": commit,
            "tracked_dirty": bool(worktree_dirty or index_dirty),
            "untracked_runtime_sources": untracked_runtime_sources,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "tracked_dirty": None,
            "untracked_runtime_sources": None,
        }


def assert_reference_recipe(cfg):
    """Prevent a radius-1 replication from silently becoming a knob sweep."""

    mismatches = {
        name: (getattr(cfg, name), expected)
        for name, expected in REFERENCE_RECIPE.items()
        if getattr(cfg, name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in sorted(mismatches.items())
        )
        raise ValueError(f"AFE2 reference recipe mismatch: {details}")


def named_seed(base_seed, *parts):
    """Stable independent RNG stream key for gather/update/eval purposes."""

    payload = ":".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


# ------------------------------------------------------------------ representation / uncertainty
@torch.no_grad()
def embed_queries(policy, store, cfg, device, ids=None, chunk=512):
    """Re-embed stored queries with the CURRENT phi_s^(n) -> normalized z [N,32] (cpu float32)."""
    ids = list(range(len(store))) if ids is None else ids
    Zs = []
    for i0 in range(0, len(ids), chunk):
        part = ids[i0:i0 + chunk]
        sids = [store.q_sid[q] for q in part]
        G = store.grid3_of(sids).to(device)
        L = torch.stack([torch.from_numpy(store.ctx_low5[s]) for s in sids]).to(device)
        Hh = torch.stack([torch.from_numpy(store.ctx_hist[s].astype(np.float32)) for s in sids]).to(device)
        U = torch.stack([torch.from_numpy(store.q_U[q]) for q in part]).to(device)
        Zs.append(AC.frozen_feat(policy, U, G, L, Hh, s=cfg.s).cpu())
    return torch.cat(Zs) if Zs else torch.zeros(0, policy.repr_dim or policy.width)


def rebuild_A(policy, store, cfg, device):
    """Round-start rebuild: A^(n) = I + lam^-1 sum_i z_i^(n) z_i^(n)T over the WHOLE archive,
    with z re-embedded under the current representation. Returns (blr, spectrum diagnostics)."""
    dim = policy.repr_dim or policy.width
    blr = AC.BLRSigma(dim=dim, lam=cfg.lam)
    diag = dict(
        n=0,
        S_eff_rank=0.0,
        A_eff_rank=float(dim),
        A_eigenvalues=[1.0] * int(dim),
        A_eig_top=1.0,
        A_eig_med=1.0,
    )
    if len(store) == 0:
        return blr, diag
    Z = embed_queries(policy, store, cfg, device).to(torch.float64)
    S = Z.T @ Z                                     # query mass in the current feature space
    A = torch.eye(dim, dtype=torch.float64) + S / cfg.lam
    blr.A_inv = torch.linalg.inv(A)
    blr.n = Z.shape[0]
    ev_s = torch.linalg.eigvalsh(S).clamp_min(0)
    ev_a = 1.0 + ev_s / cfg.lam
    diag = dict(
        n=int(Z.shape[0]),
        S_eff_rank=float(
            ev_s.sum() ** 2 / (ev_s ** 2).sum().clamp_min(1e-12)
        ),
        A_eff_rank=float(
            ev_a.sum() ** 2 / (ev_a ** 2).sum().clamp_min(1e-12)
        ),
        A_eigenvalues=[float(value) for value in ev_a.cpu()],
        A_eig_top=float(ev_a.max()),
        A_eig_med=float(ev_a.median()),
    )
    return blr, diag


@torch.no_grad()
def rep_probe_build(policy, env, cfg, device, n_ctx=24, n_plans=8, seed=20260716):
    """Fixed probe set for representation cosine drift: (c,U) pairs sampled ONCE from the round-0
    policy at fixed audit-like contexts. Returns tensors + their phi^(0) features."""
    ctxs = AC.build_audit_contexts(env, [0.1, 0.5, 1.0], n_pos=n_ctx // 4, seed=seed)[:n_ctx]
    G, L, Hh, U = [], [], [], []
    with AC.isolated_random_state(seed):
        for c in ctxs:
            gT = torch.tensor(c["grid"], device=device)
            lT = torch.tensor(c["low5"], device=device)
            hT = torch.tensor(c["hist"], device=device)
            Uc = policy.sample_window(gT, lT, hT, n=n_plans, temp=1.0, nfe=cfg.nfe)
            for j in range(n_plans):
                G.append(torch.tensor(c["grid"])); L.append(torch.tensor(c["low5"]))
                Hh.append(torch.tensor(c["hist"])); U.append(Uc[j].cpu())
    G = torch.stack(G).to(device); L = torch.stack(L).to(device)
    Hh = torch.stack(Hh).to(device); U = torch.stack(U).to(device)
    f0 = AC.frozen_feat(policy, U, G, L, Hh, s=cfg.s).cpu()
    return dict(G=G, L=L, H=Hh, U=U, f0=f0)


@torch.no_grad()
def rep_cos_drift(policy, probe, cfg):
    f = AC.frozen_feat(policy, probe["U"], probe["G"], probe["L"], probe["H"], s=cfg.s).cpu()
    return float((f * probe["f0"]).sum(1).mean())   # both normalized -> mean cosine


# ------------------------------------------------------------------ acquisition step (shared)
def acquire_and_execute(policy, blr, env, cfg, st, hist, g, store, round_i, ep, t, device,
                        collect=True, viz=None):
    """Acquire B queries and execute one full- or terminal-prefix-certified plan.

    The cumulative training-positive label remains the full-H verifier result.
    A certified prefix ending in the absorbing goal set is execution-admissible
    but never relabels its unverified suffix as positive training data.
    ``collect=False`` stores and updates nothing.
    """
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal_np = env.goal.detach().cpu().numpy()
    grid_np = GF.axis_grid(st[:2], obs, rr)
    l5_np = GF.low5(st, goal_np, g)
    h_np = GF.hist_pad(np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)), GF.K_HIST)
    gT = torch.tensor(grid_np, device=device)
    lT = torch.tensor(l5_np, device=device)
    hT = torch.tensor(h_np, device=device)
    Ucand = policy.sample_window(gT, lT, hT, n=cfg.K, temp=cfg.temp, nfe=cfg.nfe)
    Z = AC.frozen_feat(policy, Ucand, gT, lT, hT, s=cfg.s)
    sig = blr.sigma(Z)
    w = torch.exp(((sig - sig.max()) / max(cfg.beta, 1e-6)).clamp(-30, 30))
    pi = (w / w.sum()).to(torch.float64)
    ess = float((pi.sum() ** 2) / (pi ** 2).sum())                    # ESS in [1,K]
    ent = float(-(pi * (pi + 1e-30).log()).sum() / np.log(cfg.K))     # normalized entropy
    drawn = torch.multinomial(pi.float(), min(cfg.B, cfg.K), replacement=False).tolist()
    uplift = float(sig[drawn].mean() - sig.mean())
    sid = store.add_step_ctx(st, grid_np, l5_np, h_np, (round_i, ep, t)) if collect else -1
    best = None
    n_err = 0
    dres = []
    for j in drawn:
        U_np = Ucand[j].detach().cpu().numpy()
        seg = GR.window_positions(st, U_np, env.dt)
        v = AC.verify_plan_with_terminal(
            st,
            U_np,
            env,
            g,
            goal_np,
            reach=cfg.reach,
            n_theta=cfg.n_theta,
        )
        if v["reason"] == "socp_error":             # spec: update NOTHING on socp_error
            n_err += 1
            dres.append(dict(j=j, full_y=None, exec_y=0, qid=-1, v=v))
            continue
        qid = -1
        if collect:
            qid = store.add_query(sid, U_np, v, float(sig[j]), g, round_i, seg)
            blr.update(Z[j:j + 1])
        dres.append(dict(j=j, full_y=v["y"], exec_y=v["exec_y"], qid=qid, v=v))
        if v["exec_y"] and (best is None or v["exec_prog"] > best[0]):
            best = (v["exec_prog"], qid, U_np, j)
    if viz is not None:
        segsK = GR.di_rollout_batch(st, Ucand.detach().cpu().numpy(), env.dt).astype(np.float16)
        viz.append(dict(t=t, gamma=g, state=st.copy(), segsK=segsK,
                        drawn=[d["j"] for d in dres],
                        y=[(-1 if d["full_y"] is None else d["full_y"]) for d in dres],
                        exec_y=[d["exec_y"] for d in dres],
                        terminal_rescue=[bool(d["v"]["terminal_rescue"]) for d in dres],
                        terminal_tau=[d["v"]["terminal_tau"] for d in dres],
                        n_socp_solve=sum(int(d["v"]["n_socp_solve"]) for d in dres),
                        sel=(best[3] if best is not None else -1),
                        sig_q=[float(q) for q in np.quantile(sig.numpy(), [0.1, 0.5, 0.9])],
                        sigB_q=[float(q) for q in np.quantile(sig[drawn].numpy(), [0.1, 0.5, 0.9])],
                        min_margin=float(np.nanmin(
                            [d["v"]["exec_margin"] for d in dres if d["exec_y"]]
                        ) if any(d["exec_y"] for d in dres) else np.nan)))
    selected_terminal_rescue = bool(
        best is not None
        and any(
            d["j"] == best[3] and d["v"]["terminal_rescue"]
            for d in dres
        )
    )
    full_positive_available = any(d["full_y"] == 1 for d in dres)
    selected_terminal_required = bool(
        selected_terminal_rescue and not full_positive_available
    )
    stats = dict(ess=ess, ent=ent, uplift=uplift, n_err=n_err,
                 n_socp_solve=sum(int(d["v"]["n_socp_solve"]) for d in dres),
                 verifier_seconds=sum(float(d["v"]["verifier_seconds"]) for d in dres),
                 n_terminal_error=sum(
                     d["v"]["terminal_reason"] == "socp_error" for d in dres
                 ),
                 n_pos=sum(1 for d in dres if d["full_y"] == 1),
                 n_exec_pos=sum(1 for d in dres if d["exec_y"] == 1),
                 n_terminal_rescue=sum(1 for d in dres if d["v"]["terminal_rescue"]),
                 n_terminal_reverify=sum(1 for d in dres if d["v"]["terminal_reverify"]),
                 selected_terminal_rescue=selected_terminal_rescue,
                 selected_terminal_required=selected_terminal_required,
                 full_positive_available=full_positive_available,
                 n_drawn=len(dres),
                 sig_all=[float(q) for q in np.quantile(sig.numpy(), [0.1, 0.5, 0.9])],
                 sig_sel=[float(q) for q in np.quantile(sig[drawn].numpy(), [0.1, 0.5, 0.9])])
    return best, stats


def run_episode(policy, blr, env, cfg, g, store, round_i, ep, device, collect=True, viz=None,
                rollout_seed=None):
    """One expert-free shielded episode at gamma g. Ends on reach / NO_VERIFIED_POSITIVE / timeout.
    Executing only certified first actions => dead (collision/OOB) should be ~impossible; counted."""
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal_np = env.goal.detach().cpu().numpy()
    st = env.x0.detach().cpu().numpy().astype(np.float32).copy()
    hist, path = [], [st[:2].copy()]
    clear_min = (
        float((np.linalg.norm(st[:2][None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min())
        if obs.size else float("inf")
    )
    step_stats = []
    status, term_t = "timeout", None
    collision = bool(clear_min < 0.0)
    oob = bool(
        (st[:2] < -cfg.taskspace_epsilon).any()
        or (st[:2] > GM.GRID_M + cfg.taskspace_epsilon).any()
    )
    if collision or oob or np.linalg.norm(st[:2] - goal_np) < cfg.reach:
        status = "collision" if collision else ("oob" if oob else "reached")
        return dict(gamma=g, path=np.asarray(path, np.float32), status=status, term_t=0,
                    steps=0, clear_min=clear_min, collision=collision, oob=oob,
                    step_stats=step_stats)
    for t in range(cfg.T):
        if rollout_seed is None:
            best, stats = acquire_and_execute(
                policy, blr, env, cfg, st, hist, g, store,
                round_i, ep, t, device, collect=collect, viz=viz,
            )
        else:
            with AC.isolated_random_state(named_seed(rollout_seed, "control", t)):
                best, stats = acquire_and_execute(
                    policy, blr, env, cfg, st, hist, g, store,
                    round_i, ep, t, device, collect=collect, viz=viz,
                )
        step_stats.append(stats)
        if best is None:                            # spec: terminate, never call an expert
            status, term_t = "nvp", t
            break
        a = best[2][0]
        if collect and best[1] >= 0:
            store.mark_executed(best[1])
        st = di_step(st, np.asarray(a, np.float32), dt=env.dt)
        hist.append(np.asarray(a, np.float32))
        path.append(st[:2].copy())
        if obs.size:
            clear_min = min(clear_min, float((np.linalg.norm(st[:2][None] - obs[:, :2], axis=1)
                                              - obs[:, 2] - rr).min()))
        collision = bool(clear_min < 0.0)
        oob = bool(
            (st[:2] < -cfg.taskspace_epsilon).any()
            or (st[:2] > GM.GRID_M + cfg.taskspace_epsilon).any()
        )
        if collision or oob:
            status, term_t = ("collision" if collision else "oob"), t + 1
            break
        if np.linalg.norm(st[:2] - goal_np) < cfg.reach:
            status, term_t = "reached", t + 1
            break
    return dict(gamma=g, path=np.asarray(path, np.float32), status=status, term_t=term_t,
                steps=len(path) - 1, clear_min=(clear_min if np.isfinite(clear_min) else np.nan),
                collision=collision, oob=oob, step_stats=step_stats)


# ------------------------------------------------------------------ updates (the two arms)
def update_round(policy, opt, store, cfg, device, rng):
    """Both arms: uniform replay over CUMULATIVE D+. prox: l_CFM + ||th-th_n||^2/2eta, lr 2e-5,
    stop at fstep>=0.03 or 40 steps. afe: plain l_CFM, lr 1e-4, exactly 250 steps, no prox."""
    if store.n_pos() == 0:
        return None
    policy.train()
    groups = {k: list(m.parameters()) for k, m in policy.module_groups().items()}
    g_before = {k: torch.sqrt(sum((p.detach() ** 2).sum() for p in ps)).item()
                for k, ps in groups.items()}
    snap = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
    trainable = [p for p in policy.parameters() if p.requires_grad]
    refs = [p.detach().clone() for p in trainable] if cfg.arm == "prox" else None
    n_steps = cfg.prox_max_inner if cfg.arm == "prox" else cfg.afe_steps
    probe = v_before = None
    drawn_ids = {}
    cfm_hist, fstep_hist = [], []
    gnorm = {k: [] for k in groups}
    stop = "all_steps"
    for k_step in range(n_steps):
        G, L, Hh, U, ids = store.sample_pos(cfg.batch, rng)
        for q in ids:
            drawn_ids[q] = drawn_ids.get(q, 0) + 1
        G, L, Hh, U = G.to(device), L.to(device), Hh.to(device), U.to(device)
        if probe is None:
            na = min(U.shape[0], 128)
            xa = 0.5 * (U[:na] / policy.u_max).reshape(na, policy.d)
            ta = torch.full((na,), 0.5, device=device)
            ctxa = policy.ctx_from(G[:na], L[:na], Hh[:na]).detach()
            with torch.no_grad():
                v_before = policy.forward(xa, ta, policy._expand_ctx(ctxa, na)).detach()
            probe = (xa, ta, ctxa, na)
        cfm = policy.cfm_loss(U, policy.ctx_from(G, L, Hh))
        loss = cfm
        if cfg.arm == "prox":
            loss = loss + sum(((p - r) ** 2).sum() for p, r in zip(trainable, refs)) / (2.0 * cfg.prox_eta)
        opt.zero_grad()
        loss.backward()
        for kg, ps in groups.items():
            gnorm[kg].append(float(sum((p.grad ** 2).sum() for p in ps if p.grad is not None)) ** 0.5)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        cfm_hist.append(float(cfm.detach()))
        xa, ta, ctxa, na = probe
        with torch.no_grad():
            va = policy.forward(xa, ta, policy._expand_ctx(ctxa, na))
            fstep = float((va - v_before).norm(dim=1).mean() /
                          v_before.norm(dim=1).mean().clamp_min(1e-9))
        fstep_hist.append(fstep)
        if cfg.arm == "prox" and fstep >= cfg.prox_fstep_stop:
            stop = "fstep_bound"
            break
    rel_dp = {}
    for kg, ps in groups.items():
        num = torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(ps, snap[kg]))).item()
        rel_dp[kg] = num / max(g_before[kg], 1e-12)
    return dict(steps=len(cfm_hist), stop=stop, cfm=float(np.mean(cfm_hist)),
                cfm_first=cfm_hist[0], cfm_last=cfm_hist[-1],
                fstep_final=fstep_hist[-1], fstep_max=max(fstep_hist),
                grad_norm={k: float(np.mean(v)) for k, v in gnorm.items()},
                rel_param_change=rel_dp, drawn_ids=drawn_ids, n_distinct=len(drawn_ids))


# ------------------------------------------------------------------ controller evaluation
def controller_eval(policy, blr, env, cfg, device, round_i):
    """Expert-free verified controller, NO fallback, fixed-index equal-count rollouts: the SAME
    M_eval rollout seeds per gamma at every round (paired across rounds). Nothing is stored;
    A is used read-only."""
    dummy = AC.DStore()                             # scratch store; discarded (collect=False anyway)
    rows = {}
    for g in cfg.gammas:
        recs = []
        for m in range(cfg.M_eval):
            seed = named_seed(cfg.seed, "controller_eval", str(g), m)
            r = run_episode(policy, blr, env, cfg, float(g), dummy, round_i, -1, device,
                            collect=False, viz=None, rollout_seed=seed)
            recs.append(r)
        n = len(recs)
        rows[str(g)] = dict(
            SR=sum(r["status"] == "reached" for r in recs) / n,
            CR=sum(r["collision"] or r["oob"] for r in recs) / n,
            collision=sum(r["collision"] for r in recs) / n,
            OOB=sum(r["oob"] for r in recs) / n,
            NVP=sum(r["status"] == "nvp" for r in recs) / n,
            TO=sum(r["status"] == "timeout" for r in recs) / n,
            clear=float(np.nanmean([r["clear_min"] for r in recs])),
            time=float(np.mean([r["steps"] * env.dt for r in recs if r["status"] == "reached"])
                       if any(r["status"] == "reached" for r in recs) else np.nan),
            terminal_rescue_steps=sum(
                int(s["selected_terminal_rescue"])
                for r in recs for s in r["step_stats"]
            ),
            terminal_rescue_episodes=sum(
                any(s["selected_terminal_rescue"] for s in r["step_stats"])
                for r in recs
            ),
            terminal_required_steps=sum(
                int(s["selected_terminal_required"])
                for r in recs for s in r["step_stats"]
            ),
            terminal_required_episodes=sum(
                any(s["selected_terminal_required"] for s in r["step_stats"])
                for r in recs
            ),
            reached_without_terminal_rescue=sum(
                r["status"] == "reached"
                and not any(s["selected_terminal_rescue"] for s in r["step_stats"])
                for r in recs
            ),
            reached_with_terminal_rescue=sum(
                r["status"] == "reached"
                and any(s["selected_terminal_rescue"] for s in r["step_stats"])
                for r in recs
            ),
            clear_values=[float(r["clear_min"]) for r in recs],
            time_success_values=[
                float(r["steps"] * env.dt) for r in recs if r["status"] == "reached"
            ],
            status_values=[str(r["status"]) for r in recs],
            nvp_t=[int(r["term_t"]) for r in recs if r["status"] == "nvp"])
    pooled = dict(SR=float(np.mean([v["SR"] for v in rows.values()])),
                  CR=float(np.mean([v["CR"] for v in rows.values()])),
                  NVP=float(np.mean([v["NVP"] for v in rows.values()])))
    return rows, pooled


# ------------------------------------------------------------------ beta calibration
def calibrate_beta(policy, env, cfg, device, betas=BC.CANDIDATES, log=print):
    """Calibrate beta once on radius-1 round-0 pools under beta-neutral acquisition.

    Uniform B-without-replacement queries evolve A during this dry pass, so no
    candidate beta changes the pools on which it is evaluated. The chosen beta
    is closest to the center of the declared ESS/K band among in-band
    candidates; calibration fails if none exists.
    """

    policy.eval()
    blr = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    raw_sigs = []
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal_np = env.goal.detach().cpu().numpy()
    for ep, g in enumerate(cfg.gammas):
        st = env.x0.detach().cpu().numpy().astype(np.float32).copy()
        hist = []
        for t in range(cfg.T):
            grid_np = GF.axis_grid(st[:2], obs, rr)
            l5 = GF.low5(st, goal_np, float(g))
            h_np = GF.hist_pad(np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)), GF.K_HIST)
            gT = torch.tensor(grid_np, device=device); lT = torch.tensor(l5, device=device)
            hT = torch.tensor(h_np, device=device)
            with AC.isolated_random_state(named_seed(cfg.seed, "beta_calibration", ep, t)):
                Ucand = policy.sample_window(
                    gT, lT, hT, n=cfg.K, temp=cfg.temp, nfe=cfg.nfe
                )
                Z = AC.frozen_feat(policy, Ucand, gT, lT, hT, s=cfg.s)
                sig = blr.sigma(Z)
                drawn = torch.randperm(cfg.K)[:cfg.B].tolist()
            raw_sigs.append(sig.numpy().copy())
            best = None
            for j in drawn:
                U_np = Ucand[j].detach().cpu().numpy()
                v = AC.verify_plan_with_terminal(
                    st,
                    U_np,
                    env,
                    float(g),
                    goal_np,
                    reach=cfg.reach,
                    n_theta=cfg.n_theta,
                )
                if v["reason"] == "socp_error":
                    continue
                blr.update(Z[j:j + 1])
                if v["exec_y"] and (best is None or v["exec_prog"] > best[0]):
                    best = (v["exec_prog"], U_np)
            if best is None:
                break
            st = di_step(st, np.asarray(best[1][0], np.float32), dt=env.dt)
            hist.append(np.asarray(best[1][0], np.float32))
            if np.linalg.norm(st[:2] - goal_np) < cfg.reach:
                break
        log(f"[calib] gamma {g}: {len(raw_sigs)} cumulative sigma pools")
    if not raw_sigs:
        raise RuntimeError("beta calibration produced no round-0 candidate pools")
    table = {}
    for b in betas:
        esss = []
        for sg in raw_sigs:
            w = np.exp(np.clip((sg - sg.max()) / max(b, 1e-9), -30, 30))
            p = w / w.sum()
            esss.append(1.0 / (p ** 2).sum() / cfg.K)
        table[b] = dict(ess_med=float(np.median(esss)), ess_p10=float(np.quantile(esss, 0.1)),
                        ess_p90=float(np.quantile(esss, 0.9)))
        log(f"[calib] beta {b}: median ESS/K {table[b]['ess_med']:.3f} "
            f"(p10 {table[b]['ess_p10']:.3f} p90 {table[b]['ess_p90']:.3f})")
    table_json = {str(beta): row for beta, row in table.items()}
    try:
        pick = BC.select_beta(table_json)
    except ValueError:
        pick = None
        log("[calib] no beta is in-band; recording failure and refusing a nearest fallback")
    log(f"[calib] chosen beta = {pick} (band [0.25,0.5]; {len(raw_sigs)} pools over 7 gammas)")
    return pick, table, len(raw_sigs)


# ------------------------------------------------------------------ run
def run_afe2(
    policy,
    env,
    cfg,
    device,
    outdir,
    log=print,
    *,
    checkpoint_path=None,
    checkpoint_sha256=None,
    checkpoint_model_sha256=None,
    beta_calibration=None,
    beta_calibration_sha256=None,
    reference_recipe_locked=False,
    source_git_state=None,
):
    if (
        reference_recipe_locked
        and os.path.isdir(outdir)
        and os.listdir(outdir)
    ):
        raise RuntimeError(f"locked AFE2 run requires an empty output directory: {outdir}")
    os.makedirs(outdir, exist_ok=True)
    store = AC.DStore()
    opt = torch.optim.Adam(policy.parameters(),
                           lr=(cfg.prox_lr if cfg.arm == "prox" else cfg.afe_lr))
    policy.eval()
    audit_ctxs = AC.build_audit_contexts(env, cfg.gammas, n_pos=cfg.audit_pos)
    probe0 = rep_probe_build(policy, env, cfg, device)
    goal_np = env.goal.detach().cpu().numpy()
    profile = get_scene_profile(cfg.scene_profile)
    scene = scene_snapshot(env, profile)
    assert_scene_snapshot(scene)
    source_git_state = _git_state() if source_git_state is None else source_git_state
    recipe = dict(algorithm="afe2_terminal_aware_two_arm_2026_07_16", arm=cfg.arm,
                  representation="EVOLVING phi_s^(n) (init pretrained); A rebuilt from the full "
                                 "archive at every round start; sequential updates within round",
                  acquisition_semantics=(
                      "one pre-step sigma/pi over K, then one B-without-replacement draw; "
                      "completed verifier rows update A after selection (the exact e97eead "
                      "Claude AFE2 behavior, not sequential re-scoring)"
                  ),
                  verifier_budget_semantics=(
                      "B counts candidate query objects; a full-H rejection that predicts a goal "
                      "hit receives one additional prefix check, so SOCP-solve count/time are "
                      "variable and logged separately"
                  ),
                  execution=(
                      "argmax terminal-aware progress among full-H positives or certified prefixes "
                      "that first enter the absorbing original goal set; full-H progress remains "
                      "stored separately; terminal-prefix-only plans do not "
                      "enter D+; NO_VERIFIED_POSITIVE terminates; NO expert/fallback anywhere"
                  ),
                  terminal_mode=cfg.terminal_mode,
                  safety_claim="certified safety through the first hitting time of the goal set",
                  taskspace_contract={
                      "legacy_tolerance": cfg.taskspace_epsilon,
                      "accepted_coordinate_interval": [
                          -cfg.taskspace_epsilon,
                          float(GM.GRID_M + cfg.taskspace_epsilon),
                      ],
                      "note": (
                          "preserved from Claude e97eead; this is not exact [0,5] containment"
                      ),
                  },
                  socp_error=(
                      "a full-H label error is not stored and does not update A; a prefix-only "
                      "error rejects execution but retains the already valid full-H negative row"
                  ),
                  K=cfg.K, B=cfg.B, beta=cfg.beta, lam=cfg.lam, s=cfg.s,
                  beta_protocol=(
                      "one shared radius-1 round-0 beta-neutral ESS calibration; median is "
                      "control-step-pool-weighted across the fixed gamma sweep"
                  ),
                  beta_calibration=beta_calibration,
                  beta_calibration_sha256=beta_calibration_sha256,
                  rng_streams=(
                      "SHA256-keyed independent streams: gather(round,gamma,t), replay(round), "
                      "update(round), controller_eval(gamma,index), audit, representation probe"
                  ),
                  update=("prox: lr %g eta %g stop fstep>=%g or %d" %
                          (cfg.prox_lr, cfg.prox_eta, cfg.prox_fstep_stop, cfg.prox_max_inner)
                          if cfg.arm == "prox" else
                          "afe: lr %g, %d steps, no prox" % (cfg.afe_lr, cfg.afe_steps)),
                  batch=cfg.batch, rounds=cfg.rounds, gamma_sweep="all 7 every round, fixed order",
                  T=cfg.T, reach=cfg.reach, M_eval=cfg.M_eval, seed=cfg.seed,
                  scene=scene,
                  source_checkpoint=(
                      None if checkpoint_path is None else os.path.abspath(checkpoint_path)
                  ),
                  source_checkpoint_sha256=checkpoint_sha256,
                  source_checkpoint_model_sha256=checkpoint_model_sha256,
                  source_git_commit=source_git_state["commit"],
                  source_git_tracked_dirty=source_git_state["tracked_dirty"],
                  source_git_untracked_runtime_sources=source_git_state[
                      "untracked_runtime_sources"
                  ],
                  runtime={
                      "python": sys.version,
                      "numpy": np.__version__,
                      "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda,
                      "device": str(device),
                      "cuda_device": (
                          torch.cuda.get_device_name(device)
                          if str(device).startswith("cuda") else None
                      ),
                  },
                  module_provenance={
                      "trainer": _module_provenance(sys.modules[__name__]),
                      "afe_core": _module_provenance(AC),
                      "afe2_calibration": _module_provenance(BC),
                      "grid_metrics2": _module_provenance(GM2),
                      "verifier_polytope": _module_provenance(GM2.VP),
                      "grid_hp_expt": _module_provenance(HP),
                      "grid_rollout": _module_provenance(GR),
                  },
                  reference_behavior_commit=REFERENCE_BEHAVIOR_COMMIT,
                  intentional_reference_deviation=(
                      "terminal-aware prefix eligibility/progress replaces fixed-H evaluation "
                      "after a plan reaches the unchanged goal set; beta is recalibrated once on "
                      "the radius-1 checkpoint; acquisition form and both update recipes match"
                  ),
                  reference_recipe_locked=bool(reference_recipe_locked),
                  reference_recipe=REFERENCE_RECIPE,
                  no_curriculum=True, no_anchor=True, no_collapse_rollback=True)
    with open(os.path.join(outdir, "recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2)

    def probe_write(rec):
        with open(os.path.join(outdir, "probe.jsonl"), "a") as f:
            f.write(json.dumps(_json_safe(rec), allow_nan=False) + "\n")

    # ---- round 0: pretrained baseline, both evaluation modes
    blr0 = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    audit0 = AC.run_audit(policy, audit_ctxs, env, goal_np, device, n_plans=cfg.audit_plans,
                          nfe=cfg.nfe, n_theta=cfg.n_theta,
                          seed=named_seed(cfg.seed, "audit"))
    rows0, pooled0 = controller_eval(policy, blr0, env, cfg, device, 0)
    log(f"[{cfg.arm}] round000 BASELINE V {audit0['V']:.3f} (adv {audit0.get('V_adverse', float('nan')):.3f}) | "
        f"ctrl SR {pooled0['SR']:.2f} CR {pooled0['CR']:.2f} NVP {pooled0['NVP']:.2f}", flush=True)
    probe_write(dict(round=0, arm=cfg.arm, V=audit0["V"], V_gamma=audit0["V_gamma"],
                     V_counts_gamma=audit0["counts_gamma"],
                     V_counts_gamma_adverse=audit0.get("counts_gamma_adverse"),
                     V_adverse=audit0.get("V_adverse"), V_gamma_adverse=audit0.get("V_gamma_adverse"),
                     ctrl=rows0, ctrl_pooled=pooled0, n_D=0, n_Dpos=0, rep_cos=1.0))
    HT._save_hp_atomic(policy, os.path.join(outdir, "ckpt_0.pt"),
                       extra={"iter": 0, "recipe": recipe, "resumable": False})

    completed_round = 0
    for n in range(1, cfg.rounds + 1):
        t0 = time.time()
        policy.eval()                                              # theta/phi FIXED during gather
        blr, spec_start = rebuild_A(policy, store, cfg, device)    # current round-start phi
        viz = []
        eps = []
        q_start = len(store)
        for ep, g in enumerate(cfg.gammas):                        # complete fixed sweep
            eps.append(run_episode(
                policy,
                blr,
                env,
                cfg,
                float(g),
                store,
                n,
                ep,
                device,
                collect=True,
                viz=viz,
                rollout_seed=named_seed(cfg.seed, "gather", n, ep),
            ))
        t_gather = time.time() - t0
        # Recompute diagnostics from the complete post-gather archive while the
        # representation is still the one that generated/acquired this round.
        _blr_gather_end, spec_gather_end = rebuild_A(
            policy, store, cfg, device
        )
        # per-gamma gather stats
        per_g = {}
        for g, r in zip(cfg.gammas, eps):
            ss = r["step_stats"]
            per_g[str(g)] = dict(status=r["status"], steps=r["steps"], term_t=r["term_t"],
                                 clear=r["clear_min"],
                                 n_q=sum(s["n_drawn"] for s in ss),
                                 n_pos=sum(s["n_pos"] for s in ss),
                                 n_exec_pos=sum(s["n_exec_pos"] for s in ss),
                                 n_terminal_rescue=sum(s["n_terminal_rescue"] for s in ss),
                                 n_terminal_reverify=sum(s["n_terminal_reverify"] for s in ss),
                                 n_selected_terminal_rescue=sum(
                                     int(s["selected_terminal_rescue"]) for s in ss
                                 ),
                                 n_selected_terminal_required=sum(
                                     int(s["selected_terminal_required"]) for s in ss
                                 ),
                                 n_socp_solve=sum(s["n_socp_solve"] for s in ss),
                                 verifier_seconds=sum(s["verifier_seconds"] for s in ss),
                                 n_err=sum(s["n_err"] for s in ss),
                                 n_terminal_error=sum(s["n_terminal_error"] for s in ss))
        all_ss = [s for r in eps for s in r["step_stats"]]
        t0 = time.time()
        replay_rng = np.random.default_rng(named_seed(cfg.seed, "replay", n))
        with AC.isolated_random_state(named_seed(cfg.seed, "update", n)):
            upd = update_round(policy, opt, store, cfg, device, replay_rng)
        t_upd = time.time() - t0
        policy.eval()
        audit = AC.run_audit(policy, audit_ctxs, env, goal_np, device, n_plans=cfg.audit_plans,
                             nfe=cfg.nfe, n_theta=cfg.n_theta,
                             seed=named_seed(cfg.seed, "audit"))
        blr_eval, spec_post = rebuild_A(policy, store, cfg, device)  # checkpoint-compatible A
        rows, pooled = controller_eval(policy, blr_eval, env, cfg, device, n)
        drawn = (upd or {}).get("drawn_ids", {})
        tr_gamma_draws = {}
        tr_gamma_distinct = {}
        for q, count in drawn.items():
            gq = str(round(store.q_gamma[q], 2))
            tr_gamma_draws[gq] = tr_gamma_draws.get(gq, 0) + int(count)
            tr_gamma_distinct[gq] = tr_gamma_distinct.get(gq, 0) + 1
        pos_prog = np.asarray([store.q_prog[q] for q in store.pos_ids], float)
        rec = dict(round=n, arm=cfg.arm, n_D=len(store), n_Dpos=store.n_pos(),
                   per_gamma=per_g,
                   ess_med=float(np.median([s["ess"] for s in all_ss])) / cfg.K,
                   ent_med=float(np.median([s["ent"] for s in all_ss])),
                   uplift_med=float(np.median([s["uplift"] for s in all_ss])),
                   sig_all_med=float(np.median([s["sig_all"][1] for s in all_ss])),
                   sig_sel_med=float(np.median([s["sig_sel"][1] for s in all_ss])),
                   A_n=spec_post["n"], A_eff_rank=spec_post["A_eff_rank"],
                   A_eig_top=spec_post["A_eig_top"], A_eig_med=spec_post["A_eig_med"],
                   S_eff_rank=spec_post["S_eff_rank"],
                   A_eigenvalues=spec_post["A_eigenvalues"],
                   A_round_start=spec_start, A_gather_end=spec_gather_end,
                   rep_cos=rep_cos_drift(policy, probe0, cfg),
                   dither_cum=float((pos_prog < cfg.dither_bar).mean()) if pos_prog.size else None,
                   V=audit["V"], V_gamma=audit["V_gamma"],
                   V_counts_gamma=audit["counts_gamma"], V_adverse=audit.get("V_adverse"),
                   V_counts_gamma_adverse=audit.get("counts_gamma_adverse"),
                   V_gamma_adverse=audit.get("V_gamma_adverse"),
                   ctrl=rows, ctrl_pooled=pooled,
                   trained_draws_gamma=tr_gamma_draws,
                   trained_distinct_gamma=tr_gamma_distinct,
                   t_gather=round(t_gather, 1), t_update=round(t_upd, 1))
        if upd is not None:
            rec.update(steps=upd["steps"], stop=upd["stop"], cfm=upd["cfm"],
                       cfm_first=upd["cfm_first"], cfm_last=upd["cfm_last"],
                       fstep_final=upd["fstep_final"], fstep_max=upd["fstep_max"],
                       grad_norm=upd["grad_norm"], rel_param_change=upd["rel_param_change"],
                       n_train_distinct=upd["n_distinct"])
        probe_write(rec)
        torch.save(dict(round=n, viz=viz, eps=[{k: v for k, v in r.items() if k != "step_stats"}
                                               for r in eps],
                        A_inv=blr_eval.A_inv.clone(),
                        A_representation="post-update checkpoint representation",
                        A_diagnostics=spec_post,
                        scene=scene,
                        audit=audit,
                        train_ids=np.asarray(sorted(drawn.keys()), np.int64),
                        train_counts=np.asarray([drawn[k] for k in sorted(drawn)], np.int64),
                        goal=goal_np, x0=env.x0.detach().cpu().numpy()),
                   os.path.join(outdir, "viz_db", f"round{n}.pt")
                   if os.path.isdir(os.path.join(outdir, "viz_db")) or
                   (os.makedirs(os.path.join(outdir, "viz_db"), exist_ok=True) or True)
                   else None)
        HT._save_hp_atomic(policy, os.path.join(outdir, f"ckpt_{n}.pt"),
                           extra={"iter": n, "recipe": recipe, "resumable": False})
        completed_round = n
        log(f"[{cfg.arm}] round{n:03d} D {len(store)} D+ {store.n_pos()} | "
            f"ESS/K {rec['ess_med']:.2f} uplift {rec['uplift_med']:.3f} "
            f"S-effR {spec_post['S_eff_rank']:.1f} "
            f"cos {rec['rep_cos']:.3f} | upd {0 if upd is None else upd['steps']}st "
            f"fstep {rec.get('fstep_final', 0):.3f} | V {audit['V']:.3f} | "
            f"ctrl SR {pooled['SR']:.2f} NVP {pooled['NVP']:.2f} | {t_gather:.0f}s+{t_upd:.0f}s",
            flush=True)
    if completed_round != cfg.rounds:
        with open(os.path.join(outdir, "INCOMPLETE.json"), "w") as f:
            json.dump(
                {
                    "status": "INCOMPLETE",
                    "completed_round": completed_round,
                    "required_round": cfg.rounds,
                    "scene_sha256": scene["sha256"],
                    "checkpoint_sha256": checkpoint_sha256,
                },
                f,
                indent=2,
            )
        store.save(os.path.join(outdir, "dstore_incomplete.pt"))
        raise RuntimeError(
            f"AFE2 stopped after round {completed_round}; round {cfg.rounds} is required"
        )
    final_path = os.path.join(outdir, "final.pt")
    store_path = os.path.join(outdir, "dstore.pt")
    HT._save_hp_atomic(policy, final_path,
                       extra={"iter": completed_round, "recipe": recipe, "resumable": False})
    store.save(store_path)
    required_artifacts = [
        "recipe.json",
        "probe.jsonl",
        "final.pt",
        "dstore.pt",
        *[f"ckpt_{round_i}.pt" for round_i in range(cfg.rounds + 1)],
        *[f"viz_db/round{round_i}.pt" for round_i in range(1, cfg.rounds + 1)],
    ]
    artifact_sha256 = {}
    for relative in required_artifacts:
        path = os.path.join(outdir, relative)
        if not os.path.isfile(path):
            raise RuntimeError(f"AFE2 completion artifact missing: {path}")
        artifact_sha256[relative] = _sha256_file(path)
    with open(os.path.join(outdir, "COMPLETE.json"), "w") as f:
        json.dump(
            {
                "status": "COMPLETE",
                "completed_round": completed_round,
                "scene_sha256": scene["sha256"],
                "checkpoint_sha256": checkpoint_sha256,
                "source_git_commit": source_git_state["commit"],
                "artifact_sha256": artifact_sha256,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    log(f"[{cfg.arm}] DONE {completed_round} rounds", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--expected-ckpt-sha256",
        default=None,
        help="required by locked runs; must equal the exact checkpoint file SHA-256",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", choices=["prox", "afe"], default="prox")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--beta", type=float, default=None, help="fixed beta from --calibrate")
    ap.add_argument(
        "--beta-calibration",
        default=None,
        help="calibration JSON generated once by --calibrate and shared by both locked arms",
    )
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--M-eval", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--afe-steps", type=int, default=250)
    ap.add_argument("--afe-lr", type=float, default=1e-4)
    ap.add_argument("--prox-lr", type=float, default=2e-5)
    ap.add_argument("--prox-eta", type=float, default=0.01)
    ap.add_argument(
        "--scene-profile",
        choices=sorted(SCENE_PROFILES),
        required=True,
        help="explicit task adapter (no default): claude_grid_v1 or codex_radius1_v1",
    )
    ap.add_argument(
        "--wall-plugs", type=int, default=None,
        help="legacy compatibility assertion; must equal the selected scene profile",
    )
    ap.add_argument(
        "--start-eps", type=float, default=None,
        help="legacy compatibility assertion; must equal both profile start coordinates",
    )
    ap.add_argument(
        "--goal-xy", type=float, nargs=2, default=None,
        help="legacy compatibility assertion; must equal the selected profile goal",
    )
    ap.add_argument("--seed", type=int, default=910)
    ap.add_argument("--calibrate", action="store_true", help="ESS beta calibration only")
    ap.add_argument(
        "--lock-reference-recipe",
        action="store_true",
        help=(
            "lock all e97eead arm/training/acquisition values except beta, which must come "
            "from the shared radius-1 calibration"
        ),
    )
    args = ap.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_sha256 = _sha256_file(args.ckpt)
    if args.expected_ckpt_sha256 is not None and (
        args.expected_ckpt_sha256.lower() != checkpoint_sha256
    ):
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"actual {checkpoint_sha256}, expected {args.expected_ckpt_sha256.lower()}"
        )
    if args.lock_reference_recipe and args.expected_ckpt_sha256 is None:
        raise ValueError("locked AFE2 runs require --expected-ckpt-sha256")
    policy, ck = HP.load_hp(args.ckpt, device="cpu")
    policy = policy.to(dev)
    if getattr(policy, "repr_dim", None) != 32:
        raise ValueError(
            "AFE2 comparison requires the declared 32-D representation; "
            f"checkpoint repr_dim={getattr(policy, 'repr_dim', None)!r}"
        )
    frozen_parameters = [name for name, value in policy.named_parameters() if not value.requires_grad]
    if frozen_parameters:
        raise ValueError(
            "AFE2 requires every model parameter trainable; frozen parameters: "
            + ", ".join(frozen_parameters[:8])
        )
    profile = get_scene_profile(args.scene_profile)
    # The identical beta-calibration rule applies independently to every (scene, checkpoint);
    # checkpoint provenance is enforced uniformly by --expected-ckpt-sha256 (the previous
    # radius1-only promoted-checkpoint gate imported a module absent from this branch).
    checkpoint_model_sha256 = None
    if args.wall_plugs is not None and args.wall_plugs != profile.wall_plugs:
        raise ValueError("--wall-plugs disagrees with --scene-profile")
    if args.start_eps is not None and not (
        np.isclose(args.start_eps, profile.start[0])
        and np.isclose(args.start_eps, profile.start[1])
    ):
        raise ValueError("--start-eps disagrees with --scene-profile")
    if args.goal_xy is not None and not np.allclose(args.goal_xy, profile.goal):
        raise ValueError("--goal-xy disagrees with --scene-profile")
    env = build_scene(profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    cfg = AFE2Config(rounds=args.rounds, K=args.K, B=args.B, lam=args.lam, T=args.T,
                     reach=args.reach, M_eval=args.M_eval, arm=args.arm, batch=args.batch,
                     afe_steps=args.afe_steps, afe_lr=args.afe_lr, prox_lr=args.prox_lr,
                     prox_eta=args.prox_eta, wall_plugs=profile.wall_plugs,
                     start_eps=profile.start[0], goal_xy=profile.goal,
                     scene_profile=profile.name, seed=args.seed)
    beta_calibration = None
    beta_calibration_sha256 = None
    if args.beta_calibration is not None:
        if args.beta is not None or args.calibrate:
            raise ValueError("--beta-calibration cannot be combined with --beta or --calibrate")
        with open(args.beta_calibration) as stream:
            beta_calibration = json.load(stream)
        beta_calibration_sha256 = _sha256_file(args.beta_calibration)
        expected_calibration = {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_model_sha256": checkpoint_model_sha256,
            "scene_sha256": scene_snapshot(env, profile)["sha256"],
            "lam": cfg.lam,
            "K": cfg.K,
            "B": cfg.B,
            "seed": cfg.seed,
        }
        cfg.beta = BC.validate_success(beta_calibration, expected_calibration)
    elif args.beta is not None:
        cfg.beta = args.beta
    elif args.lock_reference_recipe and not args.calibrate:
        raise ValueError("locked AFE2 arms require --beta-calibration")
    if args.lock_reference_recipe:
        assert_reference_recipe(cfg)
    source_git_state = _git_state()
    if args.lock_reference_recipe and (
        source_git_state["commit"] is None
        or source_git_state["tracked_dirty"] is not False
        or source_git_state["untracked_runtime_sources"] != []
    ):
        raise RuntimeError(
            "--lock-reference-recipe requires committed source files and a clean git tree; "
            f"untracked runtime sources={source_git_state['untracked_runtime_sources']}"
        )
    if beta_calibration is not None and (
        beta_calibration.get("source_git_commit") != source_git_state["commit"]
    ):
        raise ValueError(
            "beta calibration was not produced by the current committed trainer source"
        )
    print(f"[afe2] arm {cfg.arm} K{cfg.K} B{cfg.B} beta {cfg.beta} lam {cfg.lam} "
          f"scene {profile.name} EVOLVING-rep rebuild-A expert-free", flush=True)
    if args.calibrate:
        if os.path.isdir(args.outdir) and os.listdir(args.outdir):
            raise RuntimeError("beta calibration requires a new or empty output directory")
        pick, table, npools = calibrate_beta(policy, env, cfg, dev)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "beta_calibration.json"), "w") as f:
            json.dump(_json_safe(dict(
                status=(BC.SUCCESS_STATUS if pick is not None else BC.FAILURE_STATUS),
                chosen=pick,
                candidates=list(BC.CANDIDATES),
                target_ess_band=list(BC.ESS_BAND),
                selection=BC.SELECTION,
                acquisition=BC.ACQUISITION,
                pool_weighting=BC.POOL_WEIGHTING,
                table={str(k): v for k, v in table.items()},
                n_pools=npools,
                scene_sha256=scene_snapshot(env, profile)["sha256"],
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_model_sha256=checkpoint_model_sha256,
                source_git_commit=source_git_state["commit"],
                lam=cfg.lam,
                K=cfg.K,
                B=cfg.B,
                seed=cfg.seed,
            )), f, indent=2, sort_keys=True, allow_nan=False)
        if pick is None:
            raise RuntimeError(
                "no declared beta candidate reached ESS/K in [0.25,0.5]; "
                "failure table was persisted; refusing an ad-hoc nearest fallback"
            )
        return
    run_afe2(
        policy,
        env,
        cfg,
        dev,
        args.outdir,
        checkpoint_path=args.ckpt,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_model_sha256=checkpoint_model_sha256,
        beta_calibration=beta_calibration,
        beta_calibration_sha256=beta_calibration_sha256,
        reference_recipe_locked=args.lock_reference_recipe,
        source_git_state=source_git_state,
    )


if __name__ == "__main__":
    main()
