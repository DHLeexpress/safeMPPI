"""AFE2: corrected two-arm 10-round study (user spec 2026-07-16b).

Differences from grid_expand_afe.py (v1), all per spec:
  * EVOLVING representation phi_s^(n): sigma features come from the CURRENT policy (initialized at
    the pretrained phi_s^(0)); encoder+trunk+head all trainable. No permanently frozen phi0.
  * Cumulative raw query archive D_n; at the START of every round, re-embed EVERY stored query with
    phi_s^(n) and REBUILD A = I + lam^-1 sum z z^T from scratch; theta and phi are held fixed during
    the round's gathering while A updates sequentially after each successful full-verifier query.
    A is never carried across a representation update. socp_error queries update NOTHING.
  * EXPERT-FREE: no SafeMPPI, no fallback action, ever (expansion AND evaluation). If none of the B
    queried plans is SOCP-positive, the rollout TERMINATES with NO_VERIFIED_POSITIVE.
  * Execution among SOCP-positive queries: fixed nominal J_exec = maximum progress (this study).
  * Complete fixed gamma sweep every round: one episode per gamma, all seven gammas, fixed order.
  * Two arms sharing acquisition/representation/seeds exactly:
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
import json
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_scene as GS
import grid_hp_expt as HP
import grid_expand_hardtail as HT              # reuse: _apply_wall_plugs, _save_hp_atomic
from di_grid_viz import di_step

import afe_core as AC


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
    # environment
    wall_plugs: int = 8
    start_eps: float = 0.3
    goal_xy: tuple = (4.7, 4.7)
    seed: int = 910
    max_hours: float = 12.0


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
    diag = dict(n=0, eff_rank=float("nan"), eig_top=float("nan"), eig_med=float("nan"))
    if len(store) == 0:
        return blr, diag
    Z = embed_queries(policy, store, cfg, device).to(torch.float64)
    S = Z.T @ Z                                     # query mass in the current feature space
    A = torch.eye(dim, dtype=torch.float64) + S / cfg.lam
    blr.A_inv = torch.linalg.inv(A)
    blr.n = Z.shape[0]
    ev = torch.linalg.eigvalsh(S).clamp_min(0)
    diag = dict(n=int(Z.shape[0]),
                eff_rank=float((ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12))),
                eig_top=float(ev.max()), eig_med=float(ev.median()))
    return blr, diag


@torch.no_grad()
def rep_probe_build(policy, env, cfg, device, n_ctx=24, n_plans=8, seed=20260716):
    """Fixed probe set for representation cosine drift: (c,U) pairs sampled ONCE from the round-0
    policy at fixed audit-like contexts. Returns tensors + their phi^(0) features."""
    ctxs = AC.build_audit_contexts(env, [0.1, 0.5, 1.0], n_pos=n_ctx // 4, seed=seed)[:n_ctx]
    G, L, Hh, U = [], [], [], []
    cpu_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    for c in ctxs:
        gT = torch.tensor(c["grid"], device=device)
        lT = torch.tensor(c["low5"], device=device)
        hT = torch.tensor(c["hist"], device=device)
        Uc = policy.sample_window(gT, lT, hT, n=n_plans, temp=1.0, nfe=cfg.nfe)
        for j in range(n_plans):
            G.append(torch.tensor(c["grid"])); L.append(torch.tensor(c["low5"]))
            Hh.append(torch.tensor(c["hist"])); U.append(Uc[j].cpu())
    torch.random.set_rng_state(cpu_state)
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
    """One control step: sample K -> sigma (current rep, current A) -> draw B ~ pi -> full-verify
    each (socp_error updates NOTHING) -> execute argmax-progress positive, else NO_VERIFIED_POSITIVE.
    collect=False (controller evaluation): NOTHING is stored and A is NOT updated."""
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
    dres = []                                       # (j, y or None(err), qid or -1, v)
    for j in drawn:
        U_np = Ucand[j].detach().cpu().numpy()
        seg = GR.window_positions(st, U_np, env.dt)
        v = AC.verify_plan(st, U_np, env, g, goal_np, n_theta=cfg.n_theta)
        if v["reason"] == "socp_error":             # spec: update NOTHING on socp_error
            n_err += 1
            dres.append((j, None, -1, v))
            continue
        qid = -1
        if collect:
            qid = store.add_query(sid, U_np, v, float(sig[j]), g, round_i, seg)
            blr.update(Z[j:j + 1])
        dres.append((j, v["y"], qid, v))
        if v["y"] and (best is None or v["prog"] > best[0]):
            best = (v["prog"], qid, U_np, j)
    if viz is not None:
        segsK = GR.di_rollout_batch(st, Ucand.detach().cpu().numpy(), env.dt).astype(np.float16)
        viz.append(dict(t=t, gamma=g, state=st.copy(), segsK=segsK,
                        drawn=[d[0] for d in dres], y=[(-1 if d[1] is None else d[1]) for d in dres],
                        sel=(best[3] if best is not None else -1),
                        sig_q=[float(q) for q in np.quantile(sig.numpy(), [0.1, 0.5, 0.9])],
                        sigB_q=[float(q) for q in np.quantile(sig[drawn].numpy(), [0.1, 0.5, 0.9])],
                        min_margin=float(np.nanmin([d[3]["margin"] for d in dres])
                                         if any(d[1] == 1 for d in dres) else np.nan)))
    stats = dict(ess=ess, ent=ent, uplift=uplift, n_err=n_err,
                 n_pos=sum(1 for d in dres if d[1] == 1), n_drawn=len(dres),
                 sig_all=[float(q) for q in np.quantile(sig.numpy(), [0.1, 0.5, 0.9])],
                 sig_sel=[float(q) for q in np.quantile(sig[drawn].numpy(), [0.1, 0.5, 0.9])])
    return best, stats


def run_episode(policy, blr, env, cfg, g, store, round_i, ep, device, collect=True, viz=None,
                rollout_seed=None):
    """One expert-free shielded episode at gamma g. Ends on reach / NO_VERIFIED_POSITIVE / timeout.
    Executing only certified first actions => dead (collision/OOB) should be ~impossible; counted."""
    if rollout_seed is not None:
        torch.manual_seed(rollout_seed)
        np.random.seed(rollout_seed % (2 ** 31))
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal_np = env.goal.detach().cpu().numpy()
    st = env.x0.detach().cpu().numpy().astype(np.float32).copy()
    hist, path = [], [st[:2].copy()]
    clear_min = float("inf")
    step_stats = []
    status, term_t = "timeout", None
    for t in range(cfg.T):
        best, stats = acquire_and_execute(policy, blr, env, cfg, st, hist, g, store,
                                          round_i, ep, t, device, collect=collect, viz=viz)
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
        clear_min = min(clear_min, float((np.linalg.norm(st[:2][None] - obs[:, :2], axis=1)
                                          - obs[:, 2] - rr).min()))
        if np.linalg.norm(st[:2] - goal_np) < cfg.reach:
            status, term_t = "reached", t + 1
            break
        if (st[:2] < -GM.EPS_TASK).any() or (st[:2] > GM.GRID_M + GM.EPS_TASK).any() or clear_min < 0:
            status, term_t = "dead", t + 1
            break
    return dict(gamma=g, path=np.asarray(path, np.float32), status=status, term_t=term_t,
                steps=len(path) - 1, clear_min=(clear_min if np.isfinite(clear_min) else np.nan),
                step_stats=step_stats)


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
            seed = 20260716000 + int(round(g * 100)) * 1000 + m   # fixed-index: same every round
            r = run_episode(policy, blr, env, cfg, float(g), dummy, round_i, -1, device,
                            collect=False, viz=None, rollout_seed=seed)
            recs.append(r)
        n = len(recs)
        rows[str(g)] = dict(
            SR=sum(r["status"] == "reached" for r in recs) / n,
            CR=sum(r["status"] == "dead" for r in recs) / n,
            NVP=sum(r["status"] == "nvp" for r in recs) / n,
            TO=sum(r["status"] == "timeout" for r in recs) / n,
            clear=float(np.nanmean([r["clear_min"] for r in recs])),
            time=float(np.mean([r["steps"] * env.dt for r in recs if r["status"] == "reached"])
                       if any(r["status"] == "reached" for r in recs) else np.nan),
            nvp_t=[int(r["term_t"]) for r in recs if r["status"] == "nvp"])
    pooled = dict(SR=float(np.mean([v["SR"] for v in rows.values()])),
                  CR=float(np.mean([v["CR"] for v in rows.values()])),
                  NVP=float(np.mean([v["NVP"] for v in rows.values()])))
    return rows, pooled


# ------------------------------------------------------------------ beta calibration
def calibrate_beta(policy, env, cfg, device, betas=(0.01, 0.02, 0.05), log=print):
    """Dry representative round-0 pass (1 episode per gamma, sequential A from empty, current rep,
    expert-free execution); pick the fixed beta whose median ESS/K over all-K pools is in
    [0.25, 0.5]. Data discarded. Never selects on the absolute magnitude of sigma."""
    store = AC.DStore()
    blr = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    sig_pools = []

    class _Tap(list):
        pass
    viz = _Tap()
    for ep, g in enumerate(cfg.gammas):
        r = run_episode(policy, blr, env, cfg, float(g), store, 0, ep, device, collect=True, viz=viz)
        log(f"[calib] dry ep gamma {g}: {r['status']} @{r['steps']} steps")
    for v in viz:
        sig_pools.append(np.array(v["sig_q"]))       # quantiles kept for the record
    # exact per-step ESS per candidate beta from the recorded pools is not reconstructable from
    # quantiles; recompute on a second short pass storing raw sigma vectors:
    blr2 = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    store2 = AC.DStore()
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
            Ucand = policy.sample_window(gT, lT, hT, n=cfg.K, temp=cfg.temp, nfe=cfg.nfe)
            Z = AC.frozen_feat(policy, Ucand, gT, lT, hT, s=cfg.s)
            sig = blr2.sigma(Z)
            raw_sigs.append(sig.numpy().copy())
            best = None
            drawn = torch.multinomial(torch.full((cfg.K,), 1.0 / cfg.K), cfg.B,
                                      replacement=False).tolist()   # neutral draw for calibration
            for j in drawn:
                U_np = Ucand[j].detach().cpu().numpy()
                v = AC.verify_plan(st, U_np, env, float(g), goal_np, n_theta=cfg.n_theta)
                if v["reason"] == "socp_error":
                    continue
                blr2.update(Z[j:j + 1])
                if v["y"] and (best is None or v["prog"] > best[0]):
                    best = (v["prog"], U_np)
            if best is None:
                break
            st = di_step(st, np.asarray(best[1][0], np.float32), dt=env.dt)
            hist.append(np.asarray(best[1][0], np.float32))
            if np.linalg.norm(st[:2] - goal_np) < cfg.reach:
                break
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
    ok = [b for b in betas if 0.25 <= table[b]["ess_med"] <= 0.5]
    pick = ok[0] if ok else min(betas, key=lambda b: abs(table[b]["ess_med"] - 0.375))
    log(f"[calib] chosen beta = {pick} (band [0.25,0.5]; {len(raw_sigs)} pools over 7 gammas)")
    return pick, table, len(raw_sigs)


# ------------------------------------------------------------------ run
def run_afe2(policy, env, cfg, device, outdir, log=print):
    os.makedirs(outdir, exist_ok=True)
    t_start = time.time()
    store = AC.DStore()
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(policy.parameters(),
                           lr=(cfg.prox_lr if cfg.arm == "prox" else cfg.afe_lr))
    audit_ctxs = AC.build_audit_contexts(env, cfg.gammas, n_pos=cfg.audit_pos)
    probe0 = rep_probe_build(policy, env, cfg, device)
    goal_np = env.goal.detach().cpu().numpy()
    recipe = dict(algorithm="afe2_corrected_two_arm_2026_07_16b", arm=cfg.arm,
                  representation="EVOLVING phi_s^(n) (init pretrained); A rebuilt from the full "
                                 "archive at every round start; sequential updates within round",
                  execution="argmax progress among SOCP-positive queried plans; "
                            "NO_VERIFIED_POSITIVE terminates; NO expert/fallback anywhere",
                  socp_error="updates nothing (not stored, no A update)",
                  K=cfg.K, B=cfg.B, beta=cfg.beta, lam=cfg.lam, s=cfg.s,
                  update=("prox: lr %g eta %g stop fstep>=%g or %d" %
                          (cfg.prox_lr, cfg.prox_eta, cfg.prox_fstep_stop, cfg.prox_max_inner)
                          if cfg.arm == "prox" else
                          "afe: lr %g, %d steps, no prox" % (cfg.afe_lr, cfg.afe_steps)),
                  batch=cfg.batch, rounds=cfg.rounds, gamma_sweep="all 7 every round, fixed order",
                  T=cfg.T, reach=cfg.reach, M_eval=cfg.M_eval, seed=cfg.seed,
                  no_curriculum=True, no_anchor=True, no_collapse_rollback=True)
    with open(os.path.join(outdir, "recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2)

    def probe_write(rec):
        with open(os.path.join(outdir, "probe.jsonl"), "a") as f:
            f.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v)
                                for k, v in rec.items()}) + "\n")

    # ---- round 0: pretrained baseline, both evaluation modes
    blr0 = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    audit0 = AC.run_audit(policy, audit_ctxs, env, goal_np, device, n_plans=cfg.audit_plans,
                          nfe=cfg.nfe, n_theta=cfg.n_theta, seed=cfg.seed)
    rows0, pooled0 = controller_eval(policy, blr0, env, cfg, device, 0)
    log(f"[{cfg.arm}] round000 BASELINE V {audit0['V']:.3f} (adv {audit0.get('V_adverse', float('nan')):.3f}) | "
        f"ctrl SR {pooled0['SR']:.2f} CR {pooled0['CR']:.2f} NVP {pooled0['NVP']:.2f}", flush=True)
    probe_write(dict(round=0, arm=cfg.arm, V=audit0["V"], V_gamma=audit0["V_gamma"],
                     V_adverse=audit0.get("V_adverse"), V_gamma_adverse=audit0.get("V_gamma_adverse"),
                     ctrl=rows0, ctrl_pooled=pooled0, n_D=0, n_Dpos=0, rep_cos=1.0))
    HT._save_hp_atomic(policy, os.path.join(outdir, "ckpt_0.pt"),
                       extra={"iter": 0, "recipe": recipe, "resumable": False})

    for n in range(1, cfg.rounds + 1):
        t0 = time.time()
        blr, spec = rebuild_A(policy, store, cfg, device)          # rebuild under phi^(n)
        policy.eval()                                              # theta/phi FIXED during gather
        viz = []
        eps = []
        q_start = len(store)
        for ep, g in enumerate(cfg.gammas):                        # complete fixed sweep
            eps.append(run_episode(policy, blr, env, cfg, float(g), store, n, ep, device,
                                   collect=True, viz=viz))
        t_gather = time.time() - t0
        # per-gamma gather stats
        per_g = {}
        for g, r in zip(cfg.gammas, eps):
            ss = r["step_stats"]
            per_g[str(g)] = dict(status=r["status"], steps=r["steps"], term_t=r["term_t"],
                                 clear=r["clear_min"],
                                 n_q=sum(s["n_drawn"] for s in ss),
                                 n_pos=sum(s["n_pos"] for s in ss),
                                 n_err=sum(s["n_err"] for s in ss))
        all_ss = [s for r in eps for s in r["step_stats"]]
        t0 = time.time()
        upd = update_round(policy, opt, store, cfg, device, rng)
        t_upd = time.time() - t0
        audit = AC.run_audit(policy, audit_ctxs, env, goal_np, device, n_plans=cfg.audit_plans,
                             nfe=cfg.nfe, n_theta=cfg.n_theta, seed=cfg.seed)
        blr_eval, _ = rebuild_A(policy, store, cfg, device)        # eval uses post-update rep
        rows, pooled = controller_eval(policy, blr_eval, env, cfg, device, n)
        drawn = (upd or {}).get("drawn_ids", {})
        tr_gamma = {}
        for q in drawn:
            gq = str(round(store.q_gamma[q], 2))
            tr_gamma[gq] = tr_gamma.get(gq, 0) + 1
        pos_prog = np.asarray([store.q_prog[q] for q in store.pos_ids], float)
        rec = dict(round=n, arm=cfg.arm, n_D=len(store), n_Dpos=store.n_pos(),
                   per_gamma=per_g,
                   ess_med=float(np.median([s["ess"] for s in all_ss])) / cfg.K,
                   ent_med=float(np.median([s["ent"] for s in all_ss])),
                   uplift_med=float(np.median([s["uplift"] for s in all_ss])),
                   sig_all_med=float(np.median([s["sig_all"][1] for s in all_ss])),
                   sig_sel_med=float(np.median([s["sig_sel"][1] for s in all_ss])),
                   A_n=spec["n"], A_eff_rank=spec["eff_rank"], A_eig_top=spec["eig_top"],
                   A_eig_med=spec["eig_med"], rep_cos=rep_cos_drift(policy, probe0, cfg),
                   dither_cum=float((pos_prog < cfg.dither_bar).mean()) if pos_prog.size else None,
                   V=audit["V"], V_gamma=audit["V_gamma"], V_adverse=audit.get("V_adverse"),
                   V_gamma_adverse=audit.get("V_gamma_adverse"),
                   ctrl=rows, ctrl_pooled=pooled, trained_gamma=tr_gamma,
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
                        A_inv=blr.A_inv.clone(), audit=audit,
                        train_ids=np.asarray(sorted(drawn.keys()), np.int64),
                        train_counts=np.asarray([drawn[k] for k in sorted(drawn)], np.int64),
                        goal=goal_np, x0=env.x0.detach().cpu().numpy()),
                   os.path.join(outdir, "viz_db", f"round{n}.pt")
                   if os.path.isdir(os.path.join(outdir, "viz_db")) or
                   (os.makedirs(os.path.join(outdir, "viz_db"), exist_ok=True) or True)
                   else None)
        HT._save_hp_atomic(policy, os.path.join(outdir, f"ckpt_{n}.pt"),
                           extra={"iter": n, "recipe": recipe, "resumable": False})
        log(f"[{cfg.arm}] round{n:03d} D {len(store)} D+ {store.n_pos()} | "
            f"ESS/K {rec['ess_med']:.2f} uplift {rec['uplift_med']:.3f} effR {spec['eff_rank']:.1f} "
            f"cos {rec['rep_cos']:.3f} | upd {0 if upd is None else upd['steps']}st "
            f"fstep {rec.get('fstep_final', 0):.3f} | V {audit['V']:.3f} | "
            f"ctrl SR {pooled['SR']:.2f} NVP {pooled['NVP']:.2f} | {t_gather:.0f}s+{t_upd:.0f}s",
            flush=True)
        if (time.time() - t_start) / 3600 > cfg.max_hours:
            log("max_hours reached", flush=True)
            break
    HT._save_hp_atomic(policy, os.path.join(outdir, "final.pt"),
                       extra={"iter": n, "recipe": recipe, "resumable": False})
    store.save(os.path.join(outdir, "dstore.pt"))
    log(f"[{cfg.arm}] DONE {n} rounds", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", choices=["prox", "afe"], default="prox")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--beta", type=float, default=None, help="fixed beta from --calibrate")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--M-eval", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--afe-steps", type=int, default=250)
    ap.add_argument("--afe-lr", type=float, default=1e-4)
    ap.add_argument("--prox-lr", type=float, default=2e-5)
    ap.add_argument("--prox-eta", type=float, default=0.01)
    ap.add_argument("--wall-plugs", type=int, default=8)
    ap.add_argument("--start-eps", type=float, default=0.3)
    ap.add_argument("--goal-xy", type=float, nargs=2, default=[4.7, 4.7])
    ap.add_argument("--seed", type=int, default=910)
    ap.add_argument("--calibrate", action="store_true", help="ESS beta calibration only")
    args = ap.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ck = HP.load_hp(args.ckpt, device="cpu")
    policy = policy.to(dev)
    env = GS.make_grid()
    env = HT._apply_wall_plugs(env, args.wall_plugs)
    env.x0 = torch.tensor([args.start_eps, args.start_eps, 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=env.goal.dtype)
    GM2.GOAL_XY = np.array([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=float)
    cfg = AFE2Config(rounds=args.rounds, K=args.K, B=args.B, lam=args.lam, T=args.T,
                     reach=args.reach, M_eval=args.M_eval, arm=args.arm, batch=args.batch,
                     afe_steps=args.afe_steps, afe_lr=args.afe_lr, prox_lr=args.prox_lr,
                     prox_eta=args.prox_eta, wall_plugs=args.wall_plugs,
                     start_eps=args.start_eps, goal_xy=tuple(args.goal_xy), seed=args.seed)
    if args.beta is not None:
        cfg.beta = args.beta
    print(f"[afe2] arm {cfg.arm} K{cfg.K} B{cfg.B} beta {cfg.beta} lam {cfg.lam} "
          f"EVOLVING-rep rebuild-A expert-free", flush=True)
    if args.calibrate:
        pick, table, npools = calibrate_beta(policy, env, cfg, dev)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "beta_calibration.json"), "w") as f:
            json.dump(dict(chosen=pick, table={str(k): v for k, v in table.items()},
                           n_pools=npools), f, indent=2)
        return
    run_afe2(policy, env, cfg, dev, args.outdir)


if __name__ == "__main__":
    main()
