"""Core objects of the minimal AFE-style Safe Flow Expansion (user redesign 2026-07-16).

One object identity: the PLANNED action window U_t = (u_{t|t},...,u_{t+9|t}) with its context
c_t = (grid, low5, hist) is the single thing that is (1) sampled from the flow, (2) uncertainty-scored,
(3) FULLY verified (whole predicted DI rollout in-bounds AND SOCP-certified at its gamma) BEFORE
execution, (4) stored in D_n / A_n regardless of label, and (5) replayed for training when positive.
Safety margin m and progress r are stored SEPARATELY; progress is never part of the safety label y.

Pieces here:
  BLRSigma        cumulative A_n = I + lam^-1 sum z z^T over every fully-verified query (pos+neg),
                  on the FROZEN pretrained representation z = phi_s^0/||.|| (32-d); sigma^2 = z^T A_n^-1 z.
                  Rank-1 Sherman-Morrison in float64. Empty => sigma == 1. Monotone (no eviction).
  DStore          append-only query store, normalized by control step (B queries share one context).
  verify_plan     the deterministic full verifier on ONE planned window (in-bounds + SOCP + m, r).
  SafeMPPIFallback certified backup controller (the SafeMPPI expert generator, one plan per step).
  build_audit_contexts / run_audit
                  fixed held-out rho_eval (positions x gammas, zero vel, empty hist); UNTILTED plan
                  samples, fully verified -> per-gamma V_hat and V_hat^prog. Never added to buffers.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # codex_overnight/
_REV = os.path.dirname(_HERE)                               # rev_expansion/
_WORK = os.path.dirname(_REV)                               # overnight_run_07_06/
sys.path.insert(0, _WORK)                                   # shared grid code
sys.path.insert(0, _REV)                                    # rev_expansion helpers
sys.path.insert(0, _HERE)                                   # local copies ALWAYS win (grid_metrics2!)

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_scene as GS


# ------------------------------------------------------------------ uncertainty
class BLRSigma:
    """sigma_n(U,c) from the cumulative 32x32 design matrix on the frozen phi^0 feature.

    A_n = I + lam^-1 sum_i z_i z_i^T over every window actually submitted to the full verifier
    (positive or negative).  sigma^2 = z^T A_n^-1 z, matching the linear-kernel GP posterior std
    (verified == BLR to 5e-7 previously).  No cap, no eviction, no decimation: variance reduction
    is monotone by construction.
    """

    def __init__(self, dim=32, lam=1e-2):
        self.dim = int(dim)
        self.lam = float(lam)
        self.A_inv = torch.eye(self.dim, dtype=torch.float64)
        self.n = 0

    @torch.no_grad()
    def sigma(self, Z):
        """Z [B,dim] (normalized, any device/dtype) -> posterior std [B] float32 (cpu)."""
        Zd = Z.detach().to("cpu", torch.float64)
        q = torch.einsum("bi,ij,bj->b", Zd, self.A_inv, Zd).clamp_min(0.0)
        return q.sqrt().to(torch.float32)

    @torch.no_grad()
    def update(self, Z):
        """Rank-1 Sherman-Morrison for each row of Z: A += lam^-1 z z^T."""
        for z in Z.detach().to("cpu", torch.float64):
            Av = self.A_inv @ z
            self.A_inv -= torch.outer(Av, Av) / (self.lam + float(z @ Av))
            self.n += 1

    def state_dict(self):
        return dict(A_inv=self.A_inv.clone(), n=int(self.n), lam=self.lam, dim=self.dim)

    def load_state_dict(self, st):
        self.A_inv = st["A_inv"].clone()
        self.n = int(st["n"])
        self.lam = float(st["lam"])
        self.dim = int(st["dim"])


@torch.no_grad()
def frozen_feat(phi0, U, gT, lT, hT, s=0.9):
    """Normalized frozen-representation feature z = phi_s^0(U,c)/||.|| -> [B,32]."""
    f = phi0.phi_s_at(U, gT, lT, hT, s=s)
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-9)


# ------------------------------------------------------------------ verifier
def verify_plan(state4, U_np, env, gamma, goal_np, n_theta=180):
    """Full deterministic verifier on ONE planned window, called BEFORE execution.

    y = 1  iff the whole predicted DI rollout of the plan is in task space AND SOCP-certified at
    this gamma (alpha_t = (1-gamma)^t).  Progress r = d0 - dH (net approach over the plan) and the
    SOCP face margin m are returned separately and are NOT part of y.
    """
    st = np.asarray(state4, dtype=np.float32)
    seg = GR.window_positions(st, U_np, env.dt)
    d = np.linalg.norm(np.vstack([st[:2][None], seg]) - goal_np[None], axis=1)
    prog = float(d[0] - d[-1])
    d0 = float(d[0])
    if not GM.in_taskspace(seg):
        return dict(y=0, margin=float("nan"), resid=float("nan"), prog=prog, d0=d0, reason="oob")
    try:
        ok, margin, resid = GM2.window_socp_stats(st, U_np, env, float(gamma), n_theta=n_theta)
    except RuntimeError:
        # Defined-but-conservative behavior on verifier edge cases: count as unsafe, never crash
        # an overnight run. The counter surfaces in probe.jsonl so a systematic issue is visible.
        return dict(y=0, margin=float("nan"), resid=float("nan"), prog=prog, d0=d0,
                    reason="socp_error")
    return dict(y=int(bool(ok)), margin=float(margin), resid=float(resid), prog=prog, d0=d0,
                reason=("ok" if ok else "socp_fail"))


def prog_bar_ok(prog, d0, delta=GM2.DELTA_PROG):
    """The calibrated net-progress bar used by V_hat^prog: r >= min(delta, 0.5 d0)."""
    return bool(prog >= min(delta, 0.5 * d0))


# ------------------------------------------------------------------ query store
class DStore:
    """Append-only D_n = {(c_i, U_i, y_i, m_i, r_i)} over fully-verified planned windows.

    Normalized by control step: the B queries of one step share one stored context (grid kept as the
    H_P channel only, float16 -- the ONLY grid channel this model's ctx_from reads; reconstruction
    is exact for training).  Positives never leave; replay is uniform over the cumulative D+.
    """

    def __init__(self):
        # per-step context tables
        self.ctx_state = []      # np float32 [4]
        self.ctx_hp = []         # np float16 [1,32,32]  (grid channel 2)
        self.ctx_low5 = []       # np float32 [5]
        self.ctx_hist = []       # np float16 [K_HIST,2]
        self.ctx_meta = []       # (round, episode, t)
        # per-query tables
        self.q_sid = []          # context row of each query
        self.q_U = []            # np float32 [H,2]
        self.q_y = []
        self.q_margin = []
        self.q_resid = []
        self.q_prog = []
        self.q_d0 = []
        self.q_sigma = []
        self.q_gamma = []
        self.q_round = []
        self.q_exec = []         # 1 if this plan's first action was executed
        self.q_seg = []          # np float16 [H,2] planned positions (viz)
        self.pos_ids = []        # indices into the query tables with y==1

    def add_step_ctx(self, state4, grid_np, low5_np, hist_np, meta):
        self.ctx_state.append(np.asarray(state4, np.float32).copy())
        self.ctx_hp.append(np.asarray(grid_np[2:3], np.float16).copy())
        self.ctx_low5.append(np.asarray(low5_np, np.float32).copy())
        self.ctx_hist.append(np.asarray(hist_np, np.float16).copy())
        self.ctx_meta.append(tuple(meta))
        return len(self.ctx_state) - 1

    def add_query(self, sid, U_np, v, sigma, gamma, round_i, seg):
        qid = len(self.q_sid)
        self.q_sid.append(int(sid))
        self.q_U.append(np.asarray(U_np, np.float32).copy())
        self.q_y.append(int(v["y"]))
        self.q_margin.append(float(v["margin"]))
        self.q_resid.append(float(v["resid"]))
        self.q_prog.append(float(v["prog"]))
        self.q_d0.append(float(v["d0"]))
        self.q_sigma.append(float(sigma))
        self.q_gamma.append(float(gamma))
        self.q_round.append(int(round_i))
        self.q_exec.append(0)
        self.q_seg.append(np.asarray(seg, np.float16).copy())
        if v["y"]:
            self.pos_ids.append(qid)
        return qid

    def mark_executed(self, qid):
        self.q_exec[qid] = 1

    def __len__(self):
        return len(self.q_sid)

    def n_pos(self):
        return len(self.pos_ids)

    def grid3_of(self, sids):
        """Reconstruct [B,3,32,32] float32 grids (channels 0/1 zero; the model reads only ch2)."""
        hp = torch.stack([torch.from_numpy(self.ctx_hp[s].astype(np.float32)) for s in sids])
        B = hp.shape[0]
        g = torch.zeros(B, 3, hp.shape[2], hp.shape[3], dtype=torch.float32)
        g[:, 2:3] = hp
        return g

    def sample_pos(self, nb, rng):
        """Uniform-with-replacement draw over the CUMULATIVE D+ -> (G,L,H,U) cpu tensors + qids."""
        if not self.pos_ids:
            return None
        ids = [self.pos_ids[i] for i in rng.integers(0, len(self.pos_ids), nb)]
        sids = [self.q_sid[q] for q in ids]
        G = self.grid3_of(sids)
        L = torch.stack([torch.from_numpy(self.ctx_low5[s]) for s in sids])
        H = torch.stack([torch.from_numpy(self.ctx_hist[s].astype(np.float32)) for s in sids])
        U = torch.stack([torch.from_numpy(self.q_U[q]) for q in ids])
        return G, L, H, U, ids

    def round_slice(self, round_i):
        """Query ids belonging to one round (for per-round stats/viz)."""
        return [q for q in range(len(self.q_sid)) if self.q_round[q] == round_i]

    def save(self, path):
        torch.save(dict(
            ctx_state=np.stack(self.ctx_state) if self.ctx_state else np.zeros((0, 4), np.float32),
            ctx_hp=np.stack(self.ctx_hp) if self.ctx_hp else np.zeros((0, 1, 32, 32), np.float16),
            ctx_low5=np.stack(self.ctx_low5) if self.ctx_low5 else np.zeros((0, 5), np.float32),
            ctx_hist=np.stack(self.ctx_hist) if self.ctx_hist else np.zeros((0, GF.K_HIST, 2), np.float16),
            ctx_meta=np.asarray(self.ctx_meta, np.int32) if self.ctx_meta else np.zeros((0, 3), np.int32),
            q_sid=np.asarray(self.q_sid, np.int64), q_U=np.stack(self.q_U) if self.q_U else np.zeros((0, GF.H_PRED, 2), np.float32),
            q_y=np.asarray(self.q_y, np.int8), q_margin=np.asarray(self.q_margin, np.float32),
            q_resid=np.asarray(self.q_resid, np.float32), q_prog=np.asarray(self.q_prog, np.float32),
            q_d0=np.asarray(self.q_d0, np.float32), q_sigma=np.asarray(self.q_sigma, np.float32),
            q_gamma=np.asarray(self.q_gamma, np.float32), q_round=np.asarray(self.q_round, np.int32),
            q_exec=np.asarray(self.q_exec, np.int8),
            q_seg=np.stack(self.q_seg) if self.q_seg else np.zeros((0, GF.H_PRED, 2), np.float16),
        ), path)


# ------------------------------------------------------------------ certified fallback
class SafeMPPIFallback:
    """Certified SafeMPPI backup: when no drawn plan verifies safe, plan ONE action with the same
    SafeMPPI controller that generated the pretraining demos (mode1 config, walls included via
    planner_obstacles).  Fresh adapter per episode (its internal warm start is trajectory-local)."""

    def __init__(self, env):
        self.cfg = GS.mode1_config()
        self.goal_t = env.goal.detach().cpu().float()
        self.obs_plan = GS.planner_obstacles(env)
        self.ad = None

    def reset(self):
        from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
        self.ad = SafeMPPIAdapter(**self.cfg)

    def plan(self, state4, gamma, seed):
        a, _ = self.ad.plan(torch.tensor(np.asarray(state4, np.float32)), self.goal_t,
                            self.obs_plan, gamma=float(gamma), seed=int(seed))
        return a.detach().cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------ audit (rho_eval)
def build_audit_contexts(env, gammas, n_pos=12, seed=20260716, min_clear=0.05, v_adverse=0.65):
    """Fixed held-out rho_eval: n_pos free-space positions (position 0 = the episode start), each in
    TWO velocity conditions -- rest (v=0) and ADVERSE (moving at v_adverse toward the nearest
    obstacle: where window certification is actually hard; rest-only audits sit at ~99% ceiling) --
    crossed with every gamma, empty history.  Fixed across the whole run and across arms/seeds;
    audit samples are NEVER added to D_n or A_n."""
    rng = np.random.default_rng(seed)
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    x0 = env.x0.detach().cpu().numpy()
    pos = [np.array([x0[0], x0[1]], np.float32)]
    while len(pos) < n_pos:
        p = rng.uniform(0.1, 4.9, 2).astype(np.float32)
        clr = (np.linalg.norm(p[None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min()
        if clr > min_clear:
            pos.append(p)
    goal_np = env.goal.detach().cpu().numpy()
    ctxs = []
    for pi, p in enumerate(pos):
        j = int(np.argmin(np.linalg.norm(obs[:, :2] - p[None], axis=1) - obs[:, 2]))
        d = obs[j, :2] - p
        d = d / (np.linalg.norm(d) + 1e-9)
        for vk, v in (("rest", np.zeros(2, np.float32)),
                      ("adverse", (v_adverse * d).astype(np.float32))):
            st = np.array([p[0], p[1], v[0], v[1]], np.float32)
            grid_np = GF.axis_grid(st[:2], obs, rr)
            h_np = GF.hist_pad(np.zeros((0, 2)), GF.K_HIST)
            for g in gammas:
                ctxs.append(dict(pos_id=pi, vel=vk, state=st, gamma=float(g), grid=grid_np,
                                 low5=GF.low5(st, goal_np, float(g)), hist=h_np))
    return ctxs


@torch.no_grad()
def run_audit(policy, ctxs, env, goal_np, device, n_plans=4, nfe=8, n_theta=180, seed=0):
    """UNTILTED audit: for each rho_eval context sample n_plans plans from the CURRENT flow at
    temp=1 and fully verify each.  Returns per-gamma V_hat (mean y), V_hat^prog (y AND calibrated
    net-progress bar), pooled numbers, and the per-context matrix for viz."""
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)   # isolate audit RNG from training
    cpu_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    per = []
    try:
        for c in ctxs:
            gT = torch.tensor(c["grid"], device=device)
            lT = torch.tensor(c["low5"], device=device)
            hT = torch.tensor(c["hist"], device=device)
            U = policy.sample_window(gT, lT, hT, n=n_plans, temp=1.0, nfe=nfe)
            ys, ps = [], []
            for j in range(U.shape[0]):
                v = verify_plan(c["state"], U[j].detach().cpu().numpy(), env, c["gamma"], goal_np,
                                n_theta=n_theta)
                ys.append(v["y"])
                ps.append(int(v["y"] and prog_bar_ok(v["prog"], v["d0"])))
            per.append(dict(pos_id=c["pos_id"], vel=c.get("vel", "rest"), gamma=c["gamma"],
                            V=float(np.mean(ys)), Vprog=float(np.mean(ps))))
    finally:
        torch.random.set_rng_state(cpu_state)
        del g_cpu
    out = dict(per=per)
    gammas = sorted({p["gamma"] for p in per})
    out["V_gamma"] = {str(g): float(np.mean([p["V"] for p in per if p["gamma"] == g])) for g in gammas}
    out["Vprog_gamma"] = {str(g): float(np.mean([p["Vprog"] for p in per if p["gamma"] == g]))
                          for g in gammas}
    out["V"] = float(np.mean([p["V"] for p in per]))
    out["Vprog"] = float(np.mean([p["Vprog"] for p in per]))
    for vk in ("rest", "adverse"):
        sub = [p for p in per if p.get("vel", "rest") == vk]
        if sub:
            out[f"V_{vk}"] = float(np.mean([p["V"] for p in sub]))
            out[f"V_gamma_{vk}"] = {str(g): float(np.mean([p["V"] for p in sub if p["gamma"] == g]))
                                    for g in gammas}
    return out
