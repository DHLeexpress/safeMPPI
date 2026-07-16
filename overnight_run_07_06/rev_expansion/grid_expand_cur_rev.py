"""FRESH-ONLY 2-class curriculum safe-flow expansion (rev_expansion, user 2026-07-08).

Copy of overnight_run_07_06/grid_expand_cur.py, redesigned:
  - NO persistent positive buffer / no old pile. Every outer iter gathers K *valid* fresh rollouts and the
    update trains on THOSE windows only (batch composition fully known & controllable — paper-clean).
  - Per-window validity (from the FIXED valid2, net-progress only): taskspace ∧ SOCP(traj) ∧ net-progress≥0.10.
  - 2 classes (drop mid): frontier = high-σ OR low-margin OR HIGH net-progress (≥ prog_floor); easy = the rest.
  - VALIDITY floor on net-progress (--valid-prog-floor) REJECTS safe-stationary windows before they are gathered:
    a SOCP-safe but barely-moving window (prog ~0.1) is safe-not-performant; training on it teaches "stay put"
    (→ CR≈0 but SR≪1, the it600 origin-collapse death-spiral). Rejecting them at the gate breaks that spiral.
    (valid but gentle). σ = GP novelty vs a rolling query buffer.
  - Dynamic batch honoring the easy:frontier mix, availability-capped (e.g. 10 easy → 4 frontier at 7:3 → 14),
    replace=True within a class; + ~demo_frac demo windows. 0 valid ⇒ skip the update.
  - inner-steps 1 / 2 / 1 (early/mid/cool) to guard gradient blow-up on tiny batches. No warm-up gate.
  - viz_db (labels+scores) saved every viz_db_every (=100) iters.

Path shim: local (edited) grid_metrics2 wins; everything else from the parent overnight_run_07_06.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # rev_expansion/
_PARENT = os.path.dirname(_HERE)                            # overnight_run_07_06/
sys.path.insert(0, _PARENT)                                 # _paths, grid_rollout, grid_expand, ... from parent
sys.path.insert(0, _HERE)                                   # local edited grid_metrics2 wins

import argparse
import json
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
import grid_rollout as GR
import grid_expand as GE
import grid_expand2 as GX2          # state_from_low5
import grid_metrics as GM
import grid_metrics2 as GM2         # local COPY with the net-progress-only approach_ok fix
import grid_hp_expt as HP
from uncertainty import GPUncertainty
import sr_cr_eval as SR


@dataclass
class CurConfig:
    iters: int = 1000
    # exploration (σ-tilt)
    N: int = 64
    temp: float = 1.0
    s: float = 0.9
    churn: float = 0.05
    nfe_explore: int = 6
    safe_filter: bool = True
    # GP σ estimator
    kernel: str = "rbf"
    ell: float = 0.2
    lam: float = 1e-2
    gp_buf: int = 384
    qbuf_cap: int = 500
    # FRESH-ONLY curriculum
    rollouts_per_iter: int = 10     # K valid rollouts gathered per outer iter (phase-scaled ⌈K/2⌉ early/cool)
    prog_floor: float = 0.3         # frontier if window net-progress (d0-dH) >= prog_floor
    valid_prog_floor: float = 0.15  # REJECT windows below this net-progress (safe-stationary trap; 0 = off, valid2's 0.10 bar)
    min_rollouts: int = 1           # gather AT LEAST this many valid rollouts (LOCKED recipe = 1; 4 was the failed uni_C knob)
    traj_prog_min: float = 0.0      # dither gate (LOCKED recipe = 0/off; 1.0 was the failed uni_C knob)
    # ---- warm-up noise fixes (user 2026-07-09: noisy near-origin initial windows hammered as easy) ----
    strat_rid: bool = False         # batch draw round-robins across source rollouts (prob #1 at the BATCH level)
    easy_sig_abs: float = 0.0       # ABSOLUTE σ cap: σ >= this can NEVER be easy (quantile split lies when ALL are noisy); 0=off
    easy_demo_backfill: bool = False  # fill the easy shortfall with DEMO windows (true-easy anchor), not by shrinking the batch
    easy_skip_first: int = 0        # windows with in-traj index < this are NEVER easy (the noisy initial escape part)
    probe_escape: int = 0           # every N iters: M faithful rollouts -> origin-escape stability probe (0=off)
    probe_cov: int = 0              # every N iters: M=50 faithful @γ0.5 -> instantaneous SR/CR/staircase-coverage
    log_comp_every: int = 0         # composition/rid-diversity log line every N iters (micro mode: 1; 0=off)
    # ---- pile revival (user 2026-07-09: fresh_frac<1 + bounded-staleness pile + no-GD warm-up) ----
    fresh_frac: float = 1.0         # fresh share of the fresh-part batch; rest drawn from the pile (1.0 = fresh-only)
    warmup_gather: int = 0          # first N iters: gather->pile only, NO gradient step (GP σ warms up too)
    pile_cap: int = 3000            # FIFO cap -> pile holds only the last ~10-20 iters (staleness BOUNDED)
    pile_replace: bool = False      # False = least-recently-used draw: every sample gets a turn before any repeats
    pile_relabel_every: int = 10    # recompute σ-dependent labels of the whole pile every N iters (labels refresh)
    batch_cap: int = 32             # TOTAL batch (demo + fresh); demo = round(demo_frac*batch_cap)
    lr: float = 1e-4
    q_lo: float = 0.33              # low-margin quantile -> frontier
    q_hi: float = 0.67             # high-σ quantile -> frontier
    use_sigma: bool = True          # σ criterion in frontier labeling (--sigma-off disables it, for ablation)
    mix_start: tuple = (0.7, 0.3)   # easy / frontier (2-class)
    mix_end: tuple = (0.5, 0.5)
    beta_steps: tuple = (1.0, 0.5, 0.2, 0.1)
    beta_fracs: tuple = (0.0, 0.25, 0.5, 0.75)
    beta_smooth: str = ""
    beta_hi: float = 1.0
    beta_lo: float = 0.1
    viz_db_every: int = 100
    cooldown_frac: float = 0.75
    cooldown_lr_mult: float = 0.3
    inner_steps: int = 4            # mid-phase inner steps (user: focus on every window via more passes)
    early_inner: int = 2            # early/cooldown = 2
    cooldown_inner: int = 2
    early_frac: float = 0.1
    enc_grad_clip: float = 5.0
    easy_strict: bool = True        # kept for parity (unused in 2-class labeling; sweep passes --easy-strict)
    # measurement (SR/CR primary)
    measure_every: int = 100
    M_measure: int = 25
    reach: float = 0.1
    T: int = 250
    # anchors / misc
    demo_frac: float = 0.0
    lwf_eta: float = 0.0
    demo_cap: int = 1200
    gammas: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    ckpt_every: int = 500
    collapse_frac: float = 0.45
    collapse_patience: int = 3
    collapse_min_iter: int = 600


# ---------------------------------------------------------------- labeling & sampling
def _window_progress(low5, U, env):
    """Net progress d0-dH of a single window (goal = GM2.GOAL_XY), plus the distance array for approach_ok."""
    st = GX2.state_from_low5(low5)
    seg = GR.window_positions(st, U, env.dt)
    pts = np.vstack([np.asarray(st, float)[:2][None, :], seg])
    d = np.linalg.norm(pts - GM2.GOAL_XY[None], axis=1)
    return float(d[0] - d[-1]), pts, d


def _sigma_of(policy, unc, data, cfg, device):
    """GP novelty σ of every window in `data` (chunked); zeros when the GP buffer isn't ready."""
    n = data["U"].shape[0]
    sig = []
    try:
        with torch.no_grad():
            for i in range(0, n, 2048):
                ctx = policy.ctx_from(data["grid"][i:i + 2048].to(device), data["low5"][i:i + 2048].to(device),
                                      data["hist"][i:i + 2048].to(device))
                phi = policy.phi_s(data["U"][i:i + 2048].to(device), ctx, s=cfg.s)
                sig.append(unc.sigma(phi).detach().cpu().numpy())
        return np.concatenate(sig) if sig else np.zeros(n)
    except Exception:
        return np.zeros(n)                                     # GP buffer not ready yet -> σ criterion inert


def _front_mask(sigma, margin, prog, widx, cfg):
    """The ONE frontier rule (shared by fresh labeling and pile relabeling): high-σ OR low-margin OR
    high-progress, plus the absolute-σ and skip-first overrides."""
    n = len(sigma)
    front = np.zeros(n, bool)
    if cfg.use_sigma and sigma.max() - sigma.min() > 1e-6:     # σ criterion (disabled by --sigma-off ablation)
        front |= sigma >= np.quantile(sigma, cfg.q_hi)
    front |= margin <= np.quantile(margin, cfg.q_lo)
    front |= prog >= cfg.prog_floor                            # frontier = high-σ OR low-margin OR high-progress
    if cfg.easy_sig_abs > 0:                                   # ABSOLUTE novelty gate: quantiles force 2/3 of an
        if sigma.max() - sigma.min() <= 1e-9:                  # all-noisy pool into easy — fix the semantics:
            front[:] = True                                    # GP not ready -> novelty unknown -> NOTHING is easy
        else:
            front |= sigma >= cfg.easy_sig_abs                 # genuinely-noisy window can never be easy
    if cfg.easy_skip_first > 0 and widx is not None:           # initial escape windows (noisy, near origin):
        front |= widx < cfg.easy_skip_first                    # frontier (30% share), never hammered as easy
    return front


def label_fresh(policy, unc, fresh, env, cfg, device):
    """2-class labels over THIS iter's fresh valid windows. frontier = high-σ OR low-margin OR net-progress ≥
    prog_floor; easy = the rest. Returns (easy_row_idx, frontier_row_idx, scores dict)."""
    n = fresh["U"].shape[0]
    sigma = _sigma_of(policy, unc, fresh, cfg, device)
    Ln, Un = fresh["low5"].numpy(), fresh["U"].numpy()
    jerk = (np.linalg.norm(np.diff(Un, n=2, axis=1), axis=2).mean(axis=1)
            if Un.shape[1] >= 3 else np.zeros(n))
    net = Un.sum(axis=1); rg = Ln[:, :2]
    mono = (net * rg).sum(1) / (np.linalg.norm(net, axis=1) * np.linalg.norm(rg, axis=1) + 1e-9)
    margin = np.array([GM2.window_min_clearance(GX2.state_from_low5(Ln[j]), Un[j], env) for j in range(n)])
    margin = np.nan_to_num(np.clip(margin, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0)
    prog = fresh["prog"]
    front = _front_mask(sigma, margin, prog, fresh.get("widx"), cfg)
    easy = ~front
    scores = dict(sigma=sigma, margin=margin, jerk=jerk, mono=mono, prog=prog)
    return np.where(easy)[0], np.where(front)[0], scores


def _fresh_batch_plan(n_e, n_f, mix, cap):
    """Largest batch (≤cap) honoring the easy:frontier ratio given availability. e.g. n_e=10,n_f=5 @ 7:3 ->
    B=min(10/.7, 5/.3)=14.3 -> 14 -> (10 easy, 4 frontier). If one class empty, use the other alone."""
    e_frac, f_frac = float(mix[0]), float(mix[1])
    if n_e == 0 and n_f == 0:
        return 0, 0
    if n_e == 0:
        return 0, min(n_f, cap)
    if n_f == 0:
        return min(n_e, cap), 0
    cands = []
    if e_frac > 0:
        cands.append(n_e / e_frac)
    if f_frac > 0:
        cands.append(n_f / f_frac)
    B = min(cands) if cands else (n_e + n_f)
    B = int(min(B, cap))
    if B <= 0:
        return 0, 0
    ne = int(round(e_frac * B)); nf = B - ne
    return ne, nf


def _draw_strat(idx_pool, n, rids):
    """Stratified class draw (prob #1, batch level): round-robin ONE window per source rollout until n drawn,
    so the CFM update sees every gathered trajectory even when one dominates the window count."""
    by = {}
    for i in idx_pool:
        by.setdefault(int(rids[i]), []).append(int(i))
    groups = list(by.values())
    np.random.shuffle(groups)
    out, k = [], 0
    while len(out) < n:
        out.append(int(np.random.choice(groups[k % len(groups)])))
        k += 1
    return np.asarray(out, dtype=int)


class Pile:
    """Persistent positive pile, REVIVED with bounded staleness (user 2026-07-09). Differences vs the old 60k
    pile: (a) FIFO cap ~3k -> holds only the last ~10-20 iters' gathers, never trains on ancient behavior;
    (b) WITHOUT-replacement (least-recently-used) draws -> every sample gets a turn before any repeats
    ('refresh sometimes every samples'); (c) σ-dependent labels RECOMPUTED every pile_relabel_every iters with
    the current policy/GP (margin/prog are geometric -> stored once); windows migrate frontier->easy as the
    policy masters them."""

    def __init__(self, cap):
        self.cap = cap
        self.T = None                    # dict(grid, low5, hist, U) torch tensors
        self.margin = self.prog = self.widx = self.rid = self.it = self.use = None
        self.label = None                # 'easy' / 'frontier' per window

    def __len__(self):
        return 0 if self.T is None else self.T["U"].shape[0]

    def count(self, pool):
        return 0 if self.T is None else int((self.label == pool).sum())

    def add(self, fresh, easy_idx, frontier_idx, scores, t):
        n = fresh["U"].shape[0]
        lab = np.array(["easy"] * n, dtype=object); lab[frontier_idx] = "frontier"
        rid_g = t * 1000 + fresh.get("rid", np.zeros(n, int))   # globally-unique rollout id across iters
        new = dict(grid=fresh["grid"], low5=fresh["low5"], hist=fresh["hist"], U=fresh["U"])
        if self.T is None:
            self.T = {k: v.clone() for k, v in new.items()}
            self.margin = scores["margin"].copy(); self.prog = scores["prog"].copy()
            self.widx = fresh.get("widx", np.zeros(n, int)).copy(); self.rid = np.asarray(rid_g, int)
            self.it = np.full(n, t, int); self.use = np.zeros(n, float); self.label = lab
        else:
            self.T = {k: torch.cat([self.T[k], new[k]]) for k in self.T}
            self.margin = np.concatenate([self.margin, scores["margin"]])
            self.prog = np.concatenate([self.prog, scores["prog"]])
            self.widx = np.concatenate([self.widx, fresh.get("widx", np.zeros(n, int))])
            self.rid = np.concatenate([self.rid, np.asarray(rid_g, int)])
            self.it = np.concatenate([self.it, np.full(n, t, int)])
            self.use = np.concatenate([self.use, np.zeros(n, float)])
            self.label = np.concatenate([self.label, lab])
        if len(self) > self.cap:                                # FIFO: evict the OLDEST windows
            k = len(self) - self.cap
            self.T = {kk: v[k:] for kk, v in self.T.items()}
            for a in ("margin", "prog", "widx", "rid", "it", "use", "label"):
                setattr(self, a, getattr(self, a)[k:])

    def draw(self, pool, n, replace=False):
        idx = np.where(self.label == pool)[0] if self.T is not None else np.array([], int)
        if n <= 0 or len(idx) == 0:
            return np.array([], int)
        if replace:
            return np.random.choice(idx, n, replace=True)
        key = self.use[idx] + np.random.rand(len(idx))          # least-used first, random tie-break
        take = idx[np.argsort(key)[:min(n, len(idx))]]
        self.use[take] += 1.0
        return take

    def relabel(self, policy, unc, cfg, device):
        if self.T is None or len(self) == 0:
            return
        sigma = _sigma_of(policy, unc, self.T, cfg, device)
        front = _front_mask(sigma, self.margin, self.prog, self.widx, cfg)
        self.label = np.array(["easy"] * len(self), dtype=object)
        self.label[front] = "frontier"


def _grad_rms(params):
    vals = [float(p.grad.pow(2).mean()) for p in params if p.grad is not None]
    return float(np.sqrt(np.mean(vals))) if vals else 0.0


def update_flow_fresh(policy, opt, fresh, easy_idx, frontier_idx, mix, n_steps, cfg,
                      field_params, enc_params, device, demo=None, teacher=None, pile=None):
    nd_demo = int(round(cfg.demo_frac * cfg.batch_cap)) if (cfg.demo_frac > 0 and demo is not None) else 0
    fresh_target = cfg.batch_cap - nd_demo                  # fresh part of the TOTAL batch (e.g. 32-8=24)
    n_e, n_f = len(easy_idx), len(frontier_idx)
    pile_on = pile is not None and len(pile) > 0 and (cfg.fresh_frac < 1.0 or fresh is None)
    ne_pl = nf_pl = 0
    if pile_on:                                             # fresh takes its fresh_frac share; the pile fills
        tgt_e = int(round(float(mix[0]) * fresh_target))    # the rest; each backfills the other's shortfall
        tgt_f = fresh_target - tgt_e
        ne_fr = min(n_e, int(round(cfg.fresh_frac * tgt_e)))
        nf_fr = min(n_f, int(round(cfg.fresh_frac * tgt_f)))
        ne_pl = min(pile.count("easy"), tgt_e - ne_fr)
        nf_pl = min(pile.count("frontier"), tgt_f - nf_fr)
        ne_fr = min(n_e, tgt_e - ne_pl); nf_fr = min(n_f, tgt_f - nf_pl)
        ne = ne_fr + ne_pl; nf = nf_fr + nf_pl
        if cfg.easy_demo_backfill and demo is not None:
            nd_demo += fresh_target - ne - nf
    elif cfg.easy_demo_backfill and demo is not None:       # easy shortfall -> DEMO windows (the true-easy
        tgt_e = int(round(float(mix[0]) * fresh_target))    # anchor), NOT a shrunken batch / extra frontier
        tgt_f = fresh_target - tgt_e
        ne = ne_fr = min(n_e, tgt_e); nf = nf_fr = min(n_f, tgt_f)
        nd_demo += fresh_target - ne - nf
    else:
        ne, nf = _fresh_batch_plan(n_e, n_f, mix, fresh_target)
        ne_fr, nf_fr = ne, nf
    B = ne + nf
    if B == 0:
        return None
    nd = demo["U"].shape[0] if demo is not None else 0
    rids = fresh.get("rid") if fresh is not None else None
    policy.train()
    losses, fgr, egr, rid_ns, rid_doms = [], [], [], [], []
    for _ in range(n_steps):
        Gs, Ls, Hs, Us, rid_all = [], [], [], [], []
        parts = []
        if ne_fr > 0:
            parts.append(_draw_strat(easy_idx, ne_fr, rids) if (cfg.strat_rid and rids is not None)
                         else np.random.choice(easy_idx, ne_fr, replace=True))
        if nf_fr > 0:
            parts.append(_draw_strat(frontier_idx, nf_fr, rids) if (cfg.strat_rid and rids is not None)
                         else np.random.choice(frontier_idx, nf_fr, replace=True))
        if parts:
            bi_np = np.concatenate(parts)
            bi = torch.as_tensor(bi_np, dtype=torch.long)
            Gs.append(fresh["grid"][bi]); Ls.append(fresh["low5"][bi])
            Hs.append(fresh["hist"][bi]); Us.append(fresh["U"][bi])
            if rids is not None:
                rid_all.append(rids[bi_np])
        if ne_pl > 0 or nf_pl > 0:                          # pile part: LRU without-replacement draw
            pi_np = np.concatenate([pile.draw("easy", ne_pl, cfg.pile_replace),
                                    pile.draw("frontier", nf_pl, cfg.pile_replace)]).astype(int)
            if len(pi_np):
                pi = torch.as_tensor(pi_np, dtype=torch.long)
                Gs.append(pile.T["grid"][pi]); Ls.append(pile.T["low5"][pi])
                Hs.append(pile.T["hist"][pi]); Us.append(pile.T["U"][pi])
                rid_all.append(pile.rid[pi_np])
        G = torch.cat(Gs).to(device); L = torch.cat(Ls).to(device)
        H = torch.cat(Hs).to(device); U = torch.cat(Us).to(device)
        if rid_all:                                         # rid-diversity stats of THIS update's non-demo part
            _, cnts = np.unique(np.concatenate(rid_all), return_counts=True)
            rid_ns.append(len(cnts)); rid_doms.append(float(cnts.max()) / cnts.sum())
        if nd_demo > 0:                                        # δ anchor: mix pretraining-demo windows in
            di = torch.randint(0, nd, (nd_demo,))
            G = torch.cat([G, demo["grid"][di].to(device)]); L = torch.cat([L, demo["low5"][di].to(device)])
            H = torch.cat([H, demo["hist"][di].to(device)]); U = torch.cat([U, demo["U"][di].to(device)])
        loss = policy.cfm_loss(U, policy.ctx_from(G, L, H))
        if cfg.lwf_eta > 0 and teacher is not None and demo is not None:   # η anchor: LwF on demo contexts
            nl = min(nd, cfg.batch_cap)
            li = torch.randint(0, nd, (nl,))
            Gd, Ld, Hd = demo["grid"][li].to(device), demo["low5"][li].to(device), demo["hist"][li].to(device)
            Ud = demo["U"][li].to(device); B_ = Ud.shape[0]
            x1 = (Ud / policy.u_max).reshape(B_, policy.d); x0 = torch.randn_like(x1)
            tau = torch.rand(B_, device=x1.device).clamp(1e-4, 1.0)
            x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
            v_s = policy.forward(x_tau, tau, policy._expand_ctx(policy.ctx_from(Gd, Ld, Hd), B_))
            with torch.no_grad():
                v_t = teacher.forward(x_tau, tau, teacher._expand_ctx(teacher.ctx_from(Gd, Ld, Hd), B_))
            loss = loss + cfg.lwf_eta * ((v_s - v_t) ** 2).mean()
        opt.zero_grad(); loss.backward()
        fgr.append(_grad_rms(field_params)); egr.append(_grad_rms(enc_params))
        if cfg.enc_grad_clip > 0 and enc_params:
            torch.nn.utils.clip_grad_norm_(enc_params, cfg.enc_grad_clip)
        opt.step(); losses.append(float(loss))
    return dict(loss=float(np.mean(losses)) if losses else float("nan"),
                field_grad_rms=float(np.mean(fgr)) if fgr else 0.0,
                enc_grad_rms=float(np.mean(egr)) if egr else 0.0, batch=(ne, nf, nd_demo),
                n_pile=ne_pl + nf_pl, pile_batch=(ne_pl, nf_pl),
                rid_n=float(np.mean(rid_ns)) if rid_ns else float("nan"),
                rid_dom=float(np.mean(rid_doms)) if rid_doms else float("nan"))


def _gather_fresh(policy, unc, env, cfg, gammas, beta, K, target_e, target_f, qbuf, covered, device):
    """Roll out until the batch's fresh quota (target_e easy + target_f frontier, by the cheap prog proxy) is
    filled — capped at K valid rollouts (attempt cap 2K). Early-stop is the speedup: one rollout already yields
    ~hundreds of valid windows, so ~1-2 rollouts fill a 24-window fresh batch. A window is valid if its
    trajectory is SOCP-safe AND the window is in taskspace AND net-progress ≥ 0.10. proxy: frontier≈prog≥
    prog_floor, easy≈prog<prog_floor (σ/margin only MOVE windows into frontier, so easy proxy is an upper
    bound; the batch is availability-capped so any shortfall is handled)."""
    if qbuf is not None:
        qfeat = GE._buffer_feat(policy, qbuf, "phi_s", cfg.s, cfg.gp_buf, device)
        if qfeat is not None:
            unc.set_buffer(qfeat)
    goal_np = env.goal.detach().cpu().numpy()
    gG, gL, gH, gU, prog, rid, widx = [], [], [], [], [], [], []
    paths = []                                             # executed trajectories of KEPT rollouts (for viz)
    reached, coll = [], []
    valid, att, gi = 0, 0, 0
    K_eff = max(K, cfg.min_rollouts)                       # prob #1: never fewer than min_rollouts behaviors
    max_att = 2 * K_eff
    ne_prox, nf_prox = 0, 0
    while att < max_att:
        if valid >= K_eff:                                 # rollouts cap reached
            break
        if valid >= cfg.min_rollouts and ne_prox >= target_e and nf_prox >= target_f:
            break                                          # quota filled AND enough distinct rollouts
        att += 1
        g = gammas[gi % len(gammas)]; gi += 1
        out = GR.fm_deploy(policy, env, float(g), T=cfg.T,
                           tilt=dict(unc=unc, beta=beta, N=cfg.N, s=cfg.s, broad=0, feature="phi_s",
                                     temp=cfg.temp, churn=cfg.churn, safe_filter=cfg.safe_filter),
                           nfe=cfg.nfe_explore, record=True, verify_fn=GM2.window_label_cheap, device=device)
        reached.append(1.0 if out["reached"] else 0.0)
        coll.append(1.0 if SR.path_collides(out["path"], env) else 0.0)
        if not out["recs"]:
            continue
        if not GM.socp_ok(out["path"], env, float(g)):         # safety gate: SOCP-certified trajectory only
            continue
        pth = np.asarray(out["path"], dtype=float)             # AD-HOC dither gate (prob #2), TWO-TIER:
        d0T = np.linalg.norm(pth[0] - goal_np) - np.linalg.norm(pth[-1] - goal_np)
        if not out["reached"] and cfg.traj_prog_min > 0:       # gate active only when traj_prog_min > 0
            if d0T < 0.3:                                      # tier-1 hard floor: true stay-and-dither -> drop
                continue
            if d0T < cfg.traj_prog_min and valid >= cfg.min_rollouts:
                continue                                       # tier-2 soft: sub-par traj only fills the min quota
        G, L, H, U = GE._to_t(out["recs"])
        keep, wp = [], []
        for i, r in enumerate(out["recs"]):
            p_i, pts, d = _window_progress(r[1], r[3], env)
            if not GM.in_taskspace(pts):
                continue
            if not GM2.approach_ok(d):                          # net-progress ≥ 0.10 (valid2)
                continue
            if p_i < min(cfg.valid_prog_floor, 0.5 * d[0]):     # reject safe-STATIONARY (perf floor; relax near goal)
                continue
            keep.append(i); wp.append(p_i)
        if not keep:
            continue
        ki = torch.as_tensor(keep)
        gG.append(G[ki]); gL.append(L[ki]); gH.append(H[ki]); gU.append(U[ki]); prog.extend(wp)
        rid.extend([valid] * len(keep)); paths.append(pth)  # rollout id per window + executed traj (viz/diversity)
        widx.extend(keep)                                   # in-traj window index (0 = the initial escape window)
        wparr = np.asarray(wp)                              # cheap prog proxy for the early-stop quota
        nf_prox += int((wparr >= cfg.prog_floor).sum()); ne_prox += int((wparr < cfg.prog_floor).sum())
        qbuf = GE._cat(qbuf, G[ki][::3], L[ki][::3], H[ki][::3], U[ki][::3], cap=cfg.qbuf_cap)
        if out["reached"]:                                      # coverage tracking (not a gate)
            sid = GM.staircase_id(out["path"])
            if sid is not None:
                covered[g].add(sid)
        valid += 1
    if not gG:
        return None, qbuf, reached, coll, valid, att
    fresh = dict(grid=torch.cat(gG), low5=torch.cat(gL), hist=torch.cat(gH), U=torch.cat(gU),
                 prog=np.asarray(prog, dtype=float), rid=np.asarray(rid, dtype=int),
                 widx=np.asarray(widx, dtype=int), paths=paths)
    return fresh, qbuf, reached, coll, valid, att


def update_demo_only(policy, opt, cfg, field_params, enc_params, device, demo, teacher, n_steps=1):
    """RECOVERY mode (prob #2 fallback): when the traj-prog gate yields ZERO valid rollouts, train on demo
    (+LwF teacher) only — pulls the policy back toward the pretrained behavior until rollouts pass again.
    Without this the gate is a death trap: degraded policy -> no gathers -> no updates -> frozen forever."""
    nd = demo["U"].shape[0]
    policy.train()
    losses = []
    for _ in range(n_steps):
        di = torch.randint(0, nd, (cfg.batch_cap,))
        G, L, H, U = (demo["grid"][di].to(device), demo["low5"][di].to(device),
                      demo["hist"][di].to(device), demo["U"][di].to(device))
        loss = policy.cfm_loss(U, policy.ctx_from(G, L, H))
        if cfg.lwf_eta > 0 and teacher is not None:
            B_ = U.shape[0]
            x1 = (U / policy.u_max).reshape(B_, policy.d); x0 = torch.randn_like(x1)
            tau = torch.rand(B_, device=x1.device).clamp(1e-4, 1.0)
            x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
            v_s = policy.forward(x_tau, tau, policy._expand_ctx(policy.ctx_from(G, L, H), B_))
            with torch.no_grad():
                v_t = teacher.forward(x_tau, tau, teacher._expand_ctx(teacher.ctx_from(G, L, H), B_))
            loss = loss + cfg.lwf_eta * ((v_s - v_t) ** 2).mean()
        opt.zero_grad(); loss.backward()
        if cfg.enc_grad_clip > 0 and enc_params:
            torch.nn.utils.clip_grad_norm_(enc_params, cfg.enc_grad_clip)
        opt.step(); losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


def _load_demo(cfg):
    import pretrain_repr as PR
    G, L, H, U = PR.load_data("dr05_", [str(g) for g in cfg.gammas], cfg.demo_cap)
    return dict(grid=G, low5=L, hist=H, U=U)


def _escape_probe(policy, env, cfg, device, M=8, T=60, g=0.5):
    """Origin-escape stability (user 2026-07-09): M FAITHFUL rollouts truncated at T steps. Returns
    (frac that escape ||p||>1, circular std of the initial heading [rad], mean net-progress d0-dT).
    Stable escape = esc→1, hstd small-and-steady; the warm-up pathology = esc jumping + hstd large."""
    import math
    esc, heads, prog = [], [], []
    goal = env.goal.detach().cpu().numpy()
    for _ in range(M):
        out = GR.fm_deploy(policy, env, float(g), T=T, temp=1.0, nfe=cfg.nfe_explore, device=device)
        p = np.asarray(out["path"], dtype=float)
        esc.append(1.0 if (np.linalg.norm(p, axis=1) > 1.0).any() else 0.0)
        v = p[min(10, len(p) - 1)] - p[0]
        if np.linalg.norm(v) > 1e-6:
            heads.append(math.atan2(v[1], v[0]))
        prog.append(float(np.linalg.norm(p[0] - goal) - np.linalg.norm(p[-1] - goal)))
    if heads:
        R = min(1.0, math.hypot(float(np.mean(np.cos(heads))), float(np.mean(np.sin(heads)))))
        hstd = math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-9))))
    else:
        hstd = float("nan")
    return float(np.mean(esc)), float(hstd), float(np.mean(prog))


def _cov_probe(policy, env, cfg, device, M=50, g=0.5):
    """INSTANTANEOUS per-iter measurement (user 2026-07-09): M=50 faithful rollouts at one γ ->
    (SR, CR, coverage = #distinct staircase ids among the REACHED rollouts, the ids). NOT cumulative —
    this is the diversity of THIS snapshot's policy, so mode-collapse shows as cov -> 1 even while SR is high."""
    rows, _, paths = SR.eval_policy(policy, env, gammas=[g], M=M, T_max=cfg.T, reach=cfg.reach,
                                    temp=1.0, device=device, keep_paths=M, log=lambda *a, **k: None)
    goal = env.goal.detach().cpu().numpy()
    ids = set()
    for p in paths[g]:
        p = np.asarray(p, dtype=float)
        if np.linalg.norm(p[-1] - goal) < cfg.reach:
            sid = GM.staircase_id(p)
            if sid is not None:
                ids.add(sid)
    return float(rows[g]["SR"]), float(rows[g]["CR"]), len(ids), sorted(ids)


def _measure(policy, env, cfg, device):
    rows, agg, _ = SR.eval_policy(policy, env, gammas=list(cfg.gammas), M=cfg.M_measure, T_max=cfg.T,
                                  reach=cfg.reach, temp=1.0, device=device, log=lambda *a, **k: None)
    return rows, agg


def _save_viz_db(fresh, scores, easy_idx, frontier_idx, mix, path, it, cap=4096):
    """Save this iter's labeled fresh windows (easy/frontier + σ/margin/jerk/mono/prog) for the stack viz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = fresh["U"].shape[0]
    label = np.array(["easy"] * n, dtype=object); label[frontier_idx] = "frontier"
    sel = np.arange(n)
    if n > cap:                                                # stratified subsample to cap
        want_f = int(round(float(mix[1]) * cap))
        fi = np.random.choice(frontier_idx, min(want_f, len(frontier_idx)), replace=False) if len(frontier_idx) else np.array([], int)
        rem = cap - len(fi)
        ei = np.random.choice(easy_idx, min(rem, len(easy_idx)), replace=False) if len(easy_idx) else np.array([], int)
        sel = np.concatenate([ei, fi]).astype(int)
    db = dict(iter=it, mix=list(mix), label=list(label[sel]),
              grid=fresh["grid"][sel].cpu(), low5=fresh["low5"][sel].cpu(), U=fresh["U"][sel].cpu(),
              sigma=scores["sigma"][sel], margin=scores["margin"][sel], jerk=scores["jerk"][sel],
              mono=scores["mono"][sel], prog=scores["prog"][sel],
              rid=fresh.get("rid", np.zeros(n, int))[sel],           # rollout id per window (diversity check)
              widx=fresh.get("widx", np.zeros(n, int))[sel],         # in-traj window index (0 = initial escape)
              paths=[np.asarray(p) for p in fresh.get("paths", [])])  # executed trajs of the gathered rollouts
    torch.save(db, path)


# ---------------------------------------------------------------- main loop
def run_expand_cur(policy, env, cfg: CurConfig, device="cpu", outdir=None, log=print,
                   freeze_enc=True, enc_lr_mult=0.0, tag=""):
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    gammas = list(cfg.gammas)
    field_params = list(policy.trunk.parameters()) + list(policy.head.parameters())
    enc = policy.encoder_modules()
    if freeze_enc or enc_lr_mult <= 0:
        for p in enc:
            p.requires_grad_(False)
        enc_params, groups = [], [{"params": field_params, "lr": cfg.lr}]
    else:
        for p in enc:
            p.requires_grad_(True)
        enc_params = enc
        groups = [{"params": field_params, "lr": cfg.lr}, {"params": enc, "lr": cfg.lr * enc_lr_mult}]
    opt = torch.optim.Adam(groups)
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    demo = _load_demo(cfg) if (cfg.demo_frac > 0 or cfg.lwf_eta > 0) else None
    teacher = None
    if cfg.lwf_eta > 0:
        import copy
        teacher = copy.deepcopy(policy).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
    log(f"[fresh_expand{('/'+tag) if tag else ''}] iters={cfg.iters} FRESH-ONLY rollouts/iter={cfg.rollouts_per_iter} "
        f"prog_floor={cfg.prog_floor} q_hi={cfg.q_hi} q_lo={cfg.q_lo} valid_prog_floor={cfg.valid_prog_floor} "
        f"min_rollouts={cfg.min_rollouts} traj_prog_min={cfg.traj_prog_min} "
        f"mix {cfg.mix_start}->{cfg.mix_end} "
        f"inner {cfg.early_inner}/{cfg.inner_steps}/{cfg.cooldown_inner} freeze_enc={freeze_enc} "
        f"enc_lr_mult={enc_lr_mult} lr={cfg.lr} β{cfg.beta_steps} demo_frac={cfg.demo_frac} lwf_eta={cfg.lwf_eta}"
        + (f" demo={demo['U'].shape[0]}" if demo is not None else "")
        + (f" | PILE fresh_frac={cfg.fresh_frac} warmup={cfg.warmup_gather} cap={cfg.pile_cap} "
           f"replace={cfg.pile_replace} relabel_every={cfg.pile_relabel_every}" if
           (cfg.fresh_frac < 1.0 or cfg.warmup_gather > 0) else ""), flush=True)

    qbuf = None
    pile = Pile(cfg.pile_cap) if (cfg.fresh_frac < 1.0 or cfg.warmup_gather > 0) else None
    covered = {g: set() for g in gammas}
    roll_reached, roll_coll = deque(maxlen=100), deque(maxlen=100)
    history = []
    easy_idx, frontier_idx, scores = np.array([], int), np.array([], int), None
    last = dict(loss=float("nan"), field_grad_rms=0.0, enc_grad_rms=0.0, batch=(0, 0, 0))
    mix = tuple(cfg.mix_start)
    cooled = False

    rows0, agg0 = _measure(policy, env, cfg, device)
    log(f"it00000 SR {agg0['SR']:.2f} CR {agg0['CR']:.2f} | baseline "
        f"(pretrained repr{getattr(policy, 'repr_dim', '?')}, faithful temp=1)", flush=True)
    history.append(dict(iter=0, SR=agg0["SR"], CR=agg0["CR"], gdist=agg0["mean_goal_dist"],
                        rows={str(g): rows0[g] for g in gammas}, n_pos=0, beta=cfg.beta_steps[0],
                        mix=list(cfg.mix_start), n_easy=0, n_mid=0, n_frontier=0, loss=float("nan"),
                        field_grad_rms=0.0, enc_grad_rms=0.0, online_SR=0.0, online_CR=0.0,
                        covered={str(g): 0 for g in gammas}))
    best_sr = sr0 = agg0["SR"]; collapse_ct = 0
    for t in range(1, cfg.iters + 1):
        p = t / cfg.iters
        if cfg.beta_smooth == "exp":
            beta = cfg.beta_hi * (cfg.beta_lo / cfg.beta_hi) ** p
        elif cfg.beta_smooth == "aggressive":
            beta = cfg.beta_hi * (cfg.beta_lo / cfg.beta_hi) ** min(2 * p, 1.0)
        else:
            bk = max((k for k, f in enumerate(cfg.beta_fracs) if p >= f), default=0)
            beta = cfg.beta_steps[bk]
        K = cfg.rollouts_per_iter
        K_eff = int(np.ceil(K / 2)) if (p < cfg.early_frac or p >= cfg.cooldown_frac) else K
        a = float(np.clip((p - cfg.early_frac) / max(cfg.cooldown_frac - cfg.early_frac, 1e-6), 0, 1))
        mix = tuple(float(s0 * (1 - a) + e0 * a) for s0, e0 in zip(cfg.mix_start, cfg.mix_end))
        ndf = int(round(cfg.demo_frac * cfg.batch_cap)) if (cfg.demo_frac > 0 and demo is not None) else 0
        fresh_target = cfg.batch_cap - ndf                 # fresh quota for the early-stop gather
        tgt_e = int(round(mix[0] * fresh_target)); tgt_f = fresh_target - tgt_e
        fresh, qbuf, rr, rc, vr, att = _gather_fresh(policy, unc, env, cfg, gammas, beta, K_eff,
                                                     tgt_e, tgt_f, qbuf, covered, device)
        roll_reached.extend(rr); roll_coll.extend(rc)
        n_valid = 0 if fresh is None else fresh["U"].shape[0]

        inner = (cfg.early_inner if p < cfg.early_frac else
                 cfg.cooldown_inner if p >= cfg.cooldown_frac else cfg.inner_steps)
        it_batch, it_pile = (0, 0, 0), 0                   # THIS iter's actual batch draw (0s if no update)
        if fresh is not None:
            easy_idx, frontier_idx, scores = label_fresh(policy, unc, fresh, env, cfg, device)
        if t <= cfg.warmup_gather:                         # WARM-UP: gather -> pile only, NO gradient step
            if fresh is not None and pile is not None:     # (GP σ-buffer fills before the first update)
                pile.add(fresh, easy_idx, frontier_idx, scores, t)
            if t == cfg.warmup_gather:
                log(f"it{t:05d} WARM-UP done: pile {len(pile)} windows "
                    f"({pile.count('easy')}e/{pile.count('frontier')}f, "
                    f"{len(set(pile.rid.tolist()))} rollouts)", flush=True)
        elif fresh is not None:
            if p >= cfg.cooldown_frac and not cooled:
                for grp in opt.param_groups:
                    grp["lr"] *= cfg.cooldown_lr_mult
                cooled = True
            upd = update_flow_fresh(policy, opt, fresh, easy_idx, frontier_idx, mix, inner, cfg,
                                    field_params, enc_params, device, demo=demo, teacher=teacher, pile=pile)
            if upd is not None:
                last = upd
                it_batch, it_pile = upd["batch"], upd.get("n_pile", 0)
            if pile is not None:                           # add AFTER the update: the pile stays strictly older
                pile.add(fresh, easy_idx, frontier_idx, scores, t)
        elif pile is not None and len(pile) > 0:           # gather starved -> train on the (recent) pile
            upd = update_flow_fresh(policy, opt, None, np.array([], int), np.array([], int), mix, inner, cfg,
                                    field_params, enc_params, device, demo=demo, teacher=teacher, pile=pile)
            if upd is not None:
                last = upd
                it_batch, it_pile = upd["batch"], upd.get("n_pile", 0)
        elif demo is not None:                             # RECOVERY: gate starved (0 valid rollouts) ->
            rl = update_demo_only(policy, opt, cfg, field_params, enc_params, device, demo, teacher)
            if t % 25 == 0:
                log(f"it{t:05d} RECOVERY demo-only update (0 valid rollouts, loss {rl:.3f})", flush=True)
        if pile is not None and cfg.pile_relabel_every and t % cfg.pile_relabel_every == 0:
            pile.relabel(policy, unc, cfg, device)         # labels REFRESH with the current policy's σ

        if cfg.log_comp_every and t % cfg.log_comp_every == 0:   # micro diagnostics (user 2026-07-09 pattern)
            near0_e = w2_e = sig_e = sig_f = float("nan")
            if fresh is not None and scores is not None:
                n_all = fresh["U"].shape[0]
                em = np.zeros(n_all, bool); em[easy_idx] = True
                r0 = np.array([np.linalg.norm(np.asarray(GX2.state_from_low5(l), float)[:2])
                               for l in fresh["low5"].numpy()])
                near0_e = float((r0[em] < 1.0).mean()) if em.any() else float("nan")
                w2_e = float((fresh["widx"][em] < 2).mean()) if em.any() else float("nan")
                sig_e = float(scores["sigma"][em].mean()) if em.any() else float("nan")
                sig_f = float(scores["sigma"][~em].mean()) if (~em).any() else float("nan")
                comp = (f"e{len(easy_idx)}/f{len(frontier_idx)} easy(near0 {near0_e:.2f} w<2 {w2_e:.2f} "
                        f"σ {sig_e:.2f}) frontσ {sig_f:.2f} | batch rids {last.get('rid_n', float('nan')):.1f} "
                        f"dom {last.get('rid_dom', float('nan')):.2f}")
            else:
                comp = "e0/f0 (no fresh)"
            rec = dict(iter=t, beta=beta, n_easy=len(easy_idx), n_frontier=len(frontier_idx),
                       near0_e=near0_e, w2_e=w2_e, sig_e=sig_e, sig_f=sig_f,
                       rid_n=last.get("rid_n", float("nan")), rid_dom=last.get("rid_dom", float("nan")),
                       vr=vr, att=att, loss=last["loss"], fld=last["field_grad_rms"],
                       enc=last["enc_grad_rms"], lr=float(opt.param_groups[0]["lr"]),
                       batch_e=it_batch[0], batch_f=it_batch[1], batch_d=it_batch[2],
                       batch_pile=(it_pile if isinstance(it_pile, int) else 0),
                       batch_pe=(last.get("pile_batch", (0, 0))[0] if it_batch != (0, 0, 0) else 0),
                       batch_pf=(last.get("pile_batch", (0, 0))[1] if it_batch != (0, 0, 0) else 0),
                       mix_e=float(mix[0]), mix_f=float(mix[1]),
                       demo_req=int(round(cfg.demo_frac * cfg.batch_cap)) if demo is not None else 0)
            if pile is not None:
                rec.update(pile_e=pile.count("easy"), pile_f=pile.count("frontier"),
                           pile_rollouts=len(set(pile.rid.tolist())) if len(pile) else 0,
                           batch_pile=last.get("n_pile", 0), warmup=t <= cfg.warmup_gather)
                comp += f" | pile {pile.count('easy')}e/{pile.count('frontier')}f b{last.get('n_pile', 0)}"
            pr = ""
            if cfg.probe_escape and t % cfg.probe_escape == 0:
                pe, ph, pp = _escape_probe(policy, env, cfg, device)
                rec.update(esc=pe, hstd=ph, eprog=pp)
                pr = f" | esc {pe:.2f} hstd {ph:.2f}"
            if cfg.probe_cov and t % cfg.probe_cov == 0:
                s50, c50, k50, ids50 = _cov_probe(policy, env, cfg, device)
                rec.update(sr50=s50, cr50=c50, cov50=k50, ids50=ids50)
                pr += f" | SR50 {s50:.2f} CR50 {c50:.2f} cov {k50}"
            log(f"it{t:05d} COMP β {beta:.2f} {comp} | vr {vr}/{att}{pr}", flush=True)
            if outdir:
                with open(os.path.join(outdir, "probe.jsonl"), "a") as f:
                    f.write(json.dumps({k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                                        for k, v in rec.items()}) + "\n")

        if outdir and t % cfg.ckpt_every == 0:
            HP.save_hp(policy, os.path.join(outdir, f"ckpt_{t}.pt"), extra={"iter": t, "srcr": history[-1]})
        if outdir and cfg.viz_db_every and t % cfg.viz_db_every == 0 and fresh is not None and n_valid >= 8:
            _save_viz_db(fresh, scores, easy_idx, frontier_idx, mix,
                         os.path.join(outdir, "viz_db", f"it{t}.pt"), t)
        if t % cfg.measure_every == 0 or t == cfg.iters:
            rows, agg = _measure(policy, env, cfg, device)
            osr = float(np.mean(roll_reached)) if roll_reached else 0.0
            ocr = float(np.mean(roll_coll)) if roll_coll else 0.0
            ne, nf = len(easy_idx), len(frontier_idx)
            be, bf, bd = last.get("batch", (0, 0, 0))
            log(f"it{t:05d} SR {agg['SR']:.2f} CR {agg['CR']:.2f} | loss {last['loss']:.3f} "
                f"gRMS(fld {last['field_grad_rms']:.3f} enc {last['enc_grad_rms']:.3f}) | "
                f"β {beta:.2f} mix {mix[0]:.2f}/{mix[1]:.2f} lbl {ne}e/{nf}f | "
                f"batch {be}e+{bf}f+{bd}d nvalid {n_valid} vr {vr}/{att} | "
                f"on(SR {osr:.2f} CR {ocr:.2f})", flush=True)
            history.append(dict(iter=t, SR=agg["SR"], CR=agg["CR"], gdist=agg["mean_goal_dist"],
                                rows={str(g): rows[g] for g in gammas}, n_pos=n_valid, beta=beta,
                                lr=float(opt.param_groups[0]["lr"]),
                                mix=list(mix), n_easy=ne, n_mid=0, n_frontier=nf, loss=last["loss"],
                                field_grad_rms=last["field_grad_rms"], enc_grad_rms=last["enc_grad_rms"],
                                online_SR=osr, online_CR=ocr, n_valid=n_valid, valid_rollouts=vr,
                                rid_n=last.get("rid_n", float("nan")), rid_dom=last.get("rid_dom", float("nan")),
                                covered={str(gg): len(covered[gg]) for gg in gammas}))
            if outdir:                                     # live history so sweep_watch can see progress
                with open(os.path.join(outdir, "history.json"), "w") as f:
                    json.dump(history, f)
            if agg["SR"] > best_sr:
                best_sr = agg["SR"]
                if outdir:
                    HP.save_hp(policy, os.path.join(outdir, "best.pt"),
                               extra={"iter": t, "SR": agg["SR"], "CR": agg["CR"]})
            collapse_ct = (collapse_ct + 1 if (t >= cfg.collapse_min_iter and
                           agg["SR"] < cfg.collapse_frac * max(sr0, best_sr)) else 0)
            if collapse_ct >= cfg.collapse_patience:
                log(f"it{t:05d} COLLAPSED (SR {agg['SR']:.2f} < {cfg.collapse_frac}·max(SR0 {sr0:.2f}, "
                    f"best {best_sr:.2f})) — terminating early", flush=True)
                break

    if outdir:
        HP.save_hp(policy, os.path.join(outdir, "final.pt"),
                   extra={"covered": {str(g): sorted(covered[g]) for g in gammas}, "history_tail": history[-1]})
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
    return dict(history=history, covered={str(g): sorted(covered[g]) for g in gammas})


def main():
    import grid_scene as GS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--freeze", dest="freeze", action="store_true", default=True)
    ap.add_argument("--no-freeze", dest="freeze", action="store_false")
    ap.add_argument("--enc-lr-mult", type=float, default=0.3)
    ap.add_argument("--m-measure", type=int, default=25)
    ap.add_argument("--measure-every", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    # fresh-only knobs
    ap.add_argument("--rollouts-per-iter", type=int, default=10, help="MAX valid rollouts/iter (early-stop when batch fills)")
    ap.add_argument("--batch", type=int, default=32, help="total batch (demo + fresh)")
    ap.add_argument("--prog-floor", type=float, default=0.3, help="frontier if net-progress >= this")
    ap.add_argument("--valid-prog-floor", type=float, default=0.15, help="reject windows below this net-progress (safe-stationary trap; 0=off)")
    ap.add_argument("--min-rollouts", type=int, default=1, help="gather >= this many valid rollouts (LOCKED=1; 4 was the failed uni_C knob)")
    ap.add_argument("--traj-prog-min", type=float, default=0.0, help="dither gate (LOCKED=0/off; 1.0 was the failed uni_C knob)")
    ap.add_argument("--strat-rid", action="store_true", help="batch draw round-robins across source rollouts")
    ap.add_argument("--easy-sig-abs", type=float, default=0.0, help="ABSOLUTE σ cap: σ>=this can never be easy (0=off)")
    ap.add_argument("--easy-demo-backfill", action="store_true", help="fill the easy shortfall with demo windows")
    ap.add_argument("--easy-skip-first", type=int, default=0, help="in-traj window index < this is never easy")
    ap.add_argument("--probe-escape", type=int, default=0, help="origin-escape probe every N iters (0=off)")
    ap.add_argument("--probe-cov", type=int, default=0, help="M=50 faithful SR/CR/staircase-coverage probe every N iters (0=off)")
    ap.add_argument("--fresh-frac", type=float, default=1.0, help="fresh share of the fresh-part batch; rest from the pile (1.0=fresh-only)")
    ap.add_argument("--warmup-gather", type=int, default=0, help="first N iters gather->pile only, no gradient step")
    ap.add_argument("--pile-cap", type=int, default=3000, help="pile FIFO cap (staleness bound)")
    ap.add_argument("--pile-replace", action="store_true", help="pile draws WITH replacement (ablation; default LRU without-replacement)")
    ap.add_argument("--pile-relabel-every", type=int, default=10, help="recompute pile σ-labels every N iters (0=never)")
    ap.add_argument("--log-comp-every", type=int, default=0, help="composition/rid log line every N iters (0=off)")
    ap.add_argument("--frontier-qsig", type=float, default=0.67, help="high-σ quantile -> frontier")
    ap.add_argument("--frontier-qmarg", type=float, default=0.33, help="low-margin quantile -> frontier")
    ap.add_argument("--sigma-off", action="store_true", help="disable the σ criterion in frontier (ablation)")
    ap.add_argument("--mix-start", type=float, nargs=2, default=None, help="easy/frontier initial mix")
    ap.add_argument("--mix-end", type=float, nargs=2, default=None, help="easy/frontier final mix")
    ap.add_argument("--beta-steps", type=float, nargs=4, default=None)
    ap.add_argument("--beta-fracs", type=float, nargs=4, default=None)
    ap.add_argument("--beta-smooth", choices=["", "exp", "aggressive"], default="")
    ap.add_argument("--early-frac", type=float, default=None)
    ap.add_argument("--cooldown-frac", type=float, default=None)
    ap.add_argument("--inner-steps", type=int, default=None, help="mid-phase inner steps (default 4)")
    ap.add_argument("--early-inner", type=int, default=None, help="early/warmup-phase inner steps (default 2)")
    ap.add_argument("--cooldown-inner", type=int, default=None, help="cooldown-phase inner steps (default 2)")
    ap.add_argument("--demo-frac", type=float, default=0.0)
    ap.add_argument("--lwf-eta", type=float, default=0.0)
    ap.add_argument("--easy-strict", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--viz-db-every", type=int, default=100)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol, ck = HP.load_hp(args.ckpt, device=dev)
    env = GS.make_grid()
    cfg = CurConfig(iters=args.iters, M_measure=args.m_measure, measure_every=args.measure_every,
                    rollouts_per_iter=args.rollouts_per_iter, prog_floor=args.prog_floor,
                    valid_prog_floor=args.valid_prog_floor, min_rollouts=args.min_rollouts,
                    traj_prog_min=args.traj_prog_min,
                    q_hi=args.frontier_qsig, q_lo=args.frontier_qmarg, batch_cap=args.batch)
    if args.mix_start:
        cfg.mix_start = tuple(args.mix_start)
    if args.mix_end:
        cfg.mix_end = tuple(args.mix_end)
    if args.beta_steps:
        cfg.beta_steps = tuple(args.beta_steps)
    if args.beta_fracs:
        cfg.beta_fracs = tuple(args.beta_fracs)
    cfg.beta_smooth = args.beta_smooth
    if args.early_frac is not None:
        cfg.early_frac = args.early_frac
    if args.cooldown_frac is not None:
        cfg.cooldown_frac = args.cooldown_frac
    if args.inner_steps is not None:
        cfg.inner_steps = args.inner_steps
    if args.early_inner is not None:
        cfg.early_inner = args.early_inner
    if args.cooldown_inner is not None:
        cfg.cooldown_inner = args.cooldown_inner
    cfg.demo_frac = args.demo_frac
    cfg.lwf_eta = args.lwf_eta
    cfg.easy_strict = args.easy_strict
    cfg.use_sigma = not args.sigma_off
    if args.lr is not None:
        cfg.lr = args.lr
    cfg.viz_db_every = args.viz_db_every
    cfg.strat_rid = args.strat_rid
    cfg.easy_sig_abs = args.easy_sig_abs
    cfg.easy_demo_backfill = args.easy_demo_backfill
    cfg.easy_skip_first = args.easy_skip_first
    cfg.probe_escape = args.probe_escape
    cfg.probe_cov = args.probe_cov
    cfg.log_comp_every = args.log_comp_every
    cfg.fresh_frac = args.fresh_frac
    cfg.warmup_gather = args.warmup_gather
    cfg.pile_cap = args.pile_cap
    cfg.pile_replace = args.pile_replace
    cfg.pile_relabel_every = args.pile_relabel_every
    print(f"[main] ckpt {os.path.basename(args.ckpt)} repr {ck['config'].get('repr_dim')} "
          f"freeze={args.freeze} enc_lr_mult={args.enc_lr_mult} iters={args.iters} tag={args.tag}", flush=True)
    run_expand_cur(pol, env, cfg, device=dev, outdir=args.outdir, log=print,
                   freeze_enc=args.freeze, enc_lr_mult=args.enc_lr_mult, tag=args.tag)


if __name__ == "__main__":
    main()
