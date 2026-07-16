"""Curriculum safe-flow expansion (2026-07-07, user 07_06 spec). Self-contained "massive update" over
grid_expand2 (which is left untouched). Reuses grid_rollout.fm_deploy, grid_expand._cat/_to_t/_buffer_feat,
uncertainty.GPUncertainty, grid_metrics2.

Differences from grid_expand2.run_expand2:
  - defaults alpha=0, demo_frac=0, lwf_eta=0, temp=1 (faithful sampling), batch=64, slow updates.
  - WARM-UP: collect positives with NO gradient step until n_pos >= warmup_pos.
  - EASY/MID/FRONTIER curriculum: score each positive window by σ (GP novelty) / SOCP-margin / jerk(U) /
    goal-mono; batch mix ramps mix_start -> mix_end. The staircase inverse-frequency weighting is gone.
  - β (σ-tilt temperature in fm_deploy) anneals 1.0 -> 0.5 -> 0.2 -> 0.1 over iters (frontier ramp);
    cooldown lowers lr + inner-steps late; fewer Adam steps very early.
  - PRIMARY metric = origin SR (reach<=0.1) & CR via sr_cr_eval; coverage/validity2 are accumulated SILENTLY
    into history (never printed) until the user asks.
  - Encoder freeze is a knob: freeze_enc (default) OR unfrozen at enc_lr_mult with enc-grad-RMS logged + clipped.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

import _paths  # noqa: F401
import grid_rollout as GR
import grid_expand as GE
import grid_expand2 as GX2          # state_from_low5
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_hp_expt as HP
from uncertainty import GPUncertainty
import sr_cr_eval as SR


@dataclass
class CurConfig:
    iters: int = 3000
    # exploration (σ-tilt)
    N: int = 64
    temp: float = 1.0               # faithful sampling
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
    # curriculum
    warmup_pos: int = 1000
    batch: int = 64
    inner_steps: int = 4            # "slow" updates (user); 12 was too aggressive → SR collapse risk
    lr: float = 1e-4
    q_lo: float = 0.33
    q_hi: float = 0.67
    mono_thresh: float = 0.9
    easy_mode: str = "composite"   # "composite" rank OR "strict" (all-criteria AND — smaller, higher-purity easy)
    mix_start: tuple = (0.7, 0.3, 0.0)      # easy / mid / frontier
    mix_end: tuple = (0.34, 0.33, 0.33)
    beta_steps: tuple = (1.0, 0.5, 0.2, 0.1)
    beta_fracs: tuple = (0.0, 0.25, 0.5, 0.75)
    beta_smooth: str = ""          # "" = step schedule; "exp" = β_hi·(β_lo/β_hi)^p; "aggressive" = ^min(2p,1)
    beta_hi: float = 1.0
    beta_lo: float = 0.1
    viz_db_every: int = 1000       # save a labeled buffer-DB snapshot every N iters (0 = off)
    cooldown_frac: float = 0.75
    cooldown_lr_mult: float = 0.3
    cooldown_inner: int = 2
    early_frac: float = 0.1
    early_inner: int = 2
    enc_grad_clip: float = 5.0
    # measurement (SR/CR primary)
    measure_every: int = 200
    M_measure: int = 16
    reach: float = 0.1
    T: int = 250
    # buffers / misc
    cap_pos: int = 60000
    cap_neg: int = 4000
    pos_margin: float = 0.0
    alpha: float = 0.0
    demo_frac: float = 0.0         # δ anchor: fraction of each batch drawn from the PRETRAINING (dr05) demo
    lwf_eta: float = 0.0           # η anchor: LwF distillation vs the frozen pretrained teacher on demo contexts
    demo_cap: int = 1200           # per-γ demo windows loaded for the anchors
    gammas: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    ckpt_every: int = 500
    collapse_frac: float = 0.45    # terminate if SR < collapse_frac*max(SR0,bestSR) for collapse_patience measures
    collapse_patience: int = 3
    collapse_min_iter: int = 600   # never terminate before this (past warm-up + early ramp)


# ---------------------------------------------------------------- scoring & sampling
def score_positives(policy, unc, pos, env, cfg, device, cap=4000):
    """Per-window np scores over a (subsampled) view of the positive buffer."""
    n = pos["U"].shape[0]
    idx = np.arange(n) if n <= cap else np.random.choice(n, cap, replace=False)
    ti = torch.as_tensor(idx, dtype=torch.long)
    G, L, H, U = pos["grid"][ti], pos["low5"][ti], pos["hist"][ti], pos["U"][ti]
    sig = []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            ctx = policy.ctx_from(G[i:i + 2048].to(device), L[i:i + 2048].to(device), H[i:i + 2048].to(device))
            phi = policy.phi_s(U[i:i + 2048].to(device), ctx, s=cfg.s)
            sig.append(unc.sigma(phi).detach().cpu().numpy())
    sigma = np.concatenate(sig) if sig else np.zeros(len(idx))
    Un, Ln = U.numpy(), L.numpy()
    jerk = (np.linalg.norm(np.diff(Un, n=2, axis=1), axis=2).mean(axis=1)
            if Un.shape[1] >= 3 else np.zeros(len(idx)))
    net = Un.sum(axis=1); rg = Ln[:, :2]
    mono = (net * rg).sum(1) / (np.linalg.norm(net, axis=1) * np.linalg.norm(rg, axis=1) + 1e-9)
    margin = np.empty(len(idx))
    for j in range(len(idx)):
        margin[j] = GM2.window_min_clearance(GX2.state_from_low5(Ln[j]), Un[j], env)
    margin = np.nan_to_num(np.clip(margin, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0)
    return dict(sigma=sigma, margin=margin, jerk=jerk, mono=mono, idx=idx)


def curriculum_pools(scores, cfg):
    """easy = top-tercile COMPOSITE easiness among non-frontier (low σ + high margin + low jerk + high goal-
    alignment); frontier = high σ OR low SOCP-margin (user's OR). Composite ranks avoid the 4-way-AND
    starvation (a fixed mono≥0.9 threshold on a ~0.08-mean cosine admitted almost nothing)."""
    sg, mg, jk, mo, idx = (scores["sigma"], scores["margin"], scores["jerk"], scores["mono"], scores["idx"])
    n = len(idx); empty = np.array([], int)
    if n == 0:
        return empty, empty, empty
    s_hi = np.quantile(sg, cfg.q_hi); m_lo = np.quantile(mg, cfg.q_lo)
    frontier = (sg >= s_hi) | (mg <= m_lo)                          # high uncertainty OR low safety margin
    nf = ~frontier
    easy = np.zeros(n, bool)
    if nf.any():                                                   # composite easiness rank over all 4 criteria
        pr = lambda x: np.argsort(np.argsort(x)) / max(n - 1, 1)   # percentile rank in [0,1]
        easiness = ((1 - pr(sg)) + pr(mg) + (1 - pr(jk)) + pr(mo)) / 4.0   # higher = easier (low σ/jerk, high margin/mono)
        thr_q = 0.85 if cfg.easy_mode == "strict" else cfg.q_hi    # strict = purer top-15%; default top-33% (never empty)
        easy = nf & (easiness >= np.quantile(easiness[nf], thr_q))
    mid = ~easy & ~frontier
    return idx[easy], idx[mid], idx[frontier]


def _draw_batch(pools, mix, batch, all_idx):
    counts = [int(round(m * batch)) for m in mix]
    counts[int(np.argmax(mix))] += batch - sum(counts)
    nonempty = [p for p in pools if len(p) > 0]
    union = np.concatenate(nonempty) if nonempty else all_idx
    parts = []
    for k, c in enumerate(counts):
        if c <= 0:
            continue
        src = pools[k] if len(pools[k]) > 0 else union
        parts.append(np.random.choice(src, size=c, replace=True))
    return np.concatenate(parts) if parts else np.random.choice(all_idx, size=batch, replace=True)


def _grad_rms(params):
    vals = [float(p.grad.pow(2).mean()) for p in params if p.grad is not None]
    return float(np.sqrt(np.mean(vals))) if vals else 0.0


def update_flow_cur(policy, opt, pos, cfg, pools, mix, n_steps, field_params, enc_params, device,
                    demo=None, teacher=None):
    n = pos["U"].shape[0]; all_idx = np.arange(n)
    ndf = int(round(cfg.demo_frac * cfg.batch)) if (cfg.demo_frac > 0 and demo is not None) else 0
    nd = demo["U"].shape[0] if demo is not None else 0
    policy.train()
    losses, fgr, egr = [], [], []
    for _ in range(n_steps):
        bi = torch.as_tensor(_draw_batch(pools, mix, cfg.batch - ndf, all_idx), dtype=torch.long)
        G, L, H, U = (pos["grid"][bi].to(device), pos["low5"][bi].to(device),
                      pos["hist"][bi].to(device), pos["U"][bi].to(device))
        if ndf > 0:                                              # δ anchor: mix pretraining-demo windows in
            di = torch.randint(0, nd, (ndf,))
            G = torch.cat([G, demo["grid"][di].to(device)]); L = torch.cat([L, demo["low5"][di].to(device)])
            H = torch.cat([H, demo["hist"][di].to(device)]); U = torch.cat([U, demo["U"][di].to(device)])
        loss = policy.cfm_loss(U, policy.ctx_from(G, L, H))
        if cfg.lwf_eta > 0 and teacher is not None and demo is not None:   # η anchor: LwF on demo contexts
            li = torch.randint(0, nd, (cfg.batch,))
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
        fgr.append(_grad_rms(field_params)); egr.append(_grad_rms(enc_params))   # aggressiveness / enc-leakage
        if cfg.enc_grad_clip > 0 and enc_params:
            torch.nn.utils.clip_grad_norm_(enc_params, cfg.enc_grad_clip)
        opt.step(); losses.append(float(loss))
    return dict(loss=float(np.mean(losses)) if losses else float("nan"),
                field_grad_rms=float(np.mean(fgr)) if fgr else 0.0,
                enc_grad_rms=float(np.mean(egr)) if egr else 0.0)


def _load_demo(cfg):
    import pretrain_repr as PR                        # anchor demo = the pretraining (dr05) windows
    G, L, H, U = PR.load_data("dr05_", [str(g) for g in cfg.gammas], cfg.demo_cap)
    return dict(grid=G, low5=L, hist=H, U=U)


# ---------------------------------------------------------------- main loop
def _measure(policy, env, cfg, device):
    rows, agg, _ = SR.eval_policy(policy, env, gammas=list(cfg.gammas), M=cfg.M_measure, T_max=cfg.T,
                                  reach=cfg.reach, temp=1.0, device=device, log=lambda *a, **k: None)
    return rows, agg


def _save_viz_db(policy, unc, pos, neg, env, cfg, mix, device, path, it, n=64):
    """Save ~n stratified POSITIVE buffer samples (easy/mid/frontier label + σ/margin/jerk/mono) and a few
    NEGATIVES with their invalidity reason (taskspace/approach/socp) for the report viz (user 0.1)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc = score_positives(policy, unc, pos, env, cfg, device)
    easy, mid, frontier = curriculum_pools(sc, cfg)
    idx = sc["idx"]; row_of = {int(v): r for r, v in enumerate(idx)}
    counts = [max(1, int(round(m * n))) for m in mix]
    sel, labs = [], []
    for pool, lb, c in zip((easy, mid, frontier), ("easy", "mid", "frontier"), counts):
        pl = [int(v) for v in pool if int(v) in row_of]
        if pl:
            pick = np.random.choice(pl, min(c, len(pl)), replace=False)
            sel += list(pick); labs += [lb] * len(pick)
    sel = np.array(sel, int); rr = np.array([row_of[int(s)] for s in sel])
    db = dict(iter=it, mix=list(mix), label=list(labs),
              grid=pos["grid"][sel].cpu(), low5=pos["low5"][sel].cpu(), U=pos["U"][sel].cpu(),
              sigma=sc["sigma"][rr], margin=sc["margin"][rr], jerk=sc["jerk"][rr], mono=sc["mono"][rr])
    if neg is not None and neg["U"].shape[0] > 0:                # a few negatives + WHY-invalid
        ni = np.random.choice(neg["U"].shape[0], min(16, neg["U"].shape[0]), replace=False)
        reasons = []
        for j in ni:
            st = GX2.state_from_low5(neg["low5"][int(j)].numpy())
            seg = np.vstack([st[:2][None, :], GR.window_positions(st, neg["U"][int(j)].numpy(), env.dt)])
            reasons.append({k: bool(v) for k, v in GM2.criteria_status(seg, env, 0.5).items()})
        db.update(neg_low5=neg["low5"][torch.as_tensor(ni)].cpu(), neg_U=neg["U"][torch.as_tensor(ni)].cpu(),
                  neg_reason=reasons)
    torch.save(db, path)


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
    if cfg.lwf_eta > 0:                                # frozen pretrained teacher for the LwF anchor
        import copy
        teacher = copy.deepcopy(policy).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
    log(f"[expand_cur{('/'+tag) if tag else ''}] iters={cfg.iters} freeze_enc={freeze_enc} "
        f"enc_lr_mult={enc_lr_mult} enc_params={len(enc_params)} warmup_pos={cfg.warmup_pos} "
        f"batch={cfg.batch} lr={cfg.lr} β{cfg.beta_steps} demo_frac={cfg.demo_frac} lwf_eta={cfg.lwf_eta}"
        + (f" demo={demo['U'].shape[0]}" if demo is not None else ""), flush=True)

    pos = neg = qbuf = None
    covered = {g: set() for g in gammas}
    roll_reached, roll_coll = deque(maxlen=50), deque(maxlen=50)
    history = []
    pools = (np.array([], int), np.array([], int), np.array([], int))
    pools_at, cooled, warmed = -999, False, False
    last = dict(loss=float("nan"), field_grad_rms=0.0, enc_grad_rms=0.0)
    mix = tuple(cfg.mix_start)

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
        if cfg.beta_smooth == "exp":                       # smooth continuous β_hi→β_lo over the whole run
            beta = cfg.beta_hi * (cfg.beta_lo / cfg.beta_hi) ** p
        elif cfg.beta_smooth == "aggressive":              # reaches β_lo by frac 0.5, then holds
            beta = cfg.beta_hi * (cfg.beta_lo / cfg.beta_hi) ** min(2 * p, 1.0)
        else:
            bk = max((k for k, f in enumerate(cfg.beta_fracs) if p >= f), default=0)
            beta = cfg.beta_steps[bk]
        g = gammas[(t - 1) % len(gammas)]
        qfeat = GE._buffer_feat(policy, qbuf, "phi_s", cfg.s, cfg.gp_buf, device) if qbuf is not None else None
        if qfeat is not None:
            unc.set_buffer(qfeat)
        out = GR.fm_deploy(policy, env, float(g), T=cfg.T,
                           tilt=dict(unc=unc, beta=beta, N=cfg.N, s=cfg.s, broad=0, feature="phi_s",
                                     temp=cfg.temp, churn=cfg.churn, safe_filter=cfg.safe_filter),
                           nfe=cfg.nfe_explore, record=True, verify_fn=GM2.window_label_cheap, device=device)
        roll_reached.append(1.0 if out["reached"] else 0.0)
        roll_coll.append(1.0 if SR.path_collides(out["path"], env) else 0.0)
        if out["recs"]:
            G, L, H, U = GE._to_t(out["recs"])
            qbuf = GE._cat(qbuf, G[::3], L[::3], H[::3], U[::3], cap=cfg.qbuf_cap)
            if out["reached"]:                               # only reached trajectories become pos/neg
                ok2 = GM2.traj_valid2(out["path"], env, float(g))
                if ok2:                                      # SR-positive AND valid2 (safe / SOCP-certified)
                    sid = GM.staircase_id(out["path"])       # for covered[] tracking ONLY (not a pos gate)
                    if sid is not None:
                        covered[g].add(sid)
                    tg = sid if sid is not None else -1       # keep tag field; -1 = non-staircase (multimodal)
                    if cfg.pos_margin > 0:
                        keep = [i for i, r in enumerate(out["recs"])
                                if GM2.window_min_clearance(GX2.state_from_low5(r[1]), r[3], env) >= cfg.pos_margin]
                        if keep:
                            ki = torch.as_tensor(keep)
                            pos = GE._cat(pos, G[ki], L[ki], H[ki], U[ki], tags=[tg] * len(keep), cap=cfg.cap_pos)
                    else:
                        pos = GE._cat(pos, G, L, H, U, tags=[tg] * G.shape[0], cap=cfg.cap_pos)
                else:                                        # reached but not valid2 -> negative
                    neg = GE._cat(neg, G, L, H, U, cap=cfg.cap_neg)
        n_pos = 0 if pos is None else pos["U"].shape[0]

        if n_pos >= cfg.warmup_pos and not warmed:         # inspect the INITIAL gathering (σ flat, unguided)
            warmed = True
            log(f"it{t:05d} WARMUP cleared after {t} deploys: online reached {np.mean(roll_reached):.2f} "
                f"collided {np.mean(roll_coll):.2f} (σ flat early → unguided pos-gathering) n_pos {n_pos}", flush=True)
        if n_pos >= cfg.warmup_pos:                        # ---- WARM-UP gate cleared: curriculum update
            a = float(np.clip((p - cfg.early_frac) / max(cfg.cooldown_frac - cfg.early_frac, 1e-6), 0, 1))
            mix = tuple(float(s0 * (1 - a) + e0 * a) for s0, e0 in zip(cfg.mix_start, cfg.mix_end))
            if t - pools_at >= 25 or sum(len(x) for x in pools) == 0:
                pools = curriculum_pools(score_positives(policy, unc, pos, env, cfg, device), cfg)
                pools_at = t
            inner = (cfg.early_inner if p < cfg.early_frac else
                     cfg.cooldown_inner if p >= cfg.cooldown_frac else cfg.inner_steps)
            if p >= cfg.cooldown_frac and not cooled:
                for grp in opt.param_groups:
                    grp["lr"] *= cfg.cooldown_lr_mult
                cooled = True
            last = update_flow_cur(policy, opt, pos, cfg, pools, mix, inner, field_params, enc_params, device,
                                   demo=demo, teacher=teacher)

        if outdir and t % cfg.ckpt_every == 0:
            HP.save_hp(policy, os.path.join(outdir, f"ckpt_{t}.pt"), extra={"iter": t, "srcr": history[-1]})
        if outdir and cfg.viz_db_every and t % cfg.viz_db_every == 0 and pos is not None and pos["U"].shape[0] >= 64:
            _save_viz_db(policy, unc, pos, neg, env, cfg, mix, device,
                         os.path.join(outdir, "viz_db", f"it{t}.pt"), t)
        if t % cfg.measure_every == 0 or t == cfg.iters:
            rows, agg = _measure(policy, env, cfg, device)
            osr = float(np.mean(roll_reached)) if roll_reached else 0.0
            ocr = float(np.mean(roll_coll)) if roll_coll else 0.0
            ne, nm, nf = len(pools[0]), len(pools[1]), len(pools[2])
            log(f"it{t:05d} SR {agg['SR']:.2f} CR {agg['CR']:.2f} | loss {last['loss']:.3f} "
                f"gRMS(fld {last['field_grad_rms']:.3f} enc {last['enc_grad_rms']:.3f}) | "
                f"β {beta:.2f} mix {mix[0]:.2f}/{mix[1]:.2f}/{mix[2]:.2f} pools {ne}/{nm}/{nf} "
                f"npos {n_pos} | on(SR {osr:.2f} CR {ocr:.2f})", flush=True)
            history.append(dict(iter=t, SR=agg["SR"], CR=agg["CR"], gdist=agg["mean_goal_dist"],
                                rows={str(g): rows[g] for g in gammas}, n_pos=n_pos, beta=beta,
                                lr=float(opt.param_groups[0]["lr"]),
                                mix=list(mix), n_easy=ne, n_mid=nm, n_frontier=nf, loss=last["loss"],
                                field_grad_rms=last["field_grad_rms"], enc_grad_rms=last["enc_grad_rms"],
                                online_SR=osr, online_CR=ocr,
                                covered={str(gg): len(covered[gg]) for gg in gammas}))
            if agg["SR"] > best_sr:                        # keep the PEAK (the enhancement) as best.pt
                best_sr = agg["SR"]
                if outdir:
                    HP.save_hp(policy, os.path.join(outdir, "best.pt"),
                               extra={"iter": t, "SR": agg["SR"], "CR": agg["CR"]})
            collapse_ct = (collapse_ct + 1 if (t >= cfg.collapse_min_iter and
                           agg["SR"] < cfg.collapse_frac * max(sr0, best_sr)) else 0)
            if collapse_ct >= cfg.collapse_patience:       # terminate a genuinely collapsing arm early
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
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--warmup-pos", type=int, default=1000)
    ap.add_argument("--freeze", dest="freeze", action="store_true", default=True)
    ap.add_argument("--no-freeze", dest="freeze", action="store_false")
    ap.add_argument("--enc-lr-mult", type=float, default=0.3)
    ap.add_argument("--m-measure", type=int, default=16)
    ap.add_argument("--measure-every", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-steps", type=float, nargs=4, default=None,
                    help="4 β-anneal levels (default 1.0 0.5 0.2 0.1); e.g. gentle 1.0 0.7 0.5 0.3 or flat 1 1 1 1")
    ap.add_argument("--beta-fracs", type=float, nargs=4, default=None,
                    help="iter-fractions at which the 4 β steps engage (default 0 .25 .5 .75)")
    ap.add_argument("--early-frac", type=float, default=None, help="mix-ramp start + early-inner phase end (frac)")
    ap.add_argument("--cooldown-frac", type=float, default=None,
                    help="mix-ramp end + lr/inner cooldown start (frac)")
    ap.add_argument("--mix-end", type=float, nargs=3, default=None, help="easy/mid/frontier final batch mix")
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--demo-frac", type=float, default=0.0, help="δ anchor: batch fraction from dr05 pretraining data")
    ap.add_argument("--lwf-eta", type=float, default=0.0, help="η anchor: LwF weight vs frozen pretrained teacher")
    ap.add_argument("--easy-strict", action="store_true", help="stricter all-criteria-AND easy pool for the curriculum")
    ap.add_argument("--beta-smooth", choices=["", "exp", "aggressive"], default="", help="smooth β(t/iters) schedule")
    ap.add_argument("--beta-hi", type=float, default=1.0)
    ap.add_argument("--beta-lo", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=None, help="field lr (default 1e-4; A1 uses 5e-5)")
    ap.add_argument("--mix-start", type=float, nargs=3, default=None, help="easy/mid/frontier initial batch mix")
    ap.add_argument("--viz-db-every", type=int, default=1000, help="save labeled buffer-DB every N iters (0=off)")
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol, ck = HP.load_hp(args.ckpt, device=dev)
    env = GS.make_grid()
    cfg = CurConfig(iters=args.iters, warmup_pos=args.warmup_pos, M_measure=args.m_measure,
                    measure_every=args.measure_every)
    if args.beta_steps:
        cfg.beta_steps = tuple(args.beta_steps)
    if args.beta_fracs:
        cfg.beta_fracs = tuple(args.beta_fracs)
    if args.early_frac is not None:
        cfg.early_frac = args.early_frac
    if args.cooldown_frac is not None:
        cfg.cooldown_frac = args.cooldown_frac
    if args.mix_end:
        cfg.mix_end = tuple(args.mix_end)
    if args.inner_steps is not None:
        cfg.inner_steps = args.inner_steps
    cfg.demo_frac = args.demo_frac
    cfg.lwf_eta = args.lwf_eta
    if args.easy_strict:
        cfg.easy_mode = "strict"
    cfg.beta_smooth = args.beta_smooth
    cfg.beta_hi = args.beta_hi; cfg.beta_lo = args.beta_lo
    if args.lr is not None:
        cfg.lr = args.lr
    if args.mix_start:
        cfg.mix_start = tuple(args.mix_start)
    cfg.viz_db_every = args.viz_db_every
    print(f"[main] ckpt {os.path.basename(args.ckpt)} repr {ck['config'].get('repr_dim')} "
          f"freeze={args.freeze} enc_lr_mult={args.enc_lr_mult} iters={args.iters} tag={args.tag}", flush=True)
    run_expand_cur(pol, env, cfg, device=dev, outdir=args.outdir, log=print,
                   freeze_enc=args.freeze, enc_lr_mult=args.enc_lr_mult, tag=args.tag)


if __name__ == "__main__":
    main()
