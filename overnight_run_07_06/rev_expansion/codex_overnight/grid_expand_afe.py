"""Minimal AFE-style Safe Flow Expansion trainer (user redesign 2026-07-16).

Replaces the curriculum trainer (grid_expand_hardtail.py) with the endorsed minimal method:

  GATHER (shielded, per control step of R episodes/round):
    sample K plans from the current flow p_theta(.|c_t); score sigma on the FROZEN pretrained
    representation via the cumulative A_n; Gibbs-draw B without replacement
    (pi ~ exp((sigma-sigma_max)/beta), the finite-sample solution of max_q E_q[sigma] - beta KL(q||p));
    FULLY verify each drawn plan (whole-plan in-bounds + SOCP at the episode gamma) BEFORE execution;
    EVERY verified query enters D_n and A_n (positive or negative); execute the first action of the
    best-progress verified-safe plan; if none verifies, execute the certified SafeMPPI backup.

  UPDATE (once per round): uniform replay over the CUMULATIVE D+ with the single proximal objective
    l_CFM(theta) + (1/(2 eta)) ||theta - theta_n||^2.  The number of Adam steps is a solver setting:
    we stop at a functional-displacement bound (relative field change on a fixed probe batch vs the
    round start) or max_inner, and report both.

  TRACK: query acceptance a_hat_n (tilted) SEPARATELY from model validity V_hat_n on the UNTILTED
    fixed audit rho_eval (per gamma, never added to buffers) + V_hat^prog, fallback frequency (per
    gamma -- its decay IS the expansion curve), dithering share of D+ (prog < 0.05), closed-loop
    SR/CR/coverage, and per-round viz DBs recording exactly which samples trained the model.

  NO curriculum: no easy/frontier split, no quantile, no mix ratio, no gamma/mode balancing, no
  demo backfill (demo replay exists ONLY as the explicit ablation arm B via --demo-frac), sigma is
  used exactly once (acquisition).  gamma is a conditioning variable on a fixed episode rotation.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
sys.path.insert(0, _WORK)                                   # shared grid code
sys.path.insert(0, _REV)                                    # rev_expansion helpers
sys.path.insert(0, _HERE)                                   # local copies ALWAYS win (grid_metrics2!)

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_scene as GS
import grid_hp_expt as HP
import grid_expand_hardtail as HT              # reuse: _apply_wall_plugs, _preserve_torch_rng
import sr_cr_eval as SR
from di_grid_viz import di_step

import afe_core as AC


@dataclass
class AFEConfig:
    rounds: int = 100
    episodes_per_round: int = 8
    T: int = 300
    reach: float = 0.15
    # acquisition
    K: int = 64                    # candidate plans per step
    B: int = 4                     # verifier budget per step (drawn without replacement)
    beta: float = 0.2
    s: float = 0.9
    lam: float = 1e-2
    nfe: int = 8
    temp: float = 1.0
    n_theta: int = 180             # SOCP polytope resolution (the verifier's own default)
    exec_rule: str = "progress"    # progress = argmax r among verified-safe; "pi" = sample ~ pi
    gammas: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    # update (solver settings, reported not tuned)
    lr: float = 2e-5
    eta: float = 0.01              # proximal weight (1/(2 eta))||theta-theta_n||^2
    batch: int = 128
    max_inner: int = 40
    fstep_stop: float = 0.03       # stop the solver at this relative field displacement vs round start
    grad_clip: float = 1.0
    # ablation arm B
    demo_frac: float = 0.0
    demo_prefix: str = "dr05_"
    demo_cap: int = 1200
    # tracking
    audit_every: int = 5
    audit_pos: int = 12
    audit_plans: int = 4
    measure_every: int = 10
    M_measure: int = 8
    T_eval: int = 350
    ckpt_every: int = 10
    dstore_every: int = 25
    viz_every: int = 1
    dither_bar: float = 0.05
    # environment
    wall_plugs: int = 8
    start_eps: float = 0.3
    goal_xy: tuple = (4.7, 4.7)
    seed: int = 910
    max_hours: float = 20.0


# ------------------------------------------------------------------ gather
def gather_round(policy, phi0, blr, store, env, cfg, fb, round_i, device):
    """R shielded episodes; returns per-round stats + per-episode viz records."""
    policy.eval()
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal_np = env.goal.detach().cpu().numpy()
    x0 = env.x0.detach().cpu().numpy().astype(np.float32)
    ep_records = []
    n_q = n_acc = n_fb = n_exec = n_socp_err = 0
    acc_g = {float(g): [0, 0] for g in cfg.gammas}         # gamma -> [accepted, queried]
    fb_g = {float(g): [0, 0] for g in cfg.gammas}          # gamma -> [fallback steps, steps]
    sig_drawn = []
    new_pos_prog = []
    for e in range(cfg.episodes_per_round):
        g = float(cfg.gammas[((round_i - 1) * cfg.episodes_per_round + e) % len(cfg.gammas)])
        fb.reset()
        st = x0.copy()
        hist = []
        path = [st[:2].copy()]
        fb_mask = []
        reached = dead = False
        for t in range(cfg.T):
            grid_np = GF.axis_grid(st[:2], obs, rr)
            l5_np = GF.low5(st, goal_np, g)
            h_np = GF.hist_pad(np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)), GF.K_HIST)
            gT = torch.tensor(grid_np, device=device)
            lT = torch.tensor(l5_np, device=device)
            hT = torch.tensor(h_np, device=device)
            Ucand = policy.sample_window(gT, lT, hT, n=cfg.K, temp=cfg.temp, nfe=cfg.nfe)
            Z = AC.frozen_feat(phi0, Ucand, gT, lT, hT, s=cfg.s)
            sig = blr.sigma(Z)
            w = torch.exp(((sig - sig.max()) / max(cfg.beta, 1e-6)).clamp(-30, 30))
            drawn = torch.multinomial(w / w.sum(), min(cfg.B, cfg.K), replacement=False)
            sid = store.add_step_ctx(st, grid_np, l5_np, h_np, (round_i, e, t))
            best = None                                     # (prog, qid, U_np)
            safe = []                                       # (weight, qid, U_np)
            for j in drawn.tolist():
                U_np = Ucand[j].detach().cpu().numpy()
                seg = GR.window_positions(st, U_np, env.dt)
                v = AC.verify_plan(st, U_np, env, g, goal_np, n_theta=cfg.n_theta)
                qid = store.add_query(sid, U_np, v, float(sig[j]), g, round_i, seg)
                blr.update(Z[j:j + 1])
                n_q += 1
                acc_g[g][1] += 1
                sig_drawn.append(float(sig[j]))
                n_socp_err += int(v["reason"] == "socp_error")
                if v["y"]:
                    n_acc += 1
                    acc_g[g][0] += 1
                    new_pos_prog.append(v["prog"])
                    safe.append((float(w[j]), qid, U_np))
                    if best is None or v["prog"] > best[0]:
                        best = (v["prog"], qid, U_np)
            fb_g[g][1] += 1
            if safe:
                if cfg.exec_rule == "pi" and len(safe) > 1:
                    ws = np.array([sfe[0] for sfe in safe], np.float64)
                    pick = int(np.random.choice(len(safe), p=ws / ws.sum()))
                    _, qid, U_np = safe[pick]
                else:
                    _, qid, U_np = best
                a = U_np[0]
                store.mark_executed(qid)
                n_exec += 1
                fb_mask.append(False)
            else:                                           # certified backup carries the step
                try:
                    a = fb.plan(st, g, seed=cfg.seed * 1000003 + round_i * 9973 + e * 1009 + t)
                except Exception:
                    a = np.clip(-2.0 * st[2:4], -1.0, 1.0)  # brake: decelerate in place, never crash the run
                n_fb += 1
                fb_g[g][0] += 1
                fb_mask.append(True)
            st = di_step(st, np.asarray(a, np.float32), dt=env.dt)
            hist.append(np.asarray(a, np.float32))
            path.append(st[:2].copy())
            if np.linalg.norm(st[:2] - goal_np) < cfg.reach:
                reached = True
                break
            if (st[:2] < -GM.EPS_TASK).any() or (st[:2] > GM.GRID_M + GM.EPS_TASK).any():
                dead = True
                break
            if (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min() < 0.0:
                dead = True
                break
        ep_records.append(dict(gamma=g, path=np.asarray(path, np.float32), reached=bool(reached),
                               dead=bool(dead), steps=len(path) - 1,
                               fb_mask=np.asarray(fb_mask, bool)))
    return dict(n_q=n_q, n_acc=n_acc, n_fb=n_fb, n_exec=n_exec, n_socp_err=n_socp_err,
                acc_g={str(k): v for k, v in acc_g.items()},
                fb_g={str(k): v for k, v in fb_g.items()},
                sigma_drawn_mean=float(np.mean(sig_drawn)) if sig_drawn else float("nan"),
                sigma_drawn_min=float(np.min(sig_drawn)) if sig_drawn else float("nan"),
                new_pos_prog=new_pos_prog, episodes=ep_records)


# ------------------------------------------------------------------ update
def prox_update(policy, opt, store, demo, cfg, device, rng):
    """One proximal update: argmin_theta mean l_CFM over uniform D+ replay + (1/(2 eta))||theta-theta_n||^2.

    Adam-step count is a SOLVER setting: stop when the relative field displacement on a fixed probe
    batch (vs the round start) reaches fstep_stop, or at max_inner.  Reports the drawn D+ ids
    (exactly which samples trained the model this round), the loss decomposition, and the stop cause.
    """
    if store.n_pos() == 0:
        return None
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    refs = [p.detach().clone() for p in trainable]
    nb_demo = int(round(cfg.demo_frac * cfg.batch)) if (cfg.demo_frac > 0 and demo is not None) else 0
    nb_pos = cfg.batch - nb_demo
    nd = demo["U"].shape[0] if demo is not None else 0
    probe = None
    v_before = None
    drawn_ids = {}
    cfm_hist, prox_hist, fstep_hist = [], [], []
    stop = "max_inner"
    steps = 0
    for k in range(cfg.max_inner):
        out = store.sample_pos(nb_pos, rng)
        G, L, H, U, ids = out
        for q in ids:
            drawn_ids[q] = drawn_ids.get(q, 0) + 1
        G, L, H, U = G.to(device), L.to(device), H.to(device), U.to(device)
        if nb_demo > 0:
            di = torch.as_tensor(rng.integers(0, nd, nb_demo))
            G = torch.cat([G, demo["grid"][di].to(device)])
            L = torch.cat([L, demo["low5"][di].to(device)])
            H = torch.cat([H, demo["hist"][di].to(device)])
            U = torch.cat([U, demo["U"][di].to(device)])
        if probe is None:                                   # fixed functional probe = first batch
            na = min(U.shape[0], 128)
            xa = 0.5 * (U[:na] / policy.u_max).reshape(na, policy.d)
            ta = torch.full((na,), 0.5, device=device)
            ctxa = policy.ctx_from(G[:na], L[:na], H[:na]).detach()
            with torch.no_grad():
                v_before = policy.forward(xa, ta, policy._expand_ctx(ctxa, na)).detach()
            probe = (xa, ta, ctxa, na)
        cfm = policy.cfm_loss(U, policy.ctx_from(G, L, H))
        prox = sum(((p - r) ** 2).sum() for p, r in zip(trainable, refs)) / (2.0 * cfg.eta)
        loss = cfm + prox
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        steps = k + 1
        cfm_hist.append(float(cfm.detach()))
        prox_hist.append(float(prox.detach()))
        xa, ta, ctxa, na = probe
        with torch.no_grad():
            va = policy.forward(xa, ta, policy._expand_ctx(ctxa, na))
            fstep = float((va - v_before).norm(dim=1).mean() /
                          v_before.norm(dim=1).mean().clamp_min(1e-9))
        fstep_hist.append(fstep)
        if fstep >= cfg.fstep_stop:
            stop = "fstep_bound"
            break
    return dict(steps=steps, stop=stop, cfm=cfm_hist, prox=prox_hist, fstep=fstep_hist,
                fstep_final=fstep_hist[-1] if fstep_hist else 0.0,
                prox_over_cfm=(float(np.mean(prox_hist)) / max(float(np.mean(cfm_hist)), 1e-9)),
                drawn_ids=drawn_ids, n_distinct=len(drawn_ids),
                batch=(nb_pos, nb_demo))


# ------------------------------------------------------------------ closed-loop measurement
def measure_closed_loop(policy, env, cfg, device):
    with HT._preserve_torch_rng():
        rows, agg, paths = SR.eval_policy(policy, env, gammas=list(cfg.gammas), M=cfg.M_measure,
                                          T_max=cfg.T_eval, reach=cfg.reach, temp=1.0, device=device,
                                          keep_paths=cfg.M_measure, log=lambda *a, **k: None)
    goal = env.goal.detach().cpu().numpy()
    cov = {}
    for g in cfg.gammas:
        ids = set()
        for p in paths.get(g, []):
            p = np.asarray(p, float)
            if np.linalg.norm(p[-1] - goal) < cfg.reach:
                sid = GM2.staircase_id_goal(p, goal, reach=cfg.reach)
                if sid is not None:
                    ids.add(sid)
        cov[str(g)] = len(ids)
    return rows, agg, cov


# ------------------------------------------------------------------ viz db
def save_viz_db(path, round_i, store, q_start, gstats, upd, blr, audit, env):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    qids = list(range(q_start, len(store)))
    sids = [store.q_sid[q] for q in qids]
    drawn = (upd or {}).get("drawn_ids", {})
    db = dict(round=round_i,
              # this round's verified queries (the expansion mechanism)
              q_state=np.stack([store.ctx_state[s] for s in sids]) if qids else np.zeros((0, 4), np.float32),
              q_seg=np.stack([store.q_seg[q] for q in qids]) if qids else np.zeros((0, GF.H_PRED, 2), np.float16),
              q_y=np.asarray([store.q_y[q] for q in qids], np.int8),
              q_sigma=np.asarray([store.q_sigma[q] for q in qids], np.float32),
              q_gamma=np.asarray([store.q_gamma[q] for q in qids], np.float32),
              q_prog=np.asarray([store.q_prog[q] for q in qids], np.float32),
              q_margin=np.asarray([store.q_margin[q] for q in qids], np.float32),
              q_exec=np.asarray([store.q_exec[q] for q in qids], np.int8),
              # exactly which samples trained the model this round (cumulative D+ ids + draw counts)
              train_ids=np.asarray(sorted(drawn.keys()), np.int64),
              train_counts=np.asarray([drawn[k] for k in sorted(drawn.keys())], np.int64),
              train_state=(np.stack([store.ctx_state[store.q_sid[q]] for q in sorted(drawn.keys())])
                           if drawn else np.zeros((0, 4), np.float32)),
              train_y_round=np.asarray([store.q_round[q] for q in sorted(drawn.keys())], np.int32),
              train_gamma=np.asarray([store.q_gamma[q] for q in sorted(drawn.keys())], np.float32),
              # executed episodes (shielded closed loop)
              ep_paths=[e["path"] for e in gstats["episodes"]],
              ep_gamma=[e["gamma"] for e in gstats["episodes"]],
              ep_reached=[e["reached"] for e in gstats["episodes"]],
              ep_dead=[e["dead"] for e in gstats["episodes"]],
              ep_fb=[e["fb_mask"] for e in gstats["episodes"]],
              # uncertainty state (post-round) for sigma-field rendering
              A_inv=blr.A_inv.clone(), blr_n=int(blr.n),
              audit=audit,
              n_D=len(store), n_Dpos=store.n_pos(),
              goal=env.goal.detach().cpu().numpy(), x0=env.x0.detach().cpu().numpy())
    torch.save(db, path)


# ------------------------------------------------------------------ run
def run_afe(policy, phi0, env, cfg, device, outdir, log=print):
    os.makedirs(outdir, exist_ok=True)
    t_start = time.time()
    blr = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    store = AC.DStore()
    fb = AC.SafeMPPIFallback(env)
    rng = np.random.default_rng(cfg.seed)
    goal_np = env.goal.detach().cpu().numpy()
    demo = None
    if cfg.demo_frac > 0:
        import pretrain_repr as PR
        G, L, H, U = PR.load_data(cfg.demo_prefix, [str(g) for g in cfg.gammas], cfg.demo_cap)
        demo = dict(grid=G, low5=L, hist=H, U=U)
        log(f"[afe] arm-B demo replay: {U.shape[0]} windows ({cfg.demo_prefix}) frac {cfg.demo_frac}")
    # frozen-encoder training (every winning recipe; phi0 is a separate frozen copy for sigma)
    for p in policy.encoder_modules():
        p.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.lr)
    audit_ctxs = AC.build_audit_contexts(env, cfg.gammas, n_pos=cfg.audit_pos)
    recipe = dict(algorithm="afe_minimal_2026_07_16",
                  object_identity="planned_window==queried==verified==trained",
                  verifier="full window SOCP + in-bounds BEFORE execution (n_theta=%d)" % cfg.n_theta,
                  sigma="BLR on frozen phi_s^0 (s=%.2f), cumulative A_n over ALL verified queries" % cfg.s,
                  acquisition="pi ~ exp((sigma-max)/beta), B w/o replacement", K=cfg.K, B=cfg.B,
                  beta=cfg.beta, lam=cfg.lam, exec_rule=cfg.exec_rule,
                  fallback="certified SafeMPPI (mode1)",
                  update="uniform cumulative D+ replay, prox eta=%g, lr=%g, batch=%d, "
                         "stop=fstep>=%g or %d steps" % (cfg.eta, cfg.lr, cfg.batch,
                                                         cfg.fstep_stop, cfg.max_inner),
                  no_curriculum=True, demo_frac=cfg.demo_frac, gammas=list(cfg.gammas),
                  episodes_per_round=cfg.episodes_per_round, T=cfg.T, reach=cfg.reach,
                  wall_plugs=cfg.wall_plugs, start_eps=cfg.start_eps, goal_xy=list(cfg.goal_xy),
                  seed=cfg.seed, frozen_encoder=True,
                  audit=dict(every=cfg.audit_every, n_pos=cfg.audit_pos, plans=cfg.audit_plans,
                             held_out=True, never_buffered=True))
    with open(os.path.join(outdir, "recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2)

    def probe_write(rec):
        with open(os.path.join(outdir, "probe.jsonl"), "a") as f:
            f.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v)
                                for k, v in rec.items()}) + "\n")

    history = []
    # round-0 baseline: untilted audit + closed-loop measure of the pretrained
    audit0 = AC.run_audit(policy, audit_ctxs, env, goal_np, device, n_plans=cfg.audit_plans,
                          nfe=cfg.nfe, n_theta=cfg.n_theta, seed=cfg.seed)
    rows0, agg0, cov0 = measure_closed_loop(policy, env, cfg, device)
    log(f"round000 BASELINE V {audit0['V']:.3f} Vprog {audit0['Vprog']:.3f} | "
        f"SR {agg0['SR']:.2f} CR {agg0['CR']:.2f} | "
        f"Vg {' '.join(f'{k}:{v:.2f}' for k, v in audit0['V_gamma'].items())}", flush=True)
    probe_write(dict(round=0, V=audit0["V"], Vprog=audit0["Vprog"], V_gamma=audit0["V_gamma"],
                     Vprog_gamma=audit0["Vprog_gamma"], V_rest=audit0.get("V_rest"),
                     V_adverse=audit0.get("V_adverse"),
                     V_gamma_adverse=audit0.get("V_gamma_adverse"), SR=agg0["SR"], CR=agg0["CR"],
                     cov=cov0, n_D=0, n_Dpos=0))
    history.append(dict(round=0, SR=agg0["SR"], CR=agg0["CR"], V=audit0["V"],
                        rows={str(g): rows0[g] for g in cfg.gammas}, cov=cov0))

    for n in range(1, cfg.rounds + 1):
        t0 = time.time()
        q_start = len(store)
        gstats = gather_round(policy, phi0, blr, store, env, cfg, fb, n, device)
        t_gather = time.time() - t0
        t0 = time.time()
        upd = prox_update(policy, opt, store, demo, cfg, device, rng)
        t_upd = time.time() - t0
        # ---- tracking ----
        a_hat = gstats["n_acc"] / max(gstats["n_q"], 1)
        fb_rate = gstats["n_fb"] / max(gstats["n_fb"] + gstats["n_exec"], 1)
        npp = np.asarray(gstats.pop("new_pos_prog"), float)
        dither_new = float((npp < cfg.dither_bar).mean()) if npp.size else float("nan")
        all_pos_prog = np.asarray([store.q_prog[q] for q in store.pos_ids], float)
        dither_cum = float((all_pos_prog < cfg.dither_bar).mean()) if all_pos_prog.size else float("nan")
        audit = None
        if n % cfg.audit_every == 0 or n == cfg.rounds:
            audit = AC.run_audit(policy, audit_ctxs, env, goal_np, device, n_plans=cfg.audit_plans,
                                 nfe=cfg.nfe, n_theta=cfg.n_theta, seed=cfg.seed)
        rows = agg = cov = None
        if n % cfg.measure_every == 0 or n == cfg.rounds:
            rows, agg, cov = measure_closed_loop(policy, env, cfg, device)
        rec = dict(round=n, n_D=len(store), n_Dpos=store.n_pos(),
                   a_hat=a_hat, acc_g=gstats["acc_g"], fb_rate=fb_rate, fb_g=gstats["fb_g"],
                   n_q=gstats["n_q"], n_socp_err=gstats["n_socp_err"],
                   sigma_drawn_mean=gstats["sigma_drawn_mean"],
                   sigma_drawn_min=gstats["sigma_drawn_min"],
                   ep_reached=int(sum(e["reached"] for e in gstats["episodes"])),
                   ep_dead=int(sum(e["dead"] for e in gstats["episodes"])),
                   ep_steps=float(np.mean([e["steps"] for e in gstats["episodes"]])),
                   dither_new=dither_new, dither_cum=dither_cum,
                   t_gather=round(t_gather, 1), t_update=round(t_upd, 1))
        if upd is not None:
            rec.update(inner_steps=upd["steps"], stop=upd["stop"], fstep=upd["fstep_final"],
                       cfm=float(np.mean(upd["cfm"])), prox_over_cfm=upd["prox_over_cfm"],
                       n_train_distinct=upd["n_distinct"])
        if audit is not None:
            rec.update(V=audit["V"], Vprog=audit["Vprog"], V_gamma=audit["V_gamma"],
                       Vprog_gamma=audit["Vprog_gamma"], V_rest=audit.get("V_rest"),
                       V_adverse=audit.get("V_adverse"),
                       V_gamma_adverse=audit.get("V_gamma_adverse"))
        if agg is not None:
            rec.update(SR=agg["SR"], CR=agg["CR"], cov=cov)
            history.append(dict(round=n, SR=agg["SR"], CR=agg["CR"],
                                V=(audit["V"] if audit else None),
                                rows={str(g): rows[g] for g in cfg.gammas}, cov=cov))
            with open(os.path.join(outdir, "history.json"), "w") as f:
                json.dump(history, f)
        probe_write(rec)
        msg = (f"round{n:03d} D {len(store)} D+ {store.n_pos()} | a^ {a_hat:.2f} fb {fb_rate:.2f} | "
               f"upd {0 if upd is None else upd['steps']}st {rec.get('stop', '-')} "
               f"fstep {rec.get('fstep', 0):.3f} | dither {dither_new if not np.isnan(dither_new) else -1:.2f} "
               f"| {t_gather:.0f}s+{t_upd:.0f}s")
        if audit is not None:
            msg += f" | V {audit['V']:.3f} Vprog {audit['Vprog']:.3f}"
        if agg is not None:
            msg += f" | SR {agg['SR']:.2f} CR {agg['CR']:.2f}"
        log(msg, flush=True)
        if cfg.viz_every and n % cfg.viz_every == 0:
            save_viz_db(os.path.join(outdir, "viz_db", f"round{n}.pt"), n, store, q_start,
                        gstats, upd, blr, audit, env)
        if n % cfg.ckpt_every == 0:
            HT._save_hp_atomic(policy, os.path.join(outdir, f"ckpt_{n}.pt"),
                               extra={"iter": n, "recipe": recipe, "resumable": False,
                                      "afe_blr": blr.state_dict()})
        if cfg.dstore_every and n % cfg.dstore_every == 0:
            store.save(os.path.join(outdir, "dstore.pt"))
        if (time.time() - t_start) / 3600.0 > cfg.max_hours:
            log(f"round{n:03d} max_hours {cfg.max_hours} reached -- stopping gracefully", flush=True)
            break

    HT._save_hp_atomic(policy, os.path.join(outdir, "final.pt"),
                       extra={"iter": n, "recipe": recipe, "resumable": False,
                              "afe_blr": blr.state_dict(), "history_tail": history[-1]})
    store.save(os.path.join(outdir, "dstore.pt"))
    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    log(f"[afe] DONE {n} rounds, D {len(store)} (D+ {store.n_pos()}), saved final.pt", flush=True)


# ------------------------------------------------------------------ component probe
def component_probe(policy, phi0, env, cfg, device, log=print):
    """Measured (not assumed) pre-flight: module resolution, grid reconstruction exactness, SOCP and
    fallback latency, frozen-phi0 sigma separation, and eta calibration on a real mini D+."""
    import verifier_polytope as VP
    log(f"[probe] verifier_polytope <- {VP.__file__}")
    log(f"[probe] grid_metrics2   <- {GM2.__file__}")
    log(f"[probe] sr_cr_eval      <- {SR.__file__}")
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal_np = env.goal.detach().cpu().numpy()
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    grid_np = GF.axis_grid(st[:2], obs, rr)
    l5 = GF.low5(st, goal_np, 0.5)
    h_np = GF.hist_pad(np.zeros((0, 2)), GF.K_HIST)
    gT = torch.tensor(grid_np, device=device)
    lT = torch.tensor(l5, device=device)
    hT = torch.tensor(h_np, device=device)
    # (1) hp-channel reconstruction is exact for this model
    g3 = torch.zeros_like(gT)
    g3[2:3] = gT[2:3]
    with torch.no_grad():
        d = float((policy.ctx_from(gT, lT, hT) - policy.ctx_from(g3, lT, hT)).abs().max())
    log(f"[probe] ctx_from(hp-only grid) max|diff| = {d:.2e} (must be 0)")
    assert d == 0.0
    # (2) SOCP latency at both resolutions
    U = policy.sample_window(gT, lT, hT, n=20, temp=1.0, nfe=cfg.nfe)
    for nt in (180, 120):
        t0 = time.time()
        oks = [AC.verify_plan(st, U[j].cpu().numpy(), env, 0.5, goal_np, n_theta=nt)["y"]
               for j in range(U.shape[0])]
        dt = (time.time() - t0) / U.shape[0] * 1000
        log(f"[probe] verify_plan n_theta={nt}: {dt:.1f} ms/plan (acc {np.mean(oks):.2f} @ gamma .5)")
    # per-gamma acceptance of the raw pretrained at the start context
    for g in cfg.gammas:
        accs = [AC.verify_plan(st, U[j].cpu().numpy(), env, g, goal_np, n_theta=cfg.n_theta)["y"]
                for j in range(U.shape[0])]
        log(f"[probe] pretrained acceptance @start gamma {g}: {np.mean(accs):.2f}")
    # (3) fallback latency
    fbk = AC.SafeMPPIFallback(env)
    fbk.reset()
    t0 = time.time()
    for t in range(5):
        fbk.plan(st, 0.5, seed=t)
    log(f"[probe] SafeMPPI fallback: {(time.time() - t0) / 5 * 1000:.1f} ms/step")
    # (4) frozen-phi0 sigma separation: fill A_n with start-context queries, check a far context
    blr = AC.BLRSigma(dim=policy.repr_dim or policy.width, lam=cfg.lam)
    Z0 = AC.frozen_feat(phi0, U, gT, lT, hT, s=cfg.s)
    s_before = blr.sigma(Z0).mean()
    U2 = policy.sample_window(gT, lT, hT, n=100, temp=1.0, nfe=cfg.nfe)
    blr.update(AC.frozen_feat(phi0, U2, gT, lT, hT, s=cfg.s))
    s_after = blr.sigma(Z0).mean()
    st_far = np.array([4.0, 1.0, 0.0, 0.0], np.float32)     # off-diagonal far context
    gF = torch.tensor(GF.axis_grid(st_far[:2], obs, rr), device=device)
    lF = torch.tensor(GF.low5(st_far, goal_np, 0.5), device=device)
    UF = policy.sample_window(gF, lF, hT, n=20, temp=1.0, nfe=cfg.nfe)
    s_far = blr.sigma(AC.frozen_feat(phi0, UF, gF, lF, hT, s=cfg.s)).mean()
    st_diag = np.array([2.5, 2.5, 0.0, 0.0], np.float32)    # ON-diagonal (phi0 pretrained off-diag)
    gD = torch.tensor(GF.axis_grid(st_diag[:2], obs, rr), device=device)
    lD = torch.tensor(GF.low5(st_diag, goal_np, 0.5), device=device)
    UD = policy.sample_window(gD, lD, hT, n=20, temp=1.0, nfe=cfg.nfe)
    s_diag = blr.sigma(AC.frozen_feat(phi0, UD, gD, lD, hT, s=cfg.s)).mean()
    log(f"[probe] sigma: start {s_before:.3f} -> {s_after:.3f} after 100 queries there | "
        f"far-offdiag {s_far:.3f} | on-diag {s_diag:.3f} (separation = queried low, unqueried high)")
    # (5) eta calibration: mini-gather then the REAL prox solver at 3 etas from the same snapshot
    cfg_mini = copy.deepcopy(cfg)
    cfg_mini.episodes_per_round = 2
    cfg_mini.T = 60
    store = AC.DStore()
    gst = gather_round(policy, phi0, blr, store, env, cfg_mini, fbk, 1, device)
    log(f"[probe] mini-gather: {gst['n_q']} queries, {store.n_pos()} positives, "
        f"fb {gst['n_fb']}/{gst['n_fb'] + gst['n_exec']}")
    if store.n_pos() >= 32:
        snap = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        opt_probe = None
        for eta in (float("inf"), 0.1, cfg.eta, cfg.eta / 10):
            policy.load_state_dict(snap)
            trainable = [p for p in policy.parameters() if p.requires_grad]
            opt_probe = torch.optim.Adam(trainable, lr=cfg.lr)
            cfg_eta = copy.deepcopy(cfg)
            cfg_eta.eta = eta if np.isfinite(eta) else 1e18
            upd = prox_update(policy, opt_probe, store, None, cfg_eta, device,
                              np.random.default_rng(0))
            log(f"[probe] eta {eta}: {upd['steps']} steps ({upd['stop']}), fstep "
                f"{upd['fstep_final']:.4f}, prox/cfm {upd['prox_over_cfm']:.3f}, "
                f"cfm {upd['cfm'][0]:.3f}->{upd['cfm'][-1]:.3f}")
        policy.load_state_dict(snap)
    log("[probe] done")


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--T-eval", type=int, default=350)
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--n-theta", type=int, default=180)
    ap.add_argument("--exec-rule", choices=["progress", "pi"], default="progress")
    ap.add_argument("--gammas", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-inner", type=int, default=40)
    ap.add_argument("--fstep-stop", type=float, default=0.03)
    ap.add_argument("--demo-frac", type=float, default=0.0)
    ap.add_argument("--demo-prefix", default="dr05_")
    ap.add_argument("--audit-every", type=int, default=5)
    ap.add_argument("--audit-pos", type=int, default=12)
    ap.add_argument("--audit-plans", type=int, default=4)
    ap.add_argument("--measure-every", type=int, default=10)
    ap.add_argument("--M-measure", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--viz-every", type=int, default=1)
    ap.add_argument("--wall-plugs", type=int, choices=[0, 2, 4, 8], default=8)
    ap.add_argument("--start-eps", type=float, default=0.3)
    ap.add_argument("--goal-xy", type=float, nargs=2, default=[4.7, 4.7])
    ap.add_argument("--seed", type=int, default=910)
    ap.add_argument("--max-hours", type=float, default=20.0)
    ap.add_argument("--probe", action="store_true", help="component checks only, no training run")
    args = ap.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ck = HP.load_hp(args.ckpt, device="cpu")
    policy = policy.to(dev)
    phi0 = copy.deepcopy(policy).to(dev).eval()             # FROZEN representation for sigma
    for p in phi0.parameters():
        p.requires_grad_(False)
    env = GS.make_grid()
    env = HT._apply_wall_plugs(env, args.wall_plugs)
    if args.start_eps > 0:
        env.x0 = torch.tensor([args.start_eps, args.start_eps, 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=env.goal.dtype)
    GM2.GOAL_XY = np.array([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=float)
    cfg = AFEConfig(rounds=args.rounds, episodes_per_round=args.episodes, T=args.T,
                    T_eval=args.T_eval, reach=args.reach, K=args.K, B=args.B, beta=args.beta,
                    lam=args.lam, n_theta=args.n_theta, exec_rule=args.exec_rule,
                    gammas=tuple(args.gammas), lr=args.lr, eta=args.eta, batch=args.batch,
                    max_inner=args.max_inner, fstep_stop=args.fstep_stop,
                    demo_frac=args.demo_frac, demo_prefix=args.demo_prefix,
                    audit_every=args.audit_every, audit_pos=args.audit_pos,
                    audit_plans=args.audit_plans, measure_every=args.measure_every,
                    M_measure=args.M_measure, ckpt_every=args.ckpt_every, viz_every=args.viz_every,
                    wall_plugs=args.wall_plugs, start_eps=args.start_eps,
                    goal_xy=tuple(args.goal_xy), seed=args.seed, max_hours=args.max_hours)
    print(f"[afe] ckpt {os.path.basename(args.ckpt)} repr {ck['config'].get('repr_dim')} dev {dev} "
          f"K{cfg.K} B{cfg.B} beta {cfg.beta} eta {cfg.eta} exec {cfg.exec_rule} "
          f"demo {cfg.demo_frac}", flush=True)
    if args.probe:
        component_probe(policy, phi0, env, cfg, dev)
        return
    run_afe(policy, phi0, env, cfg, dev, args.outdir)


if __name__ == "__main__":
    main()
