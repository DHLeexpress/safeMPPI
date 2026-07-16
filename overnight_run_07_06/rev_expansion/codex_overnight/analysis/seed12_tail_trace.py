"""Seed-12 origin tail + near-goal overshoot: NFE=8 ODE trace and latent-tail density probe (read-only).

Answers, with numbers, the handoff question: is each faithful failure stratum caused by
(a) ORIGIN-DATA WEIGHTING (a mean shift: most base latents map OOB at the failing context), or
(b) RARE BASE-NOISE-TAIL COVERAGE (only a small latent set maps OOB and the eval seed hits it)?

Parts:
  A. Exact reproduction of the 11 fixed t104 M25 failures (seed mechanics: torch.manual_seed(seed) then
     faithful fm_deploy).  The deploy loop is re-implemented locally with IDENTICAL RNG consumption
     (one randn(1,d) per replan) so every NFE=8 Euler state x0..x8 is captured; the traced path is
     asserted equal to the true GR.fm_deploy path.
  B. Same-latent-through-checkpoints: the exact failing x0 at the step-0 origin context is integrated
     through pretrained / rollback it100 / corrected t103 / t104 (context tensors at step 0 are
     policy-independent, so the latent fiber is directly comparable).
  C. Latent-tail density: at the failing contexts (origin step 0, last in-bounds step, and each
     near-goal last in-bounds context), N fresh latents are integrated no-grad; per latent we score
     one-step OOB (executing U[0] exits the task box) and window OOB (10-step open-loop exits before
     reaching).  Small OOB fraction containing the failing latent => (b) rare tail; large => (a) shift.

Outputs: analysis/seed12_tail_trace.json / .md and figures/seed12_trace.png.
Never modifies training code, checkpoints, or Valid2.  Faithful settings only (temp=1, NFE=8, reach=.1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # analysis/
_PKG = os.path.dirname(_HERE)                               # codex_overnight/
_REV = os.path.dirname(_PKG)                                # rev_expansion/
_WORK = os.path.dirname(_REV)                               # overnight_run_07_06/
sys.path.insert(0, _WORK)
sys.path.insert(0, _REV)
sys.path.insert(0, _PKG)

import numpy as np
import torch

import _paths  # noqa: F401
import grid_scene as GS
import grid_rollout as GR
import grid_feats as GF
import grid_metrics as GM
import grid_hp_expt as HP

GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
ORIGIN_SEED = 12
NEAR_GOAL = [(0.1, 22), (0.4, 8), (0.5, 3), (0.7, 5)]        # t104 M25 near-goal overshoots
LO, HI = -float(GM.EPS_TASK), float(GM.GRID_M) + float(GM.EPS_TASK)


def _ctx_tensors(st, goal, gamma, hist, env, device):
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    grid_np = GF.axis_grid(st[:2], obs, rr)
    l5_np = GF.low5(st, goal, gamma)
    h_np = GF.hist_pad(np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)), GF.K_HIST)
    return (torch.tensor(grid_np, device=device), torch.tensor(l5_np, device=device),
            torch.tensor(h_np, device=device))


def _ctx_of(policy, gT, lT, hT):
    ctx = policy.ctx_from(gT, lT, hT)
    if ctx.shape[0] == 1:
        ctx = ctx[0]
    return ctx


@torch.no_grad()
def integrate(policy, ctx, x0, nfe=8, keep_states=False):
    """Euler NFE loop of FlowPolicy.sample for a GIVEN x0 [n,d]; consumes NO RNG."""
    n = x0.shape[0]
    ctx_e = policy._expand_ctx(ctx, n)
    x = x0.clone()
    states = [x.clone()] if keep_states else None
    for i in range(nfe):
        tau = torch.full((n,), i / nfe, device=x.device)
        x = x + (1.0 / nfe) * policy.forward(x, tau, ctx_e)
        if keep_states:
            states.append(x.clone())
    U = (x.reshape(n, policy.T, 2) * policy.u_max).clamp(-policy.u_max, policy.u_max)
    return U, states


@torch.no_grad()
def trace_deploy(policy, env, gamma, seed, T=250, nfe=8, reach=0.1, device="cuda"):
    """Faithful fm_deploy re-implementation with per-replan latent/Euler capture.
    RNG consumption is IDENTICAL to GR.fm_deploy(tilt=None): one randn(1,d) per replan step."""
    torch.manual_seed(seed)
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    hist, path, steps = [], [st[:2].copy()], []
    reached = dead = False
    for t in range(T):
        gT, lT, hT = _ctx_tensors(st, goal, gamma, hist, env, device)
        ctx = _ctx_of(policy, gT, lT, hT)
        x0 = 1.0 * torch.randn(1, policy.d, device=device)       # == FlowPolicy.sample temp*randn
        U, xs = integrate(policy, ctx, x0, nfe=nfe, keep_states=True)
        U = U[0].detach().cpu().numpy()
        a = U[0]
        steps.append(dict(state=st.copy(), grid=gT.detach().cpu().numpy(),
                          low5=lT.detach().cpu().numpy(), hist=hT.detach().cpu().numpy(),
                          x0=x0[0].detach().cpu().numpy(),
                          euler=[s[0].detach().cpu().numpy() for s in xs],
                          U=U.copy()))
        st = GR.di_step(st, np.asarray(a, np.float32), dt=env.dt)
        hist.append(np.asarray(a, np.float32))
        path.append(st[:2].copy())
        if np.linalg.norm(st[:2] - goal) < reach:
            reached = True; break
        if (st[:2] < -GM.EPS_TASK).any() or (st[:2] > GM.GRID_M + GM.EPS_TASK).any():
            dead = True; break
        if len(obs) and (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min() < 0.0:
            dead = True; break
    return dict(path=np.array(path, np.float32), reached=reached, dead=dead, steps=steps)


def verify_trace(policy, env, gamma, seed, tr, T=250, nfe=8, reach=0.1, device="cuda"):
    torch.manual_seed(seed)
    out = GR.fm_deploy(policy, env, float(gamma), T=T, temp=1.0, nfe=nfe, tilt=None,
                       reach=reach, device=device)
    same_len = len(out["path"]) == len(tr["path"])
    ok = same_len and np.allclose(out["path"], tr["path"], atol=1e-5)
    return bool(ok), out


@torch.no_grad()
def density_probe(policy, ctx, st, goal, env, n_lat=512, nfe=8, reach=0.1, probe_seed=7777,
                  device="cuda", x_fail=None):
    """N fresh latents at a fixed context: one-step OOB, window OOB-before-reach, U0 stats."""
    torch.manual_seed(probe_seed)
    X0 = torch.randn(n_lat, policy.d, device=device)
    if x_fail is not None:                                       # put the failing latent at row 0
        X0[0] = torch.as_tensor(x_fail, device=device)
    U, _ = integrate(policy, ctx, X0, nfe=nfe)
    U_np = U.detach().cpu().numpy()
    pos = GR.di_rollout_batch(st, U_np, env.dt)                  # [n, H, 2] open-loop positions
    one = st[None, :2] + env.dt * U_np[:, 0, :]                  # one-step executed position
    one_oob = (one[:, 0] < LO) | (one[:, 0] > HI) | (one[:, 1] < LO) | (one[:, 1] > HI)
    oob_t = ((pos < LO) | (pos > HI)).any(axis=2)                # [n, H]
    reach_t = np.linalg.norm(pos - goal[None, None, :], axis=2) < reach
    H = pos.shape[1]
    first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), H + 1)
    win_oob = first(oob_t) < first(reach_t)                      # exits before (ever) reaching
    return dict(one_step_oob=float(one_oob.mean()), window_oob=float(win_oob.mean()),
                u0_y_mean=float(U_np[:, 0, 1].mean()), u0_y_std=float(U_np[:, 0, 1].std()),
                u0_x_mean=float(U_np[:, 0, 0].mean()),
                fail_latent_one_step_oob=bool(one_oob[0]) if x_fail is not None else None,
                fail_latent_window_oob=bool(win_oob[0]) if x_fail is not None else None,
                fail_latent_norm_pct=float((np.linalg.norm(X0.detach().cpu().numpy(), axis=1)
                                            <= np.linalg.norm(x_fail)).mean() * 100)
                if x_fail is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-latents", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(_HERE, "seed12_tail_trace"))
    ap.add_argument("--fig", default=os.path.join(_PKG, "figures", "seed12_trace.png"))
    args = ap.parse_args()
    dev = args.device

    CKPTS = [
        ("pretrained", os.path.join(_WORK, "results", "hp_repr", "pretrained_a32uni.pt")),
        ("it100", os.path.join(_PKG, "results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt")),
        ("t103", os.path.join(_PKG, "results/p2/corrected_mode2_target50_s81_to106/ckpt_103.pt")),
        ("t104", os.path.join(_PKG, "results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt")),
    ]
    env = GS.make_grid()
    goal = env.goal.detach().cpu().numpy()
    pols = {}
    for name, p in CKPTS:
        pol, _ = HP.load_hp(p, device=dev)
        pol.eval()
        pols[name] = pol
    res = {"parts": {}, "checkpoints": {k: v for k, v in CKPTS}}

    # ---------- Part A: reproduce + trace the 11 fixed failures at t104 ----------
    print("[A] tracing t104 failures ...", flush=True)
    A = {"origin": {}, "near_goal": {}, "verify_ok": True}
    traces = {}
    for g in GAMMAS:
        tr = trace_deploy(pols["t104"], env, g, ORIGIN_SEED, device=dev)
        ok, _ = verify_trace(pols["t104"], env, g, ORIGIN_SEED, tr, device=dev)
        A["verify_ok"] &= ok
        end = tr["path"][-1]
        A["origin"][str(g)] = dict(steps=len(tr["path"]) - 1, dead=tr["dead"], reached=tr["reached"],
                                   endpoint=[float(end[0]), float(end[1])], verified=ok)
        traces[("origin", g)] = tr
        print(f"  origin seed12 γ{g}: steps={len(tr['path'])-1} dead={tr['dead']} end={end.round(3)} verify={ok}",
              flush=True)
    for g, s in NEAR_GOAL:
        tr = trace_deploy(pols["t104"], env, g, s, device=dev)
        ok, _ = verify_trace(pols["t104"], env, g, s, tr, device=dev)
        A["verify_ok"] &= ok
        end = tr["path"][-1]
        A["near_goal"][f"g{g}_s{s}"] = dict(steps=len(tr["path"]) - 1, dead=tr["dead"],
                                            reached=tr["reached"],
                                            endpoint=[float(end[0]), float(end[1])], verified=ok)
        traces[("goal", g, s)] = tr
        print(f"  near-goal γ{g} s{s}: steps={len(tr['path'])-1} dead={tr['dead']} end={end.round(3)} verify={ok}",
              flush=True)
    res["parts"]["A"] = A

    # ---------- Part B: the same origin latent through all checkpoints ----------
    print("[B] same-latent fiber across checkpoints ...", flush=True)
    B = {}
    for g in GAMMAS:
        tr = traces[("origin", g)]
        s0 = tr["steps"][0]
        gT = torch.tensor(s0["grid"], device=dev); lT = torch.tensor(s0["low5"], device=dev)
        hT = torch.tensor(s0["hist"], device=dev)
        x_fail = torch.as_tensor(s0["x0"], device=dev)[None]
        row = {}
        for name, pol in pols.items():
            ctx = _ctx_of(pol, gT, lT, hT)
            U, _ = integrate(pol, ctx, x_fail, nfe=8)
            U0 = U[0, 0].detach().cpu().numpy()
            pos = GR.di_rollout_batch(s0["state"], U[0].detach().cpu().numpy()[None], env.dt)[0]
            row[name] = dict(u0=[float(U0[0]), float(U0[1])],
                             window_min_y=float(pos[:, 1].min()),
                             window_oob=bool(((pos < LO) | (pos > HI)).any()))
        B[str(g)] = row
        print(f"  γ{g}: " + "  ".join(f"{k}: u0_y={v['u0'][1]:+.3f} minY={v['window_min_y']:+.3f}"
                                       for k, v in row.items()), flush=True)
    res["parts"]["B"] = B

    # ---------- Part C: latent-tail density at the failing contexts ----------
    print("[C] latent-tail density probes ...", flush=True)
    C = {"origin_step0": {}, "origin_lastin": {}, "near_goal_lastin": {}}
    for g in GAMMAS:
        tr = traces[("origin", g)]
        for key, si in (("origin_step0", 0), ("origin_lastin", len(tr["steps"]) - 1)):
            s = tr["steps"][si]
            gT = torch.tensor(s["grid"], device=dev); lT = torch.tensor(s["low5"], device=dev)
            hT = torch.tensor(s["hist"], device=dev)
            row = {}
            for name, pol in pols.items():
                ctx = _ctx_of(pol, gT, lT, hT)
                row[name] = density_probe(pol, ctx, s["state"], goal, env, n_lat=args.n_latents,
                                          probe_seed=7777 + int(g * 100) + si, device=dev,
                                          x_fail=s["x0"])
            C[key][str(g)] = dict(step=si, state=[float(s["state"][0]), float(s["state"][1])], **{"by_ckpt": row})
        r4 = C["origin_lastin"][str(g)]["by_ckpt"]["t104"]
        print(f"  γ{g} origin last-in-bounds: 1step_oob={r4['one_step_oob']:.3f} "
              f"win_oob={r4['window_oob']:.3f} fail∈oob={r4['fail_latent_window_oob']}", flush=True)
    for g, sd in NEAR_GOAL:
        tr = traces[("goal", g, sd)]
        s = tr["steps"][-1]
        gT = torch.tensor(s["grid"], device=dev); lT = torch.tensor(s["low5"], device=dev)
        hT = torch.tensor(s["hist"], device=dev)
        row = {}
        for name, pol in pols.items():
            ctx = _ctx_of(pol, gT, lT, hT)
            row[name] = density_probe(pol, ctx, s["state"], goal, env, n_lat=args.n_latents,
                                      probe_seed=8888 + int(g * 100) + sd, device=dev, x_fail=s["x0"])
        C["near_goal_lastin"][f"g{g}_s{sd}"] = dict(step=len(tr["steps"]) - 1,
                                                    state=[float(s["state"][0]), float(s["state"][1])],
                                                    by_ckpt=row)
        r4 = row["t104"]
        print(f"  near-goal γ{g} s{sd} last-in-bounds ({s['state'][0]:.2f},{s['state'][1]:.2f}): "
              f"1step_oob={r4['one_step_oob']:.3f} win_oob={r4['window_oob']:.3f} "
              f"fail1step={r4['fail_latent_one_step_oob']}", flush=True)
    res["parts"]["C"] = C

    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(3, 7, hspace=0.42, wspace=0.3)
    obs = env.obstacles.detach().cpu().numpy()
    for j, g in enumerate(GAMMAS):                                # row 0: origin seed-12 paths
        ax = fig.add_subplot(gs[0, j])
        for o in obs:
            ax.add_patch(plt.Circle(o[:2], o[2], color="#bbb", alpha=0.6))
        p = traces[("origin", g)]["path"]
        ax.plot(p[:, 0], p[:, 1], "-o", ms=2.5, lw=1.4, color="crimson")
        ax.axhline(-GM.EPS_TASK, color="k", ls="--", lw=0.8)
        ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.5, 2.0); ax.set_aspect("equal")
        ax.set_title(f"seed12 γ{g}  ({len(p)-1} steps)", fontsize=10)
        if j == 0:
            ax.set_ylabel("origin exit trace (t104)", fontsize=10)
    for j, g in enumerate(GAMMAS):                                # row 1: window-OOB frac per ckpt
        ax = fig.add_subplot(gs[1, j])
        names = list(pols.keys())
        v0 = [res["parts"]["C"]["origin_step0"][str(g)]["by_ckpt"][n]["window_oob"] for n in names]
        v1 = [res["parts"]["C"]["origin_lastin"][str(g)]["by_ckpt"][n]["window_oob"] for n in names]
        xx = np.arange(len(names))
        ax.bar(xx - 0.18, v0, 0.36, label="step 0", color="#4477aa")
        ax.bar(xx + 0.18, v1, 0.36, label="last in-bounds", color="#cc6677")
        ax.set_xticks(xx); ax.set_xticklabels(names, rotation=45, fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(f"γ{g} window-OOB frac", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{args.n_latents} fresh latents", fontsize=10); ax.legend(fontsize=7)
    for j, (g, sd) in enumerate(NEAR_GOAL):                       # row 2: near-goal cases
        ax = fig.add_subplot(gs[2, j])
        p = traces[("goal", g, sd)]["path"]
        for o in obs:
            ax.add_patch(plt.Circle(o[:2], o[2], color="#bbb", alpha=0.6))
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.0, color="darkorange")
        ax.plot(*p[-1], "x", ms=9, color="red")
        ax.plot(*goal, "*", ms=12, color="green")
        ax.axhline(GM.GRID_M + GM.EPS_TASK, color="k", ls="--", lw=0.8)
        ax.set_xlim(3.6, 5.4); ax.set_ylim(3.6, 5.4); ax.set_aspect("equal")
        ax.set_title(f"near-goal γ{g} s{sd}", fontsize=10)
    ax = fig.add_subplot(gs[2, 4:])
    names = list(pols.keys())
    for j, (g, sd) in enumerate(NEAR_GOAL):
        v = [res["parts"]["C"]["near_goal_lastin"][f"g{g}_s{sd}"]["by_ckpt"][n]["one_step_oob"] for n in names]
        ax.plot(names, v, "-o", label=f"γ{g} s{sd}")
    ax.set_ylabel("one-step OOB frac at last in-bounds ctx"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
    fig.suptitle("Seed-12 origin tail & near-goal overshoot — faithful NFE=8 trace + latent-tail density "
                 f"(N={args.n_latents}/context)", fontsize=13)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}.json and {args.fig}", flush=True)

    # ---------- verdict markdown ----------
    lines = ["# Seed-12 / near-goal latent-tail diagnosis", "",
             f"All 11 traced failures verified against true `GR.fm_deploy`: **{A['verify_ok']}**", "",
             "## Origin stratum (seed 12, all γ)", "",
             "| γ | exit steps | step-0 win-OOB (t104) | last-in win-OOB (t104) | 1-step OOB last-in | fail-latent in OOB set | u0_y mean (t104, step0) |",
             "|---|---|---|---|---|---|---|"]
    for g in GAMMAS:
        c0 = res["parts"]["C"]["origin_step0"][str(g)]["by_ckpt"]["t104"]
        c1 = res["parts"]["C"]["origin_lastin"][str(g)]["by_ckpt"]["t104"]
        a = A["origin"][str(g)]
        lines.append(f"| {g} | {a['steps']} | {c0['window_oob']:.3f} | {c1['window_oob']:.3f} | "
                     f"{c1['one_step_oob']:.3f} | {c1['fail_latent_window_oob']} | {c0['u0_y_mean']:+.3f} |")
    lines += ["", "## Near-goal stratum", "",
              "| case | last-in state | 1-step OOB (t104) | win-OOB (t104) | fail latent 1-step OOB | u0_y mean |",
              "|---|---|---|---|---|---|"]
    for g, sd in NEAR_GOAL:
        c = res["parts"]["C"]["near_goal_lastin"][f"g{g}_s{sd}"]
        c4 = c["by_ckpt"]["t104"]
        lines.append(f"| γ{g} s{sd} | ({c['state'][0]:.2f},{c['state'][1]:.2f}) | {c4['one_step_oob']:.3f} | "
                     f"{c4['window_oob']:.3f} | {c4['fail_latent_one_step_oob']} | {c4['u0_y_mean']:+.3f} |")
    lines += ["", "## Same-latent fiber across checkpoints (origin, u0_y / window-min-y)", "",
              "| γ | " + " | ".join(pols.keys()) + " |", "|---|" + "---|" * len(pols)]
    for g in GAMMAS:
        row = res["parts"]["B"][str(g)]
        lines.append(f"| {g} | " + " | ".join(f"{row[n]['u0'][1]:+.3f} / {row[n]['window_min_y']:+.3f}"
                                              for n in pols.keys()) + " |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out + ".md", flush=True)


if __name__ == "__main__":
    main()
