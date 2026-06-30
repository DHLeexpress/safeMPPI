"""Experiment viz: 3x5 grid (episodes x gamma) of single-integrator SafeMPPI, showing in REAL TIME the
control-space proposal -- accepted (green) / rejected (red) first-step samples, the mean (bold arrow) and the
covariance ellipse (control units, drawn at the robot) -- on top of the scene + polytope_v2 (with Suggestion 1's
velocity-predictive faces). Lets us see Suggestion 1 + mean/cov steering working.

  python overnight_run_2026-06-28/step2_sampling_grid.py --dataset ucy --predict-gain 2.0 --escape-gain 0.0
"""
from __future__ import annotations
import argparse, os, sys
from copy import deepcopy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2
from cfm_mppi.mppi.sweep import _load, velocity_inflate, _si_step, DT

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
GAMMAS = [0.1, 0.3, 0.5, 0.7, 1.0]
VSCALE = 0.6    # display scale: control (m/s) -> position units for arrows/ellipse


def norm_barrier(poly, grid):
    mr = (poly.b - poly.A @ poly.ref).clamp_min(1e-3)
    return ((poly.b.unsqueeze(0) - grid @ poly.A.T) / mr.unsqueeze(0)).min(1).values


def record(scene, gamma, cfg, predict_gain, steps=80, dev="cpu"):
    s0, goal, obs, vel = scene
    tau = float(cfg.get("horizon", 10)) * DT
    ad = SafeMPPIAdapter(**cfg)
    state = s0.astype(np.float32).copy(); T = obs.shape[0]; rec = []; reached = False
    gt = torch.tensor(goal, dtype=torch.float32, device=dev)
    for t in range(steps):
        ob = obs[min(t, T - 1)]; vl = vel[min(t, vel.shape[0] - 1)]
        ok = ~np.isnan(ob[:, :2]).any(1); ob = ob[ok]; vl = vl[ok] if vl.shape[0] == ok.shape[0] else vl
        if not reached:
            ob_plan = velocity_inflate(ob, vl, state[:2], predict_gain, tau)
            a, info = ad.plan(torch.tensor(state, dtype=torch.float32, device=dev), gt,
                              torch.tensor(ob_plan, dtype=torch.float32, device=dev), gamma=gamma,
                              obstacle_velocities=torch.tensor(vl, dtype=torch.float32, device=dev), seed=t)
            rec.append(dict(p=state[:2].copy(), v=state[2:4].copy(), crowd=ob.copy(), crowd_vel=vl.copy(),
                            samples=info["first_controls"], feas=info["feasible"],
                            mean=info["mean_control"], sigma=info["sigma"]))
            state = _si_step(state, a.detach().cpu().numpy(), DT)
            if np.linalg.norm(state[:2] - goal) < 0.5:
                reached = True
        else:
            rec.append(rec[-1] | {"p": state[:2].copy()})
    return rec, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy")
    ap.add_argument("--episodes", nargs="+", type=int, default=None)   # None = auto-pick long-navigation episodes
    ap.add_argument("--predict-gain", type=float, default=2.0)
    ap.add_argument("--escape-gain", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--show-samples", type=int, default=70)
    ap.add_argument("--sensing", type=float, default=3.5)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    base = dict(horizon=10, dt=DT, num_samples=args.samples, noise_sigma=(0.4, 0.4),
                u_min=(-2.0, -2.0), u_max=(2.0, 2.0), safety_margin=0.0, dynamics_type="singleintegrator",
                use_ho_barrier=False, eta=0.0, use_guidance=False, use_aniso_cov=False,
                barrier_topk=0, barrier_activation_radius=args.sensing, escape_gain=args.escape_gain)

    if args.episodes is None:                       # pick episodes with long ego travel + a real crowd
        cand = []
        for ep in range(60):
            s0, goal, obs, vel = _load(args.dataset, ep, 80)
            a = obs[..., :2].reshape(-1, 2); nc = int((~np.isnan(obs[:, :, 0]).all(0)).sum())
            travel = float(np.linalg.norm(goal - s0[:2]))
            if nc >= 6:
                cand.append((travel, ep))
        cand.sort(reverse=True); args.episodes = [ep for _, ep in cand[:3]]
    print("episodes:", args.episodes)

    recs = {}; scenes = {}
    for ep in args.episodes:
        s0, goal, obs, vel = _load(args.dataset, ep, 80); scenes[ep] = (s0, goal, obs, vel)
        for g in GAMMAS:
            recs[(ep, g)], _ = record((s0, goal, obs, vel), g, deepcopy(base), args.predict_gain, dev=args.device)
        print(f"recorded episode {ep}", flush=True)

    vR = args.sensing + 1.5   # robot-centered view half-width (so control samples/mean/cov are visible)
    R, C = len(args.episodes), len(GAMMAS)
    fig, axes = plt.subplots(R, C, figsize=(2.9 * C, 2.9 * R), squeeze=False)
    nF = max(len(recs[(args.episodes[0], GAMMAS[0])]), 1)

    def draw(f):
        for r, ep in enumerate(args.episodes):
            goal = scenes[ep][1]
            for c, g in enumerate(GAMMAS):
                ax = axes[r][c]; ax.clear()
                rec = recs[(ep, g)]; st = rec[min(f, len(rec) - 1)]
                p = st["p"]; peds = st["crowd"]; pv = st["crowd_vel"]
                xl = (p[0] - vR, p[0] + vR); yl = (p[1] - vR, p[1] + vR)   # robot-centered zoom
                poly, _ = build_polytope_v2(p, peds, sensing_range=args.sensing, n_base=16, margin=0.0,
                                            obstacle_velocities=pv, robot_velocity=st["v"],
                                            predict_gain=args.predict_gain, predict_tau=1.0)
                gx = np.linspace(*xl, 70); gy = np.linspace(*yl, 70); GX, GY = np.meshgrid(gx, gy)
                Hh = norm_barrier(poly, torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)).numpy().reshape(GX.shape)
                lv = sorted({round((1 - g) ** i, 4) for i in range(7)} | {0.0})
                ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.4, zorder=1)
                ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.0, zorder=2)
                for (ox, oy, rr) in peds:
                    ax.add_patch(Circle((ox, oy), rr, facecolor="#7b3294", alpha=0.4, edgecolor="#4d004b", lw=0.5, zorder=3))
                # control-space proposal at the robot (unit = control = velocity; endpoints scaled by VSCALE)
                samp = st["samples"]; feas = np.asarray(st["feas"], bool)
                idx = np.linspace(0, len(samp) - 1, min(args.show_samples, len(samp))).astype(int)
                ends = p[None, :] + VSCALE * samp[idx]                      # velocity endpoints [k,2]
                fe = feas[idx]
                ax.scatter(ends[fe, 0], ends[fe, 1], s=7, c="#1a9850", alpha=0.55, edgecolor="none", zorder=4,
                           label="accepted" if (r == 0 and c == 0) else None)
                ax.scatter(ends[~fe, 0], ends[~fe, 1], s=7, c="#d62728", alpha=0.55, edgecolor="none", zorder=4,
                           label="rejected" if (r == 0 and c == 0) else None)
                mean = st["mean"]; sig = st["sigma"]
                ax.add_patch(Ellipse((p[0] + VSCALE * mean[0], p[1] + VSCALE * mean[1]),
                                     2 * VSCALE * sig[0], 2 * VSCALE * sig[1], facecolor="none",
                                     edgecolor="#ff7f00", lw=1.4, zorder=7))
                ax.annotate("", xy=(p[0] + VSCALE * mean[0], p[1] + VSCALE * mean[1]), xytext=(p[0], p[1]),
                            arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.8), zorder=8)
                ax.scatter([p[0]], [p[1]], s=30, c="k", zorder=9)
                ax.scatter([goal[0]], [goal[1]], marker="*", s=70, c="#d62728", edgecolor="k", zorder=9)
                ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"γ={g}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"ep {ep}", fontsize=9)
        fig.suptitle(f"SI SafeMPPI control proposal (κ_predict={args.predict_gain}, escape={args.escape_gain}) · "
                     f"green=accepted red=rejected samples · orange=mean+cov (control units) · t={f}", fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=1000 // args.fps)
    p = os.path.join(FIG, "step2_sampling_grid.gif")
    anim.save(p, writer=PillowWriter(fps=args.fps), dpi=80); print("saved", p)
    draw(nF // 2); fig.savefig(os.path.join(FIG, "step2_sampling_grid.png"), dpi=120)
    print("saved", os.path.join(FIG, "step2_sampling_grid.png"))


if __name__ == "__main__":
    main()
