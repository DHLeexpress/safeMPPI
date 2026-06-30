"""Polytope SafeMPPI full grid GIF (single integrator).
Layout: 2 episodes, each with TWO rows -> (A) accept/reject trajectories on the polytope + level sets + centroid
mean-steer + executed path; (B) the control-space proposal (mean vector + covariance ellipse, time-evolving;
covariance grows when the polytope is small). Columns = gamma {0.3, 0.5, 1.0}.

  python overnight_run_2026-06-28/polytope_grid.py --episodes 0 7 --device cuda
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from polytope_explainer import rollout, H_grid, pick_navigable          # reuse the validated rollout
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from cfm_mppi.mppi.sweep import _load, DT

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
GAMMAS = [0.1, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy"); ap.add_argument("--episodes", nargs="+", type=int, default=None)
    ap.add_argument("--centroid-gain", type=float, default=0.1); ap.add_argument("--sigma-volume-gain", type=float, default=0.5)
    ap.add_argument("--centroid-horizon", type=int, default=3); ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--predict-gain", type=float, default=0.4); ap.add_argument("--sensing", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=5); ap.add_argument("--frames", type=int, default=70); ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    # 50-eps FINE-TUNED config: nominal=0 + warm-start, margin=0, polytope barrier, cg=0.1/K=3/sv=1.0/noise=0.5
    # (78% success, 82% near, 2% collision [only ep30], 86% acceptance on 50 UCY eps).
    cfg = dict(horizon=10, dt=DT, num_samples=128, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
               safety_margin=0.0, temperature=args.temperature, dynamics_type="singleintegrator",
               barrier_activation_radius=args.sensing, use_polytope_barrier=True, use_goal_nominal=False, warm_start=True,
               centroid_gain=args.centroid_gain, centroid_horizon=args.centroid_horizon,
               sigma_volume_gain=args.sigma_volume_gain, predict_gain=args.predict_gain, polytope_nbase=16)
    if args.episodes is None:                              # auto-pick 3 navigable episodes with LONG travel (so the
        cand = []                                          # ego-centric camera visibly follows the robot)
        for ep in range(80):
            s0, goal, obs, _ = _load(args.dataset, ep, 80)
            nc = int((~np.isnan(obs[:, :, 0]).all(0)).sum()); tr = float(np.linalg.norm(goal - s0[:2]))
            if 4 <= nc <= 16:
                cand.append((tr, ep))
        cand.sort(reverse=True); args.episodes = [ep for _, ep in cand[:3]] or [0, 1, 2]
    eps = args.episodes[:3]
    print("episodes:", eps)

    data = {}
    for ep in eps:
        for g in GAMMAS:
            data[(ep, g)] = rollout(args.dataset, ep, g, dict(cfg), dev=args.device)
        print(f"recorded ep {ep}", flush=True)

    vR = 2.8; R, C = 2 * len(eps), len(GAMMAS)
    fig, axes = plt.subplots(R, C, figsize=(3.3 * C, 3.0 * R), squeeze=False)
    nF = min(args.frames, max(len(data[(eps[0], GAMMAS[0])][0]), 1))

    def draw(f):
        for ei, ep in enumerate(eps):
            for ci, g in enumerate(GAMMAS):
                rec, path, goal = data[(ep, g)]; st = rec[min(f, len(rec) - 1)]; p = st["p"]
                xl = (p[0] - vR, p[0] + vR); yl = (p[1] - vR, p[1] + vR)
                # --- row A: trajectories on the polytope ---
                ax = axes[2 * ei][ci]; ax.clear()
                if st.get("poly") is not None:
                    gx = np.linspace(*xl, 80); gy = np.linspace(*yl, 80); GX, GY = np.meshgrid(gx, gy)
                    Hh = H_grid(st["poly"], GX, GY)
                    lv = sorted({round((1 - g) ** i, 4) for i in range(8)} | {0.0})
                    ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.5, zorder=1)
                    ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.3, zorder=3)
                ci_inf = st.get("crowd_infl", None)
                for i, (ox, oy, rr) in enumerate(st["crowd"]):
                    ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.6, edgecolor="#7b3294", lw=0.5, zorder=4))
                    ir = float(ci_inf[i]) if ci_inf is not None and i < len(ci_inf) else rr   # predict-inflated radius
                    if ir > rr + 1e-3:
                        ax.add_patch(Circle((ox, oy), ir, facecolor="none", edgecolor="#d62728", ls="--", lw=0.6, alpha=0.5, zorder=4))
                traj = st["traj"]; feas = st["feas"]
                for k in range(traj.shape[0]):
                    if not feas[k]:
                        xy = traj[k, :, :2]; ax.plot(xy[:, 0], xy[:, 1], "-", color="#d62728", lw=0.4, alpha=0.35, zorder=5)
                        ax.plot(xy[-1, 0], xy[-1, 1], "x", color="#d62728", ms=3, mew=0.6, zorder=6)
                for k in range(traj.shape[0]):
                    if feas[k]:
                        xy = traj[k, :, :2]; ax.plot(xy[:, 0], xy[:, 1], "-", color="#00a000", lw=0.9, alpha=0.9, zorder=7)
                cd = st.get("cdir")
                if cd is not None and np.linalg.norm(cd) > 1e-6:
                    ax.annotate("", xy=(p[0] + 1.0 * cd[0], p[1] + 1.0 * cd[1]), xytext=(p[0], p[1]),
                                arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=2.0), zorder=9)
                pu = path[:min(f, len(path) - 1) + 1]; ax.plot(pu[:, 0], pu[:, 1], "-", color="#e6191b", lw=1.4, zorder=8)
                ax.scatter([p[0]], [p[1]], s=45, c="#00a000", edgecolor="k", zorder=10)
                if xl[0] <= goal[0] <= xl[1] and yl[0] <= goal[1] <= yl[1]:
                    ax.scatter([goal[0]], [goal[1]], marker="*", s=110, c="gold", edgecolor="k", zorder=10)
                ax.text(0.02, 0.98, f"acc {st['n_acc']}\nrej {st['n_rej']}", transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
                ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if ei == 0:
                    ax.set_title(f"γ={g}", fontsize=11)
                if ci == 0:
                    ax.set_ylabel(f"ep{ep}  trajectories", fontsize=8)
                # --- row B: control-space SAMPLING mean + covariance (executed = navy x) ---
                axc = axes[2 * ei + 1][ci]; axc.clear(); mean = st.get("smean", st["mean"]); sig = st["sigma"]; ex = st["mean"]
                axc.axhline(0, color="#ddd", lw=0.6); axc.axvline(0, color="#ddd", lw=0.6)
                axc.add_patch(Ellipse((mean[0], mean[1]), 2 * sig[0], 2 * sig[1], facecolor="#ffd9b3",
                                      edgecolor="#ff7f00", lw=1.4, alpha=0.7, zorder=2))
                axc.annotate("", xy=(mean[0], mean[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.0), zorder=3)
                axc.scatter([ex[0]], [ex[1]], s=22, c="#08306b", marker="x", zorder=4)   # executed (weighted/fallback)
                sz = st.get("size"); szs = f"{sz:.2f}" if sz is not None else "-"
                axc.text(0.02, 0.98, f"size {szs}\nσ {sig[0]:.2f}", transform=axc.transAxes, va="top", fontsize=7,
                         bbox=dict(boxstyle="round", fc="white", alpha=0.7))
                axc.set_xlim(-2.3, 2.3); axc.set_ylim(-2.3, 2.3); axc.set_aspect("equal")
                axc.set_xticks([-2, 0, 2]); axc.set_yticks([-2, 0, 2]); axc.tick_params(labelsize=6)
                if ci == 0:
                    axc.set_ylabel(f"ep{ep}  mean+cov", fontsize=8)
        fig.suptitle(f"Polytope SafeMPPI · green=accepted/red ✗=rejected trajectories · orange=centroid mean-steer · "
                     f"bottom rows: control mean+cov (cov↑ when polytope small) · t={f}", fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=1000 // args.fps)
    out = os.path.join(FIG, "polytope_grid.gif"); anim.save(out, writer=PillowWriter(fps=args.fps), dpi=80)
    print("saved", out); draw(nF // 3); fig.savefig(os.path.join(FIG, "polytope_grid.png"), dpi=110); print("saved png")


if __name__ == "__main__":
    main()
