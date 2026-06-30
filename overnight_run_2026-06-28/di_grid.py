"""Double-integrator on the EVAL datasets (UCY/SDD) — same viz style as the single-integrator grid, to see DI works.
2 episodes x 2 rows (accept/reject trajectories on polytope + bimodal control mean/cov) x gamma {0.1,0.5,1.0}.
FULL-scene fixed camera.
  python overnight_run_2026-06-28/di_grid.py --dataset ucy --episodes 16 47 --cg 0.2 --sv 0.5 --aniso 2.0 --sensing 2.0
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.mppi.sweep import _load, DT
from polytope_explainer import H_grid
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
GAMMAS = [0.1, 0.5, 1.0]


def di_step(s, u):
    return np.array([s[0] + 0.1 * s[2] + 0.005 * u[0], s[1] + 0.1 * s[3] + 0.005 * u[1],
                     s[2] + 0.1 * u[0], s[3] + 0.1 * u[1]], np.float32)


def rollout(dataset, ep, g, cfg, steps=80, dev="cpu"):
    s0, goal, obs, vel = _load(dataset, ep, steps); ad = SafeMPPIAdapter(**cfg)
    st = np.array([s0[0], s0[1], 0, 0.], np.float32); rec = []; reached = False; path = [st[:2].copy()]
    gt = torch.tensor(goal, dtype=torch.float32, device=dev)
    for t in range(steps):
        ob = obs[min(t, obs.shape[0] - 1)]; vl = vel[min(t, vel.shape[0] - 1)]; ok = ~np.isnan(ob[:, :2]).any(1)
        ob = ob[ok]; vl = vl[ok]
        if not reached:
            a, info = ad.plan(torch.tensor(st, dtype=torch.float32, device=dev), gt, torch.tensor(ob, dtype=torch.float32, device=dev),
                              gamma=g, obstacle_velocities=torch.tensor(vl, dtype=torch.float32, device=dev), seed=t, return_rollouts=True)
            dr = info["debug_rollouts"]; nrej = int(info["num_barrier_violations"]); rate = float(info["infeasibility_rate"])
            ntot = int(round(nrej / rate)) if rate > 1e-9 else cfg["num_samples"]
            rec.append(dict(p=st[:2].copy(), crowd=ob.copy(), traj=dr["states"], feas=np.asarray(dr["feasible"], bool),
                            n_acc=max(0, ntot - nrej), n_rej=nrej, fc=info["first_controls"], fcf=np.asarray(info["feasible"], bool),
                            smean=info["sample_mean"], exec=info["mean_control"], sigma=info["sigma"], poly=info["polytope"],
                            cpos=info["centroid_pos"], pmix=info["mixture_p"], size=info["polytope_size"]))
            st = di_step(st, a.detach().cpu().numpy())
            if np.linalg.norm(st[:2] - goal) < 0.6:
                reached = True
        else:
            rec.append(rec[-1] | {"p": st[:2].copy()})
        path.append(st[:2].copy())
    return rec, np.array(path), goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy"); ap.add_argument("--episodes", nargs="+", type=int, default=[16, 47])
    ap.add_argument("--cg", type=float, default=0.2); ap.add_argument("--sv", type=float, default=0.5)
    ap.add_argument("--aniso", type=float, default=2.0); ap.add_argument("--sensing", type=float, default=2.0)
    ap.add_argument("--ns", type=int, default=256); ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = dict(horizon=10, dt=DT, num_samples=args.ns, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
               safety_margin=0.0, temperature=0.3, dynamics_type="doubleintegrator", barrier_activation_radius=args.sensing,
               use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, centroid_gain=args.cg, centroid_smooth=0.5,
               sigma_volume_gain=args.sv, sigma_aniso=args.aniso, control_weight=0.03, predict_gain=0.4, polytope_nbase=16)
    eps = args.episodes[:2]; data = {}
    lims = {}
    for ep in eps:
        for g in GAMMAS:
            data[(ep, g)] = rollout(args.dataset, ep, g, dict(cfg), dev=args.device)
        s0, goal, obs, _ = _load(args.dataset, ep, 80); a = obs[..., :2].reshape(-1, 2); a = a[~np.isnan(a).any(1)]
        pts = np.vstack([a, s0[:2][None], goal[None]]); pad = 1.0
        lims[ep] = ((pts[:, 0].min() - pad, pts[:, 0].max() + pad), (pts[:, 1].min() - pad, pts[:, 1].max() + pad))
        print("rolled DI ep", ep, flush=True)
    R, C = 2 * len(eps), len(GAMMAS)
    fig, axes = plt.subplots(R, C, figsize=(4.0 * C, 3.3 * R), squeeze=False)
    nF = max(len(data[(eps[0], GAMMAS[0])][0]), 1)

    def draw(f):
        for ei, ep in enumerate(eps):
            (xl, yl) = lims[ep]
            for ci, g in enumerate(GAMMAS):
                rec, path, goal = data[(ep, g)]; st = rec[min(f, len(rec) - 1)]; p = st["p"]
                ax = axes[2 * ei][ci]; ax.clear()
                if st["poly"] is not None:
                    gx = np.linspace(*xl, 100); gy = np.linspace(*yl, 80); GX, GY = np.meshgrid(gx, gy)
                    Hh = H_grid(st["poly"], GX, GY); lv = sorted({round((1 - g) ** i, 4) for i in range(8)} | {0.0})
                    ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.45, zorder=1)
                    ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.0, zorder=3)
                for (ox, oy, rr) in st["crowd"]:
                    ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.6, edgecolor="#7b3294", lw=0.4, zorder=4))
                traj = st["traj"]; feas = st["feas"]
                for k in range(traj.shape[0]):
                    if not feas[k]:
                        ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color="#d62728", lw=0.35, alpha=0.3, zorder=5)
                for k in range(traj.shape[0]):
                    if feas[k]:
                        ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color="#00a000", lw=0.7, alpha=0.85, zorder=7)
                if st["cpos"] is not None:
                    cp = st["cpos"]; ax.annotate("", xy=(cp[0], cp[1]), xytext=(p[0], p[1]), arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.6), zorder=9)
                ax.plot(path[:min(f, len(path) - 1) + 1, 0], path[:min(f, len(path) - 1) + 1, 1], "-", color="#e6191b", lw=1.5, zorder=8)
                ax.scatter([p[0]], [p[1]], s=40, c="#00a000", edgecolor="k", zorder=10)
                ax.scatter([goal[0]], [goal[1]], marker="*", s=120, c="gold", edgecolor="k", zorder=10)
                ax.text(0.02, 0.98, f"acc {st['n_acc']}/rej {st['n_rej']}\np={st['pmix']:.2f}", transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
                ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if ei == 0: ax.set_title(f"γ={g}", fontsize=11)
                if ci == 0: ax.set_ylabel(f"DI ep{ep}  trajectories", fontsize=8)
                axc = axes[2 * ei + 1][ci]; axc.clear(); fc = st["fc"]; fcf = st["fcf"]; sm = st["smean"]; sg = st["sigma"]
                axc.axhline(0, color="#ddd", lw=0.5); axc.axvline(0, color="#ddd", lw=0.5)
                axc.scatter(fc[fcf, 0], fc[fcf, 1], s=5, c="#00a000", alpha=0.5, zorder=2)
                axc.scatter(fc[~fcf, 0], fc[~fcf, 1], s=5, c="#d62728", alpha=0.4, zorder=2)
                axc.add_patch(Ellipse((sm[0], sm[1]), 2 * sg[0], 2 * sg[1], facecolor="none", edgecolor="#ff7f00", lw=1.3, zorder=3))
                axc.annotate("", xy=(sm[0], sm[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.5), zorder=4)
                axc.scatter([st["exec"][0]], [st["exec"][1]], s=28, c="#08306b", marker="x", zorder=5)
                axc.set_title(f"bimodal accel · size {st['size']:.2f} p {st['pmix']:.2f}", fontsize=8)
                axc.set_xlim(-2.3, 2.3); axc.set_ylim(-2.3, 2.3); axc.set_aspect("equal"); axc.set_xticks([-2, 0, 2]); axc.set_yticks([-2, 0, 2]); axc.tick_params(labelsize=6)
                if ci == 0: axc.set_ylabel(f"DI ep{ep}  bimodal μ/Σ", fontsize=8)
        fig.suptitle(f"Double-integrator on {args.dataset} (cg={args.cg} sv={args.sv} aniso={args.aniso} sens={args.sensing} ns={args.ns}) · "
                     f"green=accepted/red=rejected · orange=centroid+μ/Σ · navy ✗=executed accel · t={f}", fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=200)
    out = os.path.join(FIG, "di_grid.gif"); anim.save(out, writer=PillowWriter(fps=6), dpi=80)
    print("saved", out); draw(nF // 2); fig.savefig(os.path.join(FIG, "di_grid.png"), dpi=110); print("saved png")


if __name__ == "__main__":
    main()
