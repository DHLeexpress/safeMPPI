"""Double-integrator gap-threading demo with the polytope bimodal mixture proposal.
2 synthetic scenes (static single obstacle; two-obstacle narrow gap) x 2 rows (accept/reject trajectories +
control bimodal mean/cov) = 4 rows x gamma {0.1,0.5,1.0}. FULL scene (fixed camera, not ego-centric).
  python overnight_run_2026-06-28/di_gap.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from polytope_explainer import H_grid
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
GAMMAS = [0.1, 0.5, 1.0]
SCENES = [("single obstacle", np.array([6., 0.]), np.array([[3.0, 0.0, 0.6]])),
          ("narrow gap", np.array([6., 0.]), np.array([[3.0, 0.95, 0.3], [3.0, -0.95, 0.3]]))]
CFG = dict(horizon=10, dt=0.1, num_samples=256, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
           safety_margin=0.0, temperature=0.3, dynamics_type="doubleintegrator", barrier_activation_radius=3.0,
           use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, centroid_gain=0.2, centroid_smooth=0.5,
           sigma_volume_gain=0.5, sigma_aniso=2.5, predict_gain=0.0, polytope_nbase=16)


def di_step(s, u):
    return np.array([s[0] + 0.1 * s[2] + 0.005 * u[0], s[1] + 0.1 * s[3] + 0.005 * u[1], s[2] + 0.1 * u[0], s[3] + 0.1 * u[1]], np.float32)


def rollout(goal, obs, g, steps=70):
    ad = SafeMPPIAdapter(**CFG); st = np.array([0, 0, 0, 0.], np.float32); rec = []; reached = False; path = [st[:2].copy()]
    gt = torch.tensor(goal, dtype=torch.float32)
    for t in range(steps):
        if not reached:
            a, info = ad.plan(torch.tensor(st, dtype=torch.float32), gt, torch.tensor(obs, dtype=torch.float32),
                              gamma=g, seed=t, return_rollouts=True)
            dr = info["debug_rollouts"]; nrej = int(info["num_barrier_violations"]); rate = float(info["infeasibility_rate"])
            ntot = int(round(nrej / rate)) if rate > 1e-9 else CFG["num_samples"]
            rec.append(dict(p=st[:2].copy(), traj=dr["states"], feas=np.asarray(dr["feasible"], bool),
                            n_acc=max(0, ntot - nrej), n_rej=nrej, fc=info["first_controls"], fcf=np.asarray(info["feasible"], bool),
                            smean=info["sample_mean"], exec=info["mean_control"], sigma=info["sigma"], poly=info["polytope"],
                            cpos=info["centroid_pos"], pmix=info["mixture_p"], size=info["polytope_size"]))
            st = di_step(st, a.detach().cpu().numpy())
            if np.linalg.norm(st[:2] - goal) < 0.4:
                reached = True
        else:
            rec.append(rec[-1] | {"p": st[:2].copy()})
        path.append(st[:2].copy())
    return rec, np.array(path), goal, obs


def main():
    data = {}
    for name, goal, obs in SCENES:
        for g in GAMMAS:
            data[(name, g)] = rollout(goal, obs, g)
        print("rolled", name, flush=True)
    R, C = 2 * len(SCENES), len(GAMMAS)
    fig, axes = plt.subplots(R, C, figsize=(3.6 * C, 3.1 * R), squeeze=False)
    xl = (-1.0, 7.0); yl = (-3.0, 3.0)                                    # FIXED full-scene camera
    nF = max(len(data[(SCENES[0][0], GAMMAS[0])][0]), 1)

    def draw(f):
        for si, (name, goal, obs) in enumerate(SCENES):
            for ci, g in enumerate(GAMMAS):
                rec, path, goal, obs = data[(name, g)]; st = rec[min(f, len(rec) - 1)]; p = st["p"]
                # row A: trajectories on the polytope (full scene)
                ax = axes[2 * si][ci]; ax.clear()
                if st["poly"] is not None:
                    gx = np.linspace(*xl, 110); gy = np.linspace(*yl, 70); GX, GY = np.meshgrid(gx, gy)
                    Hh = H_grid(st["poly"], GX, GY)
                    lv = sorted({round((1 - g) ** i, 4) for i in range(8)} | {0.0})
                    ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.45, zorder=1)
                    ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.1, zorder=3)
                for (ox, oy, rr) in obs:
                    ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.7, edgecolor="#7b3294", lw=0.6, zorder=4))
                traj = st["traj"]; feas = st["feas"]
                for k in range(traj.shape[0]):
                    xy = traj[k, :, :2]
                    if not feas[k]:
                        ax.plot(xy[:, 0], xy[:, 1], "-", color="#d62728", lw=0.4, alpha=0.3, zorder=5)
                for k in range(traj.shape[0]):
                    if feas[k]:
                        ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color="#00a000", lw=0.7, alpha=0.85, zorder=7)
                if st["cpos"] is not None:                                 # arrow robot -> EXACT centroid
                    cp = st["cpos"]; ax.annotate("", xy=(cp[0], cp[1]), xytext=(p[0], p[1]),
                                                 arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.8), zorder=9)
                    ax.scatter([cp[0]], [cp[1]], s=20, c="#ff7f00", zorder=9)
                ax.plot(path[:min(f, len(path) - 1) + 1, 0], path[:min(f, len(path) - 1) + 1, 1], "-", color="#e6191b", lw=1.6, zorder=8)
                ax.scatter([p[0]], [p[1]], s=40, c="#00a000", edgecolor="k", zorder=10)
                ax.scatter([goal[0]], [goal[1]], marker="*", s=120, c="gold", edgecolor="k", zorder=10)
                ax.text(0.02, 0.97, f"acc {st['n_acc']}/rej {st['n_rej']}\np={st['pmix']:.2f}", transform=ax.transAxes,
                        va="top", fontsize=7, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
                ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if si == 0: ax.set_title(f"γ={g}", fontsize=11)
                if ci == 0: ax.set_ylabel(f"{name}\ntrajectories", fontsize=8)
                # row B: bimodal control-space proposal (the actual first-control samples = two clusters) + mean/cov
                axc = axes[2 * si + 1][ci]; axc.clear(); fc = st["fc"]; fcf = st["fcf"]; sm = st["smean"]; sg = st["sigma"]
                axc.axhline(0, color="#ddd", lw=0.5); axc.axvline(0, color="#ddd", lw=0.5)
                axc.scatter(fc[fcf, 0], fc[fcf, 1], s=5, c="#00a000", alpha=0.5, zorder=2)
                axc.scatter(fc[~fcf, 0], fc[~fcf, 1], s=5, c="#d62728", alpha=0.4, zorder=2)
                axc.add_patch(Ellipse((sm[0], sm[1]), 2 * sg[0], 2 * sg[1], facecolor="none", edgecolor="#ff7f00", lw=1.3, zorder=3))
                axc.annotate("", xy=(sm[0], sm[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.6), zorder=4)
                axc.scatter([st["exec"][0]], [st["exec"][1]], s=30, c="#08306b", marker="x", zorder=5)
                axc.set_title(f"bimodal samples · size {st['size']:.2f} p {st['pmix']:.2f}", fontsize=8)
                axc.set_xlim(-2.3, 2.3); axc.set_ylim(-2.3, 2.3); axc.set_aspect("equal")
                axc.set_xticks([-2, 0, 2]); axc.set_yticks([-2, 0, 2]); axc.tick_params(labelsize=6)
                if ci == 0: axc.set_ylabel(f"{name}\nbimodal μ/Σ", fontsize=8)
        fig.suptitle(f"Double-integrator gap-threading · green=accepted/red=rejected · orange=robot→EXACT centroid + "
                     f"bimodal μ/Σ · navy ✗=executed · t={f}", fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=200)
    out = os.path.join(FIG, "di_gap.gif"); anim.save(out, writer=PillowWriter(fps=6), dpi=80)
    print("saved", out); draw(nF // 2); fig.savefig(os.path.join(FIG, "di_gap.png"), dpi=110); print("saved png")


if __name__ == "__main__":
    main()
