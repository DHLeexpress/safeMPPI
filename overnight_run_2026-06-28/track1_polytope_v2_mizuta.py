"""Track 1: visualize polytope_v2 on ONE fixed real Mizuta (UCY) pedestrian episode.

Fix one frame's pedestrian layout (real clutter, not a symmetric toy), sweep the robot position across the
scene, and at each position build polytope_v2 + render its nested {H>=(1-gamma)^i} level sets. Shows the
general (non-box, non-symmetric) polytope adapting continuously to surrounding pedestrians.

  python overnight_run_2026-06-28/track1_polytope_v2_mizuta.py --R 3.5 --gamma 0.5
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
EGO = os.path.join(ROOT, "dataset", "eval80_ego_ucy.pt")
OBS = os.path.join(ROOT, "dataset", "eval80_obs_ucy.pkl")
PED_R = 0.5


def norm_barrier(poly, grid):  # robot-normalized polytope barrier (=1 at robot, 0 on boundary)
    mr = (poly.b - poly.A @ poly.ref).clamp_min(1e-3)
    val = (poly.b.unsqueeze(0) - grid @ poly.A.T) / mr.unsqueeze(0)
    return val.min(dim=1).values


def load_episode(idx):
    obs_list = pickle.load(open(OBS, "rb"))
    so = obs_list[idx % len(obs_list)].detach().cpu()           # [1,P,6,T]
    so = so[:, ~torch.isnan(so).any(dim=(0, 2, 3))]
    obs = so[0].float().numpy()                                  # [P,6,T]
    return obs                                                   # channels: 0,1=x,y ; 2,3=vx,vy


def peds_at(obs, frame):
    P = obs.shape[0]; xy = obs[:, 0:2, min(frame, obs.shape[2] - 1)]   # [P,2]
    xy = xy[~np.isnan(xy).any(1)]
    return np.concatenate([xy, np.full((len(xy), 1), PED_R)], 1)       # [P,3]


def pick_cluttered(n_scan=120):
    """Pick (episode, frame) that is CROWDED and SPREAD: most pedestrians spanning a wide area, so the
    robot sweep makes pedestrians enter/leave the sensing disk (shows continuity + a general polytope)."""
    obs_list = pickle.load(open(OBS, "rb"))
    best = (0, 40, -1.0)
    for i in range(min(n_scan, len(obs_list))):
        so = obs_list[i].detach().cpu(); so = so[:, ~torch.isnan(so).any(dim=(0, 2, 3))]
        if so.shape[1] < 6:
            continue
        o = so[0].float().numpy(); T = o.shape[2]
        for f in range(10, T - 10, 10):
            xy = o[:, 0:2, f]; xy = xy[~np.isnan(xy).any(1)]
            if len(xy) < 6:
                continue
            span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
            score = len(xy) * min(span, 12.0)                          # many peds AND spread out
            if score > best[2]:
                best = (i, f, score)
    return best[0], best[1]


def _draw(ax, peds, c, poly, gamma, xlim, ylim):
    gx = np.linspace(*xlim, 120); gy = np.linspace(*ylim, 110); GX, GY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
    H = norm_barrier(poly, grid).numpy().reshape(GX.shape)
    lv = sorted({round((1 - gamma) ** i, 4) for i in range(11)} | {0.0})
    ax.contourf(GX, GY, H, levels=lv + [1.0001], cmap="Blues", alpha=0.55, zorder=1)
    ax.contour(GX, GY, H, levels=[l for l in lv if l > 0], colors="#2166ac", linewidths=0.5, alpha=0.7, zorder=3)
    ax.contour(GX, GY, H, levels=[0.0], colors="#08306b", linewidths=1.8, zorder=3)
    for (px, py, r) in peds:
        ax.add_patch(Circle((px, py), r, facecolor="#7b3294", alpha=0.45, edgecolor="#4d004b", lw=1.0, zorder=4))
    ax.scatter([c[0]], [c[1]], s=70, c="#1a9850", edgecolor="k", marker="o", zorder=9)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=-1)   # -1 = auto-pick cluttered
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--R", type=float, default=3.5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=16)
    args = ap.parse_args()
    ep, fr = (args.episode, args.frame)
    if ep < 0 or fr < 0:
        ep, fr = pick_cluttered()
    obs = load_episode(ep); peds = peds_at(obs, fr)
    print(f"episode {ep} frame {fr}: {len(peds)} pedestrians")
    pad = args.R + 1.0
    xlim = (peds[:, 0].min() - pad, peds[:, 0].max() + pad)
    ylim = (peds[:, 1].min() - pad, peds[:, 1].max() + pad)

    # (1) static grid of robot positions across the crowd
    rows, cols = 3, 4
    rx = np.linspace(peds[:, 0].min(), peds[:, 0].max(), cols)
    ry = np.linspace(peds[:, 1].max(), peds[:, 1].min(), rows)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.9 * rows))
    for i in range(rows):
        for j in range(cols):
            c = np.array([rx[j], ry[i]])
            poly, info = build_polytope_v2(c, peds, sensing_range=args.R, n_base=args.K)
            _draw(axes[i, j], peds, c, poly, args.gamma, xlim, ylim)
            axes[i, j].set_title(f"{info['n_faces']} faces ({info['n_detected']} ped)", fontsize=7)
    fig.suptitle(f"polytope_v2 on Mizuta UCY ep{ep} fr{fr} ({len(peds)} peds, R={args.R}, K={args.K}, γ={args.gamma})\n"
                 f"robot swept across the fixed crowd · general convex polytope, no head bias / no forced symmetry",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); p = os.path.join(FIG, "track1_polytope_v2_grid.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # (2) GIF: robot walks a straight line across the scene -> faces "breathe" continuously
    cy = float(np.median(peds[:, 1]))
    path = np.stack([np.linspace(xlim[0] + 0.8, xlim[1] - 0.8, 60), np.full(60, cy)], 1)
    figg, axg = plt.subplots(figsize=(6.2, 5.0))

    def draw(t):
        axg.clear()
        c = path[t]
        poly, info = build_polytope_v2(c, peds, sensing_range=args.R, n_base=args.K)
        _draw(axg, peds, c, poly, args.gamma, xlim, ylim)
        axg.plot(path[:t + 1, 0], path[:t + 1, 1], "-", color="#1a9850", lw=1.0, alpha=0.5, zorder=8)
        axg.set_title(f"polytope_v2 · ep{ep} fr{fr} · robot moving through the crowd · {info['n_detected']} peds in range",
                      fontsize=9)
        return []
    anim = FuncAnimation(figg, draw, frames=len(path), interval=110)
    pg = os.path.join(FIG, "track1_polytope_v2_sweep.gif")
    anim.save(pg, writer=PillowWriter(fps=10), dpi=95); plt.close(figg); print("saved", pg)


if __name__ == "__main__":
    main()
