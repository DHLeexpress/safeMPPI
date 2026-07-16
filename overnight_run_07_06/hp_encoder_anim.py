"""32x32 single-H_P-layer encoder visualization (2026-07-07, user 0.1.b). The policy encoder now sees a
single scalar H_P field (channel 2 = clipped polytope value) at angular 32 × radial 32 resolution. Renders
that field across scenes in RECTANGULAR (Cartesian) axes with the ROBOT AT THE CENTER, plus an animation
stepping the robot from origin toward the goal. matplotlib only, no GPU.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

import _paths  # noqa: F401
import grid_feats as GF
import grid_scene as GS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "encoder_hp32"); os.makedirs(FIG, exist_ok=True)


def hp_field(env, c):
    """32x32 H_P field (ch2) + the robot-centered world points [n_theta,n_r,2]."""
    obs = env.obstacles.detach().cpu().numpy()
    grid = GF.axis_grid(np.asarray(c, float), obs, float(env.r_robot))   # [3,32,32]
    pts = GF.axis_polar_points(np.asarray(c, float))                     # [32,32,2] world
    return grid[2], pts


def draw_field(ax, env, c, title):
    hp, pts = hp_field(env, c)
    rel = pts - np.asarray(c, float)[None, None, :2]                     # robot-centered
    sc = ax.scatter(rel[..., 0].ravel(), rel[..., 1].ravel(), c=hp.ravel(), cmap="RdYlGn",
                    vmin=-1, vmax=1, s=14, marker="s")
    R = GF.R_SENSE
    ax.add_patch(Circle((0, 0), R, fill=False, ls="--", color="0.4"))    # sensing range
    obs = env.obstacles.detach().cpu().numpy()
    for o in obs:                                                        # true obstacles within view (robot-centered)
        d = np.linalg.norm(o[:2] - np.asarray(c, float)[:2])
        if d < R + o[2] + 0.5:
            ax.add_patch(Circle((o[0] - c[0], o[1] - c[1]), o[2], color="0.15", alpha=0.5, zorder=3))
    ax.plot(0, 0, "k*", ms=15, zorder=5)                                 # robot at center
    ax.set_xlim(-R - 0.3, R + 0.3); ax.set_ylim(-R - 0.3, R + 0.3); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    return sc


def main():
    env = GS.make_grid()
    goal = env.goal.detach().cpu().numpy()
    print(f"[enc_anim] 32x32 H_P field; sensing R={GF.R_SENSE}, N_THETA={GF.N_THETA} N_R={GF.N_R}", flush=True)

    # --- static multi-scene panel ---
    scenes = [(0.5, 0.5), (2.0, 3.5), (3.0, 3.0), (1.0, 4.0), (4.0, 2.0), (4.5, 4.5)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    sc = None
    for ax, c in zip(axes.ravel(), scenes):
        sc = draw_field(ax, env, np.array([c[0], c[1], 0, 0], float), f"robot ({c[0]},{c[1]})")
    fig.suptitle("Single-layer H_P encoder field  (32×32 polar, robot-centered Cartesian; green=free, red=barrier)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="H_P (clipped polytope value)")
    f1 = os.path.join(FIG, "hp_field_scenes.png"); fig.savefig(f1, dpi=125, bbox_inches="tight"); plt.close(fig)
    hpv, _ = hp_field(env, [2.0, 2.0, 0, 0])
    print(f"[enc_anim] H_P range [{hpv.min():.2f}, {hpv.max():.2f}] -> {f1}", flush=True)

    # --- animation: robot walks origin -> goal along a staircase-ish diagonal ---
    path = np.linspace([0.2, 0.2], goal - 0.2, 28)
    figa, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))

    def frame(k):
        axL.clear(); axR.clear()
        c = np.array([path[k, 0], path[k, 1], 0, 0], float)
        draw_field(axL, env, c, f"H_P field  step {k}/{len(path)-1}  robot ({c[0]:.1f},{c[1]:.1f})")
        # world map inset on the right
        obs = env.obstacles.detach().cpu().numpy()
        for o in obs:
            axR.add_patch(Circle((o[0], o[1]), o[2], color="0.4", zorder=2))
        axR.plot(path[:k + 1, 0], path[:k + 1, 1], "-", color="tab:blue", lw=1.5)
        axR.plot(c[0], c[1], "k*", ms=13, zorder=5); axR.plot(goal[0], goal[1], "r*", ms=15)
        axR.set_xlim(0, 5); axR.set_ylim(0, 5); axR.set_aspect("equal"); axR.set_title("world (robot ★ → goal ★)")
        return []

    anim = FuncAnimation(figa, frame, frames=len(path), blit=False)
    f2 = os.path.join(FIG, "hp_field_anim.gif")
    anim.save(f2, writer=PillowWriter(fps=4)); plt.close(figa)
    print(f"[enc_anim] animation -> {f2}", flush=True)


if __name__ == "__main__":
    main()
