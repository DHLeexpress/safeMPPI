#!/usr/bin/env python3
"""Exact v4 2x4 rollout gallery for the challenging sanity panel."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "reference"), str(ROOT), str(ROOT.parent / "codex_overnight"),
                str(ROOT.parents[1])]
import grid_scene as GS  # noqa: E402
from eval_ae import _apply_wall_plugs_eval  # noqa: E402

matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 12.5})
OUT = ROOT / "stage_results/05_sanity/viz"
GSEL = [0.1, 0.5, 1.0]
PLASMA = plt.get_cmap("plasma")
GAMMA_COLOR = {0.1: PLASMA(0.08), 0.5: PLASMA(0.55), 1.0: PLASMA(0.85)}
REACH = 0.15

EXPERT = ROOT / "stage_results/06_baselines/results/expert_m6"
PRETRAINED = ROOT / "stage_results/04_canonical/data/pretrained_m6"
KAZUKI = ROOT / "stage_results/06_baselines/results/kazuki_low_guidance_m6"
OURS = ROOT / "stage_results/05_sanity/data/eval_final_v7_ours"
ABLATIONS = [
    (ROOT / "stage_results/05_sanity/data/eval_final_v7_no_socp", "NO safety validity check"),
    (ROOT / "stage_results/05_sanity/data/eval_final_v7_no_progress", "NO progress check"),
    (ROOT / "stage_results/05_sanity/data/eval_final_v7_no_curriculum", "NO curriculum"),
]
DEMO = ROOT / "stage_results/02_demos/data"


def load_paths(directory: Path, gamma: float):
    path = directory / f"paths_g{gamma}.npz"
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    return [np.asarray(p, dtype=float) for p in data["paths"]]


def draw_scene(ax, title: str, bold=False, canonical=True):
    env = GS.make_grid(); _apply_wall_plugs_eval(env, 8)
    for obstacle in env.obstacles.numpy():
        ax.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#cccccc", zorder=1))
    if canonical:
        ax.plot(0.05, 0.05, "ks", ms=5, zorder=7)
        ax.plot(5, 5, "*", c="gold", mec="k", ms=12, zorder=7)
    ax.set_xlim(-0.45, 5.45); ax.set_ylim(-0.45, 5.45); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, pad=6, fontsize=18, fontweight="bold" if bold else "normal")


def successful(path):
    return len(path) and np.linalg.norm(path[-1] - [5, 5]) < REACH


def draw_path(ax, path, gamma, *, lw=1.4, dots=True, endpoint=True):
    if len(path) < 2:
        return
    ax.plot(path[:, 0], path[:, 1], color=GAMMA_COLOR[gamma], lw=lw, alpha=0.9, zorder=3)
    if dots:
        ax.plot(path[::3, 0], path[::3, 1], ".", color="k", ms=1.6, alpha=0.55, zorder=4)
    if endpoint and not successful(path):
        ax.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=8, mew=2.2, zorder=6)


def pick(paths, total=3):
    good = [p for p in paths if successful(p)]
    bad = [p for p in paths if not successful(p) and len(p) > 5]
    # One successful path when available, then failures: the sanity outcome stays visible.
    chosen = good[:1] + bad[:max(0, total - min(1, len(good)))]
    return (chosen or paths[:total])[:total]


def add_zoom(ax, box, paths_and_gamma, loc="lower right"):
    zoom = inset_axes(ax, width="42%", height="42%", loc=loc, borderpad=0.4)
    env = GS.make_grid(); _apply_wall_plugs_eval(env, 8)
    for obstacle in env.obstacles.numpy():
        zoom.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#cccccc", zorder=1))
    zoom.plot(5, 5, "*", c="gold", mec="k", ms=11, zorder=7)
    for path, gamma in paths_and_gamma:
        zoom.plot(path[:, 0], path[:, 1], color=GAMMA_COLOR[gamma], lw=1.6, alpha=0.95, zorder=3)
        zoom.plot(path[::2, 0], path[::2, 1], ".", color="k", ms=2.0, alpha=0.6, zorder=4)
        if not successful(path):
            zoom.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=9, mew=2.4, zorder=6)
    zoom.set_xlim(box[0], box[1]); zoom.set_ylim(box[2], box[3]); zoom.set_aspect("equal")
    zoom.set_xticks([]); zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_color("#cc3311"); spine.set_linewidth(1.6)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                           fill=False, ec="#cc3311", lw=1.3, zorder=7))


def draw_method_panel(ax, title, directory, *, bold=False, zoom=False):
    draw_scene(ax, title, bold=bold)
    failures = []
    for gamma in GSEL:
        for path in pick(load_paths(directory, gamma), total=3):
            draw_path(ax, path, gamma)
            if not successful(path):
                failures.append((path, gamma))
    if zoom and failures:
        near_goal = [(p, g) for p, g in failures if np.linalg.norm(p[-1] - [5, 5]) < 1.4]
        add_zoom(ax, (3.9, 5.45, 3.9, 5.45), (near_goal or failures)[:4], loc="lower right")


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 10.0))

    # Challenging pretraining distribution: upper-off-diagonal starts to lower-off-diagonal goals.
    data_ax = axes[0, 0]; draw_scene(data_ax, "Pre-trained data")
    cloud = np.load(DEMO / "paths_g0.5.npz", allow_pickle=True)
    data_ax.plot(cloud["starts"][:, 0], cloud["starts"][:, 1], ".",
                 color="#888888", ms=3.0, alpha=0.45, zorder=2, label="start seeds")
    data_ax.plot(cloud["goals"][:, 0], cloud["goals"][:, 1], ".",
                 color="#cc6677", ms=3.0, alpha=0.45, zorder=2, label="goal seeds")
    for gamma in GSEL:
        episode = np.load(DEMO / f"paths_g{gamma}.npz", allow_pickle=True)
        for index in (0, 100, 200):
            path = np.asarray(episode["paths"][index], dtype=float)
            data_ax.plot(path[:, 0], path[:, 1], color=GAMMA_COLOR[gamma], lw=1.25, alpha=0.8, zorder=3)
            start = episode["starts"][index]; goal = episode["goals"][index]
            data_ax.plot(start[0], start[1], "o", c="k", ms=3.5, zorder=6)
            data_ax.plot(goal[0], goal[1], "*", c="gold", mec="k", ms=5.5, zorder=6)
    data_ax.legend(loc="lower left", fontsize=8, frameon=False, handletextpad=0.3)

    draw_method_panel(axes[0, 1], "Expert", EXPERT)
    draw_method_panel(axes[0, 2], "Pretrained", PRETRAINED, zoom=True)
    draw_method_panel(axes[0, 3], r"CFM-MPPI$^{*}$", KAZUKI, zoom=True)
    for ax, (directory, title) in zip(axes[1, :3], ABLATIONS):
        draw_method_panel(ax, title, directory)
    draw_method_panel(axes[1, 3], "Ours", OURS, bold=True)

    cmap = ListedColormap([GAMMA_COLOR[g] for g in GSEL])
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm([0, 1, 2, 3], 3))
    cbar = fig.colorbar(scalar, ax=axes, location="right", fraction=0.022, pad=0.02,
                        ticks=[0.5, 1.5, 2.5])
    cbar.ax.set_yticklabels(["0.1", "0.5", "1.0"])
    cbar.set_label(r"safety level $\gamma$", fontsize=13)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rollouts_v4.{ext}", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'rollouts_v4.png'} and .pdf")


if __name__ == "__main__":
    main()
