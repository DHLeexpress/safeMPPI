#!/usr/bin/env python3
"""Overlay all random-pair SafeMPPI demonstrations in Image #1's style."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
STAGE_DIR = HERE / "stage_results" / "02_demos"
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import _paths  # noqa: F401,E402
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import gen_uniform_data as SEEDS  # noqa: E402
from viz_style import GAMMAS, GAMMA_CMAP, GAMMA_COLORS, GAMMA_NORM, gamma_boundaries  # noqa: E402


def draw_scene(ax, env, *, band: bool = True) -> None:
    ax.set_facecolor("#f7f6f4")
    if band:
        ax.add_patch(
            Polygon(
                [(-1, 0), (5, 6), (6, 6), (6, 5), (0, -1), (-1, -1)],
                closed=True,
                facecolor="#e3e1dd",
                edgecolor="none",
                alpha=0.72,
                zorder=0,
            )
        )
    for obstacle in env.obstacles.detach().cpu().numpy():
        ax.add_patch(
            Circle(
                obstacle[:2],
                obstacle[2] + float(env.r_robot),
                facecolor="#8a8a8a",
                edgecolor="none",
                zorder=3,
            )
        )
    ax.plot((0, 5, 5, 0, 0), (0, 0, 5, 5, 0), "--", color="#333333", lw=0.7, zorder=2)
    ax.set_xlim(-0.35, 5.35)
    ax.set_ylim(-0.35, 5.35)
    ax.set_aspect("equal", adjustable="box")


def load_paths(data_dir: Path):
    result = {}
    for gamma in GAMMAS:
        path = data_dir / f"paths_g{float(gamma)}.npz"
        with np.load(path, allow_pickle=True) as saved:
            result[gamma] = {
                "paths": [np.asarray(item, dtype=np.float32) for item in saved["paths"]],
                "starts": np.asarray(saved["starts"], dtype=np.float32),
                "goals": np.asarray(saved["goals"], dtype=np.float32),
                "success": np.asarray(saved["success"], dtype=bool),
                "steps": np.asarray(saved["steps"], dtype=int),
                "min_clearance": np.asarray(saved["min_clearance"], dtype=float),
            }
    return result


def render(data, output: Path, log: Path) -> None:
    env = SEEDS.make_walled_env(8)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig = plt.figure(figsize=(13, 16.2))
    layout = fig.add_gridspec(3, 4, height_ratios=(0.045, 1.55, 0.42), hspace=0.14, wspace=0.06)

    color_ax = fig.add_subplot(layout[0, 1:3])
    colorbar = mpl.colorbar.ColorbarBase(
        color_ax,
        cmap=GAMMA_CMAP,
        norm=GAMMA_NORM,
        boundaries=gamma_boundaries(),
        ticks=GAMMAS,
        spacing="uniform",
        orientation="horizontal",
        drawedges=True,
    )
    colorbar.ax.set_title(r"$\gamma$", fontsize=15)
    colorbar.ax.tick_params(labelsize=11, length=0)
    colorbar.dividers.set_color("white")
    colorbar.outline.set_linewidth(0.8)

    main_ax = fig.add_subplot(layout[1, :])
    draw_scene(main_ax, env, band=True)
    order = []
    for gamma in GAMMAS:
        order.extend((gamma, index) for index, ok in enumerate(data[gamma]["success"]) if ok)
    np.random.default_rng(0).shuffle(order)
    for gamma, index in order:
        path = data[gamma]["paths"][index]
        main_ax.plot(
            path[:, 0],
            path[:, 1],
            color=GAMMA_COLORS[gamma],
            lw=0.48,
            alpha=0.18,
            solid_capstyle="round",
            zorder=4,
        )

    first = data[GAMMAS[0]]
    main_ax.scatter(first["starts"][:, 0], first["starts"][:, 1], s=8, c="#1769aa", alpha=0.82, zorder=6)
    main_ax.scatter(first["goals"][:, 0], first["goals"][:, 1], s=8, c="#d32f2f", alpha=0.82, zorder=6)
    total_success = sum(int(data[g]["success"].sum()) for g in GAMMAS)
    main_ax.set_xlabel(r"$x$ [m]", fontsize=13)
    main_ax.set_ylabel(r"$y$ [m]", fontsize=13)
    main_ax.set_title(
        r"SafeMPPI start→goal demos: 300 shared random pairs $\times$ 7 $\gamma$ "
        f"({total_success} successful trajectories)",
        fontsize=13,
    )
    main_ax.legend(
        handles=(
            Line2D([], [], marker="o", ls="none", color="#1769aa", label="start: y>x"),
            Line2D([], [], marker="o", ls="none", color="#d32f2f", label="goal: y<x"),
        ),
        loc="center",
        fontsize=9,
        framealpha=0.92,
    )

    minis = fig.add_gridspec(1, 7, top=0.215, bottom=0.055, left=0.045, right=0.985, wspace=0.08)
    per_gamma = {}
    for column, gamma in enumerate(GAMMAS):
        ax = fig.add_subplot(minis[0, column])
        draw_scene(ax, env, band=False)
        success = data[gamma]["success"]
        for index in np.flatnonzero(success):
            path = data[gamma]["paths"][int(index)]
            ax.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=0.34, alpha=0.24, zorder=4)
        ax.scatter(
            data[gamma]["starts"][:, 0], data[gamma]["starts"][:, 1],
            s=1.0, c="#1769aa", alpha=0.5, zorder=5,
        )
        ax.scatter(
            data[gamma]["goals"][:, 0], data[gamma]["goals"][:, 1],
            s=1.0, c="#d32f2f", alpha=0.5, zorder=5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\gamma$={gamma:g}", fontsize=11, color=GAMMA_COLORS[gamma])
        per_gamma[str(gamma)] = {
            "attempted": len(success),
            "successful": int(success.sum()),
            "success_rate": float(success.mean()),
            "mean_steps_success": float(data[gamma]["steps"][success].mean()),
            "min_clearance_success": float(data[gamma]["min_clearance"][success].min()),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "output": str(output.resolve()),
                "total_successful_trajectories": total_success,
                "palette": {"gamma": "plasma_trunc", "sigma_uncertainty": "viridis"},
                "per_gamma": per_gamma,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"OVERLAY {output} ({total_success} successful trajectories)")
    print(f"LOG {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=STAGE_DIR / "data")
    parser.add_argument("--out", type=Path, default=STAGE_DIR / "viz" / "demo_300_pairs_all_gamma.png")
    parser.add_argument("--log", type=Path, default=STAGE_DIR / "logs" / "demo_overlay.json")
    args = parser.parse_args()
    render(load_paths(args.data_dir), args.out, args.log)


if __name__ == "__main__":
    main()
