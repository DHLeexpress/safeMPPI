#!/usr/bin/env python3
"""Compact paper figure comparing executed trajectory distributions by method and gamma."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import grid_metrics as GM
import grid_scene as GS


def parse_method(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("method must be LABEL=RESULT_DIR")
    label, path = text.split("=", 1)
    return label, Path(path)


def load_paths(folder, gamma):
    with np.load(folder / f"paths_g{float(gamma)}.npz", allow_pickle=True) as z:
        return [np.asarray(p) for p in z["paths"]]


def draw_scene(ax, env):
    for o in env.obstacles.detach().cpu().numpy():
        ax.add_patch(Circle(o[:2], o[2], color="#888888", alpha=.75, lw=0))
    ax.plot(0, 0, "s", color="black", ms=5, zorder=8)
    ax.plot(5, 5, "*", color="#F0C808", mec="black", ms=12, zorder=8)
    ax.set_xlim(-.2, 5.2); ax.set_ylim(-.2, 5.2); ax.set_aspect("equal")
    ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.grid(alpha=.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", type=parse_method, required=True)
    ap.add_argument("--gammas", type=float, nargs="+", default=[.1, .5, 1.0])
    ap.add_argument("--max-paths", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("figures/method_trajectory_comparison.png"))
    args = ap.parse_args()
    env = GS.make_grid(); goal = env.goal.detach().cpu().numpy()
    fig, axs = plt.subplots(len(args.method), len(args.gammas),
                            figsize=(5 * len(args.gammas), 4.7 * len(args.method)), squeeze=False,
                            constrained_layout=True)
    for i, (label, folder) in enumerate(args.method):
        for j, gamma in enumerate(args.gammas):
            ax = axs[i, j]; draw_scene(ax, env)
            paths = load_paths(folder, gamma)
            success = []
            for p in paths:
                if np.linalg.norm(p[-1] - goal) < .1:
                    success.append(p)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(success), min(len(success), args.max_paths), replace=False) if success else []
            ids = set()
            for k in idx:
                p = success[int(k)]
                ax.plot(p[:, 0], p[:, 1], color="#0072B2", alpha=.20, lw=1.0)
            for p in success:
                sid = GM.staircase_id(p)
                if sid is not None:
                    ids.add(sid)
            ax.set_title(f"{label}, γ={gamma:g}\nSR paths={len(success)}/{len(paths)}, coverage={len(ids)}")
            if i == len(args.method) - 1:
                ax.set_xlabel("x (m)")
            if j == 0:
                ax.set_ylabel("y (m)")
    fig.suptitle("Executed trajectory distributions from the OOD origin", fontsize=17)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
