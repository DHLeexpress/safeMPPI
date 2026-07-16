#!/usr/bin/env python3
"""Build the fixed off-diagonal start/goal pools for the walled scene.

The sampling rule is copied from ``rev_expansion/gen_uniform_data.py``:
a 32 x 32 grid over [0.1, 4.9], fixed +/-0.02 jitter, an excluded
``|y-x| < 1`` band, and 5 cm obstacle clearance.  Unlike the original
fixed-goal data generator, obstacle filtering happens after adding the eight
wall plugs and the remaining points are split into start (blue, y>x) and goal
(red, y<x) pools.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]  # overnight_run_07_06/
STAGE_DIR = HERE / "stage_results" / "01_seeds"
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _paths  # noqa: F401,E402 - installs the shared project import paths
import grid_scene as GS  # noqa: E402


GRID_LO = 0.1
GRID_HI = 4.9
GRID_N = 32
JITTER = 0.02
DIAGONAL_GAP = 1.0
FREE_CLEARANCE = 0.05

# Exact eight-plug geometry from grid_expand_hardtail._WALL_PLUGS8.
_WALL_STEP = 5.0 / 13.0
_WALL_PLUGS4 = (
    (_WALL_STEP, -0.2, 0.2),
    (5.0 - _WALL_STEP, 5.2, 0.2),
    (-0.2, _WALL_STEP, 0.2),
    (5.2, 5.0 - _WALL_STEP, 0.2),
)
_WALL_PLUGS8 = _WALL_PLUGS4 + (
    (0.0, -0.2, 0.2),
    (-0.2, 0.0, 0.2),
    (5.2, 5.0, 0.2),
    (5.0, 5.2, 0.2),
)


def apply_wall_plugs(env, n: int = 8):
    """Apply the trainer's wall-plug geometry to a fresh grid environment."""
    if not n:
        return env
    plugs = _WALL_PLUGS4[:2] if n == 2 else _WALL_PLUGS8 if n == 8 else _WALL_PLUGS4
    extra = torch.tensor(plugs, dtype=env.obstacles.dtype, device=env.obstacles.device)
    env.obstacles = torch.cat((env.obstacles, extra), dim=0)
    return env


def make_walled_env(wall_plugs: int = 8):
    """Return a new grid environment with the requested perimeter plugs."""
    return apply_wall_plugs(GS.make_grid(), wall_plugs)


def uniform_starts(env=None) -> np.ndarray:
    """Return the deterministic obstacle-free off-diagonal point grid.

    When no environment is supplied, the required eight-plug scene is used.
    A caller that supplies ``env`` is responsible for applying its scene
    geometry before calling this function.
    """
    if env is None:
        env = make_walled_env(8)

    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    cell = (GRID_HI - GRID_LO) / GRID_N
    centers = GRID_LO + cell * (np.arange(GRID_N) + 0.5)
    x_grid, y_grid = np.meshgrid(centers, centers)
    points = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
    points += np.random.default_rng(0).uniform(-JITTER, JITTER, points.shape)

    outside_band = np.abs(points[:, 1] - points[:, 0]) >= DIAGONAL_GAP
    clearance = (
        np.linalg.norm(points[:, None, :] - obs[None, :, :2], axis=2)
        - obs[None, :, 2]
        - rr
    )
    free = clearance.min(axis=1) > FREE_CLEARANCE
    return points[outside_band & free].astype(np.float32)


def start_goal_pools(env=None) -> tuple[np.ndarray, np.ndarray]:
    """Split the fixed grid into upper starts and lower goals."""
    points = uniform_starts(env)
    blue = points[points[:, 1] > points[:, 0]]
    red = points[points[:, 1] < points[:, 0]]
    return blue, red


def render_seeds(output: Path, env=None) -> dict[str, int]:
    """Render the seed pools and all obstacles, including wall plugs."""
    if env is None:
        env = make_walled_env(8)
    blue, red = start_goal_pools(env)
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)

    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    ax.set_facecolor("#fafafa")
    for obstacle in obs:
        ax.add_patch(
            Circle(
                (float(obstacle[0]), float(obstacle[1])),
                float(obstacle[2] + rr),
                facecolor="#777777",
                edgecolor="#4d4d4d",
                linewidth=0.35,
                alpha=0.88,
                zorder=2,
            )
        )

    # Make the excluded diagonal strip explicit without obscuring the points.
    x = np.linspace(0.0, 5.0, 400)
    ax.fill_between(
        x,
        np.maximum(0.0, x - DIAGONAL_GAP),
        np.minimum(5.0, x + DIAGONAL_GAP),
        color="#d8d8d8",
        alpha=0.28,
        label=r"excluded $|y-x|<1$",
        zorder=0,
    )
    ax.scatter(
        blue[:, 0], blue[:, 1], s=15, c="#1769aa", edgecolors="none",
        label=f"start pool: y>x (n={len(blue)})", zorder=4,
    )
    ax.scatter(
        red[:, 0], red[:, 1], s=15, c="#d32f2f", edgecolors="none",
        label=f"goal pool: y<x (n={len(red)})", zorder=4,
    )
    ax.plot((0, 5, 5, 0, 0), (0, 0, 5, 5, 0), "--", color="#222222", lw=0.9, zorder=3)

    ax.set(xlim=(-0.55, 5.55), ylim=(-0.55, 5.55), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.grid(color="white", linewidth=0.7, alpha=0.8, zorder=1)
    total = len(blue) + len(red)
    ax.set_title(f"Walled-scene start/goal seeds ({total} points, 8 plugs)")
    ax.legend(loc="center", framealpha=0.94, fontsize=9)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"total": total, "blue": len(blue), "red": len(red), "obstacles": len(obs)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STAGE_DIR / "viz" / "seeds.png")
    parser.add_argument("--log", type=Path, default=STAGE_DIR / "logs" / "seed_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = render_seeds(args.out)
    blue_pct = 100.0 * counts["blue"] / counts["total"]
    red_pct = 100.0 * counts["red"] / counts["total"]
    message = (
        f"SAVED {args.out}: {counts['total']} points; "
        f"blue={counts['blue']} ({blue_pct:.1f}%); "
        f"red={counts['red']} ({red_pct:.1f}%); "
        f"obstacles={counts['obstacles']}"
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "output": str(args.out.resolve()),
                **counts,
                "blue_fraction": counts["blue"] / counts["total"],
                "red_fraction": counts["red"] / counts["total"],
                "grid": {"lo": GRID_LO, "hi": GRID_HI, "n": GRID_N, "jitter": JITTER},
                "filters": {"diagonal_gap": DIAGONAL_GAP, "free_clearance": FREE_CLEARANCE},
                "wall_plugs": 8,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(message)
    print(f"LOG {args.log}")


if __name__ == "__main__":
    main()
