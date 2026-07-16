#!/usr/bin/env python3
"""Approval-gated Stage 1 geometry and SafeMPPI feasibility sweep.

The ID scene is the existing 8-plug stadium.  Each OOD candidate replaces only
the four central circles with one circle at (2.5, 2.5).  Start and goal remain
fixed and symmetric on the diagonal, with 0.30 m endpoint clearance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _paths  # noqa: F401,E402
import grid_scene as GS  # noqa: E402
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter  # noqa: E402

import gen_uniform_data as SEEDS  # noqa: E402
from viz_style import (  # noqa: E402
    GAMMAS,
    GAMMA_CMAP,
    GAMMA_COLORS,
    GAMMA_NORM,
    gamma_boundaries,
)


STAGE = HERE / "stage_results/01_geometry"
CENTRAL = np.asarray(((2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)), dtype=np.float32)
GIANT_CENTER = np.asarray((2.5, 2.5), dtype=np.float32)
DEFAULT_RADII = (0.90, 1.00, 1.10, 1.20, 1.28)


def make_scene(radius: float | None, start: np.ndarray, goal: np.ndarray):
    """Build the ID scene or one exact four-to-one OOD replacement."""
    env = SEEDS.make_walled_env(8)
    obs = env.obstacles.detach().cpu().numpy()
    if radius is not None:
        is_central = np.zeros(len(obs), dtype=bool)
        for center in CENTRAL:
            is_central |= np.all(np.isclose(obs[:, :2], center[None, :], atol=1e-7), axis=1)
        if int(is_central.sum()) != 4:
            raise RuntimeError(f"expected four central obstacles, found {int(is_central.sum())}")
        replacement = np.asarray([[GIANT_CENTER[0], GIANT_CENTER[1], radius]], dtype=np.float32)
        obs = np.concatenate((obs[~is_central], replacement), axis=0)
        env.obstacles = torch.as_tensor(obs, dtype=env.obstacles.dtype)
        env.obs_vel = torch.zeros(len(obs), 2, dtype=env.obstacles.dtype)
    env.x0 = torch.tensor([start[0], start[1], 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor(goal, dtype=env.goal.dtype)
    return env


def clearance(point: np.ndarray, env) -> float:
    obs = env.obstacles.detach().cpu().numpy()
    return float((np.linalg.norm(obs[:, :2] - point[None, :], axis=1)
                  - obs[:, 2] - float(env.r_robot)).min())


def select_diagonal_endpoints(target_clearance: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select symmetric points whose limiting ID obstacle clearance is the requested value."""
    dummy_start = np.asarray((0.5, 0.5), dtype=np.float32)
    dummy_goal = 5.0 - dummy_start
    env = make_scene(None, dummy_start, dummy_goal)
    ds = np.linspace(0.35, 0.90, 250_001, dtype=np.float64)
    points = np.stack((ds, ds), axis=1)
    obs = env.obstacles.detach().cpu().numpy()
    values = (np.linalg.norm(points[:, None, :] - obs[None, :, :2], axis=2)
              - obs[None, :, 2] - float(env.r_robot)).min(axis=1)
    index = int(np.abs(values - target_clearance).argmin())
    d = float(ds[index])
    start = np.asarray((d, d), dtype=np.float32)
    goal = np.asarray((5.0 - d, 5.0 - d), dtype=np.float32)
    env = make_scene(None, start, goal)
    c_start, c_goal = clearance(start, env), clearance(goal, env)
    if min(c_start, c_goal) + 1e-5 < target_clearance:
        raise RuntimeError("endpoint search fell below requested clearance")
    return start, goal, {
        "target_clearance": float(target_clearance),
        "start_clearance": c_start,
        "goal_clearance": c_goal,
        "diagonal_offset": d,
        "distance": float(np.linalg.norm(goal - start)),
    }


def nearest_surrounding_gap(env, radius: float) -> tuple[float, list[float]]:
    obs = env.obstacles.detach().cpu().numpy()
    is_giant = (np.linalg.norm(obs[:, :2] - GIANT_CENTER[None, :], axis=1) < 1e-6)
    others = obs[~is_giant]
    surface_gaps = np.linalg.norm(others[:, :2] - GIANT_CENTER[None, :], axis=1) - others[:, 2] - radius
    nearest = np.sort(surface_gaps)[:8]
    return float(nearest[0]), [float(value) for value in nearest]


def torch_step(state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
    result = torch.empty_like(state)
    result[0] = state[0] + dt * state[2] + 0.5 * dt * dt * action[0]
    result[1] = state[1] + dt * state[3] + 0.5 * dt * dt * action[1]
    result[2] = state[2] + dt * action[0]
    result[3] = state[3] + dt * action[1]
    return result


@torch.inference_mode()
def rollout_expert(env, gamma: float, seed: int, reach: float, device: torch.device) -> dict:
    config = GS.mode1_config()
    adapter = SafeMPPIAdapter(**config)
    state = env.x0.detach().to(device).float()
    goal = env.goal.detach().to(device).float()
    planner_obstacles = GS.planner_obstacles(env).to(device)
    true_obstacles = env.obstacles.detach().cpu().numpy()
    path = [state[:2].detach().cpu().numpy().copy()]
    controls = []
    dead_reason = None
    started = time.perf_counter()

    for step in range(int(env.T)):
        action, _info = adapter.plan(
            state,
            goal,
            planner_obstacles,
            gamma=float(gamma),
            seed=int(seed) * 1000 + step,
        )
        state = torch_step(state, action, float(env.dt))
        position = state[:2].detach().cpu().numpy().copy()
        controls.append(action.detach().cpu().numpy().copy())
        path.append(position)
        if float(torch.linalg.norm(state[:2] - goal)) < reach:
            break
        if (position < 0.0).any() or (position > 5.0).any():
            dead_reason = "out_of_bounds"
            break
        instant = (np.linalg.norm(true_obstacles[:, :2] - position[None, :], axis=1)
                   - true_obstacles[:, 2] - float(env.r_robot)).min()
        if instant < 0.0:
            dead_reason = "collision"
            break

    path = np.asarray(path, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    all_clearance = (np.linalg.norm(path[:, None, :] - true_obstacles[None, :, :2], axis=2)
                     - true_obstacles[None, :, 2] - float(env.r_robot))
    min_clearance = float(all_clearance.min())
    endpoint_distance = float(np.linalg.norm(path[-1] - env.goal.detach().cpu().numpy()))
    reached = endpoint_distance < reach
    collision = min_clearance < 0.0
    in_taskspace = bool(((path >= 0.0) & (path <= 5.0)).all())
    if dead_reason is None and not reached:
        dead_reason = "timeout"
    near_giant = np.linalg.norm(path - GIANT_CENTER[None, :], axis=1) < 1.9
    side_score = float(np.median(path[near_giant, 1] - path[near_giant, 0])) if near_giant.any() else math.nan
    detour_side = "upper-left" if side_score > 0.05 else "lower-right" if side_score < -0.05 else "diagonal"
    return {
        "gamma": float(gamma),
        "seed": int(seed),
        "success": bool(reached and not collision and in_taskspace),
        "reached": bool(reached),
        "collision": bool(collision),
        "in_taskspace": in_taskspace,
        "dead_reason": dead_reason,
        "steps": int(len(controls)),
        "elapsed_s": time.perf_counter() - started,
        "endpoint_distance": endpoint_distance,
        "min_clearance": min_clearance,
        "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "side_score": side_score,
        "detour_side": detour_side,
        "path": path,
        "controls": controls,
    }


def object_array(items) -> np.ndarray:
    result = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        result[index] = item
    return result


def draw_scene(axis, env, start: np.ndarray, goal: np.ndarray, reach: float, radius: float | None) -> None:
    axis.set_facecolor("#f8f7f4")
    axis.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#333333", lw=0.8, ls="--"))
    for obstacle in env.obstacles.detach().cpu().numpy():
        giant = radius is not None and np.linalg.norm(obstacle[:2] - GIANT_CENTER) < 1e-6
        axis.add_patch(Circle(
            obstacle[:2], obstacle[2] + float(env.r_robot),
            facecolor="#666666" if giant else "#929292",
            edgecolor="#b2182b" if giant else "none",
            lw=2.0 if giant else 0.0,
            alpha=0.96,
            zorder=2,
        ))
    axis.add_patch(Circle(goal, reach, fill=False, ec="#d32f2f", ls="--", lw=0.8, zorder=6))
    axis.plot(start[0], start[1], "s", c="k", ms=5.5, zorder=8)
    axis.plot(goal[0], goal[1], "*", c="gold", mec="k", ms=11, zorder=8)
    axis.set(xlim=(-0.4, 5.4), ylim=(-0.4, 5.4))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def render(summary: dict, scenes: list[dict], output_png: Path, output_pdf: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 10.2))
    for axis, scene in zip(axes.ravel(), scenes):
        draw_scene(axis, scene["env"], np.asarray(summary["start"]), np.asarray(summary["goal"]),
                   summary["reach"], scene["radius"])
        for result in scene["results"]:
            path = result["path"]
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[result["gamma"]],
                      lw=1.75 if result["success"] else 1.0,
                      alpha=0.92 if result["success"] else 0.55, zorder=4)
            if not result["success"]:
                axis.plot(path[-1, 0], path[-1, 1], "x", c="#cc3311", ms=6, mew=1.5, zorder=7)
        successes = sum(result["success"] for result in scene["results"])
        if scene["radius"] is None:
            title = f"ID stadium\nSafeMPPI {successes}/7"
        else:
            title = (f"OOD giant radius={scene['radius']:.2f} m\n"
                     f"nearest surface gap={scene['gap']:.3f} m; SafeMPPI {successes}/7")
        axis.set_title(title, fontsize=11)

    color_axis = fig.add_axes((0.35, 0.895, 0.30, 0.023))
    colorbar = mpl.colorbar.ColorbarBase(
        color_axis,
        cmap=GAMMA_CMAP,
        norm=GAMMA_NORM,
        boundaries=gamma_boundaries(),
        ticks=GAMMAS,
        spacing="uniform",
        orientation="horizontal",
        drawedges=True,
    )
    colorbar.ax.set_title(r"SafeMPPI trajectory color: safety level $\gamma$", fontsize=10, pad=2)
    colorbar.ax.tick_params(length=0, labelsize=8)
    colorbar.dividers.set_color("white")
    endpoint = summary["endpoint_geometry"]
    fig.suptitle(
        "Giant-obstacle OOD Stage 1 — same diagonal task; only the four central obstacles change",
        fontsize=14,
        y=0.993,
    )
    fig.text(
        0.5,
        0.957,
        f"start={np.round(summary['start'], 3)}   goal={np.round(summary['goal'], 3)}   "
        f"endpoint clearance={min(endpoint['start_clearance'], endpoint['goal_clearance']):.3f} m",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.03, top=0.84, wspace=0.05, hspace=0.13)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    parser.add_argument("--target-clearance", type=float, default=0.30)
    parser.add_argument("--reach", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=64000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--outdir", type=Path, default=STAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.radii) != 5:
        raise ValueError("the approval figure is a 2x3 panel and requires exactly five candidate radii")
    if sorted(args.radii) != list(args.radii) or len(set(args.radii)) != len(args.radii):
        raise ValueError("radii must be unique and increasing")
    start, goal, endpoint_geometry = select_diagonal_endpoints(args.target_clearance)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    scene_specs = [("id", None)] + [(f"r{radius:.2f}", float(radius)) for radius in args.radii]
    scenes = []
    flat_results = []
    started = time.perf_counter()
    for scene_index, (name, radius) in enumerate(scene_specs):
        env = make_scene(radius, start, goal)
        gap, nearest_gaps = (math.nan, []) if radius is None else nearest_surrounding_gap(env, radius)
        results = []
        if not args.skip_rollouts:
            # Keep the planner-noise seed matched across gamma so differences in
            # clearance and route are attributable to the safety level.
            for gamma in GAMMAS:
                result = rollout_expert(
                    env, gamma, args.seed + scene_index * 100,
                    args.reach, device,
                )
                result["scene"] = name
                result["radius"] = radius
                results.append(result)
                flat_results.append(result)
                print(
                    f"[{name}] gamma={gamma:g} success={int(result['success'])} "
                    f"steps={result['steps']} clear={result['min_clearance']:.3f} "
                    f"end={result['endpoint_distance']:.3f} side={result['detour_side']}",
                    flush=True,
                )
        scenes.append({
            "name": name,
            "radius": radius,
            "gap": gap,
            "nearest_eight_surface_gaps": nearest_gaps,
            "env": env,
            "results": results,
        })

    summary = {
        "status": "GEOMETRY_ONLY" if args.skip_rollouts else "COMPLETE_AWAITING_APPROVAL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "reach": float(args.reach),
        "wall_plugs": 8,
        "endpoint_geometry": endpoint_geometry,
        "central_removed": CENTRAL.tolist(),
        "giant_center": GIANT_CENTER.tolist(),
        "candidates": [],
    }
    for scene in scenes:
        serial_results = [
            {key: value for key, value in result.items() if key not in ("path", "controls")}
            for result in scene["results"]
        ]
        summary["candidates"].append({
            "name": scene["name"],
            "radius": scene["radius"],
            "nearest_surface_gap": None if scene["radius"] is None else scene["gap"],
            "nearest_eight_surface_gaps": scene["nearest_eight_surface_gaps"],
            "successes": int(sum(result["success"] for result in scene["results"])),
            "collisions": int(sum(result["collision"] for result in scene["results"])),
            "out_of_bounds": int(sum(not result["in_taskspace"] for result in scene["results"])),
            "results": serial_results,
        })

    data_dir = args.outdir / "data"
    log_dir = args.outdir / "logs"
    viz_dir = args.outdir / "viz"
    for directory in (data_dir, log_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if flat_results:
        np.savez_compressed(
            data_dir / "expert_radius_sweep_paths.npz",
            scenes=np.asarray([result["scene"] for result in flat_results]),
            radii=np.asarray([math.nan if result["radius"] is None else result["radius"]
                              for result in flat_results], dtype=np.float32),
            gammas=np.asarray([result["gamma"] for result in flat_results], dtype=np.float32),
            success=np.asarray([result["success"] for result in flat_results], dtype=bool),
            paths=object_array([result["path"] for result in flat_results]),
            controls=object_array([result["controls"] for result in flat_results]),
            start=start,
            goal=goal,
        )
    render(summary, scenes, viz_dir / "giant_radius_sweep.png", viz_dir / "giant_radius_sweep.pdf")
    (log_dir / "stage1_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "start": summary["start"],
        "goal": summary["goal"],
        "endpoint_clearance": min(endpoint_geometry["start_clearance"], endpoint_geometry["goal_clearance"]),
        "successes_by_scene": {row["name"]: row["successes"] for row in summary["candidates"]},
        "output": str(viz_dir / "giant_radius_sweep.png"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
