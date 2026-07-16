#!/usr/bin/env python3
"""Stage 2A: tune and inspect an optional exponential goal-retreat penalty.

This is an approval-gated recipe check, not demonstration generation.  The
fixed radius-1.2 OOD scene and the Stage-1B smooth expert are retained.  Only
the following MPPI running-cost term is swept::

    w_retreat * expm1(max(d_{t+1} - d_t, 0) / retreat_scale)

The term is intended to remove back-and-forth demonstrations while preserving
the two legitimate tangential detours around the giant obstacle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _paths  # noqa: F401,E402
import grid_metrics as GM  # noqa: E402
import grid_scene as GS  # noqa: E402
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter  # noqa: E402

from giant_obstacle_ood.stage1_geometry_sweep import (  # noqa: E402
    draw_scene,
    object_array,
    torch_step,
)
from giant_obstacle_ood.stage1b_smooth_expert import (  # noqa: E402
    GOAL,
    RADIUS,
    START,
    approach_audit,
    finite_stats,
    make_fixed_scene,
    route_mode,
    smoothness_metrics,
)
from viz_style import GAMMAS, GAMMA_COLORS  # noqa: E402


STAGE = HERE / "stage_results/02a_retreat_penalty"
BASELINE_DATA = HERE / "stage_results/01b_smooth_expert/data/expert_m2_paths.npz"
DEFAULT_WEIGHTS = (0.0, 0.01, 0.03, 0.10, 0.30, 1.0)
DEFAULT_SCALE = 0.05
DEFAULT_CAP = 6.0
SMOOTH_WEIGHT = 8.0


def retreat_audit(path: np.ndarray, goal: np.ndarray, scale: float, cap: float) -> dict:
    """Executed-path radial backtracking diagnostics (cost-independent)."""
    p = np.asarray(path, dtype=float)
    distances = np.linalg.norm(p - np.asarray(goal, dtype=float)[None, :], axis=1)
    increments = np.diff(distances)
    retreat = np.maximum(increments, 0.0)
    forward = np.maximum(-increments, 0.0)
    active = retreat > 1e-3
    longest = current = 0
    for flag in active:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    signed = np.zeros_like(increments, dtype=np.int8)
    signed[increments > 5e-3] = 1
    signed[increments < -5e-3] = -1
    nz = signed[signed != 0]
    switches = int(np.count_nonzero(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    normalized = np.minimum(retreat / max(float(scale), np.finfo(float).eps), max(float(cap), 0.0))
    return {
        "initial_goal_distance": float(distances[0]),
        "final_goal_distance": float(distances[-1]),
        "net_goal_progress": float(distances[0] - distances[-1]),
        "gross_forward_distance": float(forward.sum()),
        "retreat_distance_total": float(retreat.sum()),
        "retreat_distance_max_step": float(retreat.max(initial=0.0)),
        "retreat_step_fraction_1mm": float(active.mean()) if len(active) else 0.0,
        "retreat_steps_1mm": int(active.sum()),
        "retreat_steps_5mm": int(np.count_nonzero(retreat > 5e-3)),
        "longest_retreat_run_1mm": int(longest),
        "radial_direction_switches_5mm": switches,
        "backtrack_over_forward": float(retreat.sum() / max(forward.sum(), 1e-12)),
        "executed_exp_retreat_mass": float(np.expm1(normalized).sum()),
        "goal_distances": distances.astype(np.float32),
    }


@torch.inference_mode()
def rollout(env, gamma: float, seed: int, retreat_weight: float, scale: float,
            cap: float, reach: float, device: torch.device, certify: bool) -> dict:
    config = GS.mode1_config()
    config.update({
        "smooth_weight": SMOOTH_WEIGHT,
        "goal_retreat_exp_weight": float(retreat_weight),
        "goal_retreat_exp_scale": float(scale),
        "goal_retreat_exp_cap": float(cap),
    })
    adapter = SafeMPPIAdapter(**config)
    state = env.x0.detach().to(device).float()
    goal = env.goal.detach().to(device).float()
    planner_obstacles = GS.planner_obstacles(env).to(device)
    true_obstacles = env.obstacles.detach().cpu().numpy()
    states = [state.detach().cpu().numpy().copy()]
    controls = []
    dead_reason = None
    started = time.perf_counter()
    for step in range(int(env.T)):
        action, _ = adapter.plan(
            state,
            goal,
            planner_obstacles,
            gamma=float(gamma),
            seed=int(seed) * 1000 + step,
        )
        state = torch_step(state, action, float(env.dt))
        current = state.detach().cpu().numpy().copy()
        controls.append(action.detach().cpu().numpy().copy())
        states.append(current)
        position = current[:2]
        instant = (np.linalg.norm(true_obstacles[:, :2] - position[None, :], axis=1)
                   - true_obstacles[:, 2] - float(env.r_robot)).min()
        if instant < 0.0:
            dead_reason = "collision"
            break
        if (position < 0.0).any() or (position > 5.0).any():
            dead_reason = "out_of_bounds"
            break
        if float(np.linalg.norm(position - env.goal.detach().cpu().numpy())) < reach:
            break

    states = np.asarray(states, dtype=np.float32)
    path = states[:, :2]
    controls = np.asarray(controls, dtype=np.float32)
    clearances = (np.linalg.norm(path[:, None, :] - true_obstacles[None, :, :2], axis=2)
                  - true_obstacles[None, :, 2] - float(env.r_robot))
    goal_np = env.goal.detach().cpu().numpy()
    endpoint_distance = float(np.linalg.norm(path[-1] - goal_np))
    reached = endpoint_distance < reach
    collision = bool(clearances.min() < 0.0)
    in_taskspace = bool(((path >= 0.0) & (path <= 5.0)).all())
    success = bool(reached and not collision and in_taskspace)
    if dead_reason is None and not reached:
        dead_reason = "timeout"
    mode, side_score = route_mode(path)
    audit = retreat_audit(path, goal_np, scale, cap)
    result = {
        "gamma": float(gamma),
        "seed": int(seed),
        "retreat_weight": float(retreat_weight),
        "retreat_scale": float(scale),
        "retreat_cap": float(cap),
        "smooth_weight": SMOOTH_WEIGHT,
        "success": success,
        "reached": bool(reached),
        "collision": collision,
        "in_taskspace": in_taskspace,
        "dead_reason": dead_reason,
        "steps": int(len(controls)),
        "time_s": float(len(controls) * env.dt),
        "wall_s": time.perf_counter() - started,
        "endpoint_distance": endpoint_distance,
        "clearance_mean": float(clearances.min(axis=1).mean()),
        "min_clearance": float(clearances.min()),
        "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "route_mode": mode,
        "side_score": side_score,
        **smoothness_metrics(controls, float(env.dt)),
        **{key: value for key, value in audit.items() if key != "goal_distances"},
        "path": path,
        "states": states,
        "controls": controls,
        "goal_distances": audit["goal_distances"],
    }
    if certify:
        progress = approach_audit(path, goal_np, reach)
        result.update({
            "progress_ok": bool(progress["ok"]),
            "progress_failure_fraction": progress["failure_fraction"],
            "socp_ok": bool(GM.socp_ok(path, env, gamma)),
        })
        result["valid2"] = bool(result["success"] and result["progress_ok"] and result["socp_ok"])
    return result


def serial_result(result: dict) -> dict:
    omitted = {"path", "states", "controls", "goal_distances"}
    return {key: value for key, value in result.items() if key not in omitted}


def summarize(group: list[dict], weight: float) -> dict:
    successes = [result for result in group if result["success"]]
    source = successes or group
    row = {
        "retreat_weight": float(weight),
        "N": len(group),
        "successes": int(sum(result["success"] for result in group)),
        "collisions": int(sum(result["collision"] for result in group)),
        "timeouts": int(sum(result["dead_reason"] == "timeout" for result in group)),
        "out_of_bounds": int(sum(not result["in_taskspace"] for result in group)),
    }
    for output_key, input_key in (
        ("mean_retreat_total", "retreat_distance_total"),
        ("mean_retreat_fraction", "retreat_step_fraction_1mm"),
        ("mean_max_retreat_step", "retreat_distance_max_step"),
        ("mean_longest_retreat_run", "longest_retreat_run_1mm"),
        ("mean_direction_switches", "radial_direction_switches_5mm"),
        ("mean_backtrack_ratio", "backtrack_over_forward"),
        ("mean_time_s", "time_s"),
        ("mean_path_length", "path_length"),
        ("mean_clearance", "clearance_mean"),
        ("mean_control_delta", "control_delta_mean"),
    ):
        row[output_key] = finite_stats([result[input_key] for result in source])[0]
    row["covered_homotopies"] = len({result["route_mode"] for result in successes})
    return row


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(rows: list[dict], output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    weights = np.asarray([row["retreat_weight"] for row in rows])
    x = np.arange(len(weights))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    panels = (
        ("successes", "successes / 7"),
        ("mean_retreat_total", "executed radial retreat [m]"),
        ("mean_retreat_fraction", "fraction of retreating steps (>1 mm)"),
        ("mean_time_s", "time to goal [s]"),
        ("mean_path_length", "path length [m]"),
        ("mean_clearance", "mean nearest-obstacle clearance [m]"),
    )
    for axis, (key, label) in zip(axes.ravel(), panels):
        axis.plot(x, [row[key] for row in rows], "o-", color="#3b528b")
        axis.set_xticks(x, [f"{w:g}" for w in weights])
        axis.set_xlabel(r"retreat weight $w_r$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.suptitle(r"SafeMPPI anti-retreat tuning: $w_r\,\mathrm{expm1}([d_{t+1}-d_t]_+/0.05)$", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_paths(env, records: list[dict], weights: list[float], reach: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 7})
    fig, axes = plt.subplots(len(weights), len(GAMMAS), figsize=(17.5, 3.0 * len(weights)), squeeze=False)
    for row_index, weight in enumerate(weights):
        for col_index, gamma in enumerate(GAMMAS):
            axis = axes[row_index, col_index]
            draw_scene(axis, env, START, GOAL, reach, RADIUS)
            result = next(r for r in records if r["retreat_weight"] == weight and r["gamma"] == gamma)
            path = result["path"]
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=1.5, zorder=5)
            if not result["success"]:
                axis.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=6, mew=1.5, zorder=7)
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$", fontsize=10)
            if col_index == 0:
                axis.set_ylabel(rf"$w_r={weight:g}$", fontsize=10)
            axis.tick_params(labelbottom=False, labelleft=False, length=0)
    fig.suptitle("Matched-seed OOD expert trajectories — inspect backtracking before data generation", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.4, w_pad=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def render_distance_profiles(records: list[dict], weights: list[float], dt: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0))
    palette = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, len(weights)))
    for axis, gamma in zip(axes.ravel()[:7], GAMMAS):
        for color, weight in zip(palette, weights):
            result = next(r for r in records if r["retreat_weight"] == weight and r["gamma"] == gamma)
            d = result["goal_distances"]
            axis.plot(np.arange(len(d)) * dt, d, color=color, lw=1.2, label=f"{weight:g}")
        axis.set_title(rf"$\gamma={gamma:g}$")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("goal distance [m]")
        axis.grid(alpha=0.22)
    legend_axis = axes.ravel()[7]
    legend_axis.axis("off")
    handles = [mpl.lines.Line2D([], [], color=color, lw=2, label=rf"$w_r={weight:g}$")
               for color, weight in zip(palette, weights)]
    legend_axis.legend(handles=handles, loc="center", frameon=False, title="retreat penalty")
    fig.suptitle("Executed goal-distance traces (upward segments are the targeted hack)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def save_records(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=np.asarray([r["retreat_weight"] for r in records]),
        gammas=np.asarray([r["gamma"] for r in records]),
        seeds=np.asarray([r["seed"] for r in records]),
        success=np.asarray([r["success"] for r in records]),
        paths=object_array([r["path"] for r in records]),
        states=object_array([r["states"] for r in records]),
        controls=object_array([r["controls"] for r in records]),
        goal_distances=object_array([r["goal_distances"] for r in records]),
    )


def load_stage1b_baseline(env, scale: float, cap: float, reach: float) -> list[dict]:
    """Re-audit the locked Stage-1B M=2 paths with the new retreat metrics."""
    if not BASELINE_DATA.exists():
        raise FileNotFoundError(f"locked Stage-1B baseline is missing: {BASELINE_DATA}")
    archive = np.load(BASELINE_DATA, allow_pickle=True)
    true_obstacles = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    results = []
    for gamma, seed, path, states, controls in zip(
        archive["gammas"], archive["seeds"], archive["paths"], archive["states"], archive["controls"]
    ):
        path = np.asarray(path, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        controls = np.asarray(controls, dtype=np.float32)
        clearances = (np.linalg.norm(path[:, None, :] - true_obstacles[None, :, :2], axis=2)
                      - true_obstacles[None, :, 2] - float(env.r_robot))
        endpoint_distance = float(np.linalg.norm(path[-1] - goal))
        reached = endpoint_distance < reach
        collision = bool(clearances.min() < 0.0)
        in_taskspace = bool(((path >= 0.0) & (path <= 5.0)).all())
        mode, side_score = route_mode(path)
        audit = retreat_audit(path, goal, scale, cap)
        results.append({
            "gamma": float(gamma),
            "seed": int(seed),
            "retreat_weight": 0.0,
            "success": bool(reached and not collision and in_taskspace),
            "reached": bool(reached),
            "collision": collision,
            "in_taskspace": in_taskspace,
            "dead_reason": None if reached else "timeout",
            "steps": int(len(controls)),
            "time_s": float(len(controls) * env.dt),
            "endpoint_distance": endpoint_distance,
            "clearance_mean": float(clearances.min(axis=1).mean()),
            "min_clearance": float(clearances.min()),
            "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "route_mode": mode,
            "side_score": side_score,
            **smoothness_metrics(controls, float(env.dt)),
            **{key: value for key, value in audit.items() if key != "goal_distances"},
            "path": path,
            "states": states,
            "controls": controls,
            "goal_distances": audit["goal_distances"],
        })
    return results


def gamma_comparison_rows(baseline: list[dict], selected: list[dict]) -> list[dict]:
    rows = []
    for gamma in GAMMAS:
        old = [r for r in baseline if r["gamma"] == gamma]
        new = [r for r in selected if r["gamma"] == gamma]
        row = {"gamma": float(gamma), "M": len(new)}
        for prefix, group in (("baseline", old), ("selected", new)):
            row[f"{prefix}_SR"] = float(np.mean([r["success"] for r in group]))
            row[f"{prefix}_retreat_m"] = finite_stats([r["retreat_distance_total"] for r in group])[0]
            row[f"{prefix}_retreat_fraction"] = finite_stats([r["retreat_step_fraction_1mm"] for r in group])[0]
            row[f"{prefix}_switches"] = finite_stats([r["radial_direction_switches_5mm"] for r in group])[0]
            row[f"{prefix}_time_s"] = finite_stats([r["time_s"] for r in group])[0]
            row[f"{prefix}_path_length_m"] = finite_stats([r["path_length"] for r in group])[0]
            row[f"{prefix}_clearance_m"] = finite_stats([r["clearance_mean"] for r in group])[0]
            row[f"{prefix}_homotopies"] = len({r["route_mode"] for r in group if r["success"]})
        for name in ("retreat_m", "retreat_fraction", "switches", "time_s", "path_length_m"):
            denominator = row[f"baseline_{name}"]
            row[f"relative_change_{name}"] = ((row[f"selected_{name}"] / denominator) - 1.0
                                               if denominator else math.nan)
        rows.append(row)
    return rows


def render_selected_paths(env, baseline: list[dict], selected: list[dict], reach: float,
                          weight: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.8))
    for axis, gamma in zip(axes.ravel()[:7], GAMMAS):
        draw_scene(axis, env, START, GOAL, reach, RADIUS)
        old = [r for r in baseline if r["gamma"] == gamma]
        new = [r for r in selected if r["gamma"] == gamma]
        for result in old:
            path = result["path"]
            axis.plot(path[:, 0], path[:, 1], color="0.72", lw=1.1, alpha=0.75, zorder=3)
        for replicate, result in enumerate(new):
            path = result["path"]
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=2.0,
                      ls="-" if replicate == 0 else "--", alpha=0.95, zorder=5)
            if not result["success"]:
                axis.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=7, mew=1.5, zorder=7)
        old_retreat = finite_stats([r["retreat_distance_total"] for r in old])[0]
        new_retreat = finite_stats([r["retreat_distance_total"] for r in new])[0]
        axis.set_title(
            rf"$\gamma={gamma:g}$  SR={np.mean([r['success'] for r in new]):.0%}" "\n"
            rf"retreat {old_retreat:.2f}$\rightarrow${new_retreat:.2f} m",
            fontsize=9,
        )
    info = axes.ravel()[7]
    info.axis("off")
    info.text(
        0.03, 0.95,
        "Stage 2A candidate\n\n"
        f"retreat weight = {weight:g}\n"
        f"retreat scale = {DEFAULT_SCALE:g} m\n"
        f"smooth weight = {SMOOTH_WEIGHT:g}\n"
        "M = 2 per gamma\n\n"
        "gray = locked no-penalty expert\n"
        "color = candidate expert\n"
        "solid/dashed = replicate 1/2",
        va="top", fontsize=11,
    )
    fig.suptitle("Soft anti-retreat SafeMPPI candidate — matched against locked Stage 1B", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_selected_profiles(baseline: list[dict], selected: list[dict], dt: float,
                             output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0))
    for axis, gamma in zip(axes.ravel()[:7], GAMMAS):
        for result in [r for r in baseline if r["gamma"] == gamma]:
            d = result["goal_distances"]
            axis.plot(np.arange(len(d)) * dt, d, color="0.72", lw=1.0, alpha=0.8)
        for replicate, result in enumerate([r for r in selected if r["gamma"] == gamma]):
            d = result["goal_distances"]
            axis.plot(np.arange(len(d)) * dt, d, color=GAMMA_COLORS[gamma], lw=1.6,
                      ls="-" if replicate == 0 else "--")
        axis.set_title(rf"$\gamma={gamma:g}$")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("goal distance [m]")
        axis.grid(alpha=0.22)
    legend_axis = axes.ravel()[7]
    legend_axis.axis("off")
    handles = [
        mpl.lines.Line2D([], [], color="0.72", lw=2, label="no penalty"),
        mpl.lines.Line2D([], [], color="0.25", lw=2, label="selected penalty"),
    ]
    legend_axis.legend(handles=handles, loc="center", frameon=False)
    fig.suptitle("Goal-distance audit — upward segments expose backtracking", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_comparison(rows: list[dict], output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    gamma = np.asarray([row["gamma"] for row in rows])
    panels = (
        ("retreat_m", "radial retreat [m]"),
        ("retreat_fraction", "retreating-step fraction"),
        ("switches", "radial direction switches"),
        ("time_s", "time to goal [s]"),
        ("path_length_m", "path length [m]"),
        ("clearance_m", "mean clearance [m]"),
    )
    for axis, (key, ylabel) in zip(axes.ravel(), panels):
        axis.plot(gamma, [row[f"baseline_{key}"] for row in rows], "o--", color="0.6", label="no penalty")
        values = [row[f"selected_{key}"] for row in rows]
        axis.plot(gamma, values, color="0.35", lw=1.2, zorder=2, label="selected")
        for g, value in zip(gamma, values):
            axis.scatter(g, value, color=GAMMA_COLORS[float(g)], edgecolor="black", linewidth=0.35, s=45, zorder=4)
        axis.set_xticks(gamma)
        axis.set_xlabel(r"safety level $\gamma$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Stage 2A M=2 audit: locked expert vs soft anti-retreat candidate", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def run_tune(args: argparse.Namespace, device: torch.device) -> None:
    env = make_fixed_scene(args.max_steps)
    records = []
    started = time.perf_counter()
    for weight in args.weights:
        for gamma in GAMMAS:
            result = rollout(env, gamma, args.seed0, weight, args.scale, args.cap,
                             args.reach, device, certify=False)
            records.append(result)
            print(
                f"[tune w={weight:g}] gamma={gamma:g} success={int(result['success'])} "
                f"steps={result['steps']} retreat={result['retreat_distance_total']:.3f}m "
                f"frac={result['retreat_step_fraction_1mm']:.1%} mode={result['route_mode']} "
                f"reason={result['dead_reason']}",
                flush=True,
            )
    rows = [summarize([r for r in records if r["retreat_weight"] == weight], weight)
            for weight in args.weights]
    for subdir in ("logs", "data", "tables", "viz"):
        (args.outdir / subdir).mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "RETREAT_TUNING_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "task": {"start": START.tolist(), "goal": GOAL.tolist(), "radius": RADIUS,
                 "reach": args.reach, "max_steps": args.max_steps},
        "cost": {"formula": "w * expm1(clamp(relu(d_next-d_prev)/scale, max=cap))",
                 "weights": args.weights, "scale": args.scale, "cap": args.cap,
                 "smooth_weight": SMOOTH_WEIGHT},
        "matched_seed": args.seed0,
        "rows": rows,
        "rollouts": [serial_result(result) for result in records],
    }
    (args.outdir / "logs/tuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(rows, args.outdir / "tables/tuning_summary.csv")
    save_records(records, args.outdir / "data/tuning_paths.npz")
    render_summary(rows, args.outdir / "viz/tuning_summary.png")
    render_paths(env, records, list(args.weights), args.reach, args.outdir / "viz/tuning_paths.png")
    render_distance_profiles(records, list(args.weights), float(env.dt),
                             args.outdir / "viz/tuning_goal_distance.png")
    print(json.dumps({"status": summary["status"], "rows": rows,
                      "output": str(args.outdir / "viz/tuning_paths.png")}, indent=2), flush=True)


def run_validate(args: argparse.Namespace, device: torch.device) -> None:
    if args.M != 2:
        raise ValueError("the approval candidate is locked to M=2 per gamma")
    env = make_fixed_scene(args.max_steps)
    baseline = load_stage1b_baseline(env, args.scale, args.cap, args.reach)
    if sorted((r["gamma"], r["seed"]) for r in baseline) != sorted(
        (gamma, args.seed0 + replicate) for replicate in range(args.M) for gamma in GAMMAS
    ):
        raise RuntimeError("Stage-1B baseline seeds/gammas do not match the candidate protocol")
    locked_probe = next(r for r in baseline if r["gamma"] == 0.3 and r["seed"] == args.seed0)
    zero_probe = rollout(env, 0.3, args.seed0, 0.0, args.scale, args.cap,
                         args.reach, device, certify=False)
    def trajectory_digest(result: dict) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(result["states"], dtype=np.float32).tobytes())
        digest.update(np.asarray(result["controls"], dtype=np.float32).tobytes())
        return digest.hexdigest()
    compatibility = {
        "probe_gamma": 0.3,
        "probe_seed": args.seed0,
        "locked_stage1b_sha256": trajectory_digest(locked_probe),
        "new_zero_weight_sha256": trajectory_digest(zero_probe),
        "states_exact_equal": bool(np.array_equal(locked_probe["states"], zero_probe["states"])),
        "controls_exact_equal": bool(np.array_equal(locked_probe["controls"], zero_probe["controls"])),
    }
    compatibility["exact_match"] = bool(
        compatibility["states_exact_equal"] and compatibility["controls_exact_equal"]
    )
    if not compatibility["exact_match"]:
        raise RuntimeError("zero retreat weight changed the locked Stage-1B trajectory")
    selected = []
    started = time.perf_counter()
    for replicate in range(args.M):
        seed = args.seed0 + replicate
        for gamma in GAMMAS:
            result = rollout(env, gamma, seed, args.selected_weight, args.scale, args.cap,
                             args.reach, device, certify=True)
            result["replicate"] = replicate
            selected.append(result)
            print(
                f"[validate m={replicate + 1}/{args.M}] gamma={gamma:g} success={int(result['success'])} "
                f"steps={result['steps']} retreat={result['retreat_distance_total']:.3f}m "
                f"mode={result['route_mode']} valid2={int(result['valid2'])}",
                flush=True,
            )
    rows = gamma_comparison_rows(baseline, selected)
    old_summary = summarize(baseline, 0.0)
    new_summary = summarize(selected, args.selected_weight)
    old_modes = {r["route_mode"] for r in baseline if r["success"]}
    new_modes = {r["route_mode"] for r in selected if r["success"]}
    decision = {
        "all_14_physical_success": bool(all(r["success"] for r in selected)),
        "all_14_collision_free": bool(not any(r["collision"] for r in selected)),
        "both_homotopies_preserved": bool(new_modes == {"upper-left", "lower-right"}),
        "retreat_reduced": bool(new_summary["mean_retreat_total"] < old_summary["mean_retreat_total"]),
        "direction_switches_reduced": bool(new_summary["mean_direction_switches"] < old_summary["mean_direction_switches"]),
        "mean_time_not_slower": bool(new_summary["mean_time_s"] <= old_summary["mean_time_s"]),
        "mean_path_not_longer": bool(new_summary["mean_path_length"] <= old_summary["mean_path_length"]),
        "retreat_relative_change": float(new_summary["mean_retreat_total"] / old_summary["mean_retreat_total"] - 1.0),
        "direction_switch_relative_change": float(new_summary["mean_direction_switches"] / old_summary["mean_direction_switches"] - 1.0),
        "time_relative_change": float(new_summary["mean_time_s"] / old_summary["mean_time_s"] - 1.0),
        "path_length_relative_change": float(new_summary["mean_path_length"] / old_summary["mean_path_length"] - 1.0),
        "baseline_modes": sorted(old_modes),
        "selected_modes": sorted(new_modes),
        "selected_progress_pass_rate": float(np.mean([r["progress_ok"] for r in selected])),
        "selected_socp_pass_rate": float(np.mean([r["socp_ok"] for r in selected])),
        "selected_valid2_rate": float(np.mean([r["valid2"] for r in selected])),
    }
    for subdir in ("logs", "data", "tables", "viz"):
        (args.outdir / subdir).mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "RETREAT_CANDIDATE_M2_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "task": {"start": START.tolist(), "goal": GOAL.tolist(), "radius": RADIUS,
                 "reach": args.reach, "max_steps": args.max_steps, "M_per_gamma": args.M},
        "cost": {"formula": "w * expm1(clamp(relu(d_next-d_prev)/scale, max=cap))",
                 "selected_weight": args.selected_weight, "scale": args.scale,
                 "cap": args.cap, "smooth_weight": SMOOTH_WEIGHT},
        "baseline_source": str(BASELINE_DATA),
        "zero_weight_compatibility": compatibility,
        "matched_seeds": list(range(args.seed0, args.seed0 + args.M)),
        "baseline_overall": old_summary,
        "selected_overall": new_summary,
        "decision_audit": decision,
        "per_gamma": rows,
        "selected_rollouts": [serial_result(result) for result in selected],
    }
    (args.outdir / "logs/selected_m2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(rows, args.outdir / "tables/selected_vs_baseline_m2.csv")
    save_records(selected, args.outdir / "data/selected_m2_paths.npz")
    render_selected_paths(env, baseline, selected, args.reach, args.selected_weight,
                          args.outdir / "viz/selected_m2_by_gamma.png")
    render_selected_profiles(baseline, selected, float(env.dt),
                             args.outdir / "viz/selected_m2_goal_distance.png")
    render_comparison(rows, args.outdir / "viz/selected_vs_baseline_m2.png")
    print(json.dumps({"status": summary["status"], "baseline": old_summary,
                      "selected": new_summary, "decision": decision,
                      "output": str(args.outdir / "viz/selected_m2_by_gamma.png")}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "validate"), default="tune")
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--selected-weight", type=float, default=1.0)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument("--seed0", type=int, default=65100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=STAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")
    if args.cap <= 0.0:
        raise ValueError("--cap must be positive")
    if any(weight < 0.0 for weight in args.weights):
        raise ValueError("retreat weights must be nonnegative")
    if args.selected_weight < 0.0:
        raise ValueError("--selected-weight must be nonnegative")
    if len(set(args.weights)) != len(args.weights):
        raise ValueError("retreat weights must be unique")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.phase == "tune":
        run_tune(args, device)
    else:
        run_validate(args, device)


if __name__ == "__main__":
    main()
