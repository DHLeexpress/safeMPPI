#!/usr/bin/env python3
"""Stage 1B: smooth SafeMPPI expert on a declared-radius OOD scene.

The task is deliberately fixed to start=(0.5,0.5), goal=(4.5,4.5), and one
giant obstacle replacing the four central obstacles.  ``tune`` compares
smoothness weights with matched planner noise.  ``final`` evaluates M=2 for
every gamma with a long closed-loop horizon and saves a--e plus validity audits.
"""
from __future__ import annotations

import argparse
import csv
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
    GIANT_CENTER,
    draw_scene,
    make_scene,
    nearest_surrounding_gap,
    object_array,
    torch_step,
)
from viz_style import GAMMAS, GAMMA_COLORS  # noqa: E402


STAGE = HERE / "stage_results/01b_smooth_expert"
START = np.asarray((0.5, 0.5), dtype=np.float32)
GOAL = np.asarray((4.5, 4.5), dtype=np.float32)
RADIUS = 1.20
DEFAULT_SMOOTH_GRID = (0.12, 0.50, 1.0, 2.0, 4.0, 8.0)


def make_fixed_scene(max_steps: int):
    env = make_scene(RADIUS, START, GOAL)
    env.T = int(max_steps)
    return env


def route_mode(path: np.ndarray) -> tuple[str, float]:
    """Classify the local homotopy by which side of the diagonal skirts the giant."""
    p = np.asarray(path, dtype=float)
    radial = np.linalg.norm(p - GIANT_CENTER[None, :], axis=1)
    near = radial <= RADIUS + 0.45
    if not near.any():
        near[int(np.argmin(radial))] = True
    score = float(np.median(p[near, 1] - p[near, 0]))
    # The giant intersects the start-goal diagonal, so every collision-free
    # completion belongs to exactly one of these two homotopies.  Do not create
    # an artificial third "diagonal" mode for a median close to zero.
    if score >= 0.0:
        return "upper-left", score
    return "lower-right", score


def approach_audit(path: np.ndarray, goal: np.ndarray, reach: float, H: int = 10,
                   stride: int = 2, delta: float = 0.10) -> dict:
    """Goal-specific authoritative net-progress audit with failure telemetry."""
    p = np.asarray(path, dtype=float)
    if len(p) < H + 1:
        return {"ok": False, "windows": 0, "failures": 0, "failure_fraction": math.nan,
                "first_failure_step": None, "worst_net_progress": math.nan}
    distances = np.linalg.norm(p - goal[None, :], axis=1)
    windows = 0
    failures = []
    gains = []
    for index in range(0, len(p) - H, stride):
        window = distances[index:index + H + 1]
        d0 = float(window[0])
        if d0 < reach:
            continue
        windows += 1
        gain = d0 - float(window[-1])
        gains.append(gain)
        if gain < min(delta, 0.5 * d0):
            failures.append(index)
    return {
        "ok": not failures,
        "windows": windows,
        "failures": len(failures),
        "failure_fraction": len(failures) / max(windows, 1),
        "first_failure_step": failures[0] if failures else None,
        "worst_net_progress": min(gains) if gains else math.nan,
    }


def approach_ok(path: np.ndarray, goal: np.ndarray, reach: float, H: int = 10,
                stride: int = 2, delta: float = 0.10) -> bool:
    """Boolean convenience wrapper around :func:`approach_audit`."""
    return bool(approach_audit(path, goal, reach, H=H, stride=stride, delta=delta)["ok"])


def smoothness_metrics(controls: np.ndarray, dt: float) -> dict:
    u = np.asarray(controls, dtype=float)
    if len(u) == 0:
        return {
            "accel_rms": math.nan,
            "control_delta_mean": math.nan,
            "control_delta_rms": math.nan,
            "jerk_rms": math.nan,
            "smooth_cost_analog": math.nan,
        }
    accel_rms = float(np.sqrt(np.mean(np.sum(u * u, axis=1))))
    if len(u) == 1:
        return {
            "accel_rms": accel_rms,
            "control_delta_mean": 0.0,
            "control_delta_rms": 0.0,
            "jerk_rms": 0.0,
            "smooth_cost_analog": 0.0,
        }
    delta = np.diff(u, axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    delta_rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    return {
        "accel_rms": accel_rms,
        "control_delta_mean": float(delta_norm.mean()),
        "control_delta_rms": delta_rms,
        "jerk_rms": delta_rms / float(dt),
        "smooth_cost_analog": float(np.mean(np.sum(delta * delta, axis=1))),
    }


@torch.inference_mode()
def rollout(env, gamma: float, seed: int, smooth_weight: float, reach: float,
            device: torch.device, certify: bool) -> dict:
    config = GS.mode1_config()
    config["smooth_weight"] = float(smooth_weight)
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
        action, _info = adapter.plan(
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
    endpoint_distance = float(np.linalg.norm(path[-1] - env.goal.detach().cpu().numpy()))
    reached = endpoint_distance < reach
    collision = bool(clearances.min() < 0.0)
    in_taskspace = bool(((path >= 0.0) & (path <= 5.0)).all())
    success = bool(reached and not collision and in_taskspace)
    if dead_reason is None and not reached:
        dead_reason = "timeout"
    mode, side_score = route_mode(path)
    smooth = smoothness_metrics(controls, float(env.dt))
    result = {
        "gamma": float(gamma),
        "seed": int(seed),
        "smooth_weight": float(smooth_weight),
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
        "per_obstacle_min_mean": float(clearances.min(axis=0).mean()),
        "giant_min_clearance": float((np.linalg.norm(path - GIANT_CENTER[None, :], axis=1) - RADIUS).min()),
        "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "route_mode": mode,
        "side_score": side_score,
        **smooth,
        "path": path,
        "states": states,
        "controls": controls,
    }
    if certify:
        progress_audit = approach_audit(path, env.goal.detach().cpu().numpy(), reach)
        progress = bool(progress_audit["ok"])
        socp = bool(GM.socp_ok(path, env, gamma))
        result.update({
            "progress_ok": bool(progress),
            "progress_windows": progress_audit["windows"],
            "progress_failures": progress_audit["failures"],
            "progress_failure_fraction": progress_audit["failure_fraction"],
            "progress_first_failure_step": progress_audit["first_failure_step"],
            "progress_worst_net_progress": progress_audit["worst_net_progress"],
            "socp_ok": socp,
            "valid2": bool(success and progress and socp),
        })
    return result


def finite_stats(values) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return math.nan, math.nan
    return float(x.mean()), float(x.std(ddof=0))


def summarize_gamma(results: list[dict], gamma: float) -> dict:
    successes = [result for result in results if result["success"]]
    modes = sorted({result["route_mode"] for result in successes
                    if result["route_mode"] in ("upper-left", "lower-right")})
    row = {
        "gamma": float(gamma),
        "M": len(results),
        "a_SR": float(np.mean([result["success"] for result in results])),
        "b_CR": float(np.mean([result["collision"] for result in results])),
        "n_success": len(successes),
        "e_coverage": len(modes),
        "e_coverage_fraction": len(modes) / 2.0,
        "coverage_modes": modes,
        "timeouts": int(sum(result["dead_reason"] == "timeout" for result in results)),
        "out_of_bounds": int(sum(not result["in_taskspace"] for result in results)),
    }
    fields = (
        ("c_clearance", "clearance_mean"),
        ("min_clearance", "min_clearance"),
        ("per_obstacle_min", "per_obstacle_min_mean"),
        ("giant_min_clearance", "giant_min_clearance"),
        ("d_time_s", "time_s"),
        ("path_length", "path_length"),
        ("control_delta", "control_delta_mean"),
        ("jerk_rms", "jerk_rms"),
        ("accel_rms", "accel_rms"),
    )
    for prefix, key in fields:
        mean, std = finite_stats([result[key] for result in successes])
        row[f"{prefix}_mean"] = mean
        row[f"{prefix}_std"] = std
    if results and "valid2" in results[0]:
        row["valid2_rate"] = float(np.mean([result["valid2"] for result in results]))
        row["progress_pass_rate"] = float(np.mean([result["progress_ok"] for result in results]))
        row["progress_window_failure_fraction_mean"] = float(np.mean(
            [result["progress_failure_fraction"] for result in results]))
        row["socp_pass_rate"] = float(np.mean([result["socp_ok"] for result in results]))
    return row


def serial_result(result: dict) -> dict:
    return {key: value for key, value in result.items()
            if key not in ("path", "states", "controls")}


def matched_color_scatter(axis, xs, ys, **kwargs) -> None:
    for gamma, x, y in zip(GAMMAS, xs, ys):
        axis.scatter(x, y, color=GAMMA_COLORS[gamma], edgecolor="black", linewidth=0.35,
                     zorder=4, **kwargs)


def render_tuning(records: list[dict], output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    weights = sorted({result["smooth_weight"] for result in records})
    rows = []
    for weight in weights:
        group = [result for result in records if result["smooth_weight"] == weight]
        successes = [result for result in group if result["success"]]
        by_gamma = {result["gamma"]: result for result in group}
        rows.append({
            "weight": weight,
            "successes": sum(result["success"] for result in group),
            "collisions": sum(result["collision"] for result in group),
            "delta": finite_stats([result["control_delta_mean"] for result in successes])[0],
            "time": finite_stats([result["time_s"] for result in successes])[0],
            "clear_gap": by_gamma[0.1]["clearance_mean"] - by_gamma[1.0]["clearance_mean"],
            "time_gap": by_gamma[0.1]["time_s"] - by_gamma[1.0]["time_s"],
        })
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    x = np.asarray(weights)
    axes[0, 0].plot(x, [row["successes"] for row in rows], "o-", label="success / 7")
    axes[0, 0].plot(x, [row["collisions"] for row in rows], "s--", label="collisions")
    axes[0, 0].axhline(7, color="0.6", lw=0.8)
    axes[0, 0].set_ylabel("rollouts")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(x, [row["delta"] for row in rows], "o-", color="#0072b2")
    axes[0, 1].set_ylabel(r"mean executed $\|u_t-u_{t-1}\|_2$")
    axes[1, 0].plot(x, [row["time"] for row in rows], "o-", color="#d55e00")
    axes[1, 0].set_ylabel("mean time to goal [s]")
    axes[1, 1].plot(x, [row["clear_gap"] for row in rows], "o-", label=r"clearance: $\gamma=.1-\gamma=1$")
    axes[1, 1].plot(x, [row["time_gap"] for row in rows], "s--", label=r"time: $\gamma=.1-\gamma=1$")
    axes[1, 1].axhline(0, color="0.5", lw=0.8)
    axes[1, 1].set_ylabel("low-minus-high gamma gap")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.set_xlabel("SafeMPPI smooth weight")
        axis.grid(alpha=0.25)
    fig.suptitle("Stage 1B smoothness pilot — one matched-seed rollout per gamma", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_by_gamma(env, results: list[dict], rows: list[dict], reach: float, smooth_weight: float,
                    output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.8))
    for axis, gamma, row in zip(axes.ravel()[:7], GAMMAS, rows):
        draw_scene(axis, env, START, GOAL, reach, RADIUS)
        group = [result for result in results if result["gamma"] == gamma]
        for replicate, result in enumerate(group):
            path = result["path"]
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma],
                      lw=2.0, ls="-" if replicate == 0 else "--", alpha=0.94, zorder=4)
            if not result["success"]:
                axis.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=7, mew=1.5, zorder=7)
        axis.set_title(
            rf"$\gamma={gamma:g}$  SR={row['a_SR']:.0%}, CR={row['b_CR']:.0%}" "\n"
            rf"clear={row['c_clearance_mean']:.3f} m, time={row['d_time_s_mean']:.1f} s, cov={row['e_coverage']}/2",
            fontsize=9,
        )
    info = axes.ravel()[7]
    info.axis("off")
    gap = nearest_surrounding_gap(env, RADIUS)[0]
    info.text(
        0.03,
        0.95,
        "Locked Stage 1B\n\n"
        f"start = ({START[0]:.1f}, {START[1]:.1f})\n"
        f"goal = ({GOAL[0]:.1f}, {GOAL[1]:.1f})\n"
        f"giant radius = {RADIUS:.2f} m\n"
        f"surface gap = {gap:.3f} m\n"
        f"smooth weight = {smooth_weight:g}\n"
        f"max time = {env.T * env.dt:.0f} s\n"
        f"goal reach = {reach:.2f} m\n\n"
        "Solid/dashed = matched replicate 1/2\n"
        "Color = safety level gamma",
        va="top",
        fontsize=11,
    )
    fig.suptitle("Smooth SafeMPPI expert — M=2 trajectories per safety level", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_overlay(env, results: list[dict], reach: float, smooth_weight: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, axis = plt.subplots(figsize=(8.4, 7.4))
    draw_scene(axis, env, START, GOAL, reach, RADIUS)
    for result in results:
        path = result["path"]
        replicate = result["seed"] - min(r["seed"] for r in results)
        axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[result["gamma"]],
                  lw=1.9, ls="-" if replicate == 0 else "--", alpha=0.90, zorder=4)
    handles = [mpl.lines.Line2D([], [], color=GAMMA_COLORS[gamma], lw=2, label=rf"$\gamma={gamma:g}$")
               for gamma in GAMMAS]
    axis.legend(handles=handles, loc="upper left", ncol=2, frameon=True, fontsize=9)
    axis.set_title(
        f"Stage 1B overlay: M=2 per gamma, smooth weight={smooth_weight:g}, max time={env.T * env.dt:.0f} s",
        fontsize=12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_ae(rows: list[dict], smooth_weight: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.5))
    gamma = np.asarray([row["gamma"] for row in rows])
    panels = (
        ("a_SR", None, "(a) success rate", (0, 1.05)),
        ("b_CR", None, "(b) collision rate", (0, 1.05)),
        ("c_clearance_mean", "c_clearance_std", "(c) mean nearest-obstacle clearance [m]", None),
        ("d_time_s_mean", "d_time_s_std", "(d) time to goal [s]", None),
        ("e_coverage", None, "(e) local homotopy coverage [/2]", (-0.05, 2.15)),
        ("control_delta_mean", "control_delta_std", r"smoothness audit: mean $\|u_t-u_{t-1}\|_2$", None),
    )
    for axis, (key, err_key, title, ylim) in zip(axes.ravel(), panels):
        values = np.asarray([row[key] for row in rows], dtype=float)
        axis.plot(gamma, values, color="0.45", lw=1.2, zorder=2)
        if err_key:
            error = np.asarray([row[err_key] for row in rows], dtype=float)
            axis.errorbar(gamma, values, yerr=error, fmt="none", ecolor="0.45", capsize=3, zorder=2)
        matched_color_scatter(axis, gamma, values, s=55)
        axis.set_title(title)
        axis.set_xlabel(r"safety level $\gamma$")
        axis.set_xticks(gamma)
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
    fig.suptitle(f"SafeMPPI expert a–e, M=2 per gamma — smooth weight={smooth_weight:g}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_tune(args: argparse.Namespace, device: torch.device) -> None:
    env = make_fixed_scene(args.max_steps)
    records = []
    started = time.perf_counter()
    for weight in args.smooth_grid:
        for gamma in GAMMAS:
            result = rollout(env, gamma, args.seed0, weight, args.reach, device, certify=False)
            records.append(result)
            print(
                f"[tune w={weight:g}] gamma={gamma:g} success={int(result['success'])} "
                f"steps={result['steps']} clear={result['clearance_mean']:.3f} "
                f"du={result['control_delta_mean']:.4f} reason={result['dead_reason']}",
                flush=True,
            )
    rows = []
    for weight in args.smooth_grid:
        group = [result for result in records if result["smooth_weight"] == weight]
        successes = [result for result in group if result["success"]]
        by_gamma = {result["gamma"]: result for result in group}
        rows.append({
            "smooth_weight": float(weight),
            "successes": int(sum(result["success"] for result in group)),
            "collisions": int(sum(result["collision"] for result in group)),
            "timeouts": int(sum(result["dead_reason"] == "timeout" for result in group)),
            "mean_control_delta": finite_stats([result["control_delta_mean"] for result in successes])[0],
            "mean_jerk_rms": finite_stats([result["jerk_rms"] for result in successes])[0],
            "mean_time_s": finite_stats([result["time_s"] for result in successes])[0],
            "low_minus_high_clearance": by_gamma[0.1]["clearance_mean"] - by_gamma[1.0]["clearance_mean"],
            "low_minus_high_time_s": by_gamma[0.1]["time_s"] - by_gamma[1.0]["time_s"],
        })
    log_dir = args.outdir / "logs"
    data_dir = args.outdir / "data"
    viz_dir = args.outdir / "viz"
    for directory in (log_dir, data_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "TUNING_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "task": {"start": START.tolist(), "goal": GOAL.tolist(), "radius": RADIUS,
                 "reach": args.reach, "max_steps": args.max_steps, "max_time_s": args.max_steps * env.dt},
        "seed_matching": "same planner seed across gamma and smoothness weights",
        "rows": rows,
        "rollouts": [serial_result(result) for result in records],
    }
    (log_dir / "smoothness_tuning.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        data_dir / "smoothness_tuning_paths.npz",
        weights=np.asarray([result["smooth_weight"] for result in records]),
        gammas=np.asarray([result["gamma"] for result in records]),
        paths=object_array([result["path"] for result in records]),
        controls=object_array([result["controls"] for result in records]),
    )
    render_tuning(records, viz_dir / "smoothness_tuning.png")
    print(json.dumps({"status": summary["status"], "rows": rows,
                      "output": str(viz_dir / "smoothness_tuning.png")}, indent=2), flush=True)


def run_final(args: argparse.Namespace, device: torch.device) -> None:
    if args.smooth_weight is None:
        raise ValueError("--smooth-weight is required for --phase final")
    env = make_fixed_scene(args.max_steps)
    records = []
    started = time.perf_counter()
    for replicate in range(args.M):
        seed = args.seed0 + replicate
        for gamma in GAMMAS:
            result = rollout(env, gamma, seed, args.smooth_weight, args.reach, device, certify=True)
            result["replicate"] = replicate
            records.append(result)
            print(
                f"[final m={replicate + 1}/{args.M}] gamma={gamma:g} success={int(result['success'])} "
                f"steps={result['steps']} clear={result['clearance_mean']:.3f} "
                f"du={result['control_delta_mean']:.4f} mode={result['route_mode']} "
                f"valid2={int(result['valid2'])}",
                flush=True,
            )
    rows = [summarize_gamma([result for result in records if result["gamma"] == gamma], gamma)
            for gamma in GAMMAS]
    clearances = np.asarray([row["c_clearance_mean"] for row in rows], dtype=float)
    minimum_clearances = np.asarray([row["min_clearance_mean"] for row in rows], dtype=float)
    times = np.asarray([row["d_time_s_mean"] for row in rows], dtype=float)
    gamma_values = np.asarray(GAMMAS, dtype=float)
    trend = {
        "low_gamma_has_max_clearance": bool(int(np.nanargmax(clearances)) == 0),
        "minimum_clearance_strictly_decreases_with_gamma": bool(np.all(np.diff(minimum_clearances) < 0.0)),
        "low_gamma_is_slowest": bool(int(np.nanargmax(times)) == 0),
        "fastest_gamma_is_medium_or_high": bool(int(np.nanargmin(times)) >= 2),
        "gamma_clearance_pearson": float(np.corrcoef(gamma_values, clearances)[0, 1]),
        "gamma_time_pearson": float(np.corrcoef(gamma_values, times)[0, 1]),
        "all_gamma_SR_100": bool(all(row["a_SR"] == 1.0 for row in rows)),
        "all_gamma_CR_0": bool(all(row["b_CR"] == 0.0 for row in rows)),
        "all_gamma_SOCP": bool(all(row["socp_pass_rate"] == 1.0 for row in rows)),
        "all_gamma_valid2": bool(all(row["valid2_rate"] == 1.0 for row in rows)),
        "full_two_mode_coverage_gammas": [row["gamma"] for row in rows if row["e_coverage"] == 2],
    }
    log_dir = args.outdir / "logs"
    data_dir = args.outdir / "data"
    viz_dir = args.outdir / "viz"
    table_dir = args.outdir / "tables"
    for directory in (log_dir, data_dir, viz_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "FINAL_M2_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "task": {
            "start": START.tolist(),
            "goal": GOAL.tolist(),
            "radius": RADIUS,
            "nearest_surface_gap": nearest_surrounding_gap(env, RADIUS)[0],
            "reach": args.reach,
            "max_steps": args.max_steps,
            "max_time_s": args.max_steps * env.dt,
            "M_per_gamma": args.M,
        },
        "config_change": {"smooth_weight": float(args.smooth_weight),
                          "all_other_mode1_config_values_unchanged": True},
        "seed_matching": "replicate seeds are matched across all gamma values",
        "a_e_definition": {
            "a": "physical success rate: reached final goal disk, collision-free, in [0,5]^2",
            "b": "collision rate",
            "c": "episode mean of nearest-obstacle surface clearance, successful paths only",
            "d": "time to goal, successful paths only",
            "e": "number of local giant-obstacle homotopies covered (upper-left/lower-right), max 2",
        },
        "rows": rows,
        "gamma_intuition": trend,
        "rollouts": [serial_result(result) for result in records],
    }
    (log_dir / "expert_m2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(rows, table_dir / "expert_ae_m2.csv")
    np.savez_compressed(
        data_dir / "expert_m2_paths.npz",
        gammas=np.asarray([result["gamma"] for result in records]),
        seeds=np.asarray([result["seed"] for result in records]),
        success=np.asarray([result["success"] for result in records]),
        paths=object_array([result["path"] for result in records]),
        states=object_array([result["states"] for result in records]),
        controls=object_array([result["controls"] for result in records]),
        start=START,
        goal=GOAL,
        radius=np.asarray(RADIUS),
        smooth_weight=np.asarray(args.smooth_weight),
    )
    render_by_gamma(env, records, rows, args.reach, args.smooth_weight,
                    viz_dir / "expert_m2_by_gamma.png")
    render_overlay(env, records, args.reach, args.smooth_weight,
                   viz_dir / "expert_m2_overlay.png")
    render_ae(rows, args.smooth_weight, viz_dir / "expert_ae_m2.png")
    print(json.dumps({"status": summary["status"], "rows": rows, "gamma_intuition": trend,
                      "output": str(viz_dir / "expert_m2_by_gamma.png")}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "final"), required=True)
    parser.add_argument("--smooth-grid", type=float, nargs="+", default=list(DEFAULT_SMOOTH_GRID))
    parser.add_argument("--smooth-weight", type=float)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS,
        help="giant-obstacle radius; all other scene geometry remains fixed",
    )
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument("--seed0", type=int, default=65100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=STAGE)
    return parser.parse_args()


def main() -> None:
    global RADIUS
    args = parse_args()
    if not 0.0 < args.radius < 2.0:
        raise ValueError("--radius must lie in (0, 2)")
    # The existing implementation intentionally routes every geometry use
    # through this module constant.  Set it once, before constructing an env,
    # so a radius sweep changes geometry only and remains reproducible.
    RADIUS = float(args.radius)
    if args.M != 2 and args.phase == "final":
        raise ValueError("Stage 1B approval figure is locked to M=2 per gamma")
    if args.max_steps < 250:
        raise ValueError("Stage 1B requires a long horizon of at least 250 controls")
    if any(weight <= 0.0 for weight in args.smooth_grid):
        raise ValueError("smoothness weights must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.phase == "tune":
        run_tune(args, device)
    else:
        run_final(args, device)


if __name__ == "__main__":
    main()
