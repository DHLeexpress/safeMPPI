#!/usr/bin/env python3
"""Stage 1C: window-level valid2 audit and moving nominal/verifier GIF.

This script reuses the exact successful Stage 1B M=2 trajectories.  ``metrics``
scores every H=10 training-style sample (including terminal-padded samples) and
also reports the executed-only full-window subset.  ``gif`` synchronizes the
first matched replicate across all seven gamma values and draws the moving
nominal polytope in blue and the fitted verifier polytope in green.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _paths  # noqa: F401,E402
import grid_metrics as GM  # noqa: E402
import grid_metrics2 as GM2  # noqa: E402
import grid_rollout as GR  # noqa: E402
import grid_scene as GS  # noqa: E402
import verifier_polytope as VP  # noqa: E402
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2  # noqa: E402

from giant_obstacle_ood.stage1_geometry_sweep import draw_scene, make_scene  # noqa: E402
from giant_obstacle_ood.stage1b_smooth_expert import (  # noqa: E402
    GOAL,
    RADIUS,
    START,
)
from viz_style import GAMMAS, GAMMA_COLORS  # noqa: E402


SOURCE = HERE / "stage_results/01b_smooth_expert/data/expert_m2_paths.npz"
STAGE = HERE / "stage_results/01c_window_validity"
H = 10
BLUE = "#0072B2"
GREEN = "#009E73"
RED = "#CC3311"


def make_diagnostic_scene(max_steps: int):
    """Build the exact declared-radius scene used by the source trajectories."""

    env = make_scene(float(RADIUS), START, GOAL)
    env.T = int(max_steps)
    return env


def halfspace_polygon(A: np.ndarray, b: np.ndarray, tol: float = 1e-7) -> np.ndarray | None:
    """Bounded 2-D polygon for A x <= b, computed from feasible face intersections."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    vertices = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            matrix = np.stack((A[i], A[j]))
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) < 1e-10:
                continue
            point = np.linalg.solve(matrix, np.asarray((b[i], b[j])))
            if np.all(A @ point <= b + tol):
                vertices.append(point)
    if len(vertices) < 3:
        return None
    vertices = np.unique(np.round(np.asarray(vertices), decimals=9), axis=0)
    if len(vertices) < 3:
        return None
    center = vertices.mean(axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def load_rollouts(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=True) as data:
        records = []
        for index in range(len(data["paths"])):
            records.append({
                "gamma": float(data["gammas"][index]),
                "seed": int(data["seeds"][index]),
                "path": np.asarray(data["paths"][index], dtype=np.float32),
                "states": np.asarray(data["states"][index], dtype=np.float32),
                "controls": np.asarray(data["controls"][index], dtype=np.float32),
                "success": bool(data["success"][index]),
            })
    if len(records) != 14 or not all(record["success"] for record in records):
        raise RuntimeError("Stage 1C expects the validated 14 successful Stage 1B trajectories")
    return records


def nominal_polytope(state: np.ndarray, env):
    config = GS.mode1_config()
    obstacles = GS.planner_obstacles(env).detach().cpu().numpy()
    velocities = np.zeros((len(obstacles), 2), dtype=np.float32)
    return build_polytope_v2(
        state[:2],
        obstacles,
        sensing_range=float(config["barrier_activation_radius"]),
        n_base=int(config.get("polytope_nbase", 16)),
        margin=0.0,
        max_obstacles=12,
        obstacle_velocities=velocities,
        robot_velocity=state[2:4],
        predict_gain=float(config.get("predict_gain", 0.0)),
        predict_tau=float(config["horizon"]) * float(config["dt"]),
    )


def nominal_certificate(poly, window: np.ndarray, gamma: float) -> tuple[bool, float]:
    A = poly.A.detach().cpu().numpy().astype(float)
    b = poly.b.detach().cpu().numpy().astype(float)
    center = poly.ref.detach().cpu().numpy().astype(float)
    margins = b - A @ center
    if np.any(margins <= 0.0):
        return False, -math.inf
    values = (b[None, :] - np.asarray(window, float) @ A.T) / margins[None, :]
    barrier = values.min(axis=1)
    alpha = (1.0 - float(gamma)) ** np.arange(len(window), dtype=float)
    residual = barrier[1:] - alpha[1:]
    return bool(np.all(residual >= -1e-8)), float(residual.min())


def training_window(record: dict, index: int, env) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return state, H controls, and whether terminal padding was needed."""
    controls = record["controls"]
    state = record["states"][index]
    window_u = controls[index:index + H]
    padded = len(window_u) < H
    if padded:
        window_u = np.concatenate((window_u, np.repeat(window_u[-1:], H - len(window_u), axis=0)), axis=0)
    positions = GR.window_positions(state, window_u, float(env.dt))
    return state, np.asarray(window_u, dtype=np.float32), np.vstack((state[:2], positions))


def score_window(state: np.ndarray, window: np.ndarray, env, gamma: float,
                 padded: bool, trajectory_index: int, step: int) -> dict:
    taskspace = bool(GM.in_taskspace(window))
    distances = np.linalg.norm(window - GOAL[None, :], axis=1)
    progress = bool(GM2.approach_ok(distances))
    obstacles = env.obstacles.detach().cpu().numpy()
    socp, faces, _raw, _reff = VP.certify_window(
        window,
        obstacles,
        float(env.r_robot),
        float(gamma),
        R=2.5,
        n_theta=180,
    )
    clearance = (np.linalg.norm(window[:, None, :] - obstacles[None, :, :2], axis=2)
                 - obstacles[None, :, 2] - float(env.r_robot))
    physical = bool(clearance.min() >= 0.0)
    poly, info = nominal_polytope(state, env)
    A = poly.A.detach().cpu().numpy()
    b = poly.b.detach().cpu().numpy()
    nominal_exists = bool(info["contains_robot"] and halfspace_polygon(A, b) is not None)
    nominal_cert, nominal_residual = nominal_certificate(poly, window, gamma)
    return {
        "trajectory_index": int(trajectory_index),
        "seed": int(0),  # overwritten by caller
        "gamma": float(gamma),
        "step": int(step),
        "padded": bool(padded),
        "taskspace": taskspace,
        "progress": progress,
        "socp": bool(socp),
        "joint_valid2": bool(taskspace and progress and socp),
        "physical_collision_free": physical,
        "min_clearance": float(clearance.min()),
        "nominal_exists": nominal_exists,
        "nominal_certificate": nominal_cert,
        "nominal_residual": nominal_residual,
        "verifier_all_faces_feasible": bool(all(face.feasible and face.m > 0.0 for face in faces)),
    }


def rates(group: list[dict]) -> dict:
    n = len(group)
    keys = (
        "taskspace",
        "progress",
        "socp",
        "joint_valid2",
        "physical_collision_free",
        "nominal_exists",
        "nominal_certificate",
    )
    result = {"n": n}
    for key in keys:
        count = int(sum(record[key] for record in group))
        result[f"{key}_count"] = count
        result[f"{key}_rate"] = count / max(n, 1)
    result["nominal_implies_verifier_violations"] = int(sum(
        record["nominal_certificate"] and not record["socp"] for record in group))
    return result


def run_metrics(args: argparse.Namespace) -> None:
    env = make_diagnostic_scene(800)
    rollouts = load_rollouts(args.source)
    records = []
    started = time.perf_counter()
    for trajectory_index, rollout in enumerate(rollouts):
        controls = rollout["controls"]
        for step in range(len(controls)):
            state, _window_u, window = training_window(rollout, step, env)
            padded = step + H > len(controls)
            row = score_window(state, window, env, rollout["gamma"], padded,
                               trajectory_index, step)
            row["seed"] = rollout["seed"]
            records.append(row)
        print(
            f"[windows] gamma={rollout['gamma']:g} seed={rollout['seed']} "
            f"samples={len(controls)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    summaries = []
    for gamma in GAMMAS:
        group = [record for record in records if record["gamma"] == gamma]
        full = [record for record in group if not record["padded"]]
        padded = [record for record in group if record["padded"]]
        summaries.append({
            "gamma": float(gamma),
            "all_training_samples": rates(group),
            "executed_full_windows": rates(full),
            "terminal_padded_windows": rates(padded),
        })
    overall = {
        "all_training_samples": rates(records),
        "executed_full_windows": rates([record for record in records if not record["padded"]]),
        "terminal_padded_windows": rates([record for record in records if record["padded"]]),
    }
    summary = {
        "status": "WINDOW_AUDIT_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "source": str(args.source.resolve()),
        "task": {"start": START.tolist(), "goal": GOAL.tolist(), "radius": RADIUS,
                 "H": H, "M_per_gamma": 2},
        "sample_semantics": (
            "one H=10 control window per successful-trajectory control step; the final nine samples "
            "repeat the last available control exactly as stage2_grid_data.windows_from"
        ),
        "valid2": {
            "taskspace": "all H+1 positions inside [0,5]^2 with authoritative 0.12 tolerance",
            "progress": "goal-specific net progress >=0.10 m over H=10; starts within 0.45 m auto-pass",
            "socp": "verifier_polytope.certify_window, R=2.5, n_theta=180",
            "joint": "taskspace AND progress AND socp",
        },
        "per_gamma": summaries,
        "overall": overall,
    }
    log_dir = args.outdir / "logs"
    table_dir = args.outdir / "tables"
    data_dir = args.outdir / "data"
    viz_dir = args.outdir / "viz"
    for directory in (log_dir, table_dir, data_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (log_dir / "window_validity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (data_dir / "window_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(
        data_dir / "window_masks.npz",
        gamma=np.asarray([record["gamma"] for record in records]),
        seed=np.asarray([record["seed"] for record in records]),
        step=np.asarray([record["step"] for record in records]),
        padded=np.asarray([record["padded"] for record in records]),
        taskspace=np.asarray([record["taskspace"] for record in records]),
        progress=np.asarray([record["progress"] for record in records]),
        socp=np.asarray([record["socp"] for record in records]),
        joint_valid2=np.asarray([record["joint_valid2"] for record in records]),
        nominal_exists=np.asarray([record["nominal_exists"] for record in records]),
        nominal_certificate=np.asarray([record["nominal_certificate"] for record in records]),
    )
    render_window_metrics(summary, viz_dir / "window_validity_by_gamma.png")
    write_window_table(summary, table_dir / "window_validity.md")
    print(json.dumps({
        "status": summary["status"],
        "overall": overall,
        "per_gamma_joint_all": {
            str(row["gamma"]): row["all_training_samples"]["joint_valid2_rate"] for row in summaries
        },
        "output": str(viz_dir / "window_validity_by_gamma.png"),
    }, indent=2), flush=True)


def write_window_table(summary: dict, output: Path) -> None:
    lines = [
        "# Stage 1C window-level valid2",
        "",
        "All rates use H=10 training-style samples from the two physically successful trajectories per gamma.",
        "",
        "| gamma | samples | task | progress | SOCP | joint valid2 | nominal exists | nominal cert | N-cert => !V |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_gamma"]:
        stats = row["all_training_samples"]
        lines.append(
            f"| {row['gamma']:.1f} | {stats['n']} | {stats['taskspace_rate']:.1%} | "
            f"{stats['progress_rate']:.1%} | {stats['socp_rate']:.1%} | "
            f"{stats['joint_valid2_rate']:.1%} | {stats['nominal_exists_rate']:.1%} | "
            f"{stats['nominal_certificate_rate']:.1%} | "
            f"{stats['nominal_implies_verifier_violations']} |"
        )
    output.write_text("\n".join(lines) + "\n")


def render_window_metrics(summary: dict, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    rows = summary["per_gamma"]
    gamma = np.asarray([row["gamma"] for row in rows])
    colors = [GAMMA_COLORS[float(value)] for value in gamma]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))

    criteria = ("taskspace", "progress", "socp", "joint_valid2")
    labels = ("task space", "progress", "SOCP", "joint valid2")
    offsets = np.linspace(-0.024, 0.024, len(criteria))
    width = 0.015
    for key, label, offset in zip(criteria, labels, offsets):
        values = [row["all_training_samples"][f"{key}_rate"] for row in rows]
        axes[0, 0].bar(gamma + offset, values, width=width, label=label)
    axes[0, 0].set(title="(A) all training-style H=10 samples", ylabel="pass rate", ylim=(0, 1.05))
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)

    for key, label, marker in (("progress", "progress", "o"), ("socp", "SOCP", "s"),
                               ("joint_valid2", "joint valid2", "*")):
        full = [row["executed_full_windows"][f"{key}_rate"] for row in rows]
        padded = [row["terminal_padded_windows"][f"{key}_rate"] for row in rows]
        axes[0, 1].plot(gamma, full, marker=marker, label=f"full: {label}")
        axes[0, 1].plot(gamma, padded, marker=marker, ls="--", alpha=0.65, label=f"padded: {label}")
    axes[0, 1].set(title="(B) executed-full versus terminal-padded", ylabel="pass rate", ylim=(-0.03, 1.05))
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=7)

    for key, label, marker in (("nominal_exists", "nominal exists", "o"),
                               ("nominal_certificate", "executed window passes nominal schedule", "s"),
                               ("socp", "verifier SOCP", "^")):
        values = [row["all_training_samples"][f"{key}_rate"] for row in rows]
        axes[1, 0].plot(gamma, values, marker=marker, label=label)
    axes[1, 0].set(title="(C) existence is distinct from trajectory certification", ylabel="rate", ylim=(-0.03, 1.05))
    axes[1, 0].legend(frameon=False, fontsize=8)

    matrix = np.asarray([
        [row["all_training_samples"][f"{key}_rate"] for row in rows]
        for key in criteria
    ])
    image = axes[1, 1].imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].set_xticks(np.arange(len(gamma)), [f"{value:g}" for value in gamma])
    axes[1, 1].set_yticks(np.arange(len(labels)), labels)
    axes[1, 1].set_xlabel(r"safety level $\gamma$")
    axes[1, 1].set_title("(D) window-pass heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1, 1].text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                            color="white" if matrix[i, j] < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for axis in axes.ravel()[:3]:
        axis.set_xlabel(r"safety level $\gamma$")
        axis.set_xticks(gamma)
        axis.grid(alpha=0.25)
    fig.suptitle("Stage 1C — valid2 at the H=10 sample level inside successful expert trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def verifier_polygons(window: np.ndarray, env, gamma: float):
    obstacles = env.obstacles.detach().cpu().numpy()
    ok, faces, _raw, _reff = VP.certify_window(
        window, obstacles, float(env.r_robot), float(gamma), R=2.5, n_theta=180)
    if not all(face.feasible and face.m > 0.0 for face in faces):
        return bool(ok), None, None
    center = np.asarray(window[0], dtype=float)
    A = np.stack([face.a for face in faces])
    margins = np.asarray([face.m for face in faces])
    outer = halfspace_polygon(A, margins + A @ center)
    alpha_h = (1.0 - float(gamma)) ** (len(window) - 1)
    level = halfspace_polygon(A, (1.0 - alpha_h) * margins + A @ center)
    return bool(ok), outer, level


def nominal_polygons(state: np.ndarray, env, gamma: float):
    poly, info = nominal_polytope(state, env)
    A = poly.A.detach().cpu().numpy().astype(float)
    b = poly.b.detach().cpu().numpy().astype(float)
    center = poly.ref.detach().cpu().numpy().astype(float)
    margins = b - A @ center
    outer = halfspace_polygon(A, b)
    alpha_h = (1.0 - float(gamma)) ** H
    level = halfspace_polygon(A, b - alpha_h * margins)
    return bool(info["contains_robot"] and outer is not None), outer, level, poly


def add_polygon(axis, vertices, color: str, fill_alpha: float, linestyle: str = "-",
                linewidth: float = 1.5, zorder: int = 3) -> None:
    if vertices is None:
        return
    axis.add_patch(Polygon(vertices, closed=True, facecolor=color, alpha=fill_alpha,
                           edgecolor="none", zorder=zorder))
    axis.plot(np.r_[vertices[:, 0], vertices[0, 0]], np.r_[vertices[:, 1], vertices[0, 1]],
              color=color, ls=linestyle, lw=linewidth, zorder=zorder + 0.1)


def frame_panel(axis, record: dict, env, step: int, reach: float) -> dict:
    gamma = record["gamma"]
    states = record["states"]
    controls = record["controls"]
    current = min(step, len(states) - 1)
    state = states[current]
    draw_scene(axis, env, START, GOAL, reach, RADIUS)
    axis.plot(states[:current + 1, 0], states[:current + 1, 1],
              color=GAMMA_COLORS[gamma], lw=1.7, alpha=0.95, zorder=5)
    axis.plot(state[0], state[1], "o", color="black", ms=4.5, zorder=8)

    nominal_exists, nominal_outer, nominal_level, poly = nominal_polygons(state, env, gamma)
    # Keep the requested nominal geometry visibly on top when the fitted
    # verifier happens to share nearly the same faces.
    add_polygon(axis, nominal_outer, BLUE, 0.05, "-", 2.0, 6)
    add_polygon(axis, nominal_level, BLUE, 0.0, "--", 1.5, 6)

    arrived = current >= len(controls)
    if arrived:
        axis.set_title(rf"$\gamma={gamma:g}$  arrived at {len(controls) * env.dt:.1f} s", fontsize=9)
        return {"nominal_exists": nominal_exists, "nominal_cert": None,
                "verifier": None, "joint": None, "arrived": True}

    horizon = min(H, len(controls) - current)
    window = states[current:current + horizon + 1, :2]
    axis.plot(window[:, 0], window[:, 1], color="black", lw=1.0, marker=".", ms=2.5,
              alpha=0.75, zorder=7)
    nominal_cert, _residual = nominal_certificate(poly, window, gamma)
    verifier_ok, verifier_outer, verifier_level = verifier_polygons(window, env, gamma)
    if verifier_ok:
        add_polygon(axis, verifier_outer, GREEN, 0.10, "-", 1.4, 4)
        add_polygon(axis, verifier_level, GREEN, 0.0, "--", 1.1, 4)
    else:
        axis.text(0.03, 0.05, "verifier infeasible", transform=axis.transAxes,
                  color=RED, fontsize=7.5, weight="bold",
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": RED, "pad": 1.5}, zorder=10)
    taskspace = bool(GM.in_taskspace(window))
    progress = bool(GM2.approach_ok(np.linalg.norm(window - GOAL[None, :], axis=1)))
    joint = bool(taskspace and progress and verifier_ok)
    status = f"N-exists {int(nominal_exists)}  N-cert {int(nominal_cert)}  V {int(verifier_ok)}  valid2 {int(joint)}"
    axis.set_title(rf"$\gamma={gamma:g}$  step {current}/{len(controls)}" + "\n" + status, fontsize=8.2)
    return {"nominal_exists": nominal_exists, "nominal_cert": nominal_cert,
            "verifier": verifier_ok, "joint": joint, "arrived": False}


def run_gif(args: argparse.Namespace) -> None:
    env = make_diagnostic_scene(800)
    all_rollouts = load_rollouts(args.source)
    seed = min(record["seed"] for record in all_rollouts) if args.seed is None else args.seed
    rollouts = [record for record in all_rollouts if record["seed"] == seed]
    rollouts.sort(key=lambda record: GAMMAS.index(record["gamma"]))
    if len(rollouts) != len(GAMMAS):
        raise RuntimeError(f"expected one rollout per gamma for seed {seed}, found {len(rollouts)}")
    max_step = max(len(record["controls"]) for record in rollouts)
    frame_steps = list(range(0, max_step + 1, args.frame_stride))
    if frame_steps[-1] != max_step:
        frame_steps.append(max_step)

    viz_dir = args.outdir / "viz"
    log_dir = args.outdir / "logs"
    viz_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output = viz_dir / "nominal_blue_verifier_green_all_gamma.gif"
    poster_step = min(frame_steps, key=lambda value: abs(value - args.poster_step))
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="stage1c_frames_", dir=viz_dir) as temp_name:
        temp = Path(temp_name)
        frame_paths = []
        frame_status = []
        for frame_index, step in enumerate(frame_steps):
            mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8})
            fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.1))
            statuses = []
            for axis, record in zip(axes.ravel()[:7], rollouts):
                statuses.append(frame_panel(axis, record, env, step, args.reach))
            legend_axis = axes.ravel()[7]
            legend_axis.axis("off")
            legend_axis.text(
                0.02, 0.97,
                "Moving local certificates\n\n"
                "BLUE solid: nominal polytope\n"
                "BLUE dashed: nominal H-step level set\n"
                "GREEN solid: fitted verifier polytope\n"
                "GREEN dashed: verifier H-step level set\n"
                "BLACK dots: next executed window\n\n"
                "N-exists: geometric nominal exists\n"
                "N-cert: future executed window obeys\n"
                "         this nominal gamma schedule\n"
                "V: fitted verifier SOCP succeeds\n"
                "valid2: task AND progress AND V\n\n"
                f"matched seed {seed}\n"
                f"frame stride {args.frame_stride} controls\n"
                f"GIF interval {args.duration_ms} ms",
                va="top", fontsize=9.5,
            )
            fig.suptitle(
                f"Nominal versus verifier polytope following the robot — shared time {step * env.dt:.1f} s",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            frame_path = temp / f"frame_{frame_index:04d}.png"
            fig.savefig(frame_path, dpi=90, facecolor="white")
            if step == poster_step:
                fig.savefig(viz_dir / "nominal_verifier_poster.png", dpi=180, facecolor="white")
            plt.close(fig)
            frame_paths.append(frame_path)
            frame_status.append({"step": step, "statuses": statuses})
            if frame_index % 10 == 0 or frame_index + 1 == len(frame_steps):
                print(f"[gif] frame {frame_index + 1}/{len(frame_steps)} step={step} "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

        rgb = [Image.open(path).convert("RGB") for path in frame_paths]
        first = rgb[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=160)
        palette_source = first.copy()
        frames = [first] + [image.quantize(palette=palette_source, dither=Image.Dither.NONE)
                            for image in rgb[1:]]
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=args.duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        for image in rgb:
            image.close()

    metadata = {
        "status": "GIF_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "seed": seed,
        "gammas": list(GAMMAS),
        "frames": len(frame_steps),
        "frame_stride_controls": args.frame_stride,
        "duration_ms": args.duration_ms,
        "playback_seconds": len(frame_steps) * args.duration_ms / 1000.0,
        "poster_step": poster_step,
        "window_H": H,
        "colors": {"nominal": BLUE, "verifier": GREEN},
        "output": str(output.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }
    (log_dir / "polytope_gif_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("metrics", "gif", "all"), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--outdir", type=Path, default=STAGE)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--poster-step", type=int, default=100)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS,
        help="giant-obstacle radius used to generate --source",
    )
    return parser.parse_args()


def main() -> None:
    global RADIUS
    args = parse_args()
    if not 0.0 < args.radius < 2.0:
        raise ValueError("--radius must lie in (0, 2)")
    RADIUS = float(args.radius)
    if args.frame_stride < 1:
        raise ValueError("frame stride must be positive")
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if args.phase in ("metrics", "all"):
        run_metrics(args)
    if args.phase in ("gif", "all"):
        run_gif(args)


if __name__ == "__main__":
    main()
