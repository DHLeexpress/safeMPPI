#!/usr/bin/env python3
"""Render a slow, synchronized seven-gamma SafeMPPI expert preview.

The preview uses one deterministic start from the upper off-diagonal pool and
one deterministic goal from the lower pool.  Every panel shows the same walled
scene, the gamma-colored executed trail, the moving nominal SafeMPPI polytope
and its DTCBF level sets, the next executed H-step data window, and the fitted
Pillar-3 verifier boundary (green when certified).

This is the approval-gated preview.  It does not generate the 300-pair training
dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]  # overnight_run_07_06/
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _paths  # noqa: F401,E402 - shared project import bootstrap
# _paths intentionally promotes the shared overnight sources. Re-promote this
# stage directory so the adapted local generators win same-named imports.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import grid_scene as GS  # noqa: E402
import verifier_polytope as VP  # noqa: E402
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter  # noqa: E402
from di_grid_viz import di_step  # noqa: E402

import gen_uniform_data as SEEDS  # noqa: E402
from viz_style import (  # noqa: E402
    GAMMAS,
    GAMMA_CMAP,
    GAMMA_COLORS,
    GAMMA_NORM,
    SIGMA_CMAP_NAME,
    gamma_boundaries,
)


STAGE_DIR = HERE / "stage_results" / "02_demos"
DEFAULT_GIF = STAGE_DIR / "viz" / "fixed_pair_expert_polytopes.gif"
DEFAULT_PNG = STAGE_DIR / "viz" / "fixed_pair_expert_polytopes_final.png"
DEFAULT_DATA = STAGE_DIR / "data" / "fixed_pair_preview_paths.npz"
DEFAULT_LOG = STAGE_DIR / "logs" / "fixed_pair_preview.json"

GREEN = "#009944"
FAIL_RED = "#cc3311"
NOMINAL_BLUE = "#2166ac"


@dataclass
class ExpertRun:
    gamma: float
    states: np.ndarray
    controls: np.ndarray
    polytopes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    planner_infeasibility: np.ndarray
    elapsed_s: float


@dataclass
class CertificateFrame:
    ok: bool
    faces: list[Any]
    segment: np.ndarray
    nominal_ok: bool
    nominal_min_slack: float


def _nearest(pool: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
    idx = int(np.linalg.norm(pool - target[None, :], axis=1).argmin())
    return pool[idx].astype(np.float32).copy(), idx


def select_pair(start_target: np.ndarray, goal_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    env = SEEDS.make_walled_env(8)
    blue, red = SEEDS.start_goal_pools(env)
    start, start_idx = _nearest(blue, start_target)
    goal, goal_idx = _nearest(red, goal_target)
    return start, goal, {
        "requested_start": start_target.tolist(),
        "requested_goal": goal_target.tolist(),
        "selected_start_index": start_idx,
        "selected_goal_index": goal_idx,
        "start_pool_size": len(blue),
        "goal_pool_size": len(red),
    }


def make_pair_env(start: np.ndarray, goal: np.ndarray):
    env = SEEDS.make_walled_env(8)
    env.x0 = torch.tensor([start[0], start[1], 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor(goal, dtype=env.goal.dtype)
    return env


def rollout_expert(
    gamma: float,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    seed: int,
    reach: float,
) -> ExpertRun:
    """Run one deterministic receding-horizon SafeMPPI episode."""
    env = make_pair_env(start, goal)
    cfg = GS.mode1_config()
    adapter = SafeMPPIAdapter(**cfg)
    obs_plan = GS.planner_obstacles(env)
    state = env.x0.detach().cpu().numpy().astype(np.float32).copy()
    goal_t = env.goal.detach().cpu().float()

    states = [state.copy()]
    controls: list[np.ndarray] = []
    polytopes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    infeasibility: list[float] = []
    started = time.perf_counter()

    for step in range(env.T):
        action, info = adapter.plan(
            torch.tensor(state, dtype=torch.float32),
            goal_t,
            obs_plan,
            gamma=float(gamma),
            seed=int(seed) * 1000 + step,
        )
        raw_poly = info.get("polytope")
        if raw_poly is None:
            raise RuntimeError("mode-1 SafeMPPI did not return its nominal polytope")
        polytopes.append(tuple(np.asarray(value, dtype=np.float32) for value in raw_poly))
        infeasibility.append(float(info["infeasibility_rate"]))

        action_np = action.detach().cpu().numpy().astype(np.float32)
        controls.append(action_np)
        state = di_step(state, action_np, dt=env.dt).astype(np.float32)
        states.append(state.copy())
        if np.linalg.norm(state[:2] - goal) < reach:
            break

    return ExpertRun(
        gamma=float(gamma),
        states=np.asarray(states, dtype=np.float32),
        controls=np.asarray(controls, dtype=np.float32),
        polytopes=polytopes,
        planner_infeasibility=np.asarray(infeasibility, dtype=np.float32),
        elapsed_s=time.perf_counter() - started,
    )


def _nominal_h(poly, points: np.ndarray) -> np.ndarray:
    A, b, _center, margins = poly
    return ((b[None, :] - points @ A.T) / np.maximum(margins[None, :], 1e-6)).min(axis=1)


def _faces_h(faces, center: np.ndarray, points: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=float).reshape(-1, 2) - center[None, :2]
    values = np.full(len(rel), np.inf, dtype=float)
    used = 0
    for face in faces:
        margin = float(face.m)
        if not getattr(face, "feasible", True) or margin <= 1e-9:
            continue
        a = np.asarray(face.a, dtype=float)[:2]
        values = np.minimum(values, (margin - rel @ a) / margin)
        used += 1
    if used == 0:
        return np.full(len(rel), np.nan, dtype=float)
    return values


def _window(run: ExpertRun, step: int, horizon: int) -> np.ndarray:
    stop = min(step + horizon + 1, len(run.states))
    segment = run.states[step:stop, :2]
    if len(segment) < 2:
        segment = run.states[-2:, :2]
    return np.asarray(segment, dtype=float)


def certify_frame(run: ExpertRun, step: int, obs: np.ndarray, r_robot: float, horizon: int, n_theta: int):
    segment = _window(run, step, horizon)
    poly = run.polytopes[step]
    alpha = (1.0 - run.gamma) ** np.arange(len(segment), dtype=float)
    nominal_h = _nominal_h(poly, segment)
    nominal_slack = float(np.min(nominal_h - alpha))
    ok, faces, _raw, _r_eff = VP.certify_window(
        segment,
        obs,
        r_robot,
        run.gamma,
        R=2.0,
        n_theta=n_theta,
    )
    return CertificateFrame(
        ok=bool(ok),
        faces=faces,
        segment=segment,
        nominal_ok=nominal_slack >= -1e-6,
        nominal_min_slack=nominal_slack,
    )


def run_metrics(run: ExpertRun, env, goal: np.ndarray, reach: float) -> dict[str, Any]:
    xy = run.states[:, :2].astype(float)
    obs = env.obstacles.detach().cpu().numpy()
    clearance = (
        np.linalg.norm(xy[:, None, :] - obs[None, :, :2], axis=2)
        - obs[None, :, 2]
        - float(env.r_robot)
    )
    collision = bool((clearance.min(axis=1) < 0.0).any())
    return {
        "steps": len(run.controls),
        "elapsed_s": run.elapsed_s,
        "reached": bool(np.linalg.norm(xy[-1] - goal) < reach),
        "endpoint_distance": float(np.linalg.norm(xy[-1] - goal)),
        "collision": collision,
        "in_taskspace": bool(((xy >= 0.0) & (xy <= 5.0)).all()),
        "min_clearance": float(clearance.min()),
        "mean_planner_infeasibility_rate": float(run.planner_infeasibility.mean()),
    }


def draw_scene(ax, env, start: np.ndarray, goal: np.ndarray) -> None:
    ax.set_facecolor("#f8f7f4")
    ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#444444", lw=0.8, ls="--", zorder=2))
    for obstacle in env.obstacles.detach().cpu().numpy():
        ax.add_patch(
            Circle(
                obstacle[:2],
                obstacle[2] + float(env.r_robot),
                facecolor="#8a8a8a",
                edgecolor="#666666",
                lw=0.25,
                alpha=0.9,
                zorder=4,
            )
        )
    ax.scatter(start[0], start[1], marker="s", s=34, c="#1769aa", edgecolor="white", lw=0.6, zorder=11)
    ax.scatter(goal[0], goal[1], marker="*", s=95, c="#d32f2f", edgecolor="white", lw=0.6, zorder=11)
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(-0.4, 5.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def render_animation(
    runs: dict[float, ExpertRun],
    env,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    output_gif: Path,
    output_png: Path,
    horizon: int,
    frame_stride: int,
    fps: float,
    end_hold: int,
    n_theta: int,
) -> tuple[dict[float, dict[str, Any]], list[int]]:
    obs = env.obstacles.detach().cpu().numpy()
    r_robot = float(env.r_robot)
    max_steps = max(len(run.controls) for run in runs.values())
    frame_steps = list(range(0, max_steps, frame_stride))
    if not frame_steps or frame_steps[-1] != max_steps - 1:
        frame_steps.append(max_steps - 1)

    # Cache every expensive SOCP used by the movie. Completed trajectories hold
    # their final window while the slowest gamma finishes.
    cache: dict[tuple[float, int], CertificateFrame] = {}
    rendered_unique: dict[float, set[int]] = {g: set() for g in GAMMAS}
    for global_step in frame_steps:
        for gamma in GAMMAS:
            run = runs[gamma]
            step = min(global_step, len(run.polytopes) - 1)
            rendered_unique[gamma].add(step)
            key = (gamma, step)
            if key not in cache:
                cache[key] = certify_frame(run, step, obs, r_robot, horizon, n_theta)

    cert_summary: dict[float, dict[str, Any]] = {}
    for gamma in GAMMAS:
        frames = [cache[(gamma, step)] for step in sorted(rendered_unique[gamma])]
        cert_summary[gamma] = {
            "rendered_unique_windows": len(frames),
            "verifier_certified": int(sum(item.ok for item in frames)),
            "verifier_certified_fraction": float(np.mean([item.ok for item in frames])),
            "nominal_ruler_pass": int(sum(item.nominal_ok for item in frames)),
            "nominal_ruler_pass_fraction": float(np.mean([item.nominal_ok for item in frames])),
            "nominal_min_slack": float(min(item.nominal_min_slack for item in frames)),
        }

    grid_x = np.linspace(-0.4, 5.4, 86)
    grid_y = np.linspace(-0.4, 5.4, 86)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grid_points = np.stack((grid_X.ravel(), grid_Y.ravel()), axis=1)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 8.0))
    axes_flat = axes.ravel()
    gamma_axes = {gamma: axes_flat[i] for i, gamma in enumerate(GAMMAS)}
    overview_ax = axes_flat[-1]
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.865, wspace=0.055, hspace=0.15)

    cax = fig.add_axes((0.36, 0.905, 0.30, 0.025))
    colorbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=GAMMA_CMAP,
        norm=GAMMA_NORM,
        boundaries=gamma_boundaries(),
        ticks=GAMMAS,
        spacing="uniform",
        orientation="horizontal",
        drawedges=True,
    )
    colorbar.ax.set_title(r"$\gamma$ (trajectory color; truncated plasma)", fontsize=10, pad=4)
    colorbar.ax.tick_params(labelsize=8, length=0, pad=2)
    colorbar.dividers.set_color("white")

    legend_handles = [
        Line2D([], [], color=NOMINAL_BLUE, lw=1.2, label=r"nominal $H_P$ boundary/levels"),
        Line2D([], [], color=GREEN, lw=2.0, label="fitted verifier (certified)"),
        Line2D([], [], color="#111111", lw=1.0, marker=".", label="executed H-step data window"),
    ]

    def draw(global_step: int):
        for gamma in GAMMAS:
            ax = gamma_axes[gamma]
            ax.clear()
            draw_scene(ax, env, start, goal)
            run = runs[gamma]
            step = min(global_step, len(run.polytopes) - 1)
            trail_stop = min(global_step + 2, len(run.states))
            color = GAMMA_COLORS[gamma]
            poly = run.polytopes[step]
            cert = cache[(gamma, step)]

            nominal_H = _nominal_h(poly, grid_points).reshape(grid_X.shape)
            alpha_levels = sorted(
                set([0.0, 1.0] + [round(float((1.0 - gamma) ** i), 5) for i in range(horizon + 1)])
            )
            fill_levels = alpha_levels + ([1.0001] if alpha_levels[-1] <= 1.0 else [])
            ax.contourf(grid_X, grid_Y, nominal_H, levels=fill_levels, cmap="Blues", alpha=0.26, zorder=1)
            inner = [value for value in alpha_levels if 1e-6 < value < 1.0 - 1e-6]
            if inner:
                ax.contour(
                    grid_X,
                    grid_Y,
                    nominal_H,
                    levels=inner,
                    colors=[color],
                    linewidths=0.45,
                    alpha=0.72,
                    zorder=3,
                )
            ax.contour(
                grid_X,
                grid_Y,
                nominal_H,
                levels=[0.0],
                colors=[NOMINAL_BLUE],
                linewidths=1.15,
                linestyles="--",
                zorder=5,
            )

            verifier_H = _faces_h(cert.faces, cert.segment[0], grid_points).reshape(grid_X.shape)
            verifier_color = GREEN if cert.ok else FAIL_RED
            if np.isfinite(verifier_H).any():
                ax.contour(
                    grid_X,
                    grid_Y,
                    verifier_H,
                    levels=[0.0],
                    colors=[verifier_color],
                    linewidths=2.0,
                    zorder=6,
                )

            trail = run.states[:trail_stop, :2]
            ax.plot(trail[:, 0], trail[:, 1], color=color, lw=2.2, alpha=0.96, zorder=8)
            ax.plot(
                cert.segment[:, 0],
                cert.segment[:, 1],
                color="#111111",
                lw=0.9,
                marker=".",
                ms=2.5,
                alpha=0.82,
                zorder=9,
            )
            center = cert.segment[0]
            ax.scatter(center[0], center[1], s=35, c=[color], edgecolor="black", lw=0.6, zorder=12)
            cert_text = "SOCP OK" if cert.ok else "SOCP FAIL"
            reached = global_step >= len(run.controls) - 1
            ax.set_title(
                rf"$\gamma={gamma:g}$  {cert_text}"
                + (" · reached" if reached else f" · step {step}"),
                fontsize=9.5,
                color=color,
                pad=3,
            )

        overview_ax.clear()
        draw_scene(overview_ax, env, start, goal)
        for gamma in GAMMAS:
            run = runs[gamma]
            stop = min(global_step + 2, len(run.states))
            path = run.states[:stop, :2]
            overview_ax.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=1.7, alpha=0.9, zorder=7)
        overview_ax.set_title("synchronized expert trajectories", fontsize=9.5, pad=3)
        overview_ax.legend(handles=legend_handles, loc="lower left", fontsize=6.7, framealpha=0.9)

        seconds = min(global_step, max_steps - 1) * float(env.dt)
        fig.suptitle(
            "SafeMPPI fixed-pair preview — moving nominal polytope + DTCBF level sets\n"
            f"start ({start[0]:.3f}, {start[1]:.3f}) → goal ({goal[0]:.3f}, {goal[1]:.3f})"
            f"  · synchronized time {seconds:.1f} s  · verifier green=certified",
            fontsize=12,
            y=0.992,
        )
        return []

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    movie_steps = frame_steps + [frame_steps[-1]] * max(0, end_hold)
    animation = FuncAnimation(
        fig,
        draw,
        frames=movie_steps,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(output_gif, writer=PillowWriter(fps=fps), dpi=82)
    draw(frame_steps[-1])
    fig.savefig(output_png, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return cert_summary, frame_steps


def _object_array(items) -> np.ndarray:
    result = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        result[i] = item
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # One row inward from the wall-adjacent extreme seeds: this exact approved
    # pair is collision-free and every rendered H-step window certifies at all
    # seven gammas with the reference verifier resolution.
    parser.add_argument("--start-target", type=float, nargs=2, default=(0.3168483, 4.6734347))
    parser.add_argument("--goal-target", type=float, nargs=2, default=(4.6653547, 0.3146670))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reach", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--fps", type=float, default=2.0, help="slow GIF playback rate")
    parser.add_argument("--end-hold", type=int, default=4, help="duplicate final frames")
    parser.add_argument("--n-theta", type=int, default=180, help="verifier angular resolution")
    parser.add_argument("--out", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--final-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_target = np.asarray(args.start_target, dtype=np.float32)
    goal_target = np.asarray(args.goal_target, dtype=np.float32)
    start, goal, selection = select_pair(start_target, goal_target)
    env = make_pair_env(start, goal)

    print(
        f"fixed pair: start={start.tolist()} goal={goal.tolist()} · "
        f"7 gammas · reach={args.reach}",
        flush=True,
    )
    runs: dict[float, ExpertRun] = {}
    metrics: dict[float, dict[str, Any]] = {}
    for gamma in GAMMAS:
        run = rollout_expert(gamma, start, goal, seed=args.seed, reach=args.reach)
        runs[gamma] = run
        metrics[gamma] = run_metrics(run, env, goal, args.reach)
        row = metrics[gamma]
        print(
            f"gamma={gamma:g}: steps={row['steps']} reached={row['reached']} "
            f"collision={row['collision']} min_clearance={row['min_clearance']:.3f} m "
            f"({row['elapsed_s']:.2f} s)",
            flush=True,
        )

    args.data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data,
        gammas=np.asarray(GAMMAS, dtype=np.float32),
        start=start,
        goal=goal,
        paths=_object_array([runs[g].states for g in GAMMAS]),
        controls=_object_array([runs[g].controls for g in GAMMAS]),
        seed=int(args.seed),
        reach=float(args.reach),
    )
    print(f"DATA {args.data}", flush=True)

    certificate_summary, frame_steps = render_animation(
        runs,
        env,
        start,
        goal,
        output_gif=args.out,
        output_png=args.final_png,
        horizon=args.horizon,
        frame_stride=args.frame_stride,
        fps=args.fps,
        end_hold=args.end_hold,
        n_theta=args.n_theta,
    )

    for gamma in GAMMAS:
        metrics[gamma].update(certificate_summary[gamma])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "approval_scope": "fixed-pair preview only; no 300-pair training shards generated",
        "selection": selection,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "wall_plugs": 8,
        "obstacles": len(env.obstacles),
        "seed": int(args.seed),
        "reach": float(args.reach),
        "horizon": int(args.horizon),
        "frame_stride": int(args.frame_stride),
        "frame_steps": frame_steps,
        "fps": float(args.fps),
        "palette": {
            "gamma": "plasma sampled at 7 points over [0.02, 0.90] (Image #1)",
            "sigma_uncertainty": SIGMA_CMAP_NAME,
            "verifier_certified": GREEN,
        },
        "planner_config": GS.mode1_config(),
        "per_gamma": {str(gamma): metrics[gamma] for gamma in GAMMAS},
        "outputs": {
            "gif": str(args.out.resolve()),
            "final_png": str(args.final_png.resolve()),
            "paths": str(args.data.resolve()),
        },
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"GIF {args.out}")
    print(f"FINAL {args.final_png}")
    print(f"LOG {args.log}")


if __name__ == "__main__":
    main()
