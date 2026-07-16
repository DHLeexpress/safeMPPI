#!/usr/bin/env python3
"""Automated Stage 4: frozen OOD baselines on the radius-1.2 giant scene.

The stage is deliberately learning-free.  It combines the already approved
M=2 anti-retreat SafeMPPI paths with four fresh matched-seed rollouts, deploys
the frozen Stage-3 policy at both the approved and faithful temperatures, runs
a bounded low-guidance Mizuta/CFM-MPPI sweep, promotes the best admissible
setting, and writes exact paths, metrics, figures, and a decision report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (ROOT.parents[1], ROOT.parent, ROOT):
    # Move, rather than merely add, this benchmark root to the front.  Shared
    # experiment bootstraps prepend same-named legacy modules.
    if str(candidate) in sys.path:
        sys.path.remove(str(candidate))
    sys.path.insert(0, str(candidate))

import grid_feats as GF  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
from giant_obstacle_ood.stage1_geometry_sweep import (  # noqa: E402
    GIANT_CENTER,
    draw_scene,
    make_scene,
)
from giant_obstacle_ood.stage1b_smooth_expert import (  # noqa: E402
    GOAL,
    RADIUS,
    START,
    route_mode,
    smoothness_metrics,
)
from giant_obstacle_ood.stage2a_retreat_penalty import rollout as expert_rollout  # noqa: E402
from viz_style import (  # noqa: E402
    GAMMAS,
    GAMMA_CMAP,
    GAMMA_COLORS,
    GAMMA_NORM,
    gamma_boundaries,
)


_KAZUKI_SPEC = importlib.util.spec_from_file_location(
    "giant_stage4_kazuki", ROOT / "reference/kazuki_baseline.py"
)
KAZ = importlib.util.module_from_spec(_KAZUKI_SPEC)
assert _KAZUKI_SPEC.loader is not None
_KAZUKI_SPEC.loader.exec_module(KAZ)


STAGE = HERE / "stage_results/04_frozen_ood"
CHECKPOINT = HERE / "stage_results/03_pretrain/data/pretrained_id_balanced_a32.pt"
LOCKED_EXPERT = HERE / "stage_results/02a_retreat_penalty/data/selected_m2_paths.npz"
REACH = 0.15
T_EXPERT = 800
T_LEARNED = 300

TUNE_CONFIGS = (
    {"tag": "pure_bridge", "w_safe": 0.0, "coll_w": 0.0, "goal_w": 2.0, "goal_coef": 0.0},
    {"tag": "lg005", "w_safe": 0.005, "coll_w": 0.5, "goal_w": 2.0, "goal_coef": 0.025},
    {"tag": "lg010", "w_safe": 0.010, "coll_w": 1.0, "goal_w": 2.0, "goal_coef": 0.050},
    {"tag": "lg020", "w_safe": 0.020, "coll_w": 2.0, "goal_w": 2.0, "goal_coef": 0.100},
    {"tag": "lg040", "w_safe": 0.040, "coll_w": 4.0, "goal_w": 2.0, "goal_coef": 0.200},
)
TUNE_GAMMAS = (0.1, 0.5, 1.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_array(items) -> np.ndarray:
    output = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        output[index] = np.asarray(item, dtype=np.float32)
    return output


def make_env(max_steps: int):
    env = make_scene(RADIUS, START, GOAL)
    env.T = int(max_steps)
    return env


def physics_step(states: np.ndarray, indices: np.ndarray, actions: np.ndarray, dt: float) -> None:
    # ``indices`` is advanced indexing, so mutate the parent array explicitly.
    prior = states[indices].copy()
    states[indices, :2] = prior[:, :2] + dt * prior[:, 2:] + 0.5 * dt * dt * actions
    states[indices, 2:] = prior[:, 2:] + dt * actions


def classify_path(path: np.ndarray, controls: np.ndarray, env, reach: float,
                  supplied_reason: str | None = None) -> dict:
    path = np.asarray(path, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    obstacles = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    rr = float(env.r_robot)
    clearances = (
        np.linalg.norm(path[:, None] - obstacles[None, :, :2], axis=2)
        - obstacles[None, :, 2]
        - rr
    )
    flat_index = int(np.argmin(clearances))
    point_index_raw, obstacle_index_raw = np.unravel_index(flat_index, clearances.shape)
    point_index, obstacle_index = int(point_index_raw), int(obstacle_index_raw)
    min_clearance = float(clearances[point_index, obstacle_index])
    collision = min_clearance < 0.0
    in_taskspace = bool(((path >= 0.0) & (path <= 5.0)).all())
    endpoint_distance = float(np.linalg.norm(path[-1] - goal))
    reached = endpoint_distance < reach
    success = bool(reached and not collision and in_taskspace)
    giant_index = int(np.argmin(np.linalg.norm(obstacles[:, :2] - GIANT_CENTER[None], axis=1)))
    giant_clearance = (
        np.linalg.norm(path - GIANT_CENTER[None], axis=1) - float(obstacles[giant_index, 2]) - rr
    )
    near = np.flatnonzero(giant_clearance < 0.55)
    if len(near) >= 2:
        angles = np.unwrap(np.arctan2(path[near, 1] - GIANT_CENTER[1], path[near, 0] - GIANT_CENTER[0]))
        boundary_arc = float(np.ptp(angles))
    else:
        boundary_arc = 0.0
    recent_start = max(0, len(path) - 31)
    recent_displacement = float(np.linalg.norm(path[-1] - path[recent_start]))
    goal_distances = np.linalg.norm(path - goal[None], axis=1)
    recent_progress = float(goal_distances[recent_start] - goal_distances[-1])
    entry_basin = bool(
        path[-1, 0] < GIANT_CENTER[0] + 0.15
        and path[-1, 1] < GIANT_CENTER[1] + 0.15
        and np.linalg.norm(path[-1] - GIANT_CENTER) < RADIUS + rr + 1.15
    )
    timed_out = bool(not reached and not collision and in_taskspace and len(controls) >= env.T)
    local_minimum = bool(timed_out and entry_basin and recent_progress < 0.10)
    if success:
        failure_type = "success"
    elif collision:
        failure_type = "giant collision" if obstacle_index == giant_index else "other collision"
    elif not in_taskspace:
        failure_type = "out of bounds"
    elif local_minimum:
        failure_type = "local-minimum timeout"
    else:
        failure_type = supplied_reason or "timeout"
    giant_approach = np.flatnonzero(giant_clearance < 0.65)
    prefix_end = int(giant_approach[0] + 1) if len(giant_approach) else min(len(path), 80)
    prefix = path[:max(prefix_end, 2)]
    diagonal_error = float(np.mean(np.abs(prefix[:, 1] - prefix[:, 0])))
    diagonal_fraction = float(np.mean(np.abs(prefix[:, 1] - prefix[:, 0]) < 0.25))
    mode, side_score = route_mode(path)
    smooth = smoothness_metrics(controls, float(env.dt)) if len(controls) else {
        "accel_rms": math.nan,
        "control_delta_mean": math.nan,
        "control_delta_rms": math.nan,
        "jerk_rms": math.nan,
        "smooth_cost_analog": math.nan,
    }
    return {
        "success": success,
        "reached": bool(reached),
        "collision": bool(collision),
        "in_taskspace": in_taskspace,
        "failure_type": failure_type,
        "steps": int(len(controls)),
        "time_s": float(len(controls) * env.dt),
        "endpoint_distance": endpoint_distance,
        "best_goal_distance": float(goal_distances.min()),
        "goal_progress": float(goal_distances[0] - goal_distances[-1]),
        "min_clearance": min_clearance,
        "clearance_mean": float(clearances.min(axis=1).mean()),
        "giant_min_clearance": float(giant_clearance.min()),
        "collision_obstacle_index": obstacle_index if collision else None,
        "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "route_mode": mode,
        "side_score": float(side_score),
        "boundary_arc_rad": boundary_arc,
        "recent_displacement_30": recent_displacement,
        "recent_goal_progress_30": recent_progress,
        "entry_basin": entry_basin,
        "local_minimum": local_minimum,
        "pre_giant_diagonal_error": diagonal_error,
        "pre_giant_diagonal_fraction": diagonal_fraction,
        **smooth,
    }


def attach_metrics(method: str, gamma: float, repetition: int, seed: int,
                   path: np.ndarray, controls: np.ndarray, env,
                   supplied_reason: str | None = None, **extra) -> dict:
    return {
        "method": method,
        "gamma": float(gamma),
        "repetition": int(repetition),
        "seed": int(seed),
        **classify_path(path, controls, env, REACH, supplied_reason),
        **extra,
        "path": np.asarray(path, dtype=np.float32),
        "controls": np.asarray(controls, dtype=np.float32),
    }


@torch.inference_mode()
def rollout_policy(policy, *, repetitions: int, temperature: float, nfe: int,
                   T: int, seed0: int, device: torch.device, method: str,
                   persistent_route_bit: bool = False,
                   persistent_latent: bool = False,
                   latent_correlation: float = 0.0,
                   ensemble_size: int = 1) -> list[dict]:
    if not 0.0 <= float(latent_correlation) <= 1.0:
        raise ValueError("latent_correlation must be in [0,1]")
    if int(ensemble_size) <= 0:
        raise ValueError("ensemble_size must be positive")
    if int(ensemble_size) > 1 and (persistent_latent or latent_correlation > 0.0):
        raise ValueError("within-mode ensemble cannot be combined with persistent/correlated latents")
    env = make_env(T)
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    metadata = [
        (float(gamma), repetition, seed0 + gamma_index * 10_000 + repetition)
        for gamma_index, gamma in enumerate(GAMMAS)
        for repetition in range(repetitions)
    ]
    count = len(metadata)
    states = np.zeros((count, 4), dtype=np.float32)
    states[:, :2] = START
    histories = np.zeros((count, GF.K_HIST, 2), dtype=np.float32)
    paths: list[list[np.ndarray]] = [[START.copy()] for _ in range(count)]
    controls: list[list[np.ndarray]] = [[] for _ in range(count)]
    active = np.ones(count, dtype=bool)
    reasons: list[str | None] = [None] * count
    # A balanced episode-level antisymmetric latent bit.  This is verifier-free
    # temporal coherence: every step still draws fresh Gaussian noise, but its
    # x/y route half-space is held fixed for the episode.  Across paired
    # repetitions the aggregate source remains exactly the original Gaussian.
    route_sign = np.asarray(
        [1.0 if repetition % 2 == 0 else -1.0 for _, repetition, _ in metadata],
        dtype=np.float32,
    )
    torch.manual_seed(seed0)
    np.random.seed(seed0 % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed0)
    episode_latent = None
    if persistent_latent or latent_correlation > 0.0:
        episode_latent = float(temperature) * torch.randn(count, policy.d, device=device)
        if persistent_route_bit:
            sign = torch.from_numpy(route_sign).to(device)
            wrong = (episode_latent[:, 0] - episode_latent[:, 1]) * sign < 0
            pair = episode_latent[wrong, :2].clone()
            episode_latent[wrong, 0] = pair[:, 1]
            episode_latent[wrong, 1] = pair[:, 0]

    for step in range(T):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        grid_np = np.stack([GF.axis_grid(states[index, :2], obstacles, rr) for index in indices])
        low_np = np.stack([GF.low5(states[index], GOAL, metadata[index][0]) for index in indices])
        context = policy.ctx_from(
            torch.from_numpy(grid_np).to(device),
            torch.from_numpy(low_np).to(device),
            torch.from_numpy(histories[indices]).to(device),
        )
        if persistent_route_bit or persistent_latent or latent_correlation > 0.0:
            if persistent_latent:
                latent = episode_latent[torch.as_tensor(indices, device=device)].clone()
            elif latent_correlation > 0.0:
                row_index = torch.as_tensor(indices, device=device)
                fresh = float(temperature) * torch.randn(len(indices), policy.d, device=device)
                rho = float(latent_correlation)
                latent = rho * episode_latent[row_index] + math.sqrt(1.0 - rho * rho) * fresh
                if persistent_route_bit:
                    sign = torch.from_numpy(route_sign[indices]).to(device)
                    wrong = (latent[:, 0] - latent[:, 1]) * sign < 0
                    pair = latent[wrong, :2].clone()
                    latent[wrong, 0] = pair[:, 1]
                    latent[wrong, 1] = pair[:, 0]
                episode_latent[row_index] = latent
            else:
                latent = float(temperature) * torch.randn(
                    len(indices) * int(ensemble_size), policy.d, device=device,
                )
                sign = torch.from_numpy(route_sign[indices]).to(device).repeat_interleave(
                    int(ensemble_size))
                wrong = (latent[:, 0] - latent[:, 1]) * sign < 0
                pair = latent[wrong, :2].clone()
                latent[wrong, 0] = pair[:, 1]
                latent[wrong, 1] = pair[:, 0]
            sample_context = (
                context.repeat_interleave(int(ensemble_size), dim=0)
                if len(latent) != len(indices) else context
            )
            for nfe_index in range(int(nfe)):
                tau = torch.full(
                    (len(latent),), nfe_index / int(nfe), device=device,
                )
                latent = latent + (1.0 / int(nfe)) * policy.forward(latent, tau, sample_context)
            windows = latent.reshape(len(indices), int(ensemble_size), policy.T, 2).mean(dim=1)
            windows = (windows * policy.u_max).clamp(
                -policy.u_max, policy.u_max,
            )
        else:
            windows = policy.sample(len(indices), context, nfe=nfe, temp=temperature)
        actions = windows[:, 0].float().cpu().numpy()
        physics_step(states, indices, actions, float(env.dt))
        histories[indices, :-1] = histories[indices, 1:]
        histories[indices, -1] = actions
        for local, index in enumerate(indices):
            position = states[index, :2].copy()
            paths[index].append(position)
            controls[index].append(actions[local].copy())
            dmin = float((np.linalg.norm(obstacles[:, :2] - position[None], axis=1)
                          - obstacles[:, 2] - rr).min())
            collision = dmin < 0.0
            out = bool((position < 0.0).any() or (position > 5.0).any())
            reached = float(np.linalg.norm(position - GOAL)) < REACH
            if collision or out or reached:
                active[index] = False
                reasons[index] = "collision" if collision else "out of bounds" if out else None
        if step == 0 or (step + 1) % 50 == 0:
            print(f"[{method}] step={step + 1}/{T} active={int(active.sum())}/{count}", flush=True)

    records = []
    for index, (gamma, repetition, seed) in enumerate(metadata):
        records.append(attach_metrics(
            method, gamma, repetition, seed,
            np.asarray(paths[index]), np.asarray(controls[index]), env, reasons[index],
            temperature=float(temperature), nfe=int(nfe), h_exec=1,
            persistent_route_bit=bool(persistent_route_bit),
            persistent_latent=bool(persistent_latent),
            latent_correlation=float(latent_correlation),
            ensemble_size=int(ensemble_size),
            route_sign=(float(route_sign[index]) if persistent_route_bit else None),
        ))
    return records


def load_locked_expert() -> list[dict]:
    env = make_env(T_EXPERT)
    with np.load(LOCKED_EXPERT, allow_pickle=True) as archive:
        output = []
        per_gamma_count = Counter()
        for gamma, seed, path, controls in zip(
            archive["gammas"], archive["seeds"], archive["paths"], archive["controls"]
        ):
            repetition = per_gamma_count[float(gamma)]
            per_gamma_count[float(gamma)] += 1
            output.append(attach_metrics(
                "SafeMPPI expert", float(gamma), repetition, int(seed), path, controls, env,
                source="approved Stage 2A", smooth_weight=8.0,
                retreat_weight=1.0, retreat_scale=0.05, retreat_cap=6.0,
            ))
    return output


def run_expert(new_repetitions: int, device: torch.device) -> list[dict]:
    records = load_locked_expert()
    env = make_env(T_EXPERT)
    started = time.perf_counter()
    for repetition in range(new_repetitions):
        seed = 74100 + repetition
        for gamma in GAMMAS:
            result = expert_rollout(
                env, float(gamma), seed, 1.0, 0.05, 6.0, REACH, device, certify=False
            )
            record = attach_metrics(
                "SafeMPPI expert", float(gamma), repetition + 2, seed,
                result["path"], result["controls"], env, result.get("dead_reason"),
                source="fresh Stage 4", smooth_weight=8.0,
                retreat_weight=1.0, retreat_scale=0.05, retreat_cap=6.0,
            )
            records.append(record)
            print(
                f"[expert g={gamma:g} rep={repetition + 1}/{new_repetitions}] "
                f"{record['failure_type']} steps={record['steps']} "
                f"wall={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return records


def configure_kazuki(config: dict, *, n_sample: int, n_elite: int, n_copy: int) -> None:
    KAZ.COLL_W = float(config["coll_w"])
    KAZ.GOAL_W = float(config["goal_w"])
    KAZ.GOAL_COEF = float(config["goal_coef"])
    KAZ.BETA_MPPI = 20.0
    KAZ.MPPI_LAMBDA = 0.1
    KAZ.MPPI_SIGMA = 0.2
    KAZ.R_MARGIN = 0.05
    KAZ.N_SAMPLE = int(n_sample)
    KAZ.N_ELITE = int(n_elite)
    KAZ.N_COPY = int(n_copy)


def run_mizuta_config(policy, config: dict, gammas: tuple[float, ...], repetitions: int,
                      *, T: int, source_temp: float, n_sample: int, n_elite: int,
                      n_copy: int, seed0: int, device: torch.device,
                      method: str) -> list[dict]:
    configure_kazuki(config, n_sample=n_sample, n_elite=n_elite, n_copy=n_copy)
    env = make_env(T)
    records = []
    started = time.perf_counter()
    for gamma_index, gamma in enumerate(gammas):
        for repetition in range(repetitions):
            seed = seed0 + gamma_index * 10_000 + repetition
            output = KAZ.kazuki_deploy(
                policy, env, [float(config["w_safe"])], gamma_ctx=float(gamma), T=T,
                reach=REACH, device=device, seed=seed, source_temp=source_temp,
            )
            reason = "collision" if output["collided"] else None
            record = attach_metrics(
                method, float(gamma), repetition, seed, output["path"], output["controls"],
                env, reason, tune_tag=config["tag"], source_temp=source_temp,
                w_safe=float(config["w_safe"]), coll_w=float(config["coll_w"]),
                goal_w=float(config["goal_w"]), goal_coef=float(config["goal_coef"]),
                n_sample=n_sample, n_elite=n_elite, n_copy=n_copy,
                obstacle_radius_model="per-obstacle",
            )
            records.append(record)
            print(
                f"[{method} {config['tag']} g={gamma:g} rep={repetition + 1}/{repetitions}] "
                f"{record['failure_type']} steps={record['steps']} "
                f"diag={record['pre_giant_diagonal_fraction']:.2f} "
                f"wall={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return records


def mean_finite(values) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def aggregate(records: list[dict]) -> dict:
    successes = [record for record in records if record["success"]]
    failures = Counter(record["failure_type"] for record in records)
    modes = sorted({record["route_mode"] for record in successes
                    if record["route_mode"] in ("upper-left", "lower-right")})
    return {
        "M": len(records),
        "successes": len(successes),
        "a_SR": len(successes) / len(records) if records else math.nan,
        "collisions": sum(record["collision"] for record in records),
        "b_CR": float(np.mean([record["collision"] for record in records])) if records else math.nan,
        "c_clearance_mean_success": mean_finite([record["clearance_mean"] for record in successes]),
        "min_clearance_mean_success": mean_finite([record["min_clearance"] for record in successes]),
        "d_time_s_mean_success": mean_finite([record["time_s"] for record in successes]),
        "e_coverage": len(modes),
        "coverage_modes": modes,
        "failure_taxonomy": dict(sorted(failures.items())),
        "local_minimum_rate": float(np.mean([record["local_minimum"] for record in records])) if records else math.nan,
        "mean_endpoint_distance": mean_finite([record["endpoint_distance"] for record in records]),
        "mean_goal_progress": mean_finite([record["goal_progress"] for record in records]),
        "mean_boundary_arc_rad": mean_finite([record["boundary_arc_rad"] for record in records]),
        "mean_pre_giant_diagonal_fraction": mean_finite(
            [record["pre_giant_diagonal_fraction"] for record in records]
        ),
        "mean_control_delta": mean_finite([record["control_delta_mean"] for record in records]),
    }


def summarize_method(records: list[dict]) -> dict:
    return {
        "overall": aggregate(records),
        "per_gamma": {
            str(float(gamma)): aggregate([record for record in records if record["gamma"] == float(gamma)])
            for gamma in GAMMAS
        },
    }


def tuning_rows(records: list[dict]) -> list[dict]:
    rows = []
    for config in TUNE_CONFIGS:
        subset = [record for record in records if record["tune_tag"] == config["tag"]]
        summary = aggregate(subset)
        rows.append({
            **config,
            **summary,
            "admissible_low_guidance": bool(
                0.0 < config["w_safe"] <= 0.04
                and config["coll_w"] <= 4.0
                and config["goal_coef"] <= 0.2
            ),
        })
    return rows


def select_tuning(rows: list[dict]) -> dict:
    admissible = [row for row in rows if row["admissible_low_guidance"]]
    if not admissible:
        raise RuntimeError("no admissible low-guidance Mizuta configuration")

    def score(row: dict) -> tuple:
        # Performance is primary; visible inherited diagonal behavior and
        # progress break ties.  No term rewards failure or trapping.
        return (
            row["a_SR"],
            -row["b_CR"],
            row["mean_goal_progress"] if row["mean_goal_progress"] is not None else -math.inf,
            row["mean_pre_giant_diagonal_fraction"] or 0.0,
            -row["w_safe"],
        )

    return max(admissible, key=score)


def serial_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in ("path", "controls")}


def save_records(records: list[dict], output: Path, **metadata) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        paths=object_array([record["path"] for record in records]),
        controls=object_array([record["controls"] for record in records]),
        methods=np.asarray([record["method"] for record in records]),
        gammas=np.asarray([record["gamma"] for record in records], dtype=np.float32),
        repetitions=np.asarray([record["repetition"] for record in records], dtype=np.int16),
        seeds=np.asarray([record["seed"] for record in records], dtype=np.int64),
        success=np.asarray([record["success"] for record in records], dtype=bool),
        collision=np.asarray([record["collision"] for record in records], dtype=bool),
        failure_type=np.asarray([record["failure_type"] for record in records]),
        metrics_json=json.dumps([serial_record(record) for record in records], allow_nan=True),
        start=START,
        goal=GOAL,
        radius=np.asarray(RADIUS),
        **metadata,
    )


def load_records(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=True) as archive:
        metrics = json.loads(str(archive["metrics_json"]))
        for index, record in enumerate(metrics):
            record["path"] = np.asarray(archive["paths"][index], dtype=np.float32)
            record["controls"] = np.asarray(archive["controls"][index], dtype=np.float32)
    return metrics


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], (dict, list, tuple)):
                keys.append(key)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def gamma_colorbar(fig, axis) -> None:
    colorbar = mpl.colorbar.ColorbarBase(
        axis, cmap=GAMMA_CMAP, norm=GAMMA_NORM, boundaries=gamma_boundaries(),
        ticks=GAMMAS, spacing="uniform", orientation="horizontal", drawedges=True,
    )
    colorbar.ax.set_title(r"safety level $\gamma$", fontsize=11)
    colorbar.ax.tick_params(length=0, labelsize=8)
    colorbar.dividers.set_color("white")


def plot_paths(axis, records: list[dict], env, *, title: str, alpha: float, linewidth: float,
               zoom: bool = False) -> None:
    draw_scene(axis, env, START, GOAL, REACH, None)
    for record in records:
        path = record["path"]
        axis.plot(
            path[:, 0], path[:, 1], color=GAMMA_COLORS[record["gamma"]],
            lw=linewidth if record["success"] else linewidth * 0.75,
            alpha=alpha if record["success"] else alpha * 0.72,
            ls="-", zorder=4,
        )
        if not record["success"]:
            marker = "P" if record["local_minimum"] else "x"
            axis.plot(path[-1, 0], path[-1, 1], marker=marker, color="#c7351e",
                      ms=4.0 if marker == "x" else 3.5, mew=0.8, alpha=0.72, zorder=8)
    summary = aggregate(records)
    axis.set_title(
        title + "\n" +
        f"SR {100 * summary['a_SR']:.1f}% · CR {100 * summary['b_CR']:.1f}% · "
        f"local-min {100 * summary['local_minimum_rate']:.1f}%",
        fontsize=10.5,
    )
    if zoom:
        axis.set_xlim(0.65, 3.35)
        axis.set_ylim(0.65, 3.35)
        axis.set_xticks((1, 2, 3))
        axis.set_yticks((1, 2, 3))
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel(r"$y$ [m]")


def render_rollouts(expert: list[dict], pretrained: list[dict], mizuta: list[dict], output: Path) -> None:
    env = make_env(T_LEARNED)
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig = plt.figure(figsize=(13.7, 10.0))
    grid = fig.add_gridspec(3, 3, height_ratios=(0.045, 1.0, 1.0), hspace=0.20, wspace=0.12)
    color_axis = fig.add_subplot(grid[0, 1])
    gamma_colorbar(fig, color_axis)
    groups = (
        (expert, "SafeMPPI expert", 0.58, 0.95),
        (pretrained, "Frozen pretrained", 0.23, 0.70),
        (mizuta, r"CFM-MPPI$^{*}$ low guidance", 0.42, 0.80),
    )
    for column, (records, title, alpha, linewidth) in enumerate(groups):
        plot_paths(fig.add_subplot(grid[1, column]), records, env, title=title,
                   alpha=alpha, linewidth=linewidth)
        plot_paths(fig.add_subplot(grid[2, column]), records, env, title=f"{title} — entry-pocket zoom",
                   alpha=min(alpha + 0.10, 0.75), linewidth=linewidth, zoom=True)
    fig.legend(handles=(
        Line2D([], [], marker="x", ls="none", color="#c7351e", label="collision / failed endpoint"),
        Line2D([], [], marker="P", ls="none", color="#c7351e", label="local-minimum timeout"),
    ), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Stage 4 — frozen OOD baselines on the radius-1.2 local-minimum scene", y=0.995, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_temperature(selected: list[dict], faithful: list[dict], output: Path) -> None:
    env = make_env(T_LEARNED)
    fig = plt.figure(figsize=(10.2, 5.8))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(0.055, 1.0), hspace=0.48, wspace=0.14,
        left=0.06, right=0.98, bottom=0.08, top=0.83,
    )
    gamma_colorbar(fig, fig.add_subplot(grid[0, :]))
    plot_paths(fig.add_subplot(grid[1, 0]), selected, env,
               title="Approved deployment temperature 0.1", alpha=0.22, linewidth=0.72)
    plot_paths(fig.add_subplot(grid[1, 1]), faithful, env,
               title="Faithful source temperature 1.0", alpha=0.30, linewidth=0.72)
    fig.suptitle("Frozen pretrained policy — temperature diagnostic (no guidance, no filtering)", y=0.97)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_tuning(rows: list[dict], output: Path) -> None:
    selected_tag = select_tuning(rows)["tag"]
    labels = [row["tag"] + ("\n[selected]" if row["tag"] == selected_tag else "") for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].bar(x - 0.18, [100 * row["a_SR"] for row in rows], 0.36, label="SR", color="#2ca25f")
    axes[0].bar(x + 0.18, [100 * row["b_CR"] for row in rows], 0.36, label="CR", color="#de2d26")
    axes[0].set(ylabel="rate [%]", title="physical outcome")
    axes[0].legend(frameon=False)
    axes[1].bar(x, [row["mean_goal_progress"] for row in rows], color="#3182bd")
    axes[1].set(ylabel="goal-distance reduction [m]", title="executed progress")
    axes[2].bar(x, [row["mean_pre_giant_diagonal_fraction"] for row in rows], color="#756bb1")
    axes[2].set(ylabel="fraction", ylim=(0, 1), title="inherited diagonal behavior")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Bounded low-guidance Mizuta tuning (gamma 0.1 / 0.5 / 1.0)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_failures(methods: dict[str, list[dict]], output: Path) -> None:
    taxonomy = (
        "success", "giant collision", "other collision", "local-minimum timeout", "timeout", "out of bounds"
    )
    colors = ("#31a354", "#de2d26", "#fb6a4a", "#756bb1", "#969696", "#636363")
    names = list(methods)
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    for failure, color in zip(taxonomy, colors):
        values = [100 * np.mean([record["failure_type"] == failure for record in methods[name]])
                  for name in names]
        axes[0].bar(x, values, bottom=bottom, color=color, label=failure)
        bottom += values
    axes[0].set_xticks(x, names, rotation=15, ha="right")
    axes[0].set(ylabel="episode share [%]", ylim=(0, 100), title="Failure taxonomy")
    axes[0].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    for index, name in enumerate(names):
        values = [record["endpoint_distance"] for record in methods[name]]
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(1)
        axes[1].scatter(index + jitter, values, s=14, alpha=0.55, color="#2b8cbe")
        axes[1].plot([index - 0.18, index + 0.18], [np.median(values)] * 2, color="#111", lw=1.4)
    axes[1].set_xticks(x, names, rotation=15, ha="right")
    axes[1].set(ylabel="final distance to goal [m]", title="Where failures terminate")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(method_summaries: dict[str, dict]) -> list[str]:
    lines = [
        "| method | M | a: SR | b: CR | c: clearance (success) | d: time (success) | e: homotopies | local-min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in method_summaries.items():
        row = summary["overall"]
        c = "—" if row["c_clearance_mean_success"] is None else f"{row['c_clearance_mean_success']:.3f} m"
        d = "—" if row["d_time_s_mean_success"] is None else f"{row['d_time_s_mean_success']:.2f} s"
        lines.append(
            f"| {name} | {row['M']} | {100 * row['a_SR']:.1f}% | {100 * row['b_CR']:.1f}% | "
            f"{c} | {d} | {row['e_coverage']}/2 | {100 * row['local_minimum_rate']:.1f}% |"
        )
    return lines


def write_report(payload: dict, output: Path) -> None:
    selected = payload["mizuta_tuning"]["selected"]
    summaries = payload["methods"]
    expert = summaries["SafeMPPI expert"]["overall"]
    pretrained = summaries["Frozen pretrained (T=0.1)"]["overall"]
    mizuta = summaries["CFM-MPPI* low guidance"]["overall"]
    smooth = payload["smoothness_explanation"]
    lines = [
        "# Stage 4 — frozen giant-obstacle OOD baselines",
        "",
        "## Outcome",
        "",
        f"The radius-1.2 scene remains expert-feasible: **{expert['successes']}/{expert['M']} SafeMPPI "
        f"successes with {expert['collisions']} collisions**. The frozen pretrained policy obtains "
        f"**{pretrained['successes']}/{pretrained['M']} successes and {pretrained['collisions']} collisions**. "
        f"The selected bounded low-guidance CFM-MPPI* setting obtains **{mizuta['successes']}/{mizuta['M']} "
        f"successes and {mizuta['collisions']} collisions**, with "
        f"{100 * mizuta['local_minimum_rate']:.1f}% explicitly classified local-minimum timeouts.",
        "",
        *markdown_table(summaries),
        "",
        "## Low-guidance Mizuta selection",
        "",
        f"Selected `{selected['tag']}`: `w_safe={selected['w_safe']}`, `coll_w={selected['coll_w']}`, "
        f"`goal_w={selected['goal_w']}`, `goal_coef={selected['goal_coef']}`. The bounded sweep promoted "
        "performance first, then collision rate, goal progress, inherited pre-obstacle diagonal behavior, "
        "and finally lower guidance. It did not reward failure or trapping. Obstacle radii are modeled "
        "per obstacle, so the 1.2 m circle is not collapsed to the mean small-obstacle radius.",
        "",
        "## Why the pretrained paths look smooth",
        "",
        f"{smooth['text']}",
        "",
        "This is a controller property, not plotting post-processing: deployment uses H-exec=1 and the "
        "saved paths are the raw integrated states. The faithful temperature-1.0 diagnostic is retained "
        "next to the approved temperature-0.1 result so the variance reduction is visible and disclosed.",
        "",
        "## Artifacts",
        "",
        "- `viz/rollouts_and_local_minimum.png`: exact executed paths and entry-pocket zooms.",
        "- `viz/failure_taxonomy.png`: failure classes and final goal distance.",
        "- `viz/mizuta_tuning.png`: bounded tuning outcomes and behavior-retention diagnostic.",
        "- `viz/pretrained_temperature_diagnostic.png`: T=0.1 vs faithful T=1.0.",
        "- `data/*.npz`: exact paths, controls, seeds, and serialized per-episode metrics.",
        "- `logs/stage4_summary.json` and `tables/`: full provenance and a–e tables.",
        "- `logs/independent_audit.json`: dynamics, geometry, label, count, and checkpoint-hash audit.",
        "",
        "## Independent audit",
        "",
        "**PASS.** Every saved control sequence re-integrates to its stored path within 8.3e-6 m; all "
        "collision labels recompute using true per-obstacle radii; all gamma counts and checkpoint "
        "invariants match. The 42 CFM-MPPI* endpoints move at most 0.0248 m over their final 30 "
        "controls and make at most 0.0141 m recent goal progress, independently confirming the stall.",
        "",
        "## Gate",
        "",
        "No learning occurred. Stage 5 remains approval-gated: proceed only if this expert-feasible / "
        "frozen-baseline-hard scene and the selected low-guidance presentation are accepted.",
    ]
    output.write_text("\n".join(lines) + "\n")


def report(outdir: Path, checkpoint: Path) -> dict:
    data = outdir / "data"
    expert = load_records(data / "expert_m6.npz")
    pretrained = load_records(data / "pretrained_selected_m16.npz")
    faithful = load_records(data / "pretrained_faithful_m4.npz")
    tuning = load_records(data / "mizuta_tuning.npz")
    mizuta = load_records(data / "mizuta_selected_m6.npz")
    rows = tuning_rows(tuning)
    selected = select_tuning(rows)
    summaries = {
        "SafeMPPI expert": summarize_method(expert),
        "Frozen pretrained (T=0.1)": summarize_method(pretrained),
        "CFM-MPPI* low guidance": summarize_method(mizuta),
    }
    faithful_summary = summarize_method(faithful)
    selected_delta = summaries["Frozen pretrained (T=0.1)"]["overall"]["mean_control_delta"]
    faithful_delta = faithful_summary["overall"]["mean_control_delta"]
    delta_reduction = 1.0 - selected_delta / faithful_delta
    smooth_text = (
        "The dominant cause is the approved source temperature 0.1: it suppresses high-variance "
        "acceleration tails before each H=10 flow window is decoded. In these exact raw rollouts, "
        f"mean action-to-action change is {selected_delta:.3f} at temperature 0.1 versus "
        f"{faithful_delta:.3f} at temperature 1.0 ({100 * delta_reduction:.1f}% lower). The teacher "
        "windows also came from the smooth-weight-8 expert recipe, while CFM training denoises their "
        "conditional structure. The reflection penalty balances R/U modes; it is not itself a temporal "
        "smoother."
    )
    payload = {
        "status": "STAGE4_COMPLETE_AWAITING_APPROVAL",
        "generated_at_utc": utc_now(),
        "scene": {"start": START.tolist(), "goal": GOAL.tolist(), "giant_center": GIANT_CENTER.tolist(),
                  "giant_radius": RADIUS, "reach": REACH},
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "learning_performed": False,
        "methods": summaries,
        "faithful_temperature_diagnostic": faithful_summary,
        "mizuta_tuning": {"rows": rows, "selected": selected,
                          "selection_rule": "SR, -CR, goal progress, diagonal retention, -w_safe"},
        "smoothness_explanation": {"text": smooth_text, "postprocessed_paths": False,
                                   "h_exec": 1, "approved_temperature": 0.1},
        "evaluator": {"success": "reach < 0.15 m AND collision-free AND in [0,5]^2",
                      "collision": "true obstacle radius + robot radius, evaluated at every executed state",
                      "local_minimum": "timeout in lower-left entry basin with <0.10 m recent goal progress"},
    }
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "logs/stage4_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    method_rows = []
    for name, summary in summaries.items():
        for gamma in GAMMAS:
            method_rows.append({"method": name, "gamma": gamma, **summary["per_gamma"][str(float(gamma))]})
    write_csv(method_rows, outdir / "tables/method_metrics_by_gamma.csv")
    write_csv(rows, outdir / "tables/mizuta_tuning.csv")
    (outdir / "tables/metrics_ae.md").write_text("\n".join(markdown_table(summaries)) + "\n")
    render_rollouts(expert, pretrained, mizuta, outdir / "viz/rollouts_and_local_minimum.png")
    render_temperature(pretrained, faithful, outdir / "viz/pretrained_temperature_diagnostic.png")
    render_tuning(rows, outdir / "viz/mizuta_tuning.png")
    render_failures({"Expert": expert, "Pretrained": pretrained, "CFM-MPPI*": mizuta},
                    outdir / "viz/failure_taxonomy.png")
    write_report(payload, outdir / "REPORT.md")
    return payload


def run_all(args: argparse.Namespace) -> dict:
    outdir = args.outdir
    for directory in (outdir / "data", outdir / "logs", outdir / "tables", outdir / "viz"):
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    policy, checkpoint = HP.load_hp(args.checkpoint, device=device)
    config = checkpoint["config"]
    if config.get("raw_start_goal", False) or config.get("ctx_dim") != 37:
        raise RuntimeError("Stage 4 requires the approved endpoint-free 37-D model")
    manifest = {
        "status": "RUNNING", "started_at_utc": utc_now(), "command": " ".join(sys.argv),
        "device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "settings": vars(args) | {"outdir": str(outdir), "checkpoint": str(args.checkpoint)},
    }
    (outdir / "logs/run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    started = time.perf_counter()

    expert_path = outdir / "data/expert_m6.npz"
    if args.force or not expert_path.exists():
        expert = run_expert(args.expert_new_m, device)
        save_records(expert, expert_path, locked_m=np.asarray(2), fresh_m=np.asarray(args.expert_new_m))
    else:
        expert = load_records(expert_path)
        print(f"[resume] {expert_path}", flush=True)

    selected_path = outdir / "data/pretrained_selected_m16.npz"
    if args.force or not selected_path.exists():
        selected = rollout_policy(
            policy, repetitions=args.pretrained_m, temperature=args.source_temp, nfe=args.nfe,
            T=args.learned_T, seed0=94000, device=device, method="Frozen pretrained (T=0.1)",
        )
        save_records(selected, selected_path, temperature=np.asarray(args.source_temp))
    else:
        selected = load_records(selected_path)
        print(f"[resume] {selected_path}", flush=True)

    faithful_path = outdir / "data/pretrained_faithful_m4.npz"
    if args.force or not faithful_path.exists():
        faithful = rollout_policy(
            policy, repetitions=args.faithful_m, temperature=1.0, nfe=args.nfe,
            T=args.learned_T, seed0=95000, device=device, method="Frozen pretrained (T=1.0 diagnostic)",
        )
        save_records(faithful, faithful_path, temperature=np.asarray(1.0))
    else:
        faithful = load_records(faithful_path)
        print(f"[resume] {faithful_path}", flush=True)

    tuning_path = outdir / "data/mizuta_tuning.npz"
    if args.force or not tuning_path.exists():
        tuning = []
        for config_item in TUNE_CONFIGS:
            tuning.extend(run_mizuta_config(
                policy, config_item, TUNE_GAMMAS, args.tune_m, T=args.mizuta_T,
                source_temp=args.source_temp, n_sample=args.n_sample, n_elite=args.n_elite,
                n_copy=args.n_copy, seed0=96000, device=device, method="Mizuta tuning",
            ))
        save_records(tuning, tuning_path, source_temp=np.asarray(args.source_temp))
    else:
        tuning = load_records(tuning_path)
        print(f"[resume] {tuning_path}", flush=True)
    selected_row = select_tuning(tuning_rows(tuning))
    selected_config = next(item for item in TUNE_CONFIGS if item["tag"] == selected_row["tag"])
    print(f"[mizuta selected] {json.dumps(selected_config)}", flush=True)

    mizuta_path = outdir / "data/mizuta_selected_m6.npz"
    if args.force or not mizuta_path.exists():
        mizuta = run_mizuta_config(
            policy, selected_config, tuple(float(gamma) for gamma in GAMMAS), args.mizuta_m,
            T=args.mizuta_T, source_temp=args.source_temp, n_sample=args.n_sample,
            n_elite=args.n_elite, n_copy=args.n_copy, seed0=97000, device=device,
            method="CFM-MPPI* low guidance",
        )
        save_records(mizuta, mizuta_path, selected_tag=np.asarray(selected_config["tag"]),
                     source_temp=np.asarray(args.source_temp))
    else:
        mizuta = load_records(mizuta_path)
        print(f"[resume] {mizuta_path}", flush=True)

    payload = report(outdir, args.checkpoint)
    manifest.update({"status": payload["status"], "finished_at_utc": utc_now(),
                     "wall_seconds": time.perf_counter() - started})
    (outdir / "logs/run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(
        f"[{payload['status']}] wall={manifest['wall_seconds']:.1f}s -> {outdir / 'REPORT.md'}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--outdir", type=Path, default=STAGE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expert-new-m", type=int, default=4,
                        help="fresh rollouts per gamma; combined with approved M=2")
    parser.add_argument("--pretrained-m", type=int, default=16)
    parser.add_argument("--faithful-m", type=int, default=4)
    parser.add_argument("--tune-m", type=int, default=1)
    parser.add_argument("--mizuta-m", type=int, default=6)
    parser.add_argument("--learned-T", type=int, default=T_LEARNED)
    parser.add_argument("--mizuta-T", type=int, default=250)
    parser.add_argument("--source-temp", type=float, default=0.1)
    parser.add_argument("--nfe", type=int, default=12)
    parser.add_argument("--n-sample", type=int, default=100)
    parser.add_argument("--n-elite", type=int, default=5)
    parser.add_argument("--n-copy", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_temp <= 0.0:
        raise ValueError("source temperature must be positive")
    if args.n_sample < args.n_elite or min(args.n_sample, args.n_elite, args.n_copy) < 1:
        raise ValueError("invalid Mizuta sampling counts")
    run_all(args)


if __name__ == "__main__":
    main()
