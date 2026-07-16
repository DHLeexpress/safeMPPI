#!/usr/bin/env python3
"""Stage 2B: geometrically balanced fixed-pair ID demonstrations.

The ordinary symmetric 4x4 stadium, start=(0.5,0.5), and goal=(4.5,4.5)
are fixed.  ``generate`` builds a resumable SafeMPPI candidate census with the
approved Stage-2A anti-retreat recipe.  ``build`` (enabled only after inspecting
the census) mirror-balances monotone four-right/four-up crossing signatures,
assigns equal loss mass per selected trajectory, and writes H=10 policy data.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
import grid_feats as GF  # noqa: E402
import grid_metrics as GM  # noqa: E402
import grid_metrics2 as GM2  # noqa: E402
import grid_rollout as GR  # noqa: E402
import grid_scene as GS  # noqa: E402

from giant_obstacle_ood.stage1_geometry_sweep import (  # noqa: E402
    draw_scene,
    make_scene,
    object_array,
)
from giant_obstacle_ood.stage1b_smooth_expert import GOAL, START  # noqa: E402
from giant_obstacle_ood.stage1c_window_polytope import score_window  # noqa: E402
from giant_obstacle_ood.stage2a_retreat_penalty import (  # noqa: E402
    DEFAULT_CAP,
    DEFAULT_SCALE,
    SMOOTH_WEIGHT,
    rollout as expert_rollout,
)
from viz_style import (  # noqa: E402
    GAMMAS,
    GAMMA_CMAP,
    GAMMA_COLORS,
    GAMMA_NORM,
    gamma_boundaries,
)


STAGE = HERE / "stage_results/02b_balanced_id"
CANDIDATE_DIR = STAGE / "candidates"
RETREAT_WEIGHT = 1.0
THRESHOLDS = (1.0, 2.0, 3.0, 4.0)
ALL_SIGNATURES = tuple(
    "".join("R" if index in right else "U" for index in range(8))
    for right in itertools.combinations(range(8), 4)
)
SIGNATURE_TO_ID = {word: index for index, word in enumerate(ALL_SIGNATURES)}


def gamma_tag(gamma: float) -> str:
    return str(float(gamma))


def make_id_env(max_steps: int):
    env = make_scene(None, START, GOAL)
    env.T = int(max_steps)
    return env


def mirror_word(word: str) -> str:
    return word.translate(str.maketrans({"R": "U", "U": "R"}))


def canonical_pair(word: str) -> str:
    mirrored = mirror_word(word)
    return min(word, mirrored)


def crossing_signature(path: np.ndarray, drawdown_tol: float = 0.35,
                       tie_tol: float = 1e-4) -> dict[str, Any]:
    """First-crossing R/U word with an explicit monotonicity audit.

    Crossing times are linearly interpolated inside each executed segment, so
    checking x before y cannot create a systematic R-first bias.  Small local
    reversals are allowed, but a drawdown greater than ``drawdown_tol`` makes
    the path ineligible for the monotone geometric dataset.
    """
    xy = np.asarray(path, dtype=float)
    events: list[tuple[float, str, float]] = []
    missing = []
    for axis, letter in ((0, "R"), (1, "U")):
        values = xy[:, axis]
        for threshold in THRESHOLDS:
            found = None
            for index in range(1, len(values)):
                if values[index - 1] < threshold <= values[index]:
                    denominator = values[index] - values[index - 1]
                    fraction = ((threshold - values[index - 1]) / denominator
                                if abs(denominator) > 1e-12 else 1.0)
                    found = float(index - 1 + fraction)
                    break
            if found is None:
                missing.append(f"{letter}{threshold:g}")
            else:
                events.append((found, letter, threshold))
    events.sort(key=lambda item: item[0])
    gaps = np.diff([event[0] for event in events]) if len(events) > 1 else np.asarray([])
    minimum_gap = float(gaps.min()) if len(gaps) else math.inf
    tie_count = int(np.count_nonzero(np.abs(gaps) < tie_tol))
    max_drawdown_x = float(np.max(np.maximum.accumulate(xy[:, 0]) - xy[:, 0]))
    max_drawdown_y = float(np.max(np.maximum.accumulate(xy[:, 1]) - xy[:, 1]))
    monotone = max(max_drawdown_x, max_drawdown_y) <= drawdown_tol
    word = "".join(event[1] for event in events) if len(events) == 8 else ""
    valid_word = word in SIGNATURE_TO_ID
    eligible = bool(not missing and valid_word and monotone and tie_count == 0)
    return {
        "signature": word if valid_word else "",
        "signature_id": SIGNATURE_TO_ID.get(word, -1),
        "mirror_signature": mirror_word(word) if valid_word else "",
        "pair_signature": canonical_pair(word) if valid_word else "",
        "signature_eligible": eligible,
        "missing_crossings": missing,
        "tie_count": tie_count,
        "minimum_crossing_gap_steps": minimum_gap,
        "max_drawdown_x": max_drawdown_x,
        "max_drawdown_y": max_drawdown_y,
    }


def diagonal_side_word(path: np.ndarray) -> str:
    """Which side of y=x the path uses near each diagonal obstacle."""
    xy = np.asarray(path, dtype=float)
    letters = []
    for coordinate in THRESHOLDS:
        center = np.asarray((coordinate, coordinate))
        index = int(np.linalg.norm(xy - center[None, :], axis=1).argmin())
        letters.append("A" if xy[index, 1] >= xy[index, 0] else "B")
    return "".join(letters)


def serial_record(result: dict) -> dict:
    omitted = {"path", "states", "controls", "goal_distances"}
    return {key: value for key, value in result.items() if key not in omitted}


def enrich(result: dict) -> dict:
    geometry = crossing_signature(result["path"])
    result.update(geometry)
    result["diagonal_side_word"] = diagonal_side_word(result["path"])
    result["candidate_eligible"] = bool(result["success"] and geometry["signature_eligible"])
    return result


def candidate_path(gamma: float, directory: Path = CANDIDATE_DIR) -> Path:
    return directory / f"candidates_g{gamma_tag(gamma)}.npz"


def save_candidates(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gamma=np.asarray([r["gamma"] for r in records], dtype=np.float32),
        seed=np.asarray([r["seed"] for r in records], dtype=np.int64),
        success=np.asarray([r["success"] for r in records]),
        collision=np.asarray([r["collision"] for r in records]),
        in_taskspace=np.asarray([r["in_taskspace"] for r in records]),
        candidate_eligible=np.asarray([r["candidate_eligible"] for r in records]),
        steps=np.asarray([r["steps"] for r in records], dtype=np.int32),
        time_s=np.asarray([r["time_s"] for r in records], dtype=np.float32),
        clearance_mean=np.asarray([r["clearance_mean"] for r in records], dtype=np.float32),
        min_clearance=np.asarray([r["min_clearance"] for r in records], dtype=np.float32),
        path_length=np.asarray([r["path_length"] for r in records], dtype=np.float32),
        retreat_total=np.asarray([r["retreat_distance_total"] for r in records], dtype=np.float32),
        retreat_fraction=np.asarray([r["retreat_step_fraction_1mm"] for r in records], dtype=np.float32),
        direction_switches=np.asarray([r["radial_direction_switches_5mm"] for r in records], dtype=np.int32),
        signature=np.asarray([r["signature"] for r in records], dtype="U8"),
        signature_id=np.asarray([r["signature_id"] for r in records], dtype=np.int16),
        pair_signature=np.asarray([r["pair_signature"] for r in records], dtype="U8"),
        diagonal_side_word=np.asarray([r["diagonal_side_word"] for r in records], dtype="U4"),
        tie_count=np.asarray([r["tie_count"] for r in records], dtype=np.int16),
        min_crossing_gap=np.asarray([r["minimum_crossing_gap_steps"] for r in records], dtype=np.float32),
        max_drawdown_x=np.asarray([r["max_drawdown_x"] for r in records], dtype=np.float32),
        max_drawdown_y=np.asarray([r["max_drawdown_y"] for r in records], dtype=np.float32),
        paths=object_array([r["path"] for r in records]),
        states=object_array([r["states"] for r in records]),
        controls=object_array([r["controls"] for r in records]),
    )


def load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    archive = np.load(path, allow_pickle=True)
    records = []
    for index in range(len(archive["seed"])):
        records.append({
            "gamma": float(archive["gamma"][index]),
            "seed": int(archive["seed"][index]),
            "success": bool(archive["success"][index]),
            "collision": bool(archive["collision"][index]),
            "in_taskspace": bool(archive["in_taskspace"][index]),
            "candidate_eligible": bool(archive["candidate_eligible"][index]),
            "steps": int(archive["steps"][index]),
            "time_s": float(archive["time_s"][index]),
            "clearance_mean": float(archive["clearance_mean"][index]),
            "min_clearance": float(archive["min_clearance"][index]),
            "path_length": float(archive["path_length"][index]),
            "retreat_distance_total": float(archive["retreat_total"][index]),
            "retreat_step_fraction_1mm": float(archive["retreat_fraction"][index]),
            "radial_direction_switches_5mm": int(archive["direction_switches"][index]),
            "signature": str(archive["signature"][index]),
            "signature_id": int(archive["signature_id"][index]),
            "mirror_signature": mirror_word(str(archive["signature"][index])) if archive["signature"][index] else "",
            "pair_signature": str(archive["pair_signature"][index]),
            "diagonal_side_word": str(archive["diagonal_side_word"][index]),
            "tie_count": int(archive["tie_count"][index]),
            "minimum_crossing_gap_steps": float(archive["min_crossing_gap"][index]),
            "max_drawdown_x": float(archive["max_drawdown_x"][index]),
            "max_drawdown_y": float(archive["max_drawdown_y"][index]),
            "path": np.asarray(archive["paths"][index], dtype=np.float32),
            "states": np.asarray(archive["states"][index], dtype=np.float32),
            "controls": np.asarray(archive["controls"][index], dtype=np.float32),
        })
    return records


def candidate_summary(records: list[dict], gamma: float) -> dict:
    eligible = [r for r in records if r["candidate_eligible"]]
    counts = Counter(r["signature"] for r in eligible)
    pair_counts = {}
    for pair in sorted({canonical_pair(word) for word in counts}):
        mirror = mirror_word(pair)
        pair_counts[pair] = {
            "word": counts.get(pair, 0),
            "mirror": counts.get(mirror, 0),
            "matched_capacity_each": min(counts.get(pair, 0), counts.get(mirror, 0)),
        }
    return {
        "gamma": float(gamma),
        "candidates": len(records),
        "successes": int(sum(r["success"] for r in records)),
        "collisions": int(sum(r["collision"] for r in records)),
        "eligible": len(eligible),
        "unique_signatures": len(counts),
        "unique_mirror_pairs": int(sum(value["matched_capacity_each"] > 0 for value in pair_counts.values())),
        "total_exact_mirror_balanced_capacity": int(2 * sum(value["matched_capacity_each"] for value in pair_counts.values())),
        "signature_counts": dict(sorted(counts.items())),
        "mirror_pair_counts": pair_counts,
        "side_word_counts": dict(sorted(Counter(r["diagonal_side_word"] for r in eligible).items())),
        "mean_time_s_success": float(np.mean([r["time_s"] for r in records if r["success"]])) if any(r["success"] for r in records) else math.nan,
        "mean_retreat_m_success": float(np.mean([r["retreat_distance_total"] for r in records if r["success"]])) if any(r["success"] for r in records) else math.nan,
    }


def render_candidate_census(env, all_records: dict[float, list[dict]], output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    matrix = np.zeros((len(GAMMAS), len(ALL_SIGNATURES)), dtype=int)
    for row, gamma in enumerate(GAMMAS):
        counts = Counter(r["signature"] for r in all_records[gamma] if r["candidate_eligible"])
        matrix[row] = [counts.get(word, 0) for word in ALL_SIGNATURES]
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 7.8), gridspec_kw={"height_ratios": [1.25, 1.0]})
    vmax = max(int(matrix.max()), 1)
    image = axes[0].imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    axes[0].set_yticks(range(len(GAMMAS)), [rf"$\gamma={gamma:g}$" for gamma in GAMMAS])
    axes[0].set_xticks(range(len(ALL_SIGNATURES)), ALL_SIGNATURES, rotation=90, fontsize=6)
    axes[0].set_title("(A) eligible monotone up/right signature counts (all 70 possible words)")
    fig.colorbar(image, ax=axes[0], pad=0.01, label="candidate trajectories")

    x = np.arange(len(GAMMAS))
    summaries = [candidate_summary(all_records[gamma], gamma) for gamma in GAMMAS]
    axes[1].bar(x - 0.25, [row["candidates"] for row in summaries], 0.25, label="candidates", color="0.75")
    axes[1].bar(x, [row["eligible"] for row in summaries], 0.25, label="eligible", color="#4c78a8")
    axes[1].bar(x + 0.25, [row["total_exact_mirror_balanced_capacity"] for row in summaries], 0.25,
                label="exact mirror-balanced capacity", color="#59a14f")
    axes[1].set_xticks(x, [f"{gamma:g}" for gamma in GAMMAS])
    axes[1].set_xlabel(r"safety level $\gamma$")
    axes[1].set_ylabel("trajectories")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, ncol=3)
    axes[1].set_title("(B) physical/geometry yield and no-augmentation balancing capacity")
    fig.suptitle("Stage 2B ID SafeMPPI geometric census", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_candidate_paths(env, all_records: dict[float, list[dict]], reach: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.8))
    for axis, gamma in zip(axes.ravel()[:7], GAMMAS):
        draw_scene(axis, env, START, GOAL, reach, None)
        records = all_records[gamma]
        for result in records:
            path = result["path"]
            color = GAMMA_COLORS[gamma] if result["candidate_eligible"] else "#cc3311"
            axis.plot(path[:, 0], path[:, 1], color=color, lw=0.75,
                      alpha=0.42 if result["candidate_eligible"] else 0.8, zorder=4)
        summary = candidate_summary(records, gamma)
        axis.set_title(
            rf"$\gamma={gamma:g}$: {summary['successes']}/{summary['candidates']} success" "\n"
            f"{summary['unique_signatures']} words; mirror capacity {summary['total_exact_mirror_balanced_capacity']}",
            fontsize=9,
        )
    info = axes.ravel()[7]
    info.axis("off")
    info.text(
        0.03, 0.95,
        "ID candidate census\n\n"
        f"start = ({START[0]:.1f}, {START[1]:.1f})\n"
        f"goal = ({GOAL[0]:.1f}, {GOAL[1]:.1f})\n"
        f"smooth weight = {SMOOTH_WEIGHT:g}\n"
        f"retreat weight = {RETREAT_WEIGHT:g}\n"
        f"retreat scale = {DEFAULT_SCALE:g} m\n\n"
        "color = eligible\nred = failed geometry/physics",
        va="top", fontsize=11,
    )
    fig.suptitle("Ordinary 4x4 ID stadium — fixed-pair SafeMPPI candidates", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def candidate_quality(record: dict) -> float:
    """Prefer performable examples within a fixed gamma/signature stratum."""
    return float(
        record["retreat_distance_total"]
        + 0.005 * record["time_s"]
        + 0.002 * record["radial_direction_switches_5mm"]
    )


def select_exact_mirror_balanced(records: list[dict], n_pairs: int, quota: int) -> tuple[list[dict], dict]:
    """Select equal real-rollout counts for each word and its R/U mirror."""
    eligible = [record for record in records if record["candidate_eligible"]]
    by_word: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_word[record["signature"]].append(record)
    pair_rows = []
    for pair in sorted({canonical_pair(word) for word in by_word}):
        mirror = mirror_word(pair)
        capacity = min(len(by_word.get(pair, [])), len(by_word.get(mirror, [])))
        pair_rows.append({"pair": pair, "mirror": mirror, "capacity_each": capacity})
    supported = sorted(pair_rows, key=lambda row: (-row["capacity_each"], row["pair"]))
    retained = supported[:n_pairs]
    if len(retained) != n_pairs or any(row["capacity_each"] < quota for row in retained):
        raise RuntimeError(
            f"insufficient exact mirror support: requested {n_pairs} pairs x {quota} each; "
            f"capacities={[row['capacity_each'] for row in retained]}"
        )
    selected = []
    selected_counts: Counter[str] = Counter()
    quality_rows = []
    for pair_rank, row in enumerate(retained):
        for reflection_rank, word in enumerate((row["pair"], row["mirror"])):
            pool = sorted(by_word[word], key=lambda record: (candidate_quality(record), record["seed"]))
            chosen = pool[:quota]
            for quality_rank, record in enumerate(chosen):
                copied = dict(record)
                copied.update({
                    "selection_pair_rank": pair_rank,
                    "selection_reflection_rank": reflection_rank,
                    "selection_quality_rank": quality_rank,
                    "quality_score": candidate_quality(record),
                })
                selected.append(copied)
                selected_counts[word] += 1
                quality_rows.append(candidate_quality(record))
    expected = n_pairs * 2 * quota
    if len(selected) != expected:
        raise RuntimeError(f"selected {len(selected)} trajectories, expected {expected}")
    for word, count in selected_counts.items():
        if count != quota or selected_counts[mirror_word(word)] != count:
            raise RuntimeError(f"mirror imbalance for {word}: {count} vs {selected_counts[mirror_word(word)]}")
    selected.sort(key=lambda record: (record["selection_pair_rank"],
                                      record["selection_reflection_rank"], record["seed"]))
    audit = {
        "eligible_candidates": len(eligible),
        "candidate_unique_signatures": len(by_word),
        "available_pairs": supported,
        "retained_pairs": retained,
        "n_retained_pairs": n_pairs,
        "quota_per_signature": quota,
        "selected_trajectories": len(selected),
        "selected_signature_counts": dict(sorted(selected_counts.items())),
        "exact_reflection_count_residual_max": max(
            abs(selected_counts[word] - selected_counts[mirror_word(word)]) for word in selected_counts
        ),
        "quality_score_mean": float(np.mean(quality_rows)),
        "selected_seed_list": [int(record["seed"]) for record in selected],
    }
    return selected, audit


def uniform_window_indices(n_controls: int, count: int) -> np.ndarray:
    """Return equal-mass starts whose complete H-step targets were executed.

    Terminal padding is inappropriate for expert supervision: repeating the last
    command can create a synthetic continuation that the successful expert never
    executed.  Sampling only through ``n_controls - H`` keeps every target an
    observed, coherent SafeMPPI horizon while preserving the exact per-path quota.
    """
    n_full_windows = n_controls - GF.H_PRED + 1
    if n_full_windows < count:
        raise RuntimeError(
            f"trajectory has {n_full_windows} complete H={GF.H_PRED} windows, "
            f"fewer than equal-mass quota {count}"
        )
    indices = np.rint(np.linspace(0, n_controls - GF.H_PRED, count)).astype(np.int32)
    if len(np.unique(indices)) != count:
        raise RuntimeError("uniform window sampling produced duplicate indices")
    return indices


def build_gamma_dataset(env, selected: list[dict], gamma: float, gamma_index: int,
                        windows_per_trajectory: int, output: Path) -> tuple[dict, list[dict]]:
    """Featurize an equal number of H=10 samples from every selected path."""
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal = env.goal.detach().cpu().numpy()
    grids: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context_positions: list[np.ndarray] = []
    window_steps: list[int] = []
    window_seeds: list[int] = []
    window_pair_indices: list[int] = []
    window_trajectory_ids: list[int] = []
    window_signature_ids: list[int] = []
    window_pair_ranks: list[int] = []
    masks: dict[str, list] = {key: [] for key in (
        "padded", "taskspace", "progress", "socp", "joint_valid2",
        "physical_collision_free", "nominal_exists", "nominal_certificate",
        "min_clearance", "nominal_residual",
    )}
    mask_rows: list[dict] = []
    started = time.perf_counter()
    for local_id, record in enumerate(selected):
        states = np.asarray(record["states"], dtype=np.float32)
        controls = np.asarray(record["controls"], dtype=np.float32)
        indices = uniform_window_indices(len(controls), windows_per_trajectory)
        trajectory_id = int(gamma_index * 1_000_000 + record["seed"])
        for step in indices:
            step = int(step)
            state = states[step]
            target = controls[step:step + GF.H_PRED]
            padded = len(target) < GF.H_PRED
            if padded:
                raise RuntimeError(
                    f"internal error: selected incomplete expert window at step={step}, "
                    f"controls={len(controls)}, H={GF.H_PRED}"
                )
            target = np.asarray(target, dtype=np.float32)
            planned = GR.window_positions(state, target, float(env.dt))
            segment = np.vstack((state[:2], planned))
            scored = score_window(
                state,
                segment,
                env,
                gamma,
                padded,
                trajectory_id,
                step,
            )
            scored.update({
                "seed": int(record["seed"]),
                "signature": record["signature"],
                "signature_id": int(record["signature_id"]),
                "selection_pair_rank": int(record["selection_pair_rank"]),
            })
            grids.append(GF.axis_grid(state[:2], obstacles, rr))
            lows.append(GF.low5(state, goal, gamma))
            histories.append(GF.hist_pad(controls[max(0, step - GF.K_HIST):step], GF.K_HIST))
            targets.append(target)
            context_positions.append(state[:2].copy())
            window_steps.append(step)
            window_seeds.append(int(record["seed"]))
            window_pair_indices.append(int(record["seed"]))
            window_trajectory_ids.append(trajectory_id)
            window_signature_ids.append(int(record["signature_id"]))
            window_pair_ranks.append(int(record["selection_pair_rank"]))
            for key in masks:
                masks[key].append(scored[key])
            mask_rows.append(scored)
        print(
            f"[features gamma={gamma:g}] trajectory {local_id + 1}/{len(selected)} "
            f"seed={record['seed']} word={record['signature']} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    expected = len(selected) * windows_per_trajectory
    if len(grids) != expected:
        raise RuntimeError(f"built {len(grids)} windows, expected {expected}")
    payload = {
        "grid": torch.from_numpy(np.asarray(grids, dtype=np.float32)),
        "low5": torch.from_numpy(np.asarray(lows, dtype=np.float32)),
        "hist": torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        "U": torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        "window_starts": torch.from_numpy(np.repeat(START[None, :], expected, axis=0).astype(np.float32)),
        "window_goals": torch.from_numpy(np.repeat(GOAL[None, :], expected, axis=0).astype(np.float32)),
        "window_pair_indices": torch.tensor(window_pair_indices, dtype=torch.long),
        "window_trajectory_ids": torch.tensor(window_trajectory_ids, dtype=torch.long),
        "window_steps": torch.tensor(window_steps, dtype=torch.int32),
        "window_seeds": torch.tensor(window_seeds, dtype=torch.long),
        "window_signature_ids": torch.tensor(window_signature_ids, dtype=torch.int16),
        "window_mirror_pair_ranks": torch.tensor(window_pair_ranks, dtype=torch.int8),
        "window_context_positions": torch.from_numpy(np.asarray(context_positions, dtype=np.float32)),
        "padded_mask": torch.tensor(masks["padded"], dtype=torch.bool),
        "taskspace_mask": torch.tensor(masks["taskspace"], dtype=torch.bool),
        "progress_mask": torch.tensor(masks["progress"], dtype=torch.bool),
        "socp_mask": torch.tensor(masks["socp"], dtype=torch.bool),
        "joint_valid2_mask": torch.tensor(masks["joint_valid2"], dtype=torch.bool),
        "physical_collision_free_mask": torch.tensor(masks["physical_collision_free"], dtype=torch.bool),
        "nominal_exists_mask": torch.tensor(masks["nominal_exists"], dtype=torch.bool),
        "nominal_certificate_mask": torch.tensor(masks["nominal_certificate"], dtype=torch.bool),
        "window_min_clearance": torch.tensor(masks["min_clearance"], dtype=torch.float32),
        "nominal_residual": torch.tensor(masks["nominal_residual"], dtype=torch.float32),
        "gamma": float(gamma),
        "n_traj": len(selected),
        "n_candidates": 96,
        "windows_per_trajectory": windows_per_trajectory,
        "start": torch.from_numpy(START.copy()),
        "goal": torch.from_numpy(GOAL.copy()),
        "trajectory_seeds": torch.tensor([record["seed"] for record in selected], dtype=torch.long),
        "trajectory_signature_ids": torch.tensor([record["signature_id"] for record in selected], dtype=torch.int16),
        "trajectory_signatures": [record["signature"] for record in selected],
        "signature_vocabulary": list(ALL_SIGNATURES),
        "schema_version": "giant_ood_id_balanced_v2_full_horizon",
        "balance": {
            "actual_expert_rollouts_only": True,
            "synthetic_reflections": 0,
            "mirror_operation": "R<->U",
            "equal_windows_per_trajectory": True,
            "complete_executed_horizons_only": True,
            "terminal_padding": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload, mask_rows


def selected_metrics(selected: list[dict], gamma: float, audit: dict) -> dict:
    counts = Counter(record["signature"] for record in selected)
    return {
        "gamma": float(gamma),
        "selected": len(selected),
        "physical_successes": int(sum(record["success"] for record in selected)),
        "collisions": int(sum(record["collision"] for record in selected)),
        "unique_signatures": len(counts),
        "signature_counts": dict(sorted(counts.items())),
        "mirror_residual_max": audit["exact_reflection_count_residual_max"],
        "mean_time_s": float(np.mean([record["time_s"] for record in selected])),
        "mean_path_length_m": float(np.mean([record["path_length"] for record in selected])),
        "mean_clearance_m": float(np.mean([record["clearance_mean"] for record in selected])),
        "mean_retreat_m": float(np.mean([record["retreat_distance_total"] for record in selected])),
        "mean_direction_switches": float(np.mean([record["radial_direction_switches_5mm"] for record in selected])),
    }


def window_metrics(payload: dict, gamma: float) -> dict:
    count = len(payload["grid"])
    result = {"gamma": float(gamma), "windows": count}
    for output, key in (
        ("taskspace", "taskspace_mask"),
        ("progress", "progress_mask"),
        ("socp", "socp_mask"),
        ("joint_valid2", "joint_valid2_mask"),
        ("physical", "physical_collision_free_mask"),
        ("nominal_exists", "nominal_exists_mask"),
        ("nominal_certificate", "nominal_certificate_mask"),
    ):
        values = payload[key]
        result[f"{output}_count"] = int(values.sum())
        result[f"{output}_rate"] = float(values.float().mean())
    result["padded_count"] = int(payload["padded_mask"].sum())
    result["min_clearance_m"] = float(payload["window_min_clearance"].min())
    return result


def render_balanced_paths(env, selected_by_gamma: dict[float, list[dict]], reach: float, output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.8))
    for axis, gamma in zip(axes.ravel()[:7], GAMMAS):
        draw_scene(axis, env, START, GOAL, reach, None)
        records = selected_by_gamma[gamma]
        for result in records:
            path = result["path"]
            linestyle = "-" if result["signature"] == canonical_pair(result["signature"]) else "--"
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=1.05,
                      ls=linestyle, alpha=0.62, zorder=4)
        axis.set_title(rf"$\gamma={gamma:g}$: 24 paths, 6 words $\times$ 4", fontsize=9)
    info = axes.ravel()[7]
    info.axis("off")
    info.text(
        0.03, 0.95,
        "Balanced ID demonstrations\n\n"
        "3 mirror pairs / gamma\n"
        "4 real paths / word\n"
        "24 paths / gamma\n"
        "64 windows / path\n\n"
        "solid/dashed = R/U mirrors\n"
        "no synthetic reflection",
        va="top", fontsize=11,
    )
    fig.suptitle("Geometrically balanced SafeMPPI demonstrations — ordinary 4x4 ID stadium", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_all_gamma_overlay(env, selected_by_gamma: dict[float, list[dict]], reach: float,
                             output: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig = plt.figure(figsize=(16.0, 13.5))
    grid = fig.add_gridspec(2, 7, height_ratios=(3.45, 1.0), hspace=0.18, wspace=0.08)
    main = fig.add_subplot(grid[0, 1:6])
    draw_scene(main, env, START, GOAL, reach, None)
    for gamma in GAMMAS:
        for result in selected_by_gamma[gamma]:
            path = result["path"]
            main.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=0.85, alpha=0.24, zorder=4)
    main.set_title("168 balanced expert trajectories (24 per safety level)", fontsize=13)
    for column, gamma in enumerate(GAMMAS):
        axis = fig.add_subplot(grid[1, column])
        draw_scene(axis, env, START, GOAL, reach, None)
        for result in selected_by_gamma[gamma]:
            path = result["path"]
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=0.72, alpha=0.42, zorder=4)
        axis.set_title(rf"$\gamma={gamma:g}$", color=GAMMA_COLORS[gamma], fontsize=10)
    color_axis = fig.add_axes((0.34, 0.92, 0.32, 0.022))
    colorbar = mpl.colorbar.ColorbarBase(
        color_axis, cmap=GAMMA_CMAP, norm=GAMMA_NORM, boundaries=gamma_boundaries(),
        ticks=GAMMAS, spacing="uniform", orientation="horizontal", drawedges=True,
    )
    colorbar.ax.set_title(r"safety level $\gamma$", fontsize=10, pad=2)
    colorbar.ax.tick_params(length=0, labelsize=8)
    colorbar.dividers.set_color("white")
    fig.suptitle("Fixed-pair ID expert data — uniform up/right geometry", fontsize=16, y=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_signature_balance(all_records: dict[float, list[dict]],
                             selected_by_gamma: dict[float, list[dict]], output: Path) -> None:
    raw = np.zeros((len(GAMMAS), len(ALL_SIGNATURES)), dtype=int)
    chosen = np.zeros_like(raw)
    for row, gamma in enumerate(GAMMAS):
        raw_counts = Counter(r["signature"] for r in all_records[gamma] if r["candidate_eligible"])
        chosen_counts = Counter(r["signature"] for r in selected_by_gamma[gamma])
        raw[row] = [raw_counts.get(word, 0) for word in ALL_SIGNATURES]
        chosen[row] = [chosen_counts.get(word, 0) for word in ALL_SIGNATURES]
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 7.8))
    for axis, matrix, title, cmap in (
        (axes[0], raw, "(A) raw successful candidate frequency", "Blues"),
        (axes[1], chosen, "(B) selected: exactly four per retained word and its mirror", "Greens"),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(int(matrix.max()), 1))
        axis.set_yticks(range(len(GAMMAS)), [rf"$\gamma={gamma:g}$" for gamma in GAMMAS])
        axis.set_xticks(range(len(ALL_SIGNATURES)), ALL_SIGNATURES, rotation=90, fontsize=6)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, pad=0.01, label="trajectories")
    fig.suptitle("Geometric de-biasing audit: raw SafeMPPI frequency vs training set", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_window_validity(rows: list[dict], output: Path) -> None:
    gamma = np.asarray([row["gamma"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    keys = ("taskspace", "progress", "socp", "joint_valid2")
    labels = ("task space", "progress", "SOCP", "joint valid2")
    offsets = np.linspace(-0.024, 0.024, len(keys))
    for key, label, offset in zip(keys, labels, offsets):
        axes[0].bar(gamma + offset, [row[f"{key}_rate"] for row in rows], width=0.015, label=label)
    axes[0].set(xlabel=r"safety level $\gamma$", ylabel="pass rate", ylim=(0, 1.05),
                title="(A) selected H=10 training windows")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for key, label, marker in (
        ("physical", "physical collision-free", "o"),
        ("nominal_exists", "nominal exists", "s"),
        ("nominal_certificate", "nominal schedule", "^"),
        ("socp", "fitted verifier SOCP", "*"),
    ):
        axes[1].plot(gamma, [row[f"{key}_rate"] for row in rows], marker=marker, label=label)
    axes[1].set(xlabel=r"safety level $\gamma$", ylabel="rate", ylim=(0, 1.05),
                title="(B) physical and certificate audit")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(gamma)
        axis.grid(alpha=0.25)
    fig.suptitle("Balanced ID dataset validity masks (reported, not used to relabel expert success)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def run_generate(args: argparse.Namespace, device: torch.device) -> None:
    env = make_id_env(args.max_steps)
    all_records: dict[float, list[dict]] = {}
    started = time.perf_counter()
    for gamma in GAMMAS:
        path = candidate_path(gamma, args.candidate_dir)
        records = load_candidates(path)
        existing = {record["seed"] for record in records}
        requested = list(range(args.seed0, args.seed0 + args.seeds_per_gamma))
        for ordinal, seed in enumerate(requested, start=1):
            if seed in existing:
                continue
            result = expert_rollout(
                env,
                gamma,
                seed,
                RETREAT_WEIGHT,
                DEFAULT_SCALE,
                DEFAULT_CAP,
                args.reach,
                device,
                certify=False,
            )
            enrich(result)
            records.append(result)
            existing.add(seed)
            if len(existing.intersection(requested)) % args.progress_every == 0:
                print(
                    f"[census] gamma={gamma:g} {len(existing.intersection(requested))}/{len(requested)} "
                    f"success={sum(r['success'] for r in records if r['seed'] in requested)} "
                    f"eligible={sum(r['candidate_eligible'] for r in records if r['seed'] in requested)}",
                    flush=True,
                )
        records.sort(key=lambda record: record["seed"])
        save_candidates(records, path)
        all_records[gamma] = [record for record in records if record["seed"] in requested]
        row = candidate_summary(all_records[gamma], gamma)
        print(
            f"[saved] gamma={gamma:g} N={row['candidates']} success={row['successes']} "
            f"words={row['unique_signatures']} mirror_capacity={row['total_exact_mirror_balanced_capacity']} -> {path}",
            flush=True,
        )

    summaries = [candidate_summary(all_records[gamma], gamma) for gamma in GAMMAS]
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "logs").mkdir(parents=True, exist_ok=True)
    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)
    (args.outdir / "viz").mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ID_CANDIDATE_CENSUS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "task": {"scene": "ordinary symmetric 4x4 ID stadium", "start": START.tolist(),
                 "goal": GOAL.tolist(), "max_steps": args.max_steps, "reach": args.reach},
        "expert": {"smooth_weight": SMOOTH_WEIGHT, "retreat_weight": RETREAT_WEIGHT,
                   "retreat_scale": DEFAULT_SCALE, "retreat_cap": DEFAULT_CAP},
        "geometry": {"thresholds": THRESHOLDS, "drawdown_tol": 0.35,
                     "tie_tol": 1e-4, "possible_signatures": len(ALL_SIGNATURES),
                     "mirror": "swap R and U (reflection across y=x)"},
        "seed0": args.seed0,
        "seeds_per_gamma": args.seeds_per_gamma,
        "per_gamma": summaries,
    }
    (args.outdir / "logs/candidate_census.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.outdir / "tables/candidate_census.csv").open("w", newline="") as handle:
        fields = ("gamma", "candidates", "successes", "collisions", "eligible", "unique_signatures",
                  "unique_mirror_pairs", "total_exact_mirror_balanced_capacity", "mean_time_s_success",
                  "mean_retreat_m_success")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)
    render_candidate_census(env, all_records, args.outdir / "viz/candidate_signature_census.png")
    render_candidate_paths(env, all_records, args.reach, args.outdir / "viz/candidate_paths.png")
    print(json.dumps({"status": summary["status"], "per_gamma": summaries,
                      "output": str(args.outdir / "viz/candidate_signature_census.png")}, indent=2), flush=True)


def run_build(args: argparse.Namespace) -> None:
    env = make_id_env(args.max_steps)
    all_records: dict[float, list[dict]] = {}
    selected_by_gamma: dict[float, list[dict]] = {}
    selection_audits = []
    selected_rows = []
    window_rows = []
    all_mask_rows: list[dict] = []
    started = time.perf_counter()
    for gamma in GAMMAS:
        records = load_candidates(candidate_path(gamma, args.candidate_dir))
        expected_seeds = set(range(args.seed0, args.seed0 + args.seeds_per_gamma))
        records = [record for record in records if record["seed"] in expected_seeds]
        if len(records) != args.seeds_per_gamma or {record["seed"] for record in records} != expected_seeds:
            raise RuntimeError(f"gamma={gamma:g}: incomplete candidate pool for requested seeds")
        selected, audit = select_exact_mirror_balanced(
            records,
            args.n_mirror_pairs,
            args.quota_per_signature,
        )
        audit["gamma"] = float(gamma)
        all_records[gamma] = records
        selected_by_gamma[gamma] = selected
        selection_audits.append(audit)
        selected_rows.append(selected_metrics(selected, gamma, audit))
        print(
            f"[select] gamma={gamma:g} selected={len(selected)} "
            f"words={audit['selected_signature_counts']} mirror_residual={audit['exact_reflection_count_residual_max']}",
            flush=True,
        )

    data_dir = args.outdir / "data"
    log_dir = args.outdir / "logs"
    table_dir = args.outdir / "tables"
    viz_dir = args.outdir / "viz"
    for directory in (data_dir, log_dir, table_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for gamma_index, gamma in enumerate(GAMMAS):
        output = data_dir / f"balanced_id_windows_g{gamma_tag(gamma)}.pt"
        payload, mask_rows = build_gamma_dataset(
            env,
            selected_by_gamma[gamma],
            gamma,
            gamma_index,
            args.windows_per_trajectory,
            output,
        )
        window_rows.append(window_metrics(payload, gamma))
        all_mask_rows.extend(mask_rows)
        print(
            f"[dataset] gamma={gamma:g} windows={len(payload['grid'])} "
            f"joint={float(payload['joint_valid2_mask'].float().mean()):.1%} -> {output}",
            flush=True,
        )
        del payload

    selected_flat = [record for gamma in GAMMAS for record in selected_by_gamma[gamma]]
    np.savez_compressed(
        data_dir / "balanced_id_paths_all_gamma.npz",
        gammas=np.asarray([record["gamma"] for record in selected_flat], dtype=np.float32),
        seeds=np.asarray([record["seed"] for record in selected_flat], dtype=np.int64),
        signatures=np.asarray([record["signature"] for record in selected_flat], dtype="U8"),
        signature_ids=np.asarray([record["signature_id"] for record in selected_flat], dtype=np.int16),
        pair_ranks=np.asarray([record["selection_pair_rank"] for record in selected_flat], dtype=np.int8),
        quality_scores=np.asarray([record["quality_score"] for record in selected_flat], dtype=np.float32),
        paths=object_array([record["path"] for record in selected_flat]),
        states=object_array([record["states"] for record in selected_flat]),
        controls=object_array([record["controls"] for record in selected_flat]),
        start=START,
        goal=GOAL,
    )

    with (data_dir / "window_masks.csv").open("w", newline="") as handle:
        fields = list(all_mask_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_mask_rows)
    with (table_dir / "selected_trajectory_metrics.csv").open("w", newline="") as handle:
        scalar_fields = [key for key in selected_rows[0] if key != "signature_counts"]
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in selected_rows)
    with (table_dir / "window_validity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    signature_rows = []
    for gamma in GAMMAS:
        counts = Counter(record["signature"] for record in selected_by_gamma[gamma])
        for word, count in sorted(counts.items()):
            signature_rows.append({
                "gamma": float(gamma),
                "signature": word,
                "mirror": mirror_word(word),
                "count": count,
                "mirror_count": counts[mirror_word(word)],
            })
    with (table_dir / "selected_signature_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(signature_rows[0]))
        writer.writeheader()
        writer.writerows(signature_rows)

    total_selected = len(selected_flat)
    total_windows = sum(row["windows"] for row in window_rows)
    summary = {
        "status": "BALANCED_ID_DATASET_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "task": {"scene": "ordinary symmetric 4x4 ID stadium", "start": START.tolist(),
                 "goal": GOAL.tolist(), "max_steps": args.max_steps, "reach": args.reach},
        "expert": {"smooth_weight": SMOOTH_WEIGHT, "retreat_weight": RETREAT_WEIGHT,
                   "retreat_scale": DEFAULT_SCALE, "retreat_cap": DEFAULT_CAP},
        "candidate_protocol": {"matched_seeds_per_gamma": args.seeds_per_gamma,
                               "seed0": args.seed0, "actual_rollouts": args.seeds_per_gamma * len(GAMMAS)},
        "balance_protocol": {
            "n_mirror_pairs_per_gamma": args.n_mirror_pairs,
            "quota_per_signature": args.quota_per_signature,
            "trajectories_per_gamma": args.n_mirror_pairs * 2 * args.quota_per_signature,
            "windows_per_trajectory": args.windows_per_trajectory,
            "actual_expert_rollouts_only": True,
            "synthetic_reflections": 0,
            "quality_score": "retreat_m + 0.005*time_s + 0.002*radial_direction_switches",
            "quality_selection": "lowest score within each fixed gamma/signature stratum",
            "window_selection": "64 unique uniformly spaced executed control indices including endpoints",
        },
        "total_selected_trajectories": total_selected,
        "total_training_windows": total_windows,
        "selected_per_gamma": selected_rows,
        "selection_audits": selection_audits,
        "window_validity_per_gamma": window_rows,
        "global_audit": {
            "all_selected_physical_success": bool(all(record["success"] for record in selected_flat)),
            "selected_collisions": int(sum(record["collision"] for record in selected_flat)),
            "all_gamma_equal_trajectory_count": len({row["selected"] for row in selected_rows}) == 1,
            "all_gamma_equal_window_count": len({row["windows"] for row in window_rows}) == 1,
            "mirror_count_residual_max": max(row["mirror_residual_max"] for row in selected_rows),
            "synthetic_trajectories": 0,
        },
        "files": {str(gamma): f"data/balanced_id_windows_g{gamma_tag(gamma)}.pt" for gamma in GAMMAS},
    }
    (log_dir / "balanced_dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_balanced_paths(env, selected_by_gamma, args.reach, viz_dir / "balanced_id_paths_by_gamma.png")
    render_all_gamma_overlay(env, selected_by_gamma, args.reach, viz_dir / "balanced_id_overlay_all_gamma.png")
    render_signature_balance(all_records, selected_by_gamma, viz_dir / "signature_balance_raw_vs_selected.png")
    render_window_validity(window_rows, viz_dir / "window_validity.png")
    print(json.dumps({
        "status": summary["status"],
        "total_selected_trajectories": total_selected,
        "total_training_windows": total_windows,
        "global_audit": summary["global_audit"],
        "selected_per_gamma": selected_rows,
        "window_validity_per_gamma": window_rows,
        "output": str(viz_dir / "balanced_id_overlay_all_gamma.png"),
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "build"), default="generate")
    parser.add_argument("--seeds-per-gamma", type=int, default=96)
    parser.add_argument("--seed0", type=int, default=72000)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--n-mirror-pairs", type=int, default=3)
    parser.add_argument("--quota-per-signature", type=int, default=4)
    parser.add_argument("--windows-per-trajectory", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--outdir", type=Path, default=STAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds_per_gamma <= 0:
        raise ValueError("--seeds-per-gamma must be positive")
    if args.max_steps < 200:
        raise ValueError("--max-steps must be at least 200")
    if args.n_mirror_pairs <= 0 or args.quota_per_signature <= 0:
        raise ValueError("mirror-pair and per-signature quotas must be positive")
    if args.windows_per_trajectory <= 0:
        raise ValueError("--windows-per-trajectory must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.phase == "generate":
        run_generate(args, device)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
