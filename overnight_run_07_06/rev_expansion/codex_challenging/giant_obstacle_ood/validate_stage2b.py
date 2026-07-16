#!/usr/bin/env python3
"""Independent integrity and symmetry audit for the Stage-2B dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
H = 10
N_TRAJECTORIES = 24
WINDOWS_PER_TRAJECTORY = 64
START = np.asarray((0.5, 0.5), dtype=np.float32)
GOAL = np.asarray((4.5, 4.5), dtype=np.float32)


def tag(gamma: float) -> str:
    return str(float(gamma))


def mirror(word: str) -> str:
    return word.translate(str.maketrans({"R": "U", "U": "R"}))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def audit_gamma(stage: Path, gamma: float) -> dict:
    dataset_path = stage / "data" / f"balanced_id_windows_g{tag(gamma)}.pt"
    candidate_path = stage / "candidates" / f"candidates_g{tag(gamma)}.npz"
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    candidates = np.load(candidate_path, allow_pickle=True)

    n_windows = N_TRAJECTORIES * WINDOWS_PER_TRAJECTORY
    expected_shapes = {
        "grid": (n_windows, 3, 32, 32),
        "low5": (n_windows, 5),
        "hist": (n_windows, 16, 2),
        "U": (n_windows, H, 2),
        "window_context_positions": (n_windows, 2),
        "trajectory_seeds": (N_TRAJECTORIES,),
        "trajectory_signature_ids": (N_TRAJECTORIES,),
    }
    for key, shape in expected_shapes.items():
        assert tuple(payload[key].shape) == shape, (key, payload[key].shape, shape)
    for key in ("grid", "low5", "hist", "U", "window_context_positions",
                "window_min_clearance", "nominal_residual"):
        assert finite(payload[key]), f"non-finite values in {key} for gamma={gamma}"

    assert payload["schema_version"] == "giant_ood_id_balanced_v2_full_horizon"
    assert abs(float(payload["gamma"]) - gamma) < 1e-7
    assert int(payload["n_traj"]) == N_TRAJECTORIES
    assert int(payload["windows_per_trajectory"]) == WINDOWS_PER_TRAJECTORY
    assert torch.equal(payload["start"], torch.from_numpy(START))
    assert torch.equal(payload["goal"], torch.from_numpy(GOAL))
    assert torch.all(payload["window_starts"] == torch.from_numpy(START))
    assert torch.all(payload["window_goals"] == torch.from_numpy(GOAL))
    assert not bool(payload["padded_mask"].any())
    assert bool(payload["physical_collision_free_mask"].all())
    assert bool(payload["taskspace_mask"].all())
    assert float(payload["window_min_clearance"].min()) >= 0.0
    assert payload["balance"]["actual_expert_rollouts_only"]
    assert payload["balance"]["synthetic_reflections"] == 0
    assert payload["balance"]["complete_executed_horizons_only"]
    assert payload["balance"]["terminal_padding"] is False

    trajectory_words = list(payload["trajectory_signatures"])
    trajectory_counts = Counter(trajectory_words)
    assert len(trajectory_counts) == 6
    assert set(trajectory_counts.values()) == {4}
    assert all(trajectory_counts[word] == trajectory_counts[mirror(word)] for word in trajectory_counts)
    assert sum(word.startswith("R") for word in trajectory_words) == N_TRAJECTORIES // 2
    assert sum(word.startswith("U") for word in trajectory_words) == N_TRAJECTORIES // 2

    vocabulary = list(payload["signature_vocabulary"])
    window_words = [vocabulary[int(index)] for index in payload["window_signature_ids"]]
    window_counts = Counter(window_words)
    assert set(window_counts.values()) == {4 * WINDOWS_PER_TRAJECTORY}
    assert all(window_counts[word] == window_counts[mirror(word)] for word in window_counts)
    assert sum(word.startswith("R") for word in window_words) == n_windows // 2
    assert sum(word.startswith("U") for word in window_words) == n_windows // 2

    seed_to_candidate = {int(seed): index for index, seed in enumerate(candidates["seed"])}
    selected_seeds = [int(seed) for seed in payload["trajectory_seeds"]]
    assert len(set(selected_seeds)) == N_TRAJECTORIES
    source_target_max_abs_error = 0.0
    source_context_max_abs_error = 0.0
    for local_id, (seed, word) in enumerate(zip(selected_seeds, trajectory_words, strict=True)):
        candidate_index = seed_to_candidate[seed]
        assert bool(candidates["success"][candidate_index])
        assert not bool(candidates["collision"][candidate_index])
        assert bool(candidates["candidate_eligible"][candidate_index])
        assert str(candidates["signature"][candidate_index]) == word
        controls = np.asarray(candidates["controls"][candidate_index], dtype=np.float32)
        states = np.asarray(candidates["states"][candidate_index], dtype=np.float32)
        rows = torch.where(payload["window_seeds"] == seed)[0]
        assert len(rows) == WINDOWS_PER_TRAJECTORY
        steps = payload["window_steps"][rows].numpy().astype(int)
        assert len(np.unique(steps)) == WINDOWS_PER_TRAJECTORY
        assert int(steps.min()) == 0
        assert int(steps.max()) <= len(controls) - H
        for row, step in zip(rows.tolist(), steps, strict=True):
            source_target_max_abs_error = max(
                source_target_max_abs_error,
                float(np.max(np.abs(payload["U"][row].numpy() - controls[step:step + H]))),
            )
            source_context_max_abs_error = max(
                source_context_max_abs_error,
                float(np.max(np.abs(payload["window_context_positions"][row].numpy() - states[step, :2]))),
            )
        assert int(payload["trajectory_signature_ids"][local_id]) == vocabulary.index(word)
    assert source_target_max_abs_error == 0.0
    assert source_context_max_abs_error == 0.0

    targets = payload["U"].numpy()
    target_x = targets[..., 0].reshape(-1)
    target_y = targets[..., 1].reshape(-1)
    target_abs_mean = np.mean(np.abs(targets), axis=(0, 1))
    target_abs_axis_gap = float(
        abs(target_abs_mean[0] - target_abs_mean[1]) / max(float(target_abs_mean.mean()), 1e-12)
    )
    quantiles = np.linspace(0.0, 1.0, 101)
    target_reflection_quantile_mae = float(
        np.mean(np.abs(np.quantile(target_x, quantiles) - np.quantile(target_y, quantiles)))
    )
    contexts = payload["window_context_positions"].numpy()
    context_axis_mean_gap = float(abs(contexts[:, 0].mean() - contexts[:, 1].mean()))

    return {
        "gamma": gamma,
        "dataset": str(dataset_path),
        "sha256": sha256(dataset_path),
        "trajectories": N_TRAJECTORIES,
        "windows": n_windows,
        "trajectory_signature_counts": dict(sorted(trajectory_counts.items())),
        "window_signature_counts": dict(sorted(window_counts.items())),
        "r_first_trajectories": sum(word.startswith("R") for word in trajectory_words),
        "u_first_trajectories": sum(word.startswith("U") for word in trajectory_words),
        "r_first_windows": sum(word.startswith("R") for word in window_words),
        "u_first_windows": sum(word.startswith("U") for word in window_words),
        "padded_windows": int(payload["padded_mask"].sum()),
        "physical_safe_windows": int(payload["physical_collision_free_mask"].sum()),
        "joint_valid2_windows": int(payload["joint_valid2_mask"].sum()),
        "joint_valid2_rate": float(payload["joint_valid2_mask"].float().mean()),
        "minimum_window_clearance_m": float(payload["window_min_clearance"].min()),
        "source_target_max_abs_error": source_target_max_abs_error,
        "source_context_max_abs_error": source_context_max_abs_error,
        "target_abs_mean_x": float(target_abs_mean[0]),
        "target_abs_mean_y": float(target_abs_mean[1]),
        "target_abs_axis_relative_gap": target_abs_axis_gap,
        "target_reflection_quantile_mae": target_reflection_quantile_mae,
        "context_axis_mean_gap_m": context_axis_mean_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=Path,
        default=Path(__file__).resolve().parent / "stage_results/02b_balanced_id",
    )
    args = parser.parse_args()
    rows = [audit_gamma(args.stage, gamma) for gamma in GAMMAS]
    assert sum(row["trajectories"] for row in rows) == 168
    assert sum(row["windows"] for row in rows) == 10_752
    audit = {
        "status": "PASS",
        "checks": {
            "real_successful_expert_sources_only": True,
            "complete_executed_h10_only": True,
            "physical_safe_windows": True,
            "finite_tensors": True,
            "exact_source_reconstruction": True,
            "equal_gamma_mass": True,
            "exact_trajectory_signature_reflection_balance": True,
            "exact_window_signature_reflection_balance": True,
        },
        "totals": {
            "gammas": len(rows),
            "trajectories": sum(row["trajectories"] for row in rows),
            "windows": sum(row["windows"] for row in rows),
            "padded_windows": sum(row["padded_windows"] for row in rows),
            "physical_safe_windows": sum(row["physical_safe_windows"] for row in rows),
            "r_first_trajectories": sum(row["r_first_trajectories"] for row in rows),
            "u_first_trajectories": sum(row["u_first_trajectories"] for row in rows),
            "r_first_windows": sum(row["r_first_windows"] for row in rows),
            "u_first_windows": sum(row["u_first_windows"] for row in rows),
        },
        "continuous_symmetry": {
            "max_target_abs_axis_relative_gap": max(row["target_abs_axis_relative_gap"] for row in rows),
            "max_target_reflection_quantile_mae": max(row["target_reflection_quantile_mae"] for row in rows),
            "max_context_axis_mean_gap_m": max(row["context_axis_mean_gap_m"] for row in rows),
        },
        "per_gamma": rows,
    }
    output = args.stage / "logs" / "independent_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
