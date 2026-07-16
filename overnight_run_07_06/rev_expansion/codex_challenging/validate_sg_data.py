#!/usr/bin/env python3
"""Independent schema, goal-conditioning, and trajectory audit for Stage 2."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
STAGE = HERE / "stage_results" / "02_demos"
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import _paths  # noqa: F401,E402
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import grid_feats as GF  # noqa: E402
import gen_uniform_data as SEEDS  # noqa: E402
from viz_style import GAMMAS  # noqa: E402


def audit_gamma(data_dir: Path, manifest: dict, gamma: float, env) -> dict:
    dataset_path = data_dir / f"w8sg_windows_g{float(gamma)}.pt"
    paths_path = data_dir / f"paths_g{float(gamma)}.npz"
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    with np.load(paths_path, allow_pickle=True) as saved:
        paths = [np.asarray(path, dtype=np.float32) for path in saved["paths"]]
        controls = [np.asarray(item, dtype=np.float32) for item in saved["controls"]]
        pair_indices = np.asarray(saved["pair_indices"], dtype=int)
        starts = np.asarray(saved["starts"], dtype=np.float32)
        goals = np.asarray(saved["goals"], dtype=np.float32)
        success = np.asarray(saved["success"], dtype=bool)
        attempts = np.asarray(saved["attempts"], dtype=int)

    expected_pairs = np.arange(len(manifest["starts"]))
    assert np.array_equal(pair_indices, expected_pairs)
    assert np.array_equal(starts, manifest["starts"])
    assert np.array_equal(goals, manifest["goals"])
    assert success.all(), f"gamma {gamma}: unsuccessful saved paths"
    assert data["schema_version"] == "w8sg-v1-goal-aware"
    assert data["n_pairs"] == 300 and data["n_traj"] == 300
    assert data["grid"].shape[1:] == (3, 32, 32)
    assert data["low5"].shape[1:] == (5,)
    assert data["hist"].shape[1:] == (GF.K_HIST, 2)
    assert data["U"].shape[1:] == (GF.H_PRED, 2)
    assert len(data["grid"]) == len(data["low5"]) == len(data["hist"]) == len(data["U"])
    assert len(data["grid"]) == int(data["window_counts"].sum())
    assert torch.isfinite(data["grid"]).all()
    assert torch.isfinite(data["low5"]).all()
    assert torch.isfinite(data["hist"]).all()
    assert torch.isfinite(data["U"]).all()
    assert torch.allclose(data["low5"][:, 4], torch.full_like(data["low5"][:, 4], gamma))
    assert np.array_equal(data["pair_indices"].numpy(), expected_pairs)
    assert np.allclose(data["starts"][:, :2].numpy(), manifest["starts"], atol=0, rtol=0)
    assert np.allclose(data["goals"].numpy(), manifest["goals"], atol=0, rtol=0)

    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    cursor = 0
    max_position_error = 0.0
    max_velocity_error = 0.0
    max_control_error = 0.0
    min_clearance = float("inf")
    endpoint_max = 0.0
    all_steps = []
    for pair_index, (path, episode_controls) in enumerate(zip(paths, controls)):
        count = int(data["window_counts"][pair_index])
        assert count == len(episode_controls) == len(path) - 1
        block = slice(cursor, cursor + count)
        low = data["low5"][block].numpy()
        window_goal = data["window_goals"][block].numpy()
        window_start = data["window_starts"][block].numpy()
        reconstructed_position = window_goal - low[:, :2] * GF.R_GOAL
        reconstructed_velocity = low[:, 2:4] * GF.V_SCALE
        max_position_error = max(
            max_position_error,
            float(np.max(np.abs(reconstructed_position - path[:-1, :2]))),
        )
        max_velocity_error = max(
            max_velocity_error,
            float(np.max(np.abs(reconstructed_velocity - path[:-1, 2:4]))),
        )
        max_control_error = max(
            max_control_error,
            float(np.max(np.abs(data["U"][block, 0].numpy() - episode_controls))),
        )
        assert np.allclose(window_goal, goals[pair_index], atol=0, rtol=0)
        assert np.allclose(window_start, starts[pair_index], atol=0, rtol=0)
        assert np.array_equal(data["window_pair_indices"][block].numpy(), np.full(count, pair_index))
        assert np.allclose(path[0, :2], starts[pair_index], atol=0, rtol=0)

        xy = path[:, :2].astype(float)
        clearance = (
            np.linalg.norm(xy[:, None, :] - obs[None, :, :2], axis=2)
            - obs[None, :, 2]
            - rr
        )
        min_clearance = min(min_clearance, float(clearance.min()))
        endpoint_distance = float(np.linalg.norm(xy[-1] - goals[pair_index]))
        endpoint_max = max(endpoint_max, endpoint_distance)
        assert clearance.min() >= -1e-6
        assert endpoint_distance < 0.2 + 1e-6
        assert ((xy >= 0.0) & (xy <= 5.0)).all()
        all_steps.append(count)
        cursor += count
    assert cursor == len(data["grid"])
    assert max_position_error < 2e-6
    assert max_velocity_error < 2e-6
    assert max_control_error < 1e-7
    assert (manifest["starts"][:, 1] > manifest["starts"][:, 0]).all()
    assert (manifest["goals"][:, 1] < manifest["goals"][:, 0]).all()
    assert (np.abs(manifest["starts"][:, 1] - manifest["starts"][:, 0]) >= 1.0).all()
    assert (np.abs(manifest["goals"][:, 1] - manifest["goals"][:, 0]) >= 1.0).all()

    result = {
        "gamma": gamma,
        "pairs": len(paths),
        "successes": int(success.sum()),
        "windows": len(data["grid"]),
        "mean_steps": float(np.mean(all_steps)),
        "max_steps": int(np.max(all_steps)),
        "min_clearance": min_clearance,
        "max_endpoint_distance": endpoint_max,
        "max_goal_reconstruction_position_error": max_position_error,
        "max_velocity_reconstruction_error": max_velocity_error,
        "max_executed_control_error": max_control_error,
        "max_attempts": int(attempts.max()),
        "dataset_bytes": dataset_path.stat().st_size,
        "paths_bytes": paths_path.stat().st_size,
    }
    del data
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=STAGE / "data")
    parser.add_argument("--out", type=Path, default=STAGE / "logs" / "stage2_validation.json")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--cuda-mps", action="store_true")
    parser.add_argument("--worker-seconds", type=float)
    parser.add_argument("--gpu-utilization-max", type=float)
    parser.add_argument("--gpu-memory-max-mib", type=float)
    args = parser.parse_args()
    with np.load(args.data_dir / "random_pairs_300.npz") as saved:
        manifest = {key: np.asarray(saved[key]) for key in saved.files}
    env = SEEDS.make_walled_env(8)
    per_gamma = {}
    for gamma in GAMMAS:
        result = audit_gamma(args.data_dir, manifest, gamma, env)
        per_gamma[str(gamma)] = result
        print(
            f"gamma={gamma:g}: pairs={result['pairs']} windows={result['windows']} "
            f"steps={result['mean_steps']:.1f} min_clearance={result['min_clearance']:.4f} "
            f"goal_error={result['max_goal_reconstruction_position_error']:.2e}",
            flush=True,
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "unique_pairs": len(np.unique(manifest["start_indices"] * 284 + manifest["goal_indices"])),
        "pairs_per_gamma": 300,
        "total_trajectories": sum(item["pairs"] for item in per_gamma.values()),
        "total_successes": sum(item["successes"] for item in per_gamma.values()),
        "total_windows": sum(item["windows"] for item in per_gamma.values()),
        "total_dataset_bytes": sum(item["dataset_bytes"] for item in per_gamma.values()),
        "per_gamma": per_gamma,
    }
    worker_records = []
    for path in sorted((STAGE / "logs").glob("worker_g*_s*.log")):
        for line in reversed(path.read_text().splitlines()):
            if line.startswith("WORKER_DONE "):
                worker_records.append(json.loads(line.removeprefix("WORKER_DONE ")))
                break
    if worker_records or args.physical_gpu is not None:
        payload["execution"] = {
            "physical_gpu": args.physical_gpu,
            "cuda_mps": bool(args.cuda_mps),
            "workers": len(worker_records),
            "worker_pipeline_seconds": args.worker_seconds,
            "worker_wall_seconds_max": max((item["wall_seconds"] for item in worker_records), default=None),
            "worker_wall_seconds_mean": (
                float(np.mean([item["wall_seconds"] for item in worker_records])) if worker_records else None
            ),
            "gpu_utilization_max_pct": args.gpu_utilization_max,
            "gpu_memory_max_mib": args.gpu_memory_max_mib,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"VALIDATION PASS -> {args.out}")


if __name__ == "__main__":
    main()
