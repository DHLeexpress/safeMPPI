#!/usr/bin/env python3
"""Independent consistency audit for the automated giant-obstacle Stage 4."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from giant_obstacle_ood.stage4_frozen_ood import (
    CHECKPOINT,
    GOAL,
    GAMMAS,
    REACH,
    STAGE,
    START,
    aggregate,
    classify_path,
    load_records,
    make_env,
    select_tuning,
    tuning_rows,
)


EXPECTED_CHECKPOINT_SHA256 = "a5c8280f593fbf6ef6129dbe632f740ba3282067726d8a1d5bc7039cc7aaa236"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def integrate(controls: np.ndarray, dt: float = 0.1) -> np.ndarray:
    state = np.asarray([START[0], START[1], 0.0, 0.0], dtype=np.float64)
    path = [state[:2].copy()]
    for action in np.asarray(controls, dtype=np.float64):
        state[:2] = state[:2] + dt * state[2:] + 0.5 * dt * dt * action
        state[2:] = state[2:] + dt * action
        path.append(state[:2].copy())
    return np.asarray(path)


def audit_dataset(path: Path, expected_n: int, expected_per_gamma: int, max_steps: int) -> dict:
    records = load_records(path)
    env = make_env(max_steps)
    errors = []
    physics_errors = []
    reclassified = []
    for index, record in enumerate(records):
        executed = record["path"]
        controls = record["controls"]
        if len(executed) != len(controls) + 1:
            errors.append(f"record {index}: len(path) != len(controls)+1")
        if not np.allclose(executed[0], START, atol=1e-7):
            errors.append(f"record {index}: wrong start {executed[0]}")
        physics_error = float(np.max(np.abs(integrate(controls, float(env.dt)) - executed)))
        physics_errors.append(physics_error)
        if physics_error > 2e-5:
            errors.append(f"record {index}: dynamics error {physics_error:g}")
        fresh = classify_path(executed, controls, env, REACH)
        reclassified.append(fresh)
        for key in ("success", "collision", "in_taskspace", "failure_type", "local_minimum"):
            if fresh[key] != record[key]:
                errors.append(f"record {index}: {key} saved={record[key]} recomputed={fresh[key]}")
        for key in ("endpoint_distance", "min_clearance", "recent_displacement_30"):
            if not np.isclose(fresh[key], record[key], atol=2e-6):
                errors.append(f"record {index}: {key} mismatch")
    counts = Counter(float(record["gamma"]) for record in records)
    if len(records) != expected_n:
        errors.append(f"expected {expected_n} records, found {len(records)}")
    for gamma in GAMMAS:
        if counts[float(gamma)] != expected_per_gamma:
            errors.append(f"gamma {gamma:g}: expected {expected_per_gamma}, found {counts[float(gamma)]}")
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "records": len(records),
        "gamma_counts": {str(key): value for key, value in sorted(counts.items())},
        "max_dynamics_error": max(physics_errors, default=0.0),
        "recomputed": aggregate([{**record, **fresh} for record, fresh in zip(records, reclassified)]),
        "errors": errors,
    }


def main() -> None:
    stage_summary = json.loads((STAGE / "logs/stage4_summary.json").read_text())
    checks = {}
    checkpoint_sha = digest(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checks["checkpoint"] = {
        "sha256": checkpoint_sha,
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "hash_ok": checkpoint_sha == EXPECTED_CHECKPOINT_SHA256,
        "endpoint_free": not checkpoint["config"].get("raw_start_goal", False),
        "ctx_dim": checkpoint["config"].get("ctx_dim"),
        "created_before_stage4": CHECKPOINT.stat().st_mtime < (STAGE / "data/expert_m6.npz").stat().st_mtime,
    }
    datasets = {
        "expert": audit_dataset(STAGE / "data/expert_m6.npz", 42, 6, 800),
        "pretrained_selected": audit_dataset(STAGE / "data/pretrained_selected_m16.npz", 112, 16, 300),
        "pretrained_faithful": audit_dataset(STAGE / "data/pretrained_faithful_m4.npz", 28, 4, 300),
        "mizuta_selected": audit_dataset(STAGE / "data/mizuta_selected_m6.npz", 42, 6, 250),
    }
    tuning = load_records(STAGE / "data/mizuta_tuning.npz")
    selected = select_tuning(tuning_rows(tuning))
    mizuta_records = load_records(STAGE / "data/mizuta_selected_m6.npz")
    recent_displacements = [record["recent_displacement_30"] for record in mizuta_records]
    recent_progress = [record["recent_goal_progress_30"] for record in mizuta_records]
    checks["outcomes"] = {
        "expert_all_success": datasets["expert"]["recomputed"]["successes"] == 42,
        "expert_zero_collision": datasets["expert"]["recomputed"]["collisions"] == 0,
        "pretrained_all_fail": datasets["pretrained_selected"]["recomputed"]["successes"] == 0,
        "pretrained_all_collision": datasets["pretrained_selected"]["recomputed"]["collisions"] == 112,
        "mizuta_zero_collision": datasets["mizuta_selected"]["recomputed"]["collisions"] == 0,
        "mizuta_all_local_minimum": all(record["local_minimum"] for record in mizuta_records),
        "mizuta_max_recent_displacement_m": max(recent_displacements),
        "mizuta_max_recent_goal_progress_m": max(recent_progress),
        "selected_tag": selected["tag"],
        "selected_matches_summary": selected["tag"] == stage_summary["mizuta_tuning"]["selected"]["tag"],
    }
    artifacts = [
        STAGE / "REPORT.md",
        STAGE / "viz/rollouts_and_local_minimum.png",
        STAGE / "viz/failure_taxonomy.png",
        STAGE / "viz/mizuta_tuning.png",
        STAGE / "viz/pretrained_temperature_diagnostic.png",
        STAGE / "tables/method_metrics_by_gamma.csv",
        STAGE / "tables/metrics_ae.md",
    ]
    checks["artifacts"] = {
        "all_present_nonempty": all(path.exists() and path.stat().st_size > 0 for path in artifacts),
        "files": {str(path.relative_to(STAGE)): path.stat().st_size if path.exists() else None for path in artifacts},
        "no_learned_checkpoint_in_stage4": not any(STAGE.rglob("*.pt")),
    }
    boolean_checks = []
    boolean_checks.extend([
        checks["checkpoint"]["hash_ok"],
        checks["checkpoint"]["endpoint_free"],
        checks["checkpoint"]["ctx_dim"] == 37,
        checks["checkpoint"]["created_before_stage4"],
    ])
    boolean_checks.extend(not dataset["errors"] for dataset in datasets.values())
    boolean_checks.extend(value for key, value in checks["outcomes"].items() if isinstance(value, bool))
    boolean_checks.extend([
        checks["outcomes"]["mizuta_max_recent_displacement_m"] < 0.03,
        checks["outcomes"]["mizuta_max_recent_goal_progress_m"] < 0.10,
        checks["artifacts"]["all_present_nonempty"],
        checks["artifacts"]["no_learned_checkpoint_in_stage4"],
    ])
    status = "PASS" if all(boolean_checks) else "FAIL"
    payload = {"status": status, "checks": checks, "datasets": datasets}
    destination = STAGE / "logs/independent_audit.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "status": status,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_errors": {name: result["errors"] for name, result in datasets.items()},
        "outcomes": checks["outcomes"],
        "audit": str(destination.resolve()),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
