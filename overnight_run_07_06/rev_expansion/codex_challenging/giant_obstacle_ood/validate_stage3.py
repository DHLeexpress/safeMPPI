#!/usr/bin/env python3
"""Independent artifact audit for the selected Stage-3 ID policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
STAGE = HERE / "stage_results/03_pretrain"
DATA = HERE / "stage_results/02b_balanced_id/data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checkpoint_path = STAGE / "data/pretrained_id_balanced_a32.pt"
    metrics_path = STAGE / "logs/selected_id_metrics.json"
    rollout_path = STAGE / "data/selected_id_rollouts_m16.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    training = checkpoint["stage3_pretrain_summary"]
    metrics = json.loads(metrics_path.read_text())
    summary = metrics["summary"]

    require(config["ctx_dim"] == 37 and not config["raw_start_goal"],
            "selected policy is not the endpoint-free 37-D model")
    require(config["repr_dim"] == 32 and config["trunk_hidden"] == [160, 96],
            "selected policy is not the original A32 backbone")
    require(not config.get("use_gru", False) and not config.get("boundary_adapter", False),
            "unexpected recurrent or boundary adapter")
    require(training["model"]["fresh_from_scratch"], "checkpoint was not trained from scratch")
    require(training["dataset"]["optimization_windows"] == 8064, "wrong optimization split")
    require(training["dataset"]["monitoring_windows"] == 2688, "wrong monitoring split")
    require(not training["dataset"]["monitoring_rows_seen_by_optimizer"],
            "held-out monitoring trajectories leaked into optimization")
    require(training["training"]["symmetry_augment"], "exact reflection augmentation is absent")
    require(training["training"]["equivariance_weight"] == 1.0, "wrong selected symmetry weight")

    for source in training["dataset"]["sources"]:
        path = Path(source["path"])
        require(path.parent.resolve() == DATA.resolve(), f"foreign training source: {path}")
        require(sha256(path) == source["sha256"], f"source hash changed: {path.name}")

    settings = metrics["settings"]
    require(settings["M_per_gamma"] == 16 and settings["h_exec"] == 1,
            "selected evaluation is not the faithful M=16 receding-horizon gate")
    require(settings["temperature"] == 0.1 and metrics["plain_unguided"],
            "selected evaluation uses an unexpected sampler or guidance")
    require(summary["global_success_rate"] >= 0.90, "global ID success below 90%")
    require(summary["global_collision_rate"] <= 0.10, "global ID collision above 10%")
    require(0.20 <= summary["global_r_first_share"] <= 0.80, "global R/U modes are imbalanced")
    passing = [gamma for gamma, row in summary["per_gamma"].items() if row["gate_pass"]]
    failing = [gamma for gamma, row in summary["per_gamma"].items() if not row["gate_pass"]]
    require(passing == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7"],
            f"unexpected passing gamma set: {passing}")
    require(failing == ["1.0"], f"unexpected gate failures: {failing}")

    with np.load(rollout_path, allow_pickle=True) as rollouts:
        require(len(rollouts["gammas"]) == 112, "rollout archive is not 16 x 7")
        values, counts = np.unique(rollouts["gammas"], return_counts=True)
        require(counts.tolist() == [16] * 7, f"unequal gamma rollout counts: {counts.tolist()}")
        require(np.allclose(rollouts["start"], (0.5, 0.5)), "wrong ID start")
        require(np.allclose(rollouts["goal"], (4.5, 4.5)), "wrong ID goal")

    audit = {
        "status": "PASS_WITH_GAMMA1_CAVEAT",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "architecture": {"ctx_dim": 37, "raw_start_goal": False, "repr_dim": 32,
                         "trunk_hidden": [160, 96]},
        "dataset": {"optimization_windows": 8064, "heldout_windows": 2688,
                    "source_hashes_verified": 7},
        "rollout_gate": {
            "global_success_rate": summary["global_success_rate"],
            "global_collision_rate": summary["global_collision_rate"],
            "r_first_successes": summary["global_r_first_successes"],
            "u_first_successes": summary["global_u_first_successes"],
            "passing_gammas": passing,
            "failing_gammas": failing,
            "gamma_1_limitation": summary["per_gamma"]["1.0"],
        },
    }
    output = STAGE / "logs/independent_audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
