#!/usr/bin/env python3
"""Independent audit of the revised endpoint-free Stage 3 model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import grid_hp_expt as HP
import gen_uniform_data as SEEDS
from pretrain_sg import migrate_raw_endpoint_checkpoint
from viz_style import GAMMAS


HERE = Path(__file__).resolve().parent
STAGE = HERE / "stage_results" / "03_pretrain"
DATA = HERE / "stage_results" / "02_demos" / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "pretrained_sg_walls8.pt")
    parser.add_argument("--stage-checkpoint", type=Path, default=STAGE / "data" / "pretrained_sg_walls8.pt")
    parser.add_argument(
        "--raw-archive", type=Path,
        default=STAGE / "rejected_raw_endpoints" / "data" / "pretrained_raw_endpoints.pt",
    )
    parser.add_argument("--split", type=Path, default=STAGE / "data" / "pair_split.npz")
    parser.add_argument("--pretrain-log", type=Path, default=STAGE / "logs" / "pretrain_summary.json")
    parser.add_argument("--history", type=Path, default=STAGE / "logs" / "pretrain_history.csv")
    parser.add_argument("--eval-log", type=Path, default=STAGE / "logs" / "deployment_eval.json")
    parser.add_argument("--unseen-paths", type=Path, default=STAGE / "data" / "heldout_pair_rollouts.npz")
    parser.add_argument("--canonical-paths", type=Path, default=STAGE / "data" / "canonical_target_rollouts.npz")
    parser.add_argument("--manifest", type=Path, default=DATA / "random_pairs_300.npz")
    parser.add_argument("--out", type=Path, default=STAGE / "logs" / "stage3_validation.json")
    return parser.parse_args()


def audit_paths(path: Path, expected_start, expected_goal, expected_reach: float, errors: list[str]) -> dict:
    with np.load(path, allow_pickle=True) as saved:
        success = np.asarray(saved["success"], dtype=bool)
        collision = np.asarray(saved["collision"], dtype=bool)
        in_taskspace = np.asarray(saved["in_taskspace"], dtype=bool)
        endpoint_distance = np.asarray(saved["endpoint_distance"], dtype=float)
        min_clearance = np.asarray(saved["min_clearance"], dtype=float)
        start = np.asarray(saved["start"], dtype=float)
        goal = np.asarray(saved["goal"], dtype=float)
        reach = float(saved["reach"])
        count = len(saved["paths"])
        gammas = np.asarray(saved["gammas"], dtype=float)
    if not np.allclose(start, expected_start):
        errors.append(f"{path.name}: start differs from protocol")
    if not np.allclose(goal, expected_goal):
        errors.append(f"{path.name}: goal differs from protocol")
    if abs(reach - expected_reach) > 1e-7:
        errors.append(f"{path.name}: reach {reach} != {expected_reach}")
    if not np.isfinite(endpoint_distance).all() or not np.isfinite(min_clearance).all():
        errors.append(f"{path.name}: non-finite metric")
    if (collision[success]).any() or (~in_taskspace[success]).any():
        errors.append(f"{path.name}: reported success violates collision/task-space criteria")
    if (endpoint_distance[success] >= expected_reach).any():
        errors.append(f"{path.name}: reported success violates reach criterion")
    for gamma in GAMMAS:
        if int(np.isclose(gammas, gamma).sum()) != count // len(GAMMAS):
            errors.append(f"{path.name}: gamma {gamma} rollout count mismatch")
    return {
        "rollouts": count,
        "safe_reaches": int(success.sum()),
        "collisions": int(collision.sum()),
        "out_of_bounds": int((~in_taskspace).sum()),
        "best_endpoint_distance": float(endpoint_distance.min()),
        "minimum_clearance": float(min_clearance.min()),
    }


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    root_hash = sha256(args.checkpoint)
    stage_hash = sha256(args.stage_checkpoint)
    raw_hash = sha256(args.raw_archive)
    if root_hash != stage_hash:
        errors.append("root and stage endpoint-free checkpoints differ")
    if root_hash == raw_hash:
        errors.append("active checkpoint is still the rejected raw-endpoint checkpoint")

    policy, checkpoint = HP.load_hp(args.checkpoint, device="cpu")
    config = checkpoint["config"]
    expected_config = {
        "raw_start_goal": False,
        "ctx_dim": 37,
        "grid_hw": [32, 32],
        "schema_version": "w8sg-hp-v2-low5-only",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"config {key}={config.get(key)!r}, expected {expected!r}")
    first_linear = next(module for module in policy.trunk if isinstance(module, torch.nn.Linear))
    if first_linear.in_features != 89:
        errors.append(f"first trunk layer has {first_linear.in_features} inputs, expected 89")
    if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
        errors.append("checkpoint contains non-finite tensors")

    generator = torch.Generator().manual_seed(1)
    grid = torch.rand(4, 3, 32, 32, generator=generator)
    low5 = torch.rand(4, 5, generator=generator)
    hist = torch.rand(4, 16, 2, generator=generator)
    with torch.no_grad():
        context = policy.ctx_from(grid, low5, hist)
        hp_token = policy.hp_token(grid)
    if tuple(context.shape) != (4, 37):
        errors.append(f"context shape is {tuple(context.shape)}, expected (4,37)")
    if not torch.equal(context[:, :5], low5):
        errors.append("context does not begin with exact low5")
    if not torch.allclose(context[:, 5:], hp_token):
        errors.append("context visual suffix differs from E(H_P)")
    try:
        policy.ctx_from(grid, low5, hist, torch.zeros(4, 4))
        errors.append("ctx_from still accepts raw endpoint inputs")
    except TypeError:
        pass

    # Independently reconstruct the pre-fine-tune migration and verify that
    # the only structural edit is deletion of raw endpoint columns 25:29.
    migration_model = HP.GridHPFlowPolicy()
    migration = migrate_raw_endpoint_checkpoint(migration_model, args.raw_archive)
    raw = torch.load(args.raw_archive, map_location="cpu", weights_only=False)["state_dict"]
    migrated = migration_model.state_dict()
    if not torch.equal(migrated["trunk.0.weight"][:, :25], raw["trunk.0.weight"][:, :25]):
        errors.append("migration changed pre-endpoint first-layer columns")
    if not torch.equal(migrated["trunk.0.weight"][:, 25:], raw["trunk.0.weight"][:, 29:]):
        errors.append("migration changed post-endpoint first-layer columns")
    for key in migrated:
        if key != "trunk.0.weight" and not torch.equal(migrated[key], raw[key]):
            errors.append(f"migration changed non-structural tensor {key}")
            break

    with np.load(args.split) as split:
        train_pairs = np.asarray(split["train_pairs"], dtype=int)
        val_pairs = np.asarray(split["val_pairs"], dtype=int)
    if np.intersect1d(train_pairs, val_pairs).size:
        errors.append("train and validation pair sets overlap")
    if not np.array_equal(np.sort(np.concatenate((train_pairs, val_pairs))), np.arange(300)):
        errors.append("pair split does not cover exactly indices 0..299")

    dataset_counts = {}
    train_windows = val_windows = 0
    for gamma in GAMMAS:
        path = DATA / f"w8sg_windows_g{float(gamma)}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        pair_indices = payload["window_pair_indices"].numpy()
        val_mask = np.isin(pair_indices, val_pairs)
        dataset_counts[str(gamma)] = {
            "windows": len(pair_indices),
            "train_windows": int((~val_mask).sum()),
            "val_windows": int(val_mask.sum()),
        }
        train_windows += int((~val_mask).sum())
        val_windows += int(val_mask.sum())
        del payload

    pretrain = json.loads(args.pretrain_log.read_text())
    if pretrain.get("status") != "PASS":
        errors.append("fine-tuning summary is not PASS")
    if pretrain["model"].get("raw_start_goal") is not False:
        errors.append("fine-tuning summary still reports raw endpoints")
    if pretrain["training"]["mode"] != "remove_raw_endpoint_columns_then_finetune":
        errors.append("fine-tuning mode is not the requested migration")
    if pretrain["dataset"]["train_windows"] != train_windows or pretrain["dataset"]["val_windows"] != val_windows:
        errors.append("dataset split counts differ from fine-tuning summary")
    if pretrain["training"]["best_val_cfm"] >= pretrain["training"]["initial_val_cfm"]:
        errors.append("fine-tuning did not improve grouped-pair validation loss")
    encoder = pretrain["diagnostics"]["visual_encoder"]
    if encoder["active_dimensions_std_gt_1e-3"] != 32:
        errors.append("visual encoder has inactive token dimensions")

    with args.history.open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != pretrain["training"]["epochs"]:
        errors.append("history row count differs from epoch count")
    best_from_history = min(range(len(history)), key=lambda index: float(history[index]["val_cfm"]))
    if best_from_history != checkpoint["best_epoch"]:
        errors.append("checkpoint best epoch differs from history minimum")

    evaluation = json.loads(args.eval_log.read_text())
    if evaluation.get("status") != "PASS" or evaluation.get("raw_start_goal") is not False:
        errors.append("dual deployment evaluation is not valid endpoint-free output")
    unseen = evaluation["unseen_pair"]
    canonical = evaluation["canonical_target"]
    unseen_metrics = audit_paths(
        args.unseen_paths, unseen["start"], unseen["goal"], float(unseen["reach"]), errors
    )
    canonical_metrics = audit_paths(
        args.canonical_paths, [0.05, 0.05], [5.0, 5.0], 0.15, errors
    )
    if unseen_metrics["safe_reaches"] != unseen["total_successes"]:
        errors.append("unseen success count differs from deployment log")
    if canonical_metrics["safe_reaches"] != canonical["total_successes"]:
        errors.append("canonical success count differs from deployment log")

    with np.load(args.manifest) as manifest:
        if int(unseen["start_pool_index"]) in set(map(int, manifest["start_indices"])):
            errors.append("unseen start marginal appears in manifest")
        if int(unseen["goal_pool_index"]) in set(map(int, manifest["goal_indices"])):
            errors.append("unseen goal marginal appears in manifest")

    env = SEEDS.make_walled_env(8)
    start = np.array([0.05, 0.05])
    obstacles = env.obstacles.detach().cpu().numpy()
    start_clearance = float(
        (np.linalg.norm(start[None] - obstacles[:, :2], axis=1) - obstacles[:, 2] - float(env.r_robot)).min()
    )
    if start_clearance <= 0.0:
        errors.append("canonical cleared start is not collision-free")

    output = {
        "status": "PASS" if not errors else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "checkpoint": {
            "sha256": root_hash,
            "mirrors_identical": root_hash == stage_hash,
            "differs_from_rejected_raw_checkpoint": root_hash != raw_hash,
            "parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "context_shape": [4, 37],
            "trunk_input_dimension": first_linear.in_features,
            "raw_endpoint_argument_rejected": True,
        },
        "migration": migration,
        "split": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_windows": train_windows,
            "val_windows": val_windows,
            "per_gamma": dataset_counts,
        },
        "fine_tuning": {
            "epochs": len(history),
            "initial_val_cfm": pretrain["training"]["initial_val_cfm"],
            "best_epoch": checkpoint["best_epoch"],
            "best_val_cfm": checkpoint["best_val"],
            "relative_goal_shuffle_ratio": pretrain["diagnostics"]["shuffled_over_correct_ratio"],
            "encoder_active_dimensions": encoder["active_dimensions_std_gt_1e-3"],
            "encoder_effective_rank": encoder["covariance_effective_rank"],
        },
        "deployment": {
            "unseen_pair": unseen_metrics,
            "canonical_target": canonical_metrics,
            "canonical_start": [0.05, 0.05],
            "canonical_goal": [5.0, 5.0],
            "canonical_reach": 0.15,
            "canonical_start_clearance_m": start_clearance,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"STAGE3 REVISED VALIDATION {output['status']} -> {args.out}")
    if errors:
        for error in errors:
            print(f"  ERROR: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
