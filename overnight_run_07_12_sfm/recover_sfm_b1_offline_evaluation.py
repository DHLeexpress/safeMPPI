#!/usr/bin/env python3
"""Recover only the raw-M50 phase of a completed offline 9-arm sweep."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import time

import run_sfm_b1_offline_9arm as RUN
import run_sfm_b1_r2_9arm as BASE


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    with path.open() as stream:
        return json.load(stream)


def _validate_common_r0(path: Path) -> dict:
    payload = _read_json(path)
    records = payload.get("records", [])
    if (
        payload.get("status") != RUN.EVAL_STATUS
        or payload.get("scene_profile") != RUN.SCENE_PROFILE
        or len(records) != 1
        or int(records[0].get("round", -1)) != 0
        or records[0].get("cell", {}).get("checkpoint_sha256")
        != RUN.CHECKPOINT_SHA256
    ):
        raise RuntimeError(f"common r0 evaluation contract mismatch: {path}")
    return payload


def _load_frozen_training(run_root: Path, recovery_source: dict) -> tuple:
    declaration_path = run_root / "RUN_DECLARATION.json"
    training_path = run_root / "TRAINING_COMPLETE.json"
    declaration = _read_json(declaration_path)
    training_marker = _read_json(training_path)
    if declaration.get("status") != "SFM_B1_OFFLINE_9ARM_DECLARED":
        raise RuntimeError(f"invalid declaration status: {declaration_path}")
    if training_marker.get("status") != (
        "SFM_B1_OFFLINE_9ARM_TRAINING_COMPLETE"
    ):
        raise RuntimeError(f"training is not complete: {training_path}")
    if training_marker.get("declaration_sha256") != BASE.sha256_file(
        declaration_path
    ):
        raise RuntimeError("training marker does not authenticate declaration")

    contract = declaration.get("contract", {})
    if declaration.get("contract_sha256") != RUN._sha256_json(contract):
        raise RuntimeError("declaration contract digest mismatch")
    selector = contract.get("execution_selector", "margin")
    expected = {
        "checkpoint_sha256": RUN.CHECKPOINT_SHA256,
        "scene_profile": RUN.SCENE_PROFILE,
        "rounds": RUN.ROUNDS,
        "alphas": list(RUN.ALPHAS),
        "exposure_epochs": list(RUN.EXPOSURE_EPOCHS),
        "K": RUN.K,
        "B": RUN.B,
        "T": RUN.T,
        "H": RUN.H,
        "cap": RUN.CAP,
        "gp_lambda": RUN.GP_LAMBDA,
        "batch": RUN.BATCH,
        "lr": RUN.LR,
        "ess_target": RUN.ESS_TARGET,
        "eval_M_per_gamma": 50,
        "eval_temperature": 1.0,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"frozen contract mismatch for {key}: "
                f"{contract.get(key)!r} != {value!r}"
            )
    if selector not in ("margin", "safemppi_cost", "balanced_rank"):
        raise RuntimeError(f"unsupported frozen selector: {selector}")
    if contract.get("evaluator_sha256") != BASE.sha256_file(RUN.EVALUATOR):
        raise RuntimeError(
            "recovery evaluator differs from the frozen evaluator"
        )
    checkpoint = Path(contract["checkpoint"]).resolve()
    if BASE.sha256_file(checkpoint) != RUN.CHECKPOINT_SHA256:
        raise RuntimeError("frozen pretrained checkpoint digest mismatch")

    training_source = training_marker.get("source", {})
    if (
        training_source != contract.get("source")
        or training_source.get("commit") is None
        or recovery_source.get("commit") is None
    ):
        raise RuntimeError("training source provenance mismatch")
    arms = list(RUN.arm_grid(selector))
    verifier_workers = int(contract["verifier_workers_per_arm"])
    seed = int(contract["seed"])
    training = {
        arm.name: RUN.validate_training_arm(
            run_root / "arms" / arm.name,
            arm,
            source_commit=training_source["commit"],
            checkpoint_sha256=RUN.CHECKPOINT_SHA256,
            seed=seed,
            verifier_workers=verifier_workers,
        )
        for arm in arms
    }
    return (
        declaration_path,
        training_path,
        contract,
        training_source,
        selector,
        arms,
        training,
        checkpoint,
    )


def recover(args) -> dict:
    started = time.perf_counter()
    run_root = Path(args.run_root).resolve()
    try:
        run_root.relative_to(RUN.RESEARCH_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"--run-root must be below {RUN.RESEARCH_ROOT.resolve()}"
        ) from error
    delivery_path = run_root / "DELIVERY_COMPLETE.json"
    if delivery_path.exists():
        raise FileExistsError(f"delivery already exists: {delivery_path}")

    recovery_source = BASE.source_provenance()
    (
        declaration_path,
        training_path,
        contract,
        training_source,
        selector,
        arms,
        training,
        checkpoint,
    ) = _load_frozen_training(run_root, recovery_source)
    runtime = SimpleNamespace(
        checkpoint=str(checkpoint),
        verifier_workers=int(contract["verifier_workers_per_arm"]),
        seed=int(contract["seed"]),
        eval_ep0=int(contract["eval_ep0"]),
        eval_noise_seed=int(contract["eval_noise_seed"]),
        gpu_indices=args.gpu_indices,
        idle_memory_mib=int(args.idle_memory_mib),
        idle_utilization_percent=int(args.idle_utilization_percent),
    )
    _, _, _, gpus = RUN._select_exactly_two_gpus(runtime)
    allocation = RUN.allocate_arms(arms, gpus)
    pools = BASE.allocate_cpu_pools(
        arms, int(contract["verifier_workers_per_arm"])
    )

    common_r0_dir = run_root / "evaluation" / "common_r0"
    common_r0_metrics = common_r0_dir / "raw_m50_offline_metrics.json"
    if common_r0_metrics.is_file():
        common_payload = _validate_common_r0(common_r0_metrics)
    else:
        if common_r0_dir.exists():
            raise RuntimeError(
                f"refusing to overwrite partial common-r0 output: "
                f"{common_r0_dir}"
            )
        BASE._launch_pending(
            [{
                "arm": RUN.PhaseName("common_r0"),
                "gpu": gpus[0],
                "cpu_pool": next(iter(pools.values())),
                "command": RUN._common_r0_command(runtime, common_r0_dir),
                "target": str(common_r0_dir),
            }],
            run_root / "logs" / "evaluation_recovery_common_r0",
        )
        common_payload = _validate_common_r0(common_r0_metrics)

    jobs = RUN._phase_jobs(
        runtime, arms, gpus, allocation, pools, run_root, "evaluation",
    )
    pending = []
    for job in jobs:
        target = Path(job["target"])
        metrics = target / "raw_m50_offline_metrics.json"
        if metrics.is_file():
            continue
        if target.exists():
            raise RuntimeError(
                f"refusing to overwrite partial arm evaluation: {target}"
            )
        pending.append(job)
    if pending:
        BASE._launch_pending(
            pending, run_root / "logs" / "evaluation_recovery",
        )

    evaluations = {
        arm.name: RUN.validate_evaluation(
            run_root / "evaluation" / arm.name,
            arm,
            training[arm.name],
            eval_ep0=runtime.eval_ep0,
            eval_noise_seed=runtime.eval_noise_seed,
        )
        for arm in arms
    }
    r0_keys = {value["r0_cell_key"] for value in evaluations.values()}
    noise_hashes = {
        value["noise_bank_sha256"] for value in evaluations.values()
    }
    if len(r0_keys) != 1 or len(noise_hashes) != 1:
        raise RuntimeError("recovered evaluations do not share one raw-M50 bank")
    if next(iter(r0_keys)) != (
        common_payload["records"][0]["cell"]["cell_key"]
    ):
        raise RuntimeError("recovered arm r0 differs from common r0")

    aggregate_dir = run_root / "evaluation" / "aggregate"
    if aggregate_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite existing aggregate: {aggregate_dir}"
        )
    aggregate_result = RUN.aggregate(
        evaluations, aggregate_dir, selector=selector,
    )
    manifest = {
        "status": "SFM_B1_OFFLINE_9ARM_DELIVERY_COMPLETE",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "source": training_source,
        "recovery_source": recovery_source,
        "recovery_role": (
            "evaluation-only recovery; authenticated training checkpoints "
            "were not modified or regenerated"
        ),
        "contract": contract,
        "declaration": str(declaration_path),
        "declaration_sha256": BASE.sha256_file(declaration_path),
        "training_marker": str(training_path),
        "training_marker_sha256": BASE.sha256_file(training_path),
        "training": training,
        "evaluations": evaluations,
        "common_r0_metrics": str(common_r0_metrics),
        "common_r0_metrics_sha256": BASE.sha256_file(common_r0_metrics),
        "common_r0_cell_key": next(iter(r0_keys)),
        "common_noise_bank_sha256": next(iter(noise_hashes)),
        "aggregate": aggregate_result,
    }
    RUN._write_json(delivery_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "selector": selector,
        "wall_seconds": manifest["wall_seconds"],
        "best_screening_cell": aggregate_result["best_screening_cell"],
        "delivery": str(delivery_path),
    }, indent=2, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpu-indices", default="1,3")
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--idle-utilization-percent", type=int, default=5)
    return parser


if __name__ == "__main__":
    recover(_parser().parse_args())
