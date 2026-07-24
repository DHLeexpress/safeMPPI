#!/usr/bin/env python3
"""Frozen end-to-end launcher for the offline executed-window 9-arm study.

The two phases are deliberately separate:

1. train alpha {0,.01,.1} x exposure epochs {1,10,100} for ten rounds;
2. evaluate every r0--r10 checkpoint with the same raw temperature-one
   M=50/gamma bank and terminal-truncated executed-window Validity.

All nine jobs in a phase start concurrently on four exclusive GPUs with a
deterministic 3/2/2/2 allocation.  Any child failure stops its peers.  The
output root must not exist, so a partial study can never be mistaken for a
resumed or complete scientific run.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import run_sfm_b1_r2_9arm as BASE


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAINER = HERE / "sfm_b1_offline_exec.py"
EVALUATOR = HERE / "sfm_b1_offline_eval.py"
ALPHAS = (0.0, 0.01, 0.1)
EXPOSURE_EPOCHS = (1, 10, 100)
ROUNDS = 10
ARM_STATUS = "SFM_B1_OFFLINE_EXEC_COMPLETE"
EVAL_STATUS = "SFM_B1_OFFLINE_RAW_M50_COMPLETE"
CHECKPOINT_SHA256 = (
    "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
)
SCENE_PROFILE = "double_density_velocity_ood"
ELL = 0.24210826720721101
CAP = 512
GP_LAMBDA = 1.0e-2
K = 16
B = 4
T = 180
H = 10
BATCH = 128
LR = 1.0e-4
ESS_TARGET = 0.5
RESEARCH_ROOT = Path("/data3/research1")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str | os.PathLike[str], payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _sha256_json(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Arm:
    alpha: float
    exposure_epochs: int

    @property
    def name(self) -> str:
        alpha = str(float(self.alpha)).replace(".", "p")
        return (
            f"offline_exec_alpha{alpha}_"
            f"exposures{int(self.exposure_epochs):03d}"
        )


@dataclass(frozen=True)
class PhaseName:
    name: str


def arm_grid() -> tuple[Arm, ...]:
    return tuple(
        Arm(alpha, epochs)
        for alpha in ALPHAS
        for epochs in EXPOSURE_EPOCHS
    )


def _validated_output_root(value: str | os.PathLike[str]) -> Path:
    path = Path(value).resolve()
    research_root = RESEARCH_ROOT.resolve()
    try:
        path.relative_to(research_root)
    except ValueError as error:
        raise ValueError(
            f"--outdir must be below {research_root}, got {path}"
        ) from error
    if path.exists():
        raise FileExistsError(
            f"scientific output root must not already exist: {path}"
        )
    return path


def allocate_arms(
    arms: list[Arm], gpus: list[BASE.GPU],
) -> dict[str, list[Arm]]:
    """Use all four GPUs with the intended 3/2/2/2 workload split."""
    if len(gpus) != 4:
        raise RuntimeError(f"exactly four idle GPUs are required, got {len(gpus)}")
    if set(arms) != set(arm_grid()):
        raise ValueError("offline launcher requires the complete declared arm grid")
    ordered_gpus = sorted(gpus, key=lambda gpu: int(gpu.index))
    by_epochs = {
        epochs: sorted(
            [arm for arm in arms if arm.exposure_epochs == epochs],
            key=lambda arm: arm.alpha,
        )
        for epochs in EXPOSURE_EPOCHS
    }
    allocation = {gpu.uuid: [] for gpu in ordered_gpus}
    allocation[ordered_gpus[0].uuid].extend(by_epochs[1])
    for gpu, ten, hundred in zip(
        ordered_gpus[1:], by_epochs[10], by_epochs[100]
    ):
        allocation[gpu.uuid].extend((ten, hundred))
    counts = sorted(len(values) for values in allocation.values())
    if counts != [2, 2, 2, 3] or any(not values for values in allocation.values()):
        raise RuntimeError(f"invalid four-GPU allocation: {counts}")
    return allocation


def _trainer_command(args, arm: Arm, output: Path) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--outdir",
        str(output.resolve()),
        "--alpha",
        str(arm.alpha),
        "--exposure-epochs",
        str(arm.exposure_epochs),
        "--rounds",
        str(ROUNDS),
        "--verifier-workers",
        str(args.verifier_workers),
        "--seed",
        str(args.seed),
        "--device",
        "cuda:0",
    ]


def _evaluation_command(
    args, arm: Arm, arm_dir: Path, output: Path, *, cache_dir: Path,
) -> list[str]:
    # Use the one promoted source file for r0.  Per-arm round_00 containers
    # embed arm-specific recipe metadata and therefore have different file
    # hashes despite identical model tensors.
    checkpoints = [str(Path(args.checkpoint).resolve())] + [
        str((arm_dir / f"round_{round_i:02d}.pt").resolve())
        for round_i in range(1, ROUNDS + 1)
    ]
    labels = [f"r{round_i}" for round_i in range(ROUNDS + 1)]
    return [
        sys.executable,
        str(EVALUATOR),
        "--checkpoints",
        *checkpoints,
        "--labels",
        *labels,
        "--scene-profile",
        SCENE_PROFILE,
        "--ep0",
        str(args.eval_ep0),
        "--noise-seed",
        str(args.eval_noise_seed),
        "--device",
        "cuda:0",
        "--workers",
        str(args.verifier_workers),
        "--cache-dir",
        str(cache_dir.resolve()),
        "--output-dir",
        str(output.resolve()),
    ]


def _common_r0_command(args, output: Path) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR),
        "--checkpoints",
        str(Path(args.checkpoint).resolve()),
        "--labels",
        "r0",
        "--scene-profile",
        SCENE_PROFILE,
        "--ep0",
        str(args.eval_ep0),
        "--noise-seed",
        str(args.eval_noise_seed),
        "--device",
        "cuda:0",
        "--workers",
        str(args.verifier_workers),
        "--cache-dir",
        str((output / "cache").resolve()),
        "--output-dir",
        str(output.resolve()),
    ]


def _validate_sidecar(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    digest = BASE.sha256_file(path)
    sidecar = Path(str(path) + ".COMPLETE.json")
    if not sidecar.is_file():
        raise RuntimeError(f"missing artifact sidecar: {sidecar}")
    with sidecar.open() as stream:
        payload = json.load(stream)
    if payload.get("sha256") != digest:
        raise RuntimeError(f"artifact sidecar digest mismatch: {sidecar}")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "sidecar": str(sidecar.resolve()),
        "sidecar_sha256": BASE.sha256_file(sidecar),
        "sidecar_payload": payload,
    }


def validate_training_arm(
    arm_dir: Path,
    arm: Arm,
    *,
    source_commit: str,
    checkpoint_sha256: str,
    seed: int,
    verifier_workers: int,
) -> dict:
    marker = arm_dir / "COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"missing arm completion marker: {marker}")
    with marker.open() as stream:
        payload = json.load(stream)
    if payload.get("status") != ARM_STATUS:
        raise RuntimeError(f"invalid arm status: {marker}")
    if payload.get("experiment") != arm.name:
        raise RuntimeError(f"arm identity mismatch: {marker}")
    expected_recipe = {
        "alpha": float(arm.alpha),
        "exposure_epochs": int(arm.exposure_epochs),
        "rounds": ROUNDS,
        "K": K,
        "B": B,
        "T": T,
        "H": H,
        "batch": BATCH,
        "lr": LR,
        "ess_target": ESS_TARGET,
        "nfe": 8,
        "temp": 1.0,
        "phi_s": 0.9,
        "gp_lam": GP_LAMBDA,
        "verifier_workers": int(verifier_workers),
        "seed": int(seed),
        "scene_profile": SCENE_PROFILE,
        "smoke": False,
    }
    if payload.get("recipe") != expected_recipe:
        raise RuntimeError(f"training recipe mismatch: {marker}")
    expected_constants = {
        "ell": ELL,
        "gp_buffer_cap": CAP,
        "gp_lambda": GP_LAMBDA,
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "replay_window_rounds": 1,
        "gp_quota_semantics": (
            "exactly 73 executed D+ rows per gamma plus one rotating "
            "extra; any support shortage aborts the scientific round"
        ),
        "ess_target_semantics": (
            "mean normalized ESS over each sequential remaining pool"
        ),
    }
    if payload.get("constants") != expected_constants:
        raise RuntimeError(f"training constants mismatch: {marker}")
    if payload.get("source_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"source checkpoint mismatch: {marker}")
    source = payload.get("source", {})
    if (
        source.get("commit") != source_commit
        or source.get("tracked_worktree_clean") is not True
    ):
        raise RuntimeError(f"trainer source provenance mismatch: {marker}")
    if payload.get("scientific_role") != (
        "offline_expansion_data_collector_not_safe_controller"
    ):
        raise RuntimeError(f"collector role mismatch: {marker}")

    checkpoints = []
    for round_i in range(ROUNDS + 1):
        checkpoint = _validate_sidecar(
            arm_dir / f"round_{round_i:02d}.pt"
        )
        if checkpoint["sidecar_payload"].get("status") != "COMPLETE":
            raise RuntimeError(f"invalid checkpoint sidecar: {checkpoint['sidecar']}")
        checkpoints.append({"round": round_i, **checkpoint})

    history = payload.get("history")
    if not isinstance(history, list) or [
        int(row.get("round", -1)) for row in history
    ] != list(range(1, ROUNDS + 1)):
        raise RuntimeError(f"arm must contain rounds 1--{ROUNDS}: {marker}")
    rounds = []
    for row in history:
        round_i = int(row["round"])
        if row.get("experiment") != arm.name:
            raise RuntimeError(f"round experiment mismatch: {marker}")
        if row.get("checkpoint_sha256") != checkpoints[round_i]["sha256"]:
            raise RuntimeError(f"round checkpoint digest mismatch: {marker}")
        gp_selection = row.get("gp_selection", {})
        expected_gp_count = 0 if round_i == 1 else CAP
        per_gamma_gp = gp_selection.get("per_gamma", {})
        if (
            int(gp_selection.get("requested_cap", -1)) != CAP
            or int(gp_selection.get("quota", -1)) != CAP // 7
            or int(gp_selection.get("selected", -1)) != expected_gp_count
            or sum(int(value) for value in per_gamma_gp.values())
            != expected_gp_count
            or len(row.get("gp_buffer_ids", [])) != expected_gp_count
            or len({
                tuple(identity) for identity in row.get("gp_buffer_ids", [])
            }) != expected_gp_count
        ):
            raise RuntimeError(f"previous-round GP contract mismatch in round {round_i}")
        if round_i > 1 and gp_selection.get("unique") is not True:
            raise RuntimeError(f"GP buffer is not unique in round {round_i}")
        if round_i > 1:
            extra_gamma = (
                0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0
            )[(round_i - 2) % 7]
            expected_per_gamma = {
                str(gamma): 73 + int(gamma == extra_gamma)
                for gamma in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
            }
            if per_gamma_gp != expected_per_gamma:
                raise RuntimeError(
                    f"strict gamma GP quota mismatch in round {round_i}"
                )
        if "outcomes" in row:
            raise RuntimeError("outcomes must be stored only inside gather")
        shard = row.get("shard", {})
        shard_path = Path(shard.get("path", ""))
        shard_artifact = _validate_sidecar(shard_path)
        if shard_artifact["sidecar_payload"].get("status") != (
            "OFFLINE_EXECUTED_ROUND_SHARD_COMPLETE"
        ):
            raise RuntimeError(f"invalid executed shard sidecar: {shard_path}")
        if shard_artifact["sha256"] != shard.get("sha256"):
            raise RuntimeError(f"executed shard digest mismatch: {shard_path}")
        gather = row.get("gather", {})
        if len(gather.get("outcomes", [])) != 56:
            raise RuntimeError(f"round {round_i} must contain 56 episode outcomes")
        counts = gather.get("counts", {})
        summary = gather.get("shard", {})
        contexts = int(counts.get("contexts", -1))
        if int(counts.get("B_queries", -1)) != contexts * B:
            raise RuntimeError(f"B query accounting mismatch in round {round_i}")
        if (
            int(summary.get("contexts", -1)) != contexts
            or int(summary.get("D", -1)) != contexts
            or int(summary.get("Dplus", -1))
            + int(summary.get("Dminus", -1)) != contexts
            or int(summary.get("errors", -1)) != 0
            or int(summary.get("unresolved_contexts", -1)) != 0
        ):
            raise RuntimeError(f"executed D partition mismatch in round {round_i}")
        replay = row.get("replay", {})
        dplus = int(summary["Dplus"])
        dminus = int(summary["Dminus"])
        expected_batches = math.ceil((dplus + dminus) / BATCH)
        expected_steps = expected_batches * int(arm.exposure_epochs)
        if (
            replay.get("exact_once_per_exposure_epoch") is not True
            or int(replay.get("positive_eligible", -1)) != dplus
            or int(replay.get("negative_eligible", -1)) != dminus
            or int(replay.get("positive_total_visits", -1))
            != dplus * int(arm.exposure_epochs)
            or int(replay.get("negative_total_visits", -1))
            != dminus * int(arm.exposure_epochs)
            or int(replay.get("optimizer_steps", -1)) != expected_steps
            or bool(replay.get("negative_used_for_training"))
            != bool(float(arm.alpha) > 0.0 and dminus)
        ):
            raise RuntimeError(f"offline replay accounting mismatch in round {round_i}")
        if replay.get("visual_encoder_sha_before") != replay.get(
            "visual_encoder_sha_after"
        ):
            raise RuntimeError(f"visual encoder changed in round {round_i}")
        rounds.append({
            "round": round_i,
            "D": contexts,
            "Dplus": dplus,
            "Dminus": dminus,
            "optimizer_steps": expected_steps,
            "shard": shard_artifact,
        })
    return {
        "arm": arm.name,
        "alpha": arm.alpha,
        "exposure_epochs": arm.exposure_epochs,
        "marker": str(marker.resolve()),
        "marker_sha256": BASE.sha256_file(marker),
        "checkpoints": checkpoints,
        "rounds": rounds,
    }


def validate_evaluation(
    output: Path,
    arm: Arm,
    training: dict,
    *,
    eval_ep0: int,
    eval_noise_seed: int,
) -> dict:
    metrics = output / "raw_m50_offline_metrics.json"
    if not metrics.is_file():
        raise RuntimeError(f"missing evaluation metrics: {metrics}")
    with metrics.open() as stream:
        payload = json.load(stream)
    if (
        payload.get("status") != EVAL_STATUS
        or payload.get("scene_profile") != SCENE_PROFILE
        or int(payload.get("bank", {}).get("ep0", -1)) != int(eval_ep0)
        or int(payload.get("bank", {}).get("M_per_gamma", -1)) != 50
        or int(payload.get("noise_bank", {}).get("seed", -1))
        != int(eval_noise_seed)
        or float(payload.get("noise_bank", {}).get("temperature", -1))
        != 1.0
    ):
        raise RuntimeError(f"evaluation contract mismatch: {metrics}")
    records = payload.get("records")
    if not isinstance(records, list) or [
        int(row.get("round", -1)) for row in records
    ] != list(range(ROUNDS + 1)):
        raise RuntimeError(f"evaluation must contain r0--r{ROUNDS}: {metrics}")
    expected_hashes = [CHECKPOINT_SHA256] + [
        row["sha256"] for row in training["checkpoints"][1:]
    ]
    for record, expected_hash in zip(records, expected_hashes):
        cell = record.get("cell", {})
        if (
            cell.get("status") != "SFM_B1_OFFLINE_RAW_CELL_COMPLETE"
            or cell.get("checkpoint_sha256") != expected_hash
            or int(cell.get("M_per_gamma", -1)) != 50
            or int(cell.get("summary", {}).get("pooled", {}).get(
                "verifier_errors", -1
            )) != 0
        ):
            raise RuntimeError(f"invalid evaluation cell: {metrics}")
        pooled = cell["summary"]["pooled"]
        if not math.isclose(
            float(pooled["SR"]) + float(pooled["CR"])
            + float(pooled["timeout"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"evaluation outcomes do not partition: {metrics}")
    expected_outputs = [
        output / "raw_m50_offline_curves.png",
        output / "raw_m50_offline_curves.pdf",
        output / "raw_m50_offline_curves.figure.json",
    ]
    artifacts = [
        {"path": str(path.resolve()), "sha256": BASE.sha256_file(path)}
        for path in [metrics, *expected_outputs]
        if path.is_file()
    ]
    if len(artifacts) != 4:
        raise RuntimeError(f"missing evaluation presentation artifact: {output}")
    return {
        "arm": arm.name,
        "metrics": str(metrics.resolve()),
        "metrics_sha256": BASE.sha256_file(metrics),
        "records": records,
        "noise_bank_sha256": payload["noise_bank"]["sha256"],
        "r0_cell_key": records[0]["cell"]["cell_key"],
        "artifacts": artifacts,
    }


def _cell_row(arm: Arm, record: dict) -> dict:
    pooled = record["cell"]["summary"]["pooled"]
    clearance = pooled["successful_clearance"]["mean"]
    time_to_goal = pooled["successful_time_to_goal"]["mean"]
    return {
        "arm": arm.name,
        "alpha": float(arm.alpha),
        "exposure_epochs": int(arm.exposure_epochs),
        "round": int(record["round"]),
        "SR": float(pooled["SR"]),
        "CR": float(pooled["CR"]),
        "timeout": float(pooled["timeout"]),
        "Validity": float(pooled["Validity"]["mean"]),
        "clearance": None if clearance is None else float(clearance),
        "time_to_goal": None if time_to_goal is None else float(time_to_goal),
    }


def _screening_key(row: dict) -> tuple:
    clearance = (
        -float(row["clearance"])
        if row["clearance"] is not None else float("inf")
    )
    time_to_goal = (
        float(row["time_to_goal"])
        if row["time_to_goal"] is not None else float("inf")
    )
    return (
        float(row["CR"]),
        -float(row["Validity"]),
        -float(row["SR"]),
        clearance,
        time_to_goal,
        int(row["round"]),
        int(row["exposure_epochs"]),
        float(row["alpha"]),
    )


def _render_aggregate(rows: list[dict], output: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#0072B2", 10: "#E69F00", 100: "#CC79A7"}
    linestyles = {0.0: "-", 0.01: "--", 0.1: ":"}
    specs = (
        ("CR", "Collision rate", (-0.03, 1.03)),
        ("Validity", "Validity", (-0.03, 1.03)),
        ("clearance", "Min. clearance [m]", None),
        ("time_to_goal", "Time-to-goal [s]", None),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), squeeze=False)
    for axis, (key, title, ylim) in zip(axes.flat, specs):
        for arm in arm_grid():
            values = [
                row for row in rows if row["arm"] == arm.name
            ]
            values.sort(key=lambda row: int(row["round"]))
            axis.plot(
                [row["round"] for row in values],
                [
                    float("nan") if row[key] is None else row[key]
                    for row in values
                ],
                color=colors[arm.exposure_epochs],
                linestyle=linestyles[arm.alpha],
                linewidth=1.8,
                alpha=0.85,
            )
        axis.set_title(title)
        axis.set_xlabel("Expansion round")
        axis.set_xticks(range(ROUNDS + 1))
        axis.grid(alpha=0.25)
        if ylim is not None:
            axis.set_ylim(*ylim)
    handles = [
        plt.Line2D(
            [0], [0], color=colors[epochs], lw=2.5,
            label=f"{epochs} exposure epochs",
        )
        for epochs in EXPOSURE_EPOCHS
    ]
    handles.extend(
        plt.Line2D(
            [0], [0], color="black", linestyle=linestyles[alpha],
            lw=2.0, label=rf"$\alpha={alpha:g}$",
        )
        for alpha in ALPHAS
    )
    figure.legend(
        handles=handles, ncol=6, loc="upper center", frameon=False
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    artifacts = []
    for suffix in ("png", "pdf"):
        path = output / f"factorial_raw_m50_pooled.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        artifacts.append({
            "path": str(path.resolve()),
            "sha256": BASE.sha256_file(path),
        })
    plt.close(figure)
    return artifacts


def aggregate(evaluations: dict[str, dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for arm in arm_grid():
        rows.extend(
            _cell_row(arm, record)
            for record in evaluations[arm.name]["records"]
        )
    csv_path = output / "factorial_raw_m50_metrics.csv"
    fields = (
        "arm", "alpha", "exposure_epochs", "round",
        "SR", "CR", "timeout", "Validity", "clearance", "time_to_goal",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    candidates = [row for row in rows if int(row["round"]) > 0]
    best = min(candidates, key=_screening_key)
    figures = _render_aggregate(rows, output)
    result = {
        "status": "SFM_B1_OFFLINE_9ARM_AGGREGATE_COMPLETE",
        "selection_role": (
            "exploratory common-bank M50 screening only; not an independent "
            "confirmation or a probabilistic safety guarantee"
        ),
        "selection_rule": (
            "post-expansion only: lower CR, higher window Validity, higher SR, "
            "higher successful-only clearance, lower successful-only time, "
            "then earlier round/lower exposure/lower alpha"
        ),
        "best_screening_cell": best,
        "rows": rows,
        "artifacts": [
            {
                "path": str(csv_path.resolve()),
                "sha256": BASE.sha256_file(csv_path),
            },
            *figures,
        ],
    }
    path = output / "AGGREGATE_COMPLETE.json"
    _write_json(path, result)
    result["marker"] = str(path.resolve())
    result["marker_sha256"] = BASE.sha256_file(path)
    return result


def _select_exactly_four_gpus(args):
    gpus, processes, topology = BASE.gpu_snapshot()
    selected = BASE.select_idle_gpus(
        gpus,
        processes,
        args.gpu_indices,
        max_memory_mib=args.idle_memory_mib,
        max_utilization=args.idle_utilization_percent,
    )
    if len(selected) != 4:
        raise RuntimeError(
            f"the declared study requires four exclusive GPUs, got "
            f"{[gpu.index for gpu in selected]}"
        )
    return gpus, processes, topology, selected


def _phase_jobs(args, arms, selected, allocation, pools, outdir, phase):
    by_uuid = {gpu.uuid: gpu for gpu in selected}
    arm_gpu = {
        arm: by_uuid[uuid]
        for uuid, values in allocation.items()
        for arm in values
    }
    jobs = []
    for arm in arms:
        if phase == "training":
            target = outdir / "arms" / arm.name
            command = _trainer_command(args, arm, target)
        elif phase == "evaluation":
            target = outdir / "evaluation" / arm.name
            command = _evaluation_command(
                args,
                arm,
                outdir / "arms" / arm.name,
                target,
                cache_dir=outdir / "evaluation" / "common_r0" / "cache",
            )
        else:
            raise ValueError(phase)
        jobs.append({
            "arm": arm,
            "gpu": arm_gpu[arm],
            "cpu_pool": pools[arm.name],
            "command": command,
            "target": str(target.resolve()),
        })
    return jobs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=CHECKPOINT_SHA256,
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gpu-indices", default="0,1,2,3")
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--eval-ep0", type=int, default=260000)
    parser.add_argument("--eval-noise-seed", type=int, default=20260723)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--idle-utilization-percent", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args) -> dict:
    if not 1 <= int(args.verifier_workers) <= 8:
        raise ValueError("--verifier-workers must be in [1,8]")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_checkpoint_sha = BASE.sha256_file(checkpoint)
    if (
        args.expected_checkpoint_sha256 != CHECKPOINT_SHA256
        or observed_checkpoint_sha != CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            f"checkpoint SHA mismatch: {observed_checkpoint_sha} != "
            f"{CHECKPOINT_SHA256}"
        )
    for module in (TRAINER, EVALUATOR):
        if not module.is_file():
            raise FileNotFoundError(module)
    outdir = _validated_output_root(args.outdir)
    source = BASE.source_provenance()
    arms = list(arm_grid())
    all_gpus, processes, topology, selected = _select_exactly_four_gpus(args)
    allocation = allocate_arms(arms, selected)
    pools = BASE.allocate_cpu_pools(arms, int(args.verifier_workers))
    training_jobs = _phase_jobs(
        args, arms, selected, allocation, pools, outdir, "training",
    )
    contract = {
        "version": 1,
        "source": source,
        "launcher_sha256": BASE.sha256_file(__file__),
        "trainer_sha256": BASE.sha256_file(TRAINER),
        "evaluator_sha256": BASE.sha256_file(EVALUATOR),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": observed_checkpoint_sha,
        "scene_profile": SCENE_PROFILE,
        "rounds": ROUNDS,
        "alphas": list(ALPHAS),
        "exposure_epochs": list(EXPOSURE_EPOCHS),
        "K": K,
        "B": B,
        "T": T,
        "H": H,
        "ell": ELL,
        "cap": CAP,
        "gp_lambda": GP_LAMBDA,
        "batch": BATCH,
        "lr": LR,
        "ess_target": ESS_TARGET,
        "seed": int(args.seed),
        "eval_ep0": int(args.eval_ep0),
        "eval_noise_seed": int(args.eval_noise_seed),
        "eval_M_per_gamma": 50,
        "eval_temperature": 1.0,
        "verifier_workers_per_arm": int(args.verifier_workers),
        "gpu_indices": [gpu.index for gpu in selected],
        "gpu_uuids": [gpu.uuid for gpu in selected],
    }
    declaration = {
        "status": "SFM_B1_OFFLINE_9ARM_DECLARED",
        "created_at": _utc_now(),
        "contract": contract,
        "contract_sha256": _sha256_json(contract),
        "all_gpus": [asdict(gpu) for gpu in all_gpus],
        "compute_processes": processes,
        "topology": topology,
        "allocation": {
            gpu.index: [arm.name for arm in allocation[gpu.uuid]]
            for gpu in selected
        },
        "training_jobs": [
            {
                "arm": job["arm"].name,
                "gpu_index": job["gpu"].index,
                "gpu_uuid": job["gpu"].uuid,
                "cpu_pool": job["cpu_pool"],
                "command": job["command"],
                "target": job["target"],
            }
            for job in training_jobs
        ],
    }
    if args.dry_run:
        print(json.dumps(declaration, indent=2, allow_nan=False))
        return declaration

    outdir.mkdir(parents=True)
    declaration_path = outdir / "RUN_DECLARATION.json"
    _write_json(declaration_path, declaration)
    started = time.perf_counter()
    for job in training_jobs:
        job["log_path"] = str(
            (
                outdir / "logs" / "training"
                / f"{job['arm'].name}.log"
            ).resolve()
        )
    BASE._launch_pending(training_jobs, outdir / "logs" / "training")
    training = {
        arm.name: validate_training_arm(
            outdir / "arms" / arm.name,
            arm,
            source_commit=source["commit"],
            checkpoint_sha256=observed_checkpoint_sha,
            seed=args.seed,
            verifier_workers=args.verifier_workers,
        )
        for arm in arms
    }
    training_marker = outdir / "TRAINING_COMPLETE.json"
    _write_json(training_marker, {
        "status": "SFM_B1_OFFLINE_9ARM_TRAINING_COMPLETE",
        "finished_at": _utc_now(),
        "source": source,
        "declaration_sha256": BASE.sha256_file(declaration_path),
        "arms": training,
    })

    # Recheck exclusivity between phases.  A foreign job that appeared while
    # training ran must not be silently shared with the common-bank evaluator.
    _, _, _, evaluation_gpus = _select_exactly_four_gpus(args)
    if [gpu.uuid for gpu in evaluation_gpus] != [
        gpu.uuid for gpu in selected
    ]:
        raise RuntimeError("GPU identity changed between training and evaluation")
    evaluation_allocation = allocate_arms(arms, evaluation_gpus)
    common_r0_dir = outdir / "evaluation" / "common_r0"
    BASE._launch_pending(
        [{
            "arm": PhaseName("common_r0"),
            "gpu": evaluation_gpus[0],
            "cpu_pool": next(iter(pools.values())),
            "command": _common_r0_command(args, common_r0_dir),
            "target": str(common_r0_dir.resolve()),
        }],
        outdir / "logs" / "evaluation_common_r0",
    )
    common_r0_metrics = common_r0_dir / "raw_m50_offline_metrics.json"
    if not common_r0_metrics.is_file():
        raise RuntimeError("common r0 evaluation did not produce its metrics")
    with common_r0_metrics.open() as stream:
        common_r0_payload = json.load(stream)
    common_records = common_r0_payload.get("records", [])
    if (
        common_r0_payload.get("status") != EVAL_STATUS
        or len(common_records) != 1
        or int(common_records[0].get("round", -1)) != 0
        or common_records[0].get("cell", {}).get("checkpoint_sha256")
        != CHECKPOINT_SHA256
    ):
        raise RuntimeError("common r0 evaluation contract mismatch")
    evaluation_jobs = _phase_jobs(
        args,
        arms,
        evaluation_gpus,
        evaluation_allocation,
        pools,
        outdir,
        "evaluation",
    )
    for job in evaluation_jobs:
        job["log_path"] = str(
            (
                outdir / "logs" / "evaluation"
                / f"{job['arm'].name}.log"
            ).resolve()
        )
    BASE._launch_pending(evaluation_jobs, outdir / "logs" / "evaluation")
    evaluations = {
        arm.name: validate_evaluation(
            outdir / "evaluation" / arm.name,
            arm,
            training[arm.name],
            eval_ep0=args.eval_ep0,
            eval_noise_seed=args.eval_noise_seed,
        )
        for arm in arms
    }
    r0_cell_keys = {value["r0_cell_key"] for value in evaluations.values()}
    noise_hashes = {
        value["noise_bank_sha256"] for value in evaluations.values()
    }
    if len(r0_cell_keys) != 1 or len(noise_hashes) != 1:
        raise RuntimeError(
            "the nine evaluations do not share an identical r0/common bank"
        )
    aggregate_result = aggregate(
        evaluations, outdir / "evaluation" / "aggregate",
    )
    manifest = {
        "status": "SFM_B1_OFFLINE_9ARM_DELIVERY_COMPLETE",
        "finished_at": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "source": source,
        "contract": contract,
        "declaration": str(declaration_path.resolve()),
        "declaration_sha256": BASE.sha256_file(declaration_path),
        "training_marker": str(training_marker.resolve()),
        "training_marker_sha256": BASE.sha256_file(training_marker),
        "training": training,
        "evaluations": evaluations,
        "common_r0_metrics": str(common_r0_metrics.resolve()),
        "common_r0_metrics_sha256": BASE.sha256_file(common_r0_metrics),
        "common_r0_cell_key": next(iter(r0_cell_keys)),
        "common_noise_bank_sha256": next(iter(noise_hashes)),
        "aggregate": aggregate_result,
    }
    delivery = outdir / "DELIVERY_COMPLETE.json"
    _write_json(delivery, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "wall_seconds": manifest["wall_seconds"],
        "best_screening_cell": aggregate_result["best_screening_cell"],
        "delivery": str(delivery.resolve()),
    }, indent=2, allow_nan=False))
    return manifest


def main(argv=None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
