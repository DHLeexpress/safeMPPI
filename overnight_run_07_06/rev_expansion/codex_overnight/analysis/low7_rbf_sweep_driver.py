#!/usr/bin/env python3
"""Fail-closed 48-arm low7 RBF training and raw-M50 evaluation sweep."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Callable, Iterable


CODEX_ROOT = Path(__file__).resolve().parents[1]
TRAINER = CODEX_ROOT / "grid_expand_afe_rbf.py"
EVALUATOR = CODEX_ROOT / "paper_results" / "low7_raw_m50_eval.py"
DIAGNOSTICS = CODEX_ROOT / "analysis" / "afe_rbf_sweep_diagnostics.py"
VIDEO = CODEX_ROOT / "video_afe2.py"
SUPPORTED_SCENES = (
    "low7_radius1_canonical_v1",
    "low7_radius03_canonical_v1",
)
ELL_MULTIPLIERS = (0.5, 1.0)
NEGATIVE_ALPHAS = (0.0, 0.001, 0.005)
AFE_STEPS = (4, 16, 32, 64)
EXECUTION_RULES = (
    "nominal_hp_max_step_progress",
    "nominal_hp_max_step_margin",
)
SEED = 910
RAW_M = 50
GAMMA_COUNT = 7
EVALUATION_ROUNDS = tuple(range(0, 101, 10))
GLOBAL_RANKING_RULE = (
    "across each arm's validated rank-1 pooled true raw-M50 checkpoint: maximize SR, "
    "minimize CR, minimize timeout, maximize mean minimum clearance, prefer earlier "
    "checkpoint round; arm_id is used only as a deterministic exact-tie key"
)


@dataclass(frozen=True)
class Arm:
    lengthscale_multiplier: float
    negative_alpha: float
    afe_steps: int
    execution_rule: str

    @property
    def arm_id(self) -> str:
        execution = self.execution_rule.removeprefix("nominal_hp_max_step_")
        ell = f"{self.lengthscale_multiplier:g}".replace(".", "p")
        alpha = f"{self.negative_alpha:g}".replace(".", "p")
        return f"ell{ell}_alpha{alpha}_steps{self.afe_steps:03d}_{execution}"

    def record(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "lengthscale_multiplier": self.lengthscale_multiplier,
            "negative_alpha": self.negative_alpha,
            "negative_alpha_semantics": "paper gradient-norm normalization",
            "afe_steps": self.afe_steps,
            "execution_rule": self.execution_rule,
        }


def sweep_arms() -> list[Arm]:
    return [
        Arm(ell, alpha, steps, execution)
        for ell, alpha, steps, execution in itertools.product(
            ELL_MULTIPLIERS,
            NEGATIVE_ALPHAS,
            AFE_STEPS,
            EXECUTION_RULES,
        )
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_video(path: Path, expected_frames: int) -> dict:
    payload = json.loads(subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames",
            "-of", "json", str(path),
        ],
        text=True,
    ))
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"best video must contain exactly one video stream: {payload}")
    stream = streams[0]
    try:
        frames = int(stream["nb_read_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"best video has incomplete ffprobe metadata: {stream}") from error
    if frames != int(expected_frames) or width < 2 or height < 2:
        raise RuntimeError(
            f"best video is not the declared {expected_frames}-frame artifact: {stream}"
        )
    return {
        "codec_name": stream.get("codec_name"),
        "width": width,
        "height": height,
        "frames": frames,
    }


def write_json_new(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def source_record() -> dict:
    repository = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=CODEX_ROOT, text=True
    ).strip())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=repository).returncode != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repository
        ).returncode != 0
    )
    untracked_runtime = [
        item
        for item in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repository,
            text=True,
        ).splitlines()
        if item.endswith((".py", ".sh"))
    ]
    if tracked_dirty or untracked_runtime:
        raise RuntimeError(
            "sweep requires committed clean source; "
            f"tracked_dirty={tracked_dirty}, untracked_runtime_sources={untracked_runtime}"
        )
    runtime_sources = (
        Path(__file__).resolve(),
        CODEX_ROOT / "run_low7_rbf_sweep.sh",
        TRAINER,
        CODEX_ROOT / "afe_rbf_core.py",
        CODEX_ROOT / "afe_signed_update.py",
        CODEX_ROOT / "afe_execution.py",
        CODEX_ROOT / "afe_context.py",
        CODEX_ROOT / "grid_expand_afe2.py",
        EVALUATOR,
        DIAGNOSTICS,
        VIDEO,
    )
    return {
        "repository": str(repository),
        "git_commit": commit,
        "tracked_dirty": False,
        "untracked_runtime_sources": [],
        "runtime_source_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in runtime_sources
        },
    }


def gpu_record(physical_index: int, expected_uuid: str) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(int(physical_index)):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the requested physical index; "
            f"expected {physical_index}, got {visible!r}"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable} must equal 1")
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(int(physical_index)),
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [field.strip() for field in line.split(",", 4)]
    if len(fields) != 5 or fields[0] != str(int(physical_index)):
        raise RuntimeError(f"nvidia-smi returned an unexpected physical GPU record: {line}")
    if fields[1].lower() != expected_uuid.lower():
        raise RuntimeError(
            f"physical GPU UUID {fields[1]} != expected UUID {expected_uuid}"
        )
    active_compute_pids = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(int(physical_index)),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if active_compute_pids:
        raise RuntimeError(
            f"physical GPU {physical_index} already has active compute PIDs: "
            f"{active_compute_pids.splitlines()}"
        )
    return {
        "physical_index": int(physical_index),
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
        "memory_total_mib": int(fields[4]),
        "cuda_visible_devices": visible,
        "process_device": "cuda:0",
        "external_visible_gpu_count": 1,
        "active_compute_pids_at_launch": [],
    }


def prepare_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"sweep output root must be absent/new: {output_root}")
    for relative in ("arms", "evaluations", "diagnostics", "logs", "arm_status"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def arm_paths(output_root: Path, arm: Arm) -> dict[str, Path]:
    return {
        "run": output_root / "arms" / arm.arm_id,
        "evaluation": output_root / "evaluations" / arm.arm_id,
        "diagnostic": output_root / "diagnostics" / f"{arm.arm_id}.png",
        "train_log": output_root / "logs" / f"{arm.arm_id}.train.log",
        "evaluation_log": output_root / "logs" / f"{arm.arm_id}.evaluation.log",
        "diagnostic_log": output_root / "logs" / f"{arm.arm_id}.diagnostic.log",
        "validation_log": output_root / "logs" / f"{arm.arm_id}.validation.log",
        "status": output_root / "arm_status" / f"{arm.arm_id}.json",
    }


def trainer_command(
    arm: Arm,
    *,
    python: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    scene_profile: str,
    run_dir: Path,
    verifier_workers: int,
) -> list[str]:
    return [
        python,
        str(TRAINER),
        "--ckpt", str(checkpoint),
        "--expected-ckpt-sha256", checkpoint_sha256,
        "--scene-profile", scene_profile,
        "--outdir", str(run_dir),
        "--rounds", "100",
        "--rollout-replicas", "2",
        "--K", "64",
        "--B", "8",
        "--T", "300",
        "--M-eval", "0",
        "--batch", "128",
        "--afe-steps", str(arm.afe_steps),
        "--afe-lr", "1e-4",
        "--gp-cap", "512",
        "--gp-lam", "1e-2",
        "--acquisition-mode", "sequential",
        "--adaptive-ess-target", "0.5",
        "--replay-window", "5",
        "--gp-replay-window", "5",
        "--lengthscale-multiplier", f"{arm.lengthscale_multiplier:g}",
        "--negative-alpha", f"{arm.negative_alpha:g}",
        "--execution-rule", arm.execution_rule,
        "--conditioning-schema", "low7_closest_boundary",
        "--freeze-visual-encoder",
        "--skip-training-probes",
        "--calibration-replicas", "32",
        "--calibration-control-steps", "1",
        "--sweep-compact-artifacts",
        "--verifier-workers", str(verifier_workers),
        "--seed", str(SEED),
    ]


def pipeline_commands(
    arm: Arm,
    *,
    python: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    scene_profile: str,
    output_root: Path,
    verifier_workers: int,
) -> dict[str, list[str]]:
    paths = arm_paths(output_root, arm)
    return {
        "train": trainer_command(
            arm,
            python=python,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            scene_profile=scene_profile,
            run_dir=paths["run"],
            verifier_workers=verifier_workers,
        ),
        "evaluate": [
            python,
            str(EVALUATOR),
            "--run-root", str(paths["run"]),
            "--scene-profile", scene_profile,
            "--outdir", str(paths["evaluation"]),
            "--verifier-workers", str(verifier_workers),
        ],
        "diagnostics": [
            python,
            str(DIAGNOSTICS),
            "--run", str(paths["run"]),
            "--out", str(paths["diagnostic"]),
        ],
        "validate_evaluation": [
            python,
            str(EVALUATOR),
            "--outdir", str(paths["evaluation"]),
            "--validate-only",
        ],
    }


def _run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("x") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=CODEX_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _require_status(path: Path, expected: str) -> None:
    if not path.is_file() or load_json(path).get("status") != expected:
        raise RuntimeError(f"completion status is missing or invalid: {path}")


def run_arm_pipeline(
    arm: Arm,
    *,
    commands: dict[str, list[str]],
    paths: dict[str, Path],
) -> dict:
    started = time.perf_counter()
    print(f"[sweep] start {arm.arm_id}", flush=True)
    _run_command(commands["train"], paths["train_log"])
    _require_status(paths["run"] / "COMPLETE.json", "COMPLETE")
    _run_command(commands["evaluate"], paths["evaluation_log"])
    _run_command(commands["diagnostics"], paths["diagnostic_log"])
    if not paths["diagnostic"].is_file() or paths["diagnostic"].stat().st_size == 0:
        raise RuntimeError(f"diagnostic image is missing or empty: {paths['diagnostic']}")
    _run_command(commands["validate_evaluation"], paths["validation_log"])
    _require_status(
        paths["evaluation"] / "EVALUATION_COMPLETE.json",
        "AFE_RBF_RAW_M50_EVALUATION_DELIVERY_COMPLETE",
    )
    result = {
        "status": "ARM_PIPELINE_COMPLETE",
        **arm.record(),
        "elapsed_seconds": time.perf_counter() - started,
        "run": str(paths["run"]),
        "evaluation": str(paths["evaluation"]),
        "diagnostic": str(paths["diagnostic"]),
    }
    write_json_new(paths["status"], result)
    print(f"[sweep] complete {arm.arm_id}", flush=True)
    return result


def run_bounded(
    items: Iterable[Arm],
    max_jobs: int,
    worker: Callable[[Arm], dict],
) -> list[dict]:
    """Run at most ``max_jobs`` items, stopping new submissions after one failure."""

    if max_jobs < 1:
        raise ValueError("max jobs must be positive")
    iterator = iter(items)
    results: list[dict] = []
    first_error: BaseException | None = None
    executor = ThreadPoolExecutor(max_workers=max_jobs)
    active: dict[Future, Arm] = {}
    try:
        for _ in range(max_jobs):
            item = next(iterator, None)
            if item is not None:
                active[executor.submit(worker, item)] = item
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                active.pop(future)
                try:
                    results.append(future.result())
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is None:
                while len(active) < max_jobs:
                    item = next(iterator, None)
                    if item is None:
                        break
                    active[executor.submit(worker, item)] = item
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if first_error is not None:
        raise first_error
    return results


def global_ranking(arm_summaries: list[tuple[Arm, dict]]) -> list[dict]:
    """Rank validated per-arm best rows using pooled true M50 metrics only."""

    rows = []
    for arm, summary in arm_summaries:
        if summary.get("status") != "AFE_RBF_RAW_M50_SWEEP_COMPLETE":
            raise RuntimeError(f"evaluation summary is incomplete for {arm.arm_id}")
        if int(summary.get("M", -1)) != RAW_M:
            raise RuntimeError(f"evaluation summary is not M=50 for {arm.arm_id}")
        checkpoint_ranking = summary.get("post_hoc_ranking") or []
        if not checkpoint_ranking or int(checkpoint_ranking[0].get("rank", -1)) != 1:
            raise RuntimeError(f"evaluation ranking is invalid for {arm.arm_id}")
        best = checkpoint_ranking[0]
        if int(summary.get("post_hoc_best_round", -1)) != int(best["round"]):
            raise RuntimeError(f"evaluation best round is inconsistent for {arm.arm_id}")
        rows.append({
            **arm.record(),
            "best_round": int(best["round"]),
            "SR": float(best["SR"]),
            "CR": float(best["CR"]),
            "timeout": float(best["timeout"]),
            "mean_minimum_clearance": float(best["mean_minimum_clearance"]),
            "checkpoint_ranking": checkpoint_ranking,
        })
    rows.sort(key=lambda row: (
        -row["SR"],
        row["CR"],
        row["timeout"],
        -row["mean_minimum_clearance"],
        row["best_round"],
        row["arm_id"],
    ))
    for rank, row in enumerate(rows, start=1):
        row["overall_rank"] = rank
    return rows


def write_summary_csv(path: Path, ranking: list[dict]) -> None:
    fields = (
        "overall_rank",
        "arm_id",
        "lengthscale_multiplier",
        "negative_alpha",
        "afe_steps",
        "execution_rule",
        "best_round",
        "SR",
        "CR",
        "timeout",
        "mean_minimum_clearance",
    )
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)


def completion_artifacts(output_root: Path, arms: list[Arm]) -> list[Path]:
    paths = [
        output_root / "sweep_contract.json",
        output_root / "sweep_summary.json",
        output_root / "sweep_summary.csv",
        output_root / "best_training.mp4",
    ]
    for arm in arms:
        arm_output = arm_paths(output_root, arm)
        paths.extend((
            arm_output["run"] / "COMPLETE.json",
            arm_output["evaluation"] / "EVALUATION_COMPLETE.json",
            arm_output["diagnostic"],
        ))
    return paths


def build_contract(
    *,
    args,
    output_root: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    source: dict,
    gpu: dict,
    arms: list[Arm],
) -> dict:
    arm_contracts = []
    for arm in arms:
        paths = arm_paths(output_root, arm)
        commands = pipeline_commands(
            arm,
            python=sys.executable,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            scene_profile=args.scene_profile,
            output_root=output_root,
            verifier_workers=args.verifier_workers,
        )
        arm_contracts.append({
            **arm.record(),
            "paths": {key: str(value) for key, value in paths.items()},
            "commands": commands,
        })
    raw_rollouts_per_checkpoint = RAW_M * GAMMA_COUNT
    return {
        "status": "LOW7_RBF_48_ARM_SWEEP_CONTRACT_V1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
        },
        "gpu": gpu,
        "scene_profile": args.scene_profile,
        "seed": SEED,
        "arm_count": len(arms),
        "matrix": {
            "lengthscale_multipliers": list(ELL_MULTIPLIERS),
            "paper_norm_alphas": list(NEGATIVE_ALPHAS),
            "afe_steps": list(AFE_STEPS),
            "execution_rules": list(EXECUTION_RULES),
        },
        "fixed_recipe": {
            "acquisition": "sequential RBF with adaptive normalized ESS target 0.5",
            "cfm_replay_window": 5,
            "gp_replay_window": 5,
            "K": 64,
            "B": 8,
            "batch": 128,
            "afe_lr": 1.0e-4,
            "rounds": 100,
            "rollout_replicas": 2,
            "conditioning_schema": "low7_closest_boundary",
            "visual_encoder": "frozen",
            "training_probes": False,
            "M_eval": 0,
            "calibration_replicas": 32,
            "calibration_control_steps": 1,
            "artifact_profile": "sweep_compact",
            "gp_cap": 512,
            "gp_lam": 1.0e-2,
            "trainer_verifier_workers_per_arm": args.verifier_workers,
            "evaluator_verifier_workers_per_arm": args.verifier_workers,
        },
        "runtime_assumptions": {
            "max_parallel_arm_pipelines": args.max_jobs,
            "gpu_scheduling": (
                "all child processes see only logical cuda:0 mapped to the one authenticated "
                "physical GPU; up to max_parallel_arm_pipelines share that GPU"
            ),
            "cpu_threading": "OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1",
            "max_parallel_trainer_verifier_workers": (
                args.max_jobs * args.verifier_workers
            ),
            "max_parallel_evaluator_verifier_workers": (
                args.max_jobs * args.verifier_workers
            ),
            "max_total_parallel_verifier_workers": (
                args.max_jobs * args.verifier_workers
            ),
            "pipeline_order_per_arm": [
                "train", "raw_M50_evaluate", "render_diagnostics", "validate_evaluation"
            ],
            "failure_policy": (
                "one attempt per arm; never retry, delete, or resume a partial arm; after a "
                "failure no new arms are submitted and already-running arms are allowed to exit"
            ),
            "output_policy": "the sweep output root must be absent before launch",
            "total_training_rounds": len(arms) * 100,
            "total_optimizer_steps": sum(arm.afe_steps for arm in arms) * 100,
            "raw_evaluation_rounds_per_arm": list(EVALUATION_ROUNDS),
            "raw_rollouts_per_checkpoint": raw_rollouts_per_checkpoint,
            "raw_rollouts_per_arm": raw_rollouts_per_checkpoint * len(EVALUATION_ROUNDS),
            "raw_rollouts_total": (
                raw_rollouts_per_checkpoint * len(EVALUATION_ROUNDS) * len(arms)
            ),
            "wall_clock_estimate": (
                "not asserted: verifier yield, shared-GPU contention, and raw rollout duration "
                "are measured in artifacts instead"
            ),
            "video_policy": "render exactly one training video after all 48 arms validate",
        },
        "selection": {
            "rule": GLOBAL_RANKING_RULE,
            "source": "validated per-arm evaluation_summary.json post_hoc_ranking only",
            "trainer_probe_SR_CR_used": False,
        },
        "arms": arm_contracts,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-profile", choices=SUPPORTED_SCENES, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expected-ckpt-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--verifier-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    if args.verifier_workers < 1:
        parser.error("--verifier-workers must be positive")
    if args.physical_index < 0:
        parser.error("--physical-index cannot be negative")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_ckpt_sha256):
        parser.error("--expected-ckpt-sha256 must contain 64 hexadecimal characters")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    checkpoint = Path(args.ckpt).expanduser().resolve(strict=True)
    expected_sha = args.expected_ckpt_sha256.lower()
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(f"checkpoint SHA-256 {actual_sha} != expected {expected_sha}")
    output_root = Path(args.out).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"sweep output root must be absent/new: {output_root}")
    source = source_record()
    gpu = gpu_record(args.physical_index, args.expected_gpu_uuid)
    arms = sweep_arms()
    if len(arms) != 48 or len({arm.arm_id for arm in arms}) != 48:
        raise RuntimeError("the declared sweep matrix did not produce exactly 48 unique arms")

    prepare_output_root(output_root)
    contract = build_contract(
        args=args,
        output_root=output_root,
        checkpoint=checkpoint,
        checkpoint_sha256=actual_sha,
        source=source,
        gpu=gpu,
        arms=arms,
    )
    write_json_new(output_root / "sweep_contract.json", contract)

    commands_by_arm = {
        arm.arm_id: pipeline_commands(
            arm,
            python=sys.executable,
            checkpoint=checkpoint,
            checkpoint_sha256=actual_sha,
            scene_profile=args.scene_profile,
            output_root=output_root,
            verifier_workers=args.verifier_workers,
        )
        for arm in arms
    }

    def worker(arm: Arm) -> dict:
        return run_arm_pipeline(
            arm,
            commands=commands_by_arm[arm.arm_id],
            paths=arm_paths(output_root, arm),
        )

    run_bounded(arms, args.max_jobs, worker)

    arm_summaries = [
        (
            arm,
            load_json(arm_paths(output_root, arm)["evaluation"] / "evaluation_summary.json"),
        )
        for arm in arms
    ]
    ranking = global_ranking(arm_summaries)
    if len(ranking) != 48:
        raise RuntimeError("global evaluation ranking does not contain all 48 arms")
    best = ranking[0]
    best_arm = next(arm for arm in arms if arm.arm_id == best["arm_id"])
    best_evaluation_summary = next(
        summary for arm, summary in arm_summaries if arm == best_arm
    )
    best_video = output_root / "best_training.mp4"
    video_log = output_root / "logs" / "best_training.video.log"
    _run_command(
        [
            sys.executable,
            str(VIDEO),
            "--run", str(arm_paths(output_root, best_arm)["run"]),
            "--out", str(best_video),
            "--dense-until", "10",
            "--every-after", "10",
        ],
        video_log,
    )
    if not best_video.is_file() or best_video.stat().st_size == 0:
        raise RuntimeError("best-arm training video is missing or empty")
    video_probe = ffprobe_video(best_video, expected_frames=19)

    summary = {
        "status": "LOW7_RBF_48_ARM_SWEEP_COMPLETE",
        "scene_profile": args.scene_profile,
        "seed": SEED,
        "arm_count": len(ranking),
        "ranking_rule": GLOBAL_RANKING_RULE,
        "ranking_source": (
            "validated evaluation_summary.json rank-1 pooled raw M50 rows only; "
            "trainer probe SR/CR are excluded"
        ),
        "overall_best": best,
        "overall_ranking": ranking,
        "best_training_video": str(best_video),
        "best_training_video_probe": video_probe,
        "best_evaluation_assets": {
            "curves": best_evaluation_summary["outputs"]["curves"],
            "r0_best_final_galleries": [
                path
                for path in best_evaluation_summary["outputs"]["galleries"]
                if Path(path).name.startswith("raw_m50_r0_best_final_gallery.")
            ],
            "per_round_galleries": [
                path
                for path in best_evaluation_summary["outputs"]["galleries"]
                if Path(path).parent.name == "round_galleries"
            ],
            "gallery_indices": best_evaluation_summary["outputs"]["gallery_indices"],
        },
        "sweep_contract": str(output_root / "sweep_contract.json"),
    }
    write_json_new(output_root / "sweep_summary.json", summary)
    write_summary_csv(output_root / "sweep_summary.csv", ranking)
    delivery_paths = completion_artifacts(output_root, arms)
    missing = [str(path) for path in delivery_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"sweep delivery artifact is missing: {missing}")
    write_json_new(
        output_root / "SWEEP_COMPLETE.json",
        {
            "status": "LOW7_RBF_48_ARM_SWEEP_DELIVERY_COMPLETE",
            "source_git_commit": source["git_commit"],
            "checkpoint_sha256": actual_sha,
            "gpu_uuid": gpu["uuid"],
            "scene_profile": args.scene_profile,
            "arm_count": len(arms),
            "overall_best_arm": best["arm_id"],
            "overall_best_round": best["best_round"],
            "best_training_video_probe": video_probe,
            "artifact_sha256": {
                str(path.relative_to(output_root)): sha256_file(path)
                for path in delivery_paths
            },
        },
    )
    print(
        f"LOW7 RBF 48-ARM SWEEP COMPLETE: {output_root} best={best['arm_id']} "
        f"round={best['best_round']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
