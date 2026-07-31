#!/usr/bin/env python3
"""Evaluate a completed selector sweep with M10 -> M50 staging.

Training artifacts are immutable.  All arm/round checkpoints first share one
raw temperature-one M10 bank.  The top liveness-preserving screening cells are
then evaluated on a disjoint M50 bank.  A later cross-selector job is
responsible for the final disjoint M100 confirmation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "sfm_b1_offline_eval.py"
GAMMAS = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def validate_training(run_root: Path) -> tuple[dict, dict]:
    declaration_path = run_root / "RUN_DECLARATION.json"
    training_path = run_root / "TRAINING_COMPLETE.json"
    declaration = read_json(declaration_path)
    training = read_json(training_path)
    if declaration.get("status") != "SFM_B1_OFFLINE_9ARM_DECLARED":
        raise RuntimeError("invalid run declaration")
    if training.get("status") != "SFM_B1_OFFLINE_9ARM_TRAINING_COMPLETE":
        raise RuntimeError("training is not complete")
    if training.get("declaration_sha256") != sha256_file(declaration_path):
        raise RuntimeError("training marker does not authenticate declaration")
    contract = declaration.get("contract", {})
    if declaration.get("contract_sha256") != sha256_json(contract):
        raise RuntimeError("declaration contract digest mismatch")
    checkpoint = Path(contract["checkpoint"]).resolve()
    if sha256_file(checkpoint) != contract["checkpoint_sha256"]:
        raise RuntimeError("pretrained checkpoint digest mismatch")
    arms = training.get("arms", {})
    if len(arms) != 9:
        raise RuntimeError(f"expected 9 trained arms, got {len(arms)}")
    for name, arm in arms.items():
        checkpoints = arm.get("checkpoints", [])
        if [row.get("round") for row in checkpoints] != list(range(11)):
            raise RuntimeError(f"{name}: incomplete round checkpoints")
        for row in checkpoints:
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"{name}: checkpoint digest mismatch: {path}")
    return declaration, training


def pooled_row(arm: str, record: dict) -> dict:
    pooled = record["cell"]["summary"]["pooled"]
    clearance = pooled["successful_clearance"]["mean"]
    time_to_goal = pooled["successful_time_to_goal"]["mean"]
    return {
        "arm": arm,
        "round": int(record["round"]),
        "checkpoint": record["cell"]["checkpoint"],
        "checkpoint_sha256": record["cell"]["checkpoint_sha256"],
        "SR": float(pooled["SR"]),
        "CR": float(pooled["CR"]),
        "timeout": float(pooled["timeout"]),
        "Validity": float(pooled["Validity"]["mean"]),
        "clearance": None if clearance is None else float(clearance),
        "time_to_goal": (
            None if time_to_goal is None else float(time_to_goal)
        ),
    }


def safety_key(row: dict) -> tuple:
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
        str(row["arm"]),
    )


def fallback_key(row: dict) -> tuple:
    return (
        -float(row["SR"]),
        float(row["CR"]),
        -float(row["Validity"]),
        int(row["round"]),
        str(row["arm"]),
    )


def choose_candidates(
    rows: list[dict], r0: dict, *, top_k: int
) -> tuple[list[dict], dict]:
    post = [row for row in rows if int(row["round"]) > 0]
    eligible = [
        row for row in post if float(row["SR"]) >= float(r0["SR"])
    ]
    ordered = sorted(eligible, key=safety_key)
    fallback_used = False
    if len(ordered) < top_k:
        fallback_used = True
        seen = {
            (row["arm"], row["round"], row["checkpoint_sha256"])
            for row in ordered
        }
        for row in sorted(post, key=fallback_key):
            key = (row["arm"], row["round"], row["checkpoint_sha256"])
            if key not in seen:
                ordered.append(row)
                seen.add(key)
            if len(ordered) >= top_k:
                break
    return ordered[:top_k], {
        "rule": (
            "among post-expansion cells with SR >= common-r0 SR, minimize CR, "
            "then maximize window Validity and SR; if fewer than top-k pass "
            "the liveness gate, supplement by highest SR"
        ),
        "r0_SR_gate": float(r0["SR"]),
        "eligible_cells": len(eligible),
        "fallback_used": fallback_used,
    }


def evaluator_command(
    checkpoints: list[str],
    labels: list[str],
    *,
    scene_profile: str,
    ep0: int,
    noise_seed: int,
    m_per_gamma: int,
    workers: int,
    cache_dir: Path,
    output_dir: Path,
    temperature: float = 1.0,
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR),
        "--checkpoints",
        *checkpoints,
        "--labels",
        *labels,
        "--scene-profile",
        scene_profile,
        "--ep0",
        str(ep0),
        "--noise-seed",
        str(noise_seed),
        "--m-per-gamma",
        str(m_per_gamma),
        "--temperature",
        str(float(temperature)),
        "--device",
        "cuda:0",
        "--workers",
        str(workers),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
    ]


def run_job(
    name: str,
    command: list[str],
    *,
    gpu_index: int,
    cpu_start: int,
    cpu_count: int,
    log_dir: Path,
) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONPATH"] = str(HERE)
    cpu_end = cpu_start + cpu_count - 1
    launched = [
        "taskset", "-c", f"{cpu_start}-{cpu_end}", *command
    ]
    with log_path.open("w") as stream:
        completed = subprocess.run(
            launched,
            cwd=HERE,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{name} failed with {completed.returncode}; see {log_path}"
        )
    return {
        "name": name,
        "command": launched,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def run_parallel(
    jobs: list[dict],
    *,
    gpu_index: int,
    cpu_start: int,
    cpu_count: int,
    log_dir: Path,
) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                run_job,
                job["name"],
                job["command"],
                gpu_index=gpu_index,
                cpu_start=cpu_start + index * cpu_count,
                cpu_count=cpu_count,
                log_dir=log_dir,
            ): job["name"]
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            print(f"COMPLETE {result['name']}", flush=True)
            results.append(result)
    return sorted(results, key=lambda row: row["name"])


def run(args) -> dict:
    started_at = datetime.now(timezone.utc)
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    declaration, training = validate_training(run_root)
    contract = declaration["contract"]
    selector = str(contract["execution_selector"])
    checkpoint = str(Path(contract["checkpoint"]).resolve())
    scene_profile = str(contract["scene_profile"])

    screen_root = output_root / "screening_m10"
    screen_cache = screen_root / "cache"
    common_dir = screen_root / "common_r0"
    common_command = evaluator_command(
        [checkpoint],
        ["r0"],
        scene_profile=scene_profile,
        ep0=args.screen_ep0,
        noise_seed=args.screen_noise_seed,
        m_per_gamma=args.screen_m,
        workers=args.workers,
        cache_dir=screen_cache,
        output_dir=common_dir,
    )
    common_job = run_job(
        "screen_common_r0",
        common_command,
        gpu_index=args.gpu_index,
        cpu_start=args.cpu_start,
        cpu_count=args.workers,
        log_dir=output_root / "logs",
    )
    common_metrics = read_json(
        common_dir / f"raw_m{args.screen_m}_offline_metrics.json"
    )
    r0 = pooled_row("pretrained", common_metrics["records"][0])

    screen_jobs = []
    for arm_name, arm in sorted(training["arms"].items()):
        checkpoints = [checkpoint] + [
            row["path"] for row in arm["checkpoints"] if row["round"] > 0
        ]
        labels = ["r0"] + [
            f"r{row['round']}"
            for row in arm["checkpoints"] if row["round"] > 0
        ]
        output = screen_root / arm_name
        screen_jobs.append({
            "name": f"screen_{arm_name}",
            "output": output,
            "command": evaluator_command(
                checkpoints,
                labels,
                scene_profile=scene_profile,
                ep0=args.screen_ep0,
                noise_seed=args.screen_noise_seed,
                m_per_gamma=args.screen_m,
                workers=args.workers,
                cache_dir=screen_cache,
                output_dir=output,
            ),
        })
    screen_logs = run_parallel(
        screen_jobs,
        gpu_index=args.gpu_index,
        cpu_start=args.cpu_start,
        cpu_count=args.workers,
        log_dir=output_root / "logs",
    )
    screening_rows = []
    for job in screen_jobs:
        payload = read_json(
            job["output"]
            / f"raw_m{args.screen_m}_offline_metrics.json"
        )
        screening_rows.extend(
            pooled_row(job["name"].removeprefix("screen_"), record)
            for record in payload["records"]
            if int(record["round"]) > 0
        )
    candidates, selection = choose_candidates(
        screening_rows, r0, top_k=args.top_k
    )
    write_json(output_root / "SCREENING_COMPLETE.json", {
        "status": "SFM_B1_OFFLINE_M10_SCREENING_COMPLETE",
        "selector": selector,
        "bank": {
            "M_per_gamma": args.screen_m,
            "ep0": args.screen_ep0,
            "noise_seed": args.screen_noise_seed,
        },
        "common_r0": r0,
        "selection": selection,
        "selected_candidates": candidates,
        "rows": screening_rows,
        "logs": [common_job, *screen_logs],
    })

    confirm_root = output_root / "confirmation_m50"
    confirm_cache = confirm_root / "cache"
    confirm_jobs = []
    for index, candidate in enumerate(candidates):
        name = f"candidate_{index:02d}_{candidate['arm']}_r{candidate['round']}"
        output = confirm_root / name
        confirm_jobs.append({
            "name": name,
            "candidate": candidate,
            "output": output,
            "command": evaluator_command(
                [checkpoint, candidate["checkpoint"]],
                ["r0", f"r{candidate['round']}"],
                scene_profile=scene_profile,
                ep0=args.confirm_ep0,
                noise_seed=args.confirm_noise_seed,
                m_per_gamma=args.confirm_m,
                workers=args.workers,
                cache_dir=confirm_cache,
                output_dir=output,
            ),
        })
    confirm_logs = run_parallel(
        confirm_jobs,
        gpu_index=args.gpu_index,
        cpu_start=args.cpu_start,
        cpu_count=args.workers,
        log_dir=output_root / "logs",
    )
    confirmation_rows = []
    r0_confirm = None
    for job in confirm_jobs:
        payload = read_json(
            job["output"]
            / f"raw_m{args.confirm_m}_offline_metrics.json"
        )
        if r0_confirm is None:
            r0_confirm = pooled_row("pretrained", payload["records"][0])
        confirmation_rows.append(
            pooled_row(job["candidate"]["arm"], payload["records"][1])
        )
    winner_rows, confirmation_selection = choose_candidates(
        confirmation_rows, r0_confirm, top_k=1
    )
    winner = winner_rows[0]
    completed_at = datetime.now(timezone.utc)
    result = {
        "status": "SFM_B1_OFFLINE_SELECTOR_FUNNEL_COMPLETE",
        "selector": selector,
        "role": (
            "M10 common-bank screening followed by disjoint M50 selector "
            "confirmation; no final claim or M100 confirmation"
        ),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        "training_root": str(run_root),
        "training_marker_sha256": sha256_file(
            run_root / "TRAINING_COMPLETE.json"
        ),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "gpu_index": args.gpu_index,
        "cpu_range": [
            args.cpu_start,
            args.cpu_start + 9 * args.workers - 1,
        ],
        "screening": {
            "bank": {
                "M_per_gamma": args.screen_m,
                "ep0": args.screen_ep0,
                "noise_seed": args.screen_noise_seed,
            },
            "common_r0": r0,
            "selection": selection,
            "selected_candidates": candidates,
        },
        "confirmation": {
            "bank": {
                "M_per_gamma": args.confirm_m,
                "ep0": args.confirm_ep0,
                "noise_seed": args.confirm_noise_seed,
            },
            "common_r0": r0_confirm,
            "rows": confirmation_rows,
            "selection": confirmation_selection,
            "selector_winner": winner,
        },
        "logs": confirm_logs,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_seconds": (completed_at - started_at).total_seconds(),
        "next_step": (
            "compare selector winners and run exactly one disjoint raw-M100 "
            "confirmation on a new scenario/noise bank"
        ),
    }
    marker = output_root / "SELECTOR_FUNNEL_COMPLETE.json"
    write_json(marker, result)
    print(json.dumps({
        "status": result["status"],
        "selector": selector,
        "winner": winner,
        "marker": str(marker),
    }, indent=2, allow_nan=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--cpu-start", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--screen-m", type=int, default=10)
    parser.add_argument("--screen-ep0", type=int, default=260_000)
    parser.add_argument("--screen-noise-seed", type=int, default=2_026_072_3)
    parser.add_argument("--confirm-m", type=int, default=50)
    parser.add_argument("--confirm-ep0", type=int, default=270_000)
    parser.add_argument("--confirm-noise-seed", type=int, default=2_026_072_4)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
