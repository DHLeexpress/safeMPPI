#!/usr/bin/env python3
"""Launch, monitor, merge, and visualize the 300-pair Stage 2 run on GPU 2."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from viz_style import GAMMAS


HERE = Path(__file__).resolve().parent
STAGE = HERE / "stage_results" / "02_demos"
DATA = STAGE / "data"
WORKERS = DATA / "workers"
LOGS = STAGE / "logs"


def worker_paths(gamma: float, shard: int):
    stem = f"worker_g{float(gamma)}_s{shard}"
    return WORKERS / f"{stem}.pt", WORKERS / f"{stem}_paths.npz", LOGS / f"{stem}.log"


def gpu_sample(gpu: int) -> dict:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.check_output(command, text=True).strip().split(",")
    return {
        "time_s": time.time(),
        "utilization_pct": float(raw[0]),
        "memory_mib": float(raw[1]),
        "temperature_c": float(raw[2]),
        "power_w": float(raw[3]),
    }


def run_command(command, *, env=None, stdout=None):
    return subprocess.run(command, cwd=HERE, env=env, stdout=stdout, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument(
        "--client-visible-device",
        default=None,
        help="CUDA_VISIBLE_DEVICES for workers; use 0 when an MPS daemon remaps physical GPU 2",
    )
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--pair-seed", type=int, default=20260714)
    parser.add_argument("--shards-per-gamma", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--skip-workers", action="store_true")
    parser.add_argument("--keep-worker-data", action="store_true")
    args = parser.parse_args()

    WORKERS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    manifest = DATA / "random_pairs_300.npz"
    run_command(
        [
            sys.executable,
            "gen_sg_data.py",
            "manifest",
            "--out",
            str(manifest),
            "--pairs",
            str(args.pairs),
            "--seed",
            str(args.pair_seed),
        ]
    )

    started = time.perf_counter()
    gpu_samples = []
    processes = []
    log_handles = []
    if not args.skip_workers:
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = (
            str(args.gpu) if args.client_visible_device is None else str(args.client_visible_device)
        )
        child_env["OMP_NUM_THREADS"] = "2"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["NUMEXPR_NUM_THREADS"] = "1"
        for gamma in GAMMAS:
            for shard in range(args.shards_per_gamma):
                output, paths_output, log_path = worker_paths(gamma, shard)
                log_handle = log_path.open("w")
                command = [
                    sys.executable,
                    "gen_sg_data.py",
                    "worker",
                    "--manifest",
                    str(manifest),
                    "--gamma",
                    str(gamma),
                    "--shard-id",
                    str(shard),
                    "--num-shards",
                    str(args.shards_per_gamma),
                    "--device",
                    "cuda:0",
                    "--max-retries",
                    str(args.max_retries),
                    "--out",
                    str(output),
                    "--paths-out",
                    str(paths_output),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=HERE,
                    env=child_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((gamma, shard, process, log_path))
                log_handles.append(log_handle)

        print(f"launched {len(processes)} workers on physical GPU {args.gpu}", flush=True)
        while True:
            pending = [(g, s, p, log) for g, s, p, log in processes if p.poll() is None]
            try:
                sample = gpu_sample(args.gpu)
                gpu_samples.append(sample)
                print(
                    f"GPU{args.gpu}: util={sample['utilization_pct']:.0f}% "
                    f"mem={sample['memory_mib']:.0f} MiB power={sample['power_w']:.0f} W "
                    f"workers={len(pending)}/{len(processes)}",
                    flush=True,
                )
            except Exception as error:
                print(f"GPU monitor warning: {error}", flush=True)
            if not pending:
                break
            time.sleep(args.poll_seconds)

        for handle in log_handles:
            handle.close()
        failures = [(g, s, p.returncode, log) for g, s, p, log in processes if p.returncode != 0]
        if failures:
            raise RuntimeError(f"worker failures: {failures}")

    worker_seconds = time.perf_counter() - started
    print(f"workers complete in {worker_seconds:.1f}s; merging", flush=True)

    merge_summaries = {}
    for gamma in GAMMAS:
        shard_paths = [worker_paths(gamma, shard)[0] for shard in range(args.shards_per_gamma)]
        output = DATA / f"w8sg_windows_g{float(gamma)}.pt"
        paths_output = DATA / f"paths_g{float(gamma)}.npz"
        summary = LOGS / f"dataset_g{float(gamma)}.json"
        command = [
            sys.executable,
            "gen_sg_data.py",
            "merge",
            "--manifest",
            str(manifest),
            "--gamma",
            str(gamma),
            "--shards",
            *[str(path) for path in shard_paths],
            "--out",
            str(output),
            "--paths-out",
            str(paths_output),
            "--summary",
            str(summary),
        ]
        if not args.keep_worker_data:
            command.append("--cleanup")
        run_command(command)
        merge_summaries[str(gamma)] = json.loads(summary.read_text())

    run_command([sys.executable, "plot_sg_demo_overlay.py"])
    total_seconds = time.perf_counter() - started
    pipeline_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.gpu,
        "pairs_per_gamma": args.pairs,
        "gammas": list(GAMMAS),
        "workers": len(GAMMAS) * args.shards_per_gamma,
        "worker_seconds": worker_seconds,
        "total_seconds": total_seconds,
        "gpu_samples": gpu_samples,
        "gpu_utilization_mean": (
            sum(sample["utilization_pct"] for sample in gpu_samples) / len(gpu_samples) if gpu_samples else None
        ),
        "gpu_utilization_max": max((sample["utilization_pct"] for sample in gpu_samples), default=None),
        "gpu_memory_max_mib": max((sample["memory_mib"] for sample in gpu_samples), default=None),
        "datasets": merge_summaries,
        "overlay": str((STAGE / "viz" / "demo_300_pairs_all_gamma.png").resolve()),
    }
    summary_path = LOGS / "stage2_gpu2_pipeline.json"
    summary_path.write_text(json.dumps(pipeline_summary, indent=2, sort_keys=True) + "\n")
    print(f"PIPELINE_DONE {total_seconds:.1f}s -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
