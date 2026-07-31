#!/usr/bin/env python3
"""Continue the selected neutral arm to r100 only if calibrated M50 fails."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_sfm_neutral_gamma_temperature as GAMMA
import run_sfm_neutral_temperature_m50 as GLOBAL


HERE = Path(__file__).resolve().parent
STATUS = "SFM_NEUTRAL_AUTONOMOUS_FOLLOWUP_COMPLETE"


def _wait(path: Path, poll: int) -> dict:
    while not path.is_file():
        print(f"WAITING {path}", flush=True)
        time.sleep(int(poll))
    return GLOBAL._read(path)


def _run(command: list[str], *, log: Path, gpu: int | None = None,
         cpu_range: str | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(HERE)
    if gpu is not None:
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    launched = list(command)
    if cpu_range is not None:
        launched = ["taskset", "-c", cpu_range, *launched]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            launched, cwd=HERE, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"job failed ({completed.returncode}): {log}")


def run(args) -> dict:
    started = datetime.now(timezone.utc)
    gamma_delivery_path = Path(args.gamma_delivery).resolve()
    gamma = _wait(gamma_delivery_path, args.poll_seconds)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if gamma.get("status") != GAMMA.STATUS:
        raise RuntimeError("invalid calibrated M50 delivery")
    if gamma.get("ci_clean_four_metric_win"):
        result = {
            "status": STATUS,
            "action": "STOP_GOAL_ACHIEVED_AT_R50_OR_EARLIER",
            "gamma_delivery": str(gamma_delivery_path),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        GLOBAL._write(output / "DELIVERY_COMPLETE.json", result)
        return result

    initial = GLOBAL._read(Path(gamma["initial_delivery"]))
    arm = str(initial["selection"]["selected_expanded"]["method"])
    training_root = Path(initial["training_root"]).resolve()
    resume_root = training_root / arm
    resume_delivery = GLOBAL._read(resume_root / "DELIVERY_COMPLETE.json")
    cfg = resume_delivery["config"]
    resume_round = int(resume_delivery["rounds"])
    if resume_round != 50:
        raise RuntimeError("autonomous continuation expects an r50 source")
    scenario_ids = []
    for marker in resume_delivery["round_records"]:
        scenario_ids.extend(map(int, GLOBAL._read(Path(marker))["scenarios"]))
    scenario_ep0 = min(scenario_ids)

    r100_parent = output / "r100_training"
    r100_root = r100_parent / arm
    eval_rounds = [0, resume_round, *range(60, 101, 10)]
    train_command = [
        sys.executable, str(HERE / "sfm_b1_neutral_multiround.py"),
        "--checkpoint", resume_delivery["checkpoint"],
        "--resume-run-root", str(resume_root),
        "--output-root", str(r100_root),
        "--name", f"{arm}_continued_r100",
        "--rounds", "100",
        "--scenario-ep0", str(scenario_ep0),
        "--eval-ep0", "490000",
        "--eval-M", "20",
        "--eval-rounds", ",".join(map(str, eval_rounds)),
        "--lr", str(cfg["lr"]),
        "--inner-steps", str(cfg["inner_steps"]),
        "--ell", str(cfg["ell"]),
        "--gp-cap", str(cfg["gp_cap"]),
        "--sample-seed", str(cfg["sample_seed"]),
        "--audit-seed", str(cfg["audit_seed"]),
        "--train-seed", str(cfg["train_seed"]),
        "--probe-seed", str(cfg["probe_seed"]),
        "--noise-seed", "20260737",
        "--device", "cuda:0",
        "--workers", str(args.workers),
    ]
    _run(
        train_command, log=output / "logs" / "r50_to_r100.log",
        gpu=args.training_gpu, cpu_range="16-79",
    )
    r100_delivery = GLOBAL._read(r100_root / "DELIVERY_COMPLETE.json")
    if int(r100_delivery.get("rounds", -1)) != 100:
        raise RuntimeError("r100 continuation delivery is incomplete")

    global_root = output / "r100_global_temperature"
    global_command = [
        sys.executable, str(HERE / "run_sfm_neutral_temperature_m50.py"), "run",
        "--training-root", str(r100_parent),
        "--output-dir", str(global_root),
        "--arm-names", arm,
        "--temperatures", "0.55,0.7,0.85,1.0",
        "--gpus", "1", "3", "--workers", "32",
        "--screen-ep0", "490000", "--screen-M", "20",
        "--validation-ep0", "500000", "--validation-M", "10",
        "--validation-noise-seed", "20260738",
        "--final-ep0", "510000", "--final-noise-seed", "20260739",
        "--expected-final-round", "100",
    ]
    _run(global_command, log=output / "logs" / "r100_global_temperature.log")

    r100_gamma_root = output / "r100_gamma_temperature"
    gamma_command = [
        sys.executable, str(HERE / "run_sfm_neutral_gamma_temperature.py"),
        "--initial-delivery", str(global_root / "DELIVERY_COMPLETE.json"),
        "--output-dir", str(r100_gamma_root),
        "--temperatures", "0.55,0.7,0.85,1.0",
        "--gpus", "1", "3", "--workers", "32",
        "--final-ep0", "520000", "--final-noise-seed", "20260740",
    ]
    _run(gamma_command, log=output / "logs" / "r100_gamma_temperature.log")
    r100_gamma = GLOBAL._read(r100_gamma_root / "DELIVERY_COMPLETE.json")
    achieved = bool(r100_gamma.get("ci_clean_four_metric_win"))
    result = {
        "status": STATUS,
        "action": (
            "STOP_GOAL_ACHIEVED_AT_R100"
            if achieved else "CREATIVE_SANITY_REQUIRED"
        ),
        "selected_arm": arm,
        "r50_gamma_delivery": str(gamma_delivery_path),
        "r100_training_delivery": str(r100_root / "DELIVERY_COMPLETE.json"),
        "r100_global_delivery": str(global_root / "DELIVERY_COMPLETE.json"),
        "r100_gamma_delivery": str(r100_gamma_root / "DELIVERY_COMPLETE.json"),
        "ci_clean_four_metric_win": achieved,
        "creative_sanity_priority": [
            "nontrap_progress_gated_max_margin",
            "Dplus_only_if_postpositive_audit_supports_it",
            "encoder_unfreeze_at_0.1x_lr",
        ],
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    GLOBAL._write(output / "DELIVERY_COMPLETE.json", result)
    print(json.dumps(result, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-delivery", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
