#!/usr/bin/env python3
"""Wait for three selector studies, then run one disjoint raw-M100 result."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import run_sfm_b1_offline_eval_funnel as FUNNEL


def wait_for(path: Path, poll_seconds: int) -> dict:
    while not path.is_file():
        print(f"WAITING {path}", flush=True)
        time.sleep(poll_seconds)
    return FUNNEL.read_json(path)


def margin_rows(run_root: Path) -> tuple[list[dict], dict]:
    training = FUNNEL.read_json(run_root / "TRAINING_COMPLETE.json")
    checkpoints = {
        (name, int(row["round"])): row
        for name, arm in training["arms"].items()
        for row in arm["checkpoints"]
    }
    csv_path = (
        run_root / "evaluation" / "aggregate"
        / "factorial_raw_m50_metrics.csv"
    )
    rows = []
    with csv_path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            round_index = int(source["round"])
            if round_index == 0:
                continue
            checkpoint = checkpoints[(source["arm"], round_index)]
            rows.append({
                "arm": source["arm"],
                "round": round_index,
                "checkpoint": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "SR": float(source["SR"]),
                "CR": float(source["CR"]),
                "timeout": float(source["timeout"]),
                "Validity": float(source["Validity"]),
                "clearance": (
                    None if not source["clearance"]
                    else float(source["clearance"])
                ),
                "time_to_goal": (
                    None if not source["time_to_goal"]
                    else float(source["time_to_goal"])
                ),
            })
    common = FUNNEL.read_json(
        run_root / "evaluation" / "common_r0"
        / "raw_m50_offline_metrics.json"
    )
    r0 = FUNNEL.pooled_row("pretrained", common["records"][0])
    return rows, r0


def evaluate_candidates(
    candidates: list[dict],
    *,
    checkpoint: str,
    output_root: Path,
    gpu_index: int,
    cpu_start: int,
    workers: int,
    ep0: int,
    noise_seed: int,
) -> tuple[list[dict], dict, list[dict]]:
    cache = output_root / "cache"
    jobs = []
    for index, candidate in enumerate(candidates):
        name = f"margin_{index:02d}_{candidate['arm']}_r{candidate['round']}"
        output = output_root / name
        jobs.append({
            "name": name,
            "candidate": candidate,
            "output": output,
            "command": FUNNEL.evaluator_command(
                [checkpoint, candidate["checkpoint"]],
                ["r0", f"r{candidate['round']}"],
                scene_profile="double_density_velocity_ood",
                ep0=ep0,
                noise_seed=noise_seed,
                m_per_gamma=50,
                workers=workers,
                cache_dir=cache,
                output_dir=output,
            ),
        })
    logs = FUNNEL.run_parallel(
        jobs,
        gpu_index=gpu_index,
        cpu_start=cpu_start,
        cpu_count=workers,
        log_dir=output_root / "logs",
    )
    rows = []
    r0 = None
    for job in jobs:
        payload = FUNNEL.read_json(
            job["output"] / "raw_m50_offline_metrics.json"
        )
        if r0 is None:
            r0 = FUNNEL.pooled_row("pretrained", payload["records"][0])
        rows.append(
            FUNNEL.pooled_row(
                job["candidate"]["arm"], payload["records"][1]
            )
        )
    return rows, r0, logs


def run(args) -> dict:
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    margin_root = Path(args.margin_root).resolve()
    cost_marker = Path(args.cost_marker).resolve()
    balanced_marker = Path(args.balanced_marker).resolve()
    margin_delivery = wait_for(
        margin_root / "DELIVERY_COMPLETE.json", args.poll_seconds
    )
    cost = wait_for(cost_marker, args.poll_seconds)
    balanced = wait_for(balanced_marker, args.poll_seconds)
    if margin_delivery.get("status") != (
        "SFM_B1_OFFLINE_9ARM_DELIVERY_COMPLETE"
    ):
        raise RuntimeError("invalid margin delivery")
    for payload, selector in (
        (cost, "safemppi_cost"),
        (balanced, "balanced_rank"),
    ):
        if (
            payload.get("status")
            != "SFM_B1_OFFLINE_SELECTOR_FUNNEL_COMPLETE"
            or payload.get("selector") != selector
        ):
            raise RuntimeError(f"invalid {selector} funnel marker")

    margin_all, margin_r0_screen = margin_rows(margin_root)
    margin_candidates, margin_screen_selection = FUNNEL.choose_candidates(
        margin_all, margin_r0_screen, top_k=args.top_k
    )
    checkpoint = str(Path(
        FUNNEL.read_json(margin_root / "RUN_DECLARATION.json")
        ["contract"]["checkpoint"]
    ).resolve())
    margin_confirm, shared_r0, margin_logs = evaluate_candidates(
        margin_candidates,
        checkpoint=checkpoint,
        output_root=output_root / "margin_confirmation_m50",
        gpu_index=args.gpu_index,
        cpu_start=args.cpu_start,
        workers=args.workers,
        ep0=args.selector_ep0,
        noise_seed=args.selector_noise_seed,
    )
    margin_winners, margin_confirm_selection = FUNNEL.choose_candidates(
        margin_confirm, shared_r0, top_k=1
    )
    selector_rows = [
        margin_winners[0],
        cost["confirmation"]["selector_winner"],
        balanced["confirmation"]["selector_winner"],
    ]
    global_winners, global_selection = FUNNEL.choose_candidates(
        selector_rows, shared_r0, top_k=1
    )
    global_winner = global_winners[0]

    m100_root = output_root / "final_m100"
    m100_command = FUNNEL.evaluator_command(
        [checkpoint, global_winner["checkpoint"]],
        ["r0", f"r{global_winner['round']}"],
        scene_profile="double_density_velocity_ood",
        ep0=args.final_ep0,
        noise_seed=args.final_noise_seed,
        m_per_gamma=100,
        workers=args.workers,
        cache_dir=m100_root / "cache",
        output_dir=m100_root,
    )
    m100_log = FUNNEL.run_job(
        "final_m100",
        m100_command,
        gpu_index=args.gpu_index,
        cpu_start=args.cpu_start,
        cpu_count=args.workers,
        log_dir=output_root / "logs",
    )
    m100 = FUNNEL.read_json(
        m100_root / "raw_m100_offline_metrics.json"
    )
    r0_m100 = FUNNEL.pooled_row("pretrained", m100["records"][0])
    winner_m100 = FUNNEL.pooled_row(
        global_winner["arm"], m100["records"][1]
    )
    result = {
        "status": "SFM_B1_OFFLINE_FINAL_M100_COMPLETE",
        "role": (
            "three selectors compared on one shared disjoint M50 bank; "
            "exactly one global winner confirmed on a further disjoint M100 "
            "raw temperature-one bank"
        ),
        "source_commit": FUNNEL.subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=FUNNEL.HERE, text=True
        ).strip(),
        "margin_screening": {
            "bank": {"M_per_gamma": 50, "ep0": 260_000},
            "selection": margin_screen_selection,
            "candidates": margin_candidates,
        },
        "shared_selector_confirmation": {
            "bank": {
                "M_per_gamma": 50,
                "ep0": args.selector_ep0,
                "noise_seed": args.selector_noise_seed,
            },
            "common_r0": shared_r0,
            "margin_rows": margin_confirm,
            "margin_selection": margin_confirm_selection,
            "selector_winners": selector_rows,
            "global_selection": global_selection,
            "global_winner": global_winner,
        },
        "final_confirmation": {
            "bank": {
                "M_per_gamma": 100,
                "ep0": args.final_ep0,
                "noise_seed": args.final_noise_seed,
            },
            "pretrained_r0": r0_m100,
            "winner": winner_m100,
        },
        "artifacts": {
            "margin_logs": margin_logs,
            "m100_log": m100_log,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    marker = output_root / "FINAL_M100_COMPLETE.json"
    FUNNEL.write_json(marker, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-root", required=True)
    parser.add_argument("--cost-marker", required=True)
    parser.add_argument("--balanced-marker", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--cpu-start", type=int, default=144)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--selector-ep0", type=int, default=270_000)
    parser.add_argument("--selector-noise-seed", type=int, default=2_026_072_4)
    parser.add_argument("--final-ep0", type=int, default=280_000)
    parser.add_argument("--final-noise-seed", type=int, default=2_026_072_5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
