#!/usr/bin/env python3
"""Matched Stage-5 checkpoint/temperature evaluation on the giant scene."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import grid_hp_expt as HP  # noqa: E402
from giant_obstacle_ood.stage4_frozen_ood import (  # noqa: E402
    load_records,
    rollout_policy,
    save_records,
    summarize_method,
)
from viz_style import GAMMAS  # noqa: E402


STAGE = HERE / "stage_results/05_window_expand"
RUNS = STAGE / "runs/temp0.5_stable"
EVAL = STAGE / "evaluation"
TEMPERATURES = (0.1, 0.5, 1.0)
ARMS = ("full", "no_socp", "no_progress", "no_curriculum")


def finite(value, fallback: float) -> float:
    return float(value) if value is not None and np.isfinite(value) else float(fallback)


def screen_score(summary: dict) -> tuple:
    overall = summary["overall"]
    return (
        float(overall["a_SR"]),
        -float(overall["b_CR"]),
        int(overall["e_coverage"]),
        finite(overall.get("mean_boundary_arc_rad"), -math.inf),
        finite(overall.get("mean_goal_progress"), -math.inf),
        -finite(overall.get("mean_endpoint_distance"), math.inf),
    )


def candidate_checkpoints(run: Path) -> list[Path]:
    output = [run / "final.pt"]
    output += sorted(run.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    output += [run / "best.pt", run / "safe_best.pt"]
    unique = []
    seen = set()
    for path in output:
        if path.exists() and path.resolve() not in seen:
            unique.append(path)
            seen.add(path.resolve())
    return unique


def evaluate_checkpoint(checkpoint: Path, *, temperature: float, repetitions: int,
                        device: torch.device, method: str, seed0: int,
                        persistent_route_bit: bool = False,
                        persistent_latent: bool = False,
                        latent_correlation: float = 0.0,
                        ensemble_size: int = 1) -> tuple[list[dict], dict]:
    policy, _ = HP.load_hp(checkpoint, device=device)
    records = rollout_policy(
        policy, repetitions=repetitions, temperature=temperature, nfe=8,
        T=300, seed0=seed0, device=device, method=method,
        persistent_route_bit=persistent_route_bit,
        persistent_latent=persistent_latent,
        latent_correlation=latent_correlation,
        ensemble_size=ensemble_size,
    )
    return records, summarize_method(records)


def write_rows(summary: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for gamma in GAMMAS:
        row = summary["per_gamma"][str(float(gamma))]
        payload = {
            "gamma": float(gamma),
            "M": row["M"],
            "SR": row["a_SR"],
            "CR": row["b_CR"],
            "clearance_mean": row["c_clearance_mean_success"],
            "min_clearance_mean": row["min_clearance_mean_success"],
            "time_mean_s": row["d_time_s_mean_success"],
            "coverage": row["e_coverage"],
            "coverage_modes": row["coverage_modes"],
            "boundary_arc_mean": row["mean_boundary_arc_rad"],
            "endpoint_distance_mean": row["mean_endpoint_distance"],
            "failure_taxonomy": row["failure_taxonomy"],
        }
        (directory / f"row_g{float(gamma)}.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-m", type=int, default=3)
    parser.add_argument("--final-m", type=int, default=6)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EVAL.mkdir(parents=True, exist_ok=True)

    screen = []
    for checkpoint in candidate_checkpoints(RUNS / "full"):
        records, summary = evaluate_checkpoint(
            checkpoint, temperature=0.5, repetitions=args.screen_m, device=device,
            method=f"Ours screen {checkpoint.stem}", seed0=91500,
        )
        screen.append({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_name": checkpoint.name,
            "summary": summary,
            "score": list(screen_score(summary)),
        })
    if not screen:
        raise FileNotFoundError("no Full checkpoints found")
    selected = max(screen, key=lambda row: tuple(row["score"]))
    selected_checkpoint = Path(selected["checkpoint"])
    selection = {
        "status": "PASS",
        "screen_M_per_gamma": args.screen_m,
        "temperature": 0.5,
        "selection_order": ["SR", "-CR", "route coverage", "boundary arc", "goal progress", "-endpoint distance"],
        "selected_checkpoint": str(selected_checkpoint),
        "candidates": screen,
    }
    (EVAL / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    all_summaries: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    for arm in ARMS:
        checkpoint = selected_checkpoint if arm == "full" else RUNS / arm / "final.pt"
        records, summary = evaluate_checkpoint(
            checkpoint, temperature=0.5, repetitions=args.final_m, device=device,
            method="Ours" if arm == "full" else arm,
            seed0=92500,
        )
        directory = EVAL / arm / "temp0.5"
        directory.mkdir(parents=True, exist_ok=True)
        save_records(
            records, directory / f"rollouts_m{args.final_m}.npz",
            checkpoint=np.asarray(str(checkpoint.resolve())),
            temperature=np.asarray(0.5), matched_seed0=np.asarray(92500),
        )
        (directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_rows(summary, directory)
        all_summaries[arm] = summary
        all_records[arm] = records

    temperature_summary = {"0.5": all_summaries["full"]}
    for temperature in (0.1, 1.0):
        records, summary = evaluate_checkpoint(
            selected_checkpoint, temperature=temperature, repetitions=args.final_m,
            device=device, method=f"Ours T={temperature:g}", seed0=92500,
        )
        directory = EVAL / "full" / f"temp{temperature:g}"
        directory.mkdir(parents=True, exist_ok=True)
        save_records(
            records, directory / f"rollouts_m{args.final_m}.npz",
            checkpoint=np.asarray(str(selected_checkpoint.resolve())),
            temperature=np.asarray(temperature), matched_seed0=np.asarray(92500),
        )
        (directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        temperature_summary[str(temperature)] = summary

    mode_audit = {"status": "PASS", "arms": {}}
    alerts = []
    for arm, records in all_records.items():
        successes = [record for record in records if record["success"]]
        modes = Counter(record["route_mode"] for record in successes)
        mode_audit["arms"][arm] = {
            "successful_trajectories": len(successes),
            "successful_route_modes": dict(modes),
            "distinct_global_detour_modes": len([mode for mode in modes if mode in ("upper-left", "lower-right")]),
            "failure_taxonomy": dict(Counter(record["failure_type"] for record in records)),
        }
    ours_modes = mode_audit["arms"]["full"]["distinct_global_detour_modes"]
    ours_success = mode_audit["arms"]["full"]["successful_trajectories"]
    if ours_success > 0 and ours_modes < 2:
        alerts.append("Full has successful rollouts but only one global detour mode")
    if ours_success == 0:
        alerts.append("Full has no successful rollout in the matched M evaluation")
    mode_audit["alerts"] = alerts
    mode_audit["status"] = "ALERT" if alerts else "PASS"
    (EVAL / "route_mode_audit.json").write_text(json.dumps(mode_audit, indent=2) + "\n")
    (EVAL / "temperature_sweep_metrics.json").write_text(
        json.dumps(temperature_summary, indent=2) + "\n"
    )
    manifest = {
        "status": "PASS",
        "physical_gpu_requested": 2,
        "visible_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "selected_checkpoint": str(selected_checkpoint),
        "main_temperature": 0.5,
        "temperature_sweep": list(TEMPERATURES),
        "M_per_gamma": args.final_m,
        "mode_audit_status": mode_audit["status"],
        "mode_alerts": alerts,
    }
    (EVAL / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
