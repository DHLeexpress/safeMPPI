"""Fail-closed two-GPU alpha x replay-epoch sweep for max-margin B1.

Every arm trains completely before post-hoc temperature calibration.  A
disjoint M10 bank selects a seven-gamma temperature vector, M50 screens, and a
new M100 bank confirms the selected arm/round/vector once.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sfm_b1_curve_eval as CE
import sfm_b1_sweep as SW


ALPHAS = (0.0, 0.001, 0.01)
INNER_EPOCHS = (1, 4, 16)
OPTIMIZER_CHUNKS = 16
SWEEP_LR = 1.0e-4
OUTPUT_ROOT = "/data3/research1"
RUNTIME_GATE_ROUNDS = 3


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _validate_output_root(path: str) -> str:
    output = os.path.realpath(path)
    output_root = os.path.realpath(OUTPUT_ROOT)
    if os.path.commonpath((output, output_root)) != output_root or output == output_root:
        raise ValueError(f"all run artifacts must use a fresh directory under {OUTPUT_ROOT}")
    return output


@dataclass(frozen=True)
class SweepArm:
    alpha: float
    inner_epochs: int

    @property
    def name(self) -> str:
        return f"margin_alpha{_tag(self.alpha)}_inner{self.inner_epochs:03d}"


def arm_grid() -> tuple[SweepArm, ...]:
    return tuple(SweepArm(alpha, epochs) for alpha in ALPHAS for epochs in INNER_EPOCHS)


def _load_preflight(path, checkpoint, expected_sha256):
    observed_sha256 = SW.sha256_file(path)
    if observed_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"RBF preflight SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    with open(path) as stream:
        payload = json.load(stream)
    if payload.get("status") != "RBF_PREFLIGHT_COMPLETE":
        raise RuntimeError("RBF preflight is incomplete")
    if payload.get("checkpoint_sha256") != SW.sha256_file(checkpoint):
        raise RuntimeError("RBF preflight checkpoint does not match the sweep checkpoint")
    if int(payload.get("lengthscale_count", -1)) != 50 or float(payload.get("lambda_", -1)) != 1e-2:
        raise RuntimeError("RBF preflight is not the declared exact-50/lambda=.01 contract")
    selected = payload.get("selected", {})
    if (not selected.get("ess_solved") or not selected.get("stable_conditioning")
            or int(selected.get("cap", -1)) not in (256, 512)):
        raise RuntimeError("RBF preflight has no admissible selected configuration")
    cap512 = [
        row for row in payload.get("candidates", [])
        if int(row.get("cap", -1)) == 512
        and float(row.get("ell_multiplier", -1)) == float(selected.get("ell_multiplier", -2))
        and row.get("ess_solved") and row.get("stable_conditioning")
    ]
    if len(cap512) != 1:
        raise RuntimeError("preflight lacks one stable cap-512 row at the selected length scale")
    payload["sweep_selected"] = dict(cap512[0])
    return payload


def _pairs(jobs, logdir):
    logs = []
    for start in range(0, len(jobs), 2):
        wave = []
        for offset, (command, name) in enumerate(jobs[start:start + 2]):
            wave.append(("1" if offset == 0 else "3", command, name))
        logs.extend(SW.run_parallel(wave, logdir))
    return logs


def _selected_record(curve_dir, round_i):
    with open(os.path.join(curve_dir, "metrics.jsonl")) as stream:
        rows = [json.loads(line) for line in stream]
    matches = [row for row in rows
               if row["mode"] == "validation_selected_temperature"
               and int(row["round"]) == int(round_i)]
    if len(matches) != 1:
        raise RuntimeError(f"curve output has {len(matches)} selected records for round {round_i}")
    return matches[0]


def screening_key(row, arm):
    return (*CE.temperature_selection_key(row["summary"], row["temperature_by_gamma"]),
            abs(float(arm.alpha)), int(arm.inner_epochs), arm.name)


def _table_row(arm, record):
    pooled = record["summary"]["pooled"]
    return dict(
        arm=arm.name, alpha=arm.alpha, optimizer_chunks=OPTIMIZER_CHUNKS,
        inner_epochs=arm.inner_epochs, round=record["round"],
        temperature_by_gamma=json.dumps(record["temperature_by_gamma"], sort_keys=True),
        SR=pooled["SR"], CR=pooled["CR"], V_safe=pooled["V_safe"],
        successful_clearance=pooled["successful_clearance"]["mean"],
        successful_time_to_goal=pooled["successful_time_to_goal"]["mean"],
    )


def _plot_arm_comparison(outdir, arms, *, winner, winning_round):
    specs = (
        ("CR", "Collision rate"), ("V_safe", r"$V_{\mathrm{safe}}$"),
        ("successful_clearance", "Successful min. clearance [m]"),
        ("successful_time_to_goal", "Successful time-to-goal [s]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9), constrained_layout=True)
    colors = plt.cm.viridis_r([index / max(1, len(arms) - 1) for index in range(len(arms))])
    for arm, color in zip(arms, colors):
        with open(os.path.join(outdir, "curves", arm.name, "metrics.jsonl")) as stream:
            rows = [json.loads(line) for line in stream]
        rows = sorted(
            (row for row in rows if row["mode"] == "validation_selected_temperature"),
            key=lambda row: int(row["round"]),
        )
        rounds = [int(row["round"]) for row in rows]
        label = rf"$\alpha={arm.alpha:g},\ E={arm.inner_epochs}$"
        for axis, (metric, title) in zip(axes.flat, specs):
            values = []
            for row in rows:
                pooled = row["summary"]["pooled"]
                value = pooled[metric] if metric in ("CR", "V_safe") else pooled[metric]["mean"]
                values.append(np.nan if value is None else float(value))
            axis.plot(rounds, values, color=color, lw=1.45, alpha=.86, label=label)
            if arm == winner:
                index = rounds.index(int(winning_round))
                axis.plot(rounds[index], values[index], marker="*", ms=14,
                          color="#D55E00", mec="black", mew=.7)
            axis.set(title=title, xlabel="expansion round")
            axis.grid(alpha=.25)
            if metric in ("CR", "V_safe"):
                axis.set_ylim(-.03, 1.03)
    axes[0, 0].legend(ncol=3, fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(os.path.join(outdir, f"arm_comparison.{suffix}"), dpi=300,
                       bbox_inches="tight")
    plt.close(figure)


def _runtime_gate(args, *, ell, cap, python):
    """Benchmark both replay extremes after W=2 and cap-512 become active."""
    root = os.path.join(args.outdir, "runtime_gate")
    arm_min = os.path.join(root, "arm_alpha0_inner1")
    arm_max = os.path.join(root, "arm_alpha0p01_inner16")
    curve_dir = os.path.join(root, "curve")
    def command(outdir, name, alpha, epochs):
        return [
            python, os.path.join(SW.HERE, "sfm_b1_expand.py"),
            "--checkpoint", args.checkpoint, "--outdir", outdir,
            "--custom-name", name, "--selector", "margin", "--alpha", str(alpha),
            "--optimizer-steps", str(OPTIMIZER_CHUNKS), "--inner-epochs", str(epochs),
            "--lr", str(SWEEP_LR), "--sanity-M", "10",
            "--ell", str(ell), "--cap", str(cap),
            "--rounds", str(RUNTIME_GATE_ROUNDS), "--smoke",
            "--device", "cuda:0", "--verifier-workers", str(args.workers),
            "--scene-profile", args.scene_profile,
        ]
    logs = SW.run_parallel([
        ("1", command(arm_min, "runtime_alpha0_inner1", 0.0, 1), "runtime_train_min"),
        ("3", command(arm_max, "runtime_alpha0p01_inner16", .01, 16), "runtime_train_max"),
    ], os.path.join(root, "logs"))
    methods = []
    for directory in (arm_min, arm_max):
        with open(os.path.join(directory, "method_manifest.json")) as stream:
            methods.append(json.load(stream))
    train_round_seconds = max(
        float(record["wall_seconds"])
        for method in methods for record in method["history"]
    )
    stage_seconds = Counter()
    for method in methods:
        for record in method["history"]:
            for stage, seconds in record["gather"]["timers"].items():
                stage_seconds[stage] = max(float(stage_seconds[stage]), float(seconds))
    evaluation_started = time.perf_counter()
    logs.extend(SW.run_parallel([("1", [
        python, os.path.join(SW.HERE, "sfm_b1_curve_eval.py"), "run",
        "--checkpoint-dir", arm_max, "--scene-profile", args.scene_profile,
        "--outdir", curve_dir, "--rounds", f"0:{RUNTIME_GATE_ROUNDS}", "--device", "cuda:0",
        "--workers", str(args.workers), "--tune-M", str(args.tune_M),
        "--screen-M", str(args.screen_M),
    ], "runtime_gate_curve")], os.path.join(root, "logs")))
    eval_checkpoint_seconds = (
        time.perf_counter() - evaluation_started
    ) / float(RUNTIME_GATE_ROUNDS + 1)
    waves = math.ceil(len(arm_grid()) / 2)
    training_seconds = waves * int(args.rounds) * train_round_seconds
    evaluation_seconds = waves * (int(args.rounds) + 1) * eval_checkpoint_seconds
    confirmation_seconds = 2.0 * eval_checkpoint_seconds
    forecast_seconds = 1.25 * (training_seconds + evaluation_seconds + confirmation_seconds)
    payload = dict(
        status=("RUNTIME_GATE_PASS" if forecast_seconds <= args.max_hours * 3600
                else "RUNTIME_GATE_FAIL"),
        measured_train_round_seconds=train_round_seconds,
        measured_eval_checkpoint_seconds=eval_checkpoint_seconds,
        measured_stage_seconds=dict(stage_seconds),
        dominant_training_stage=(
            None if not stage_seconds else max(stage_seconds, key=stage_seconds.get)
        ),
        benchmark_rounds=RUNTIME_GATE_ROUNDS,
        arm_count=len(arm_grid()), parallel_waves=waves, rounds=int(args.rounds),
        forecast_components=dict(training=training_seconds, evaluation=evaluation_seconds,
                                 final_confirmation=confirmation_seconds, headroom=1.25),
        forecast_seconds=forecast_seconds, limit_seconds=float(args.max_hours) * 3600.0,
        logs=logs,
    )
    SW.write_json(os.path.join(root, "RUNTIME_FORECAST.json"), payload)
    if payload["status"] != "RUNTIME_GATE_PASS":
        SW.write_json(os.path.join(args.outdir, "BOUNDED_STOP.json"), dict(
            status="STOPPED_BEFORE_SCIENTIFIC_SWEEP", runtime_forecast=payload,
            scientific_knobs_changed=False,
        ))
        raise RuntimeError(
            f"runtime gate failed: forecast {forecast_seconds / 3600:.2f} h "
            f"> limit {args.max_hours:.2f} h"
        )
    return payload


def run(args):
    if (int(args.rounds), int(args.tune_M), int(args.screen_M), int(args.confirm_M)) != (
            20, 10, 50, 100):
        raise ValueError("scientific sweep requires rounds=20 and disjoint M10/M50/M100 banks")
    if not math.isfinite(float(args.max_hours)) or float(args.max_hours) <= 0.0:
        raise ValueError("max-hours must be finite and positive")
    if os.path.exists(args.outdir):
        raise FileExistsError(f"refusing existing output root: {args.outdir}")
    _validate_output_root(args.outdir)
    source = SW.git_frozen_source()
    gpu = SW.gpu_snapshot()
    if gpu["preexisting_processes"]:
        raise RuntimeError(f"GPU 1/3 are not exclusive: {gpu['preexisting_processes']}")
    preflight = _load_preflight(
        args.preflight, args.checkpoint, args.expected_preflight_sha256,
    )
    os.makedirs(args.outdir)
    started = time.perf_counter()
    seeds = SW.seed_bank_manifest(args.outdir, rounds=args.rounds)
    authentication = SW.authentication_manifest(
        args.outdir, args.checkpoint, args.scene_profile,
    )
    selected_rbf = preflight["sweep_selected"]
    ell, cap = float(selected_rbf["ell"]), int(selected_rbf["cap"])
    if cap != 512:
        raise AssertionError("latest uncertainty memory requires cap 512")
    python = sys.executable
    arms = arm_grid()
    recipe = dict(
        status="SFM_B1_ALPHA_INNER_EPOCH_SWEEP_DECLARED", source=source,
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=SW.sha256_file(args.checkpoint),
        scene_profile=args.scene_profile, rounds=int(args.rounds),
        fixed=dict(selector="margin", K=16, B=4, T=180, W=2, batch=128,
                   lr=SWEEP_LR, optimizer_chunks=OPTIMIZER_CHUNKS,
                   sanity_M=10, ess_target=.5, ell=ell, cap=cap,
                   gp_retained_per_round=256, gp_quantile=.75,
                   gp_gamma_quota="rotating 36/37; two-round 73/74"),
        factorial=dict(alpha=list(ALPHAS), inner_epochs=list(INNER_EPOCHS)),
        preflight_sha256=SW.sha256_file(args.preflight),
        temperature=dict(grid=list(CE.TEMPERATURES), tune_M=args.tune_M,
                         screen_M=args.screen_M, shared_across_gammas=False,
                         timing="post-expansion only"),
        evaluation=("M50 every round is screening; one disjoint M100 confirmation follows "
                    "selection; canonical temperature-1 remains a plotted control"),
        runtime_limit_hours=float(args.max_hours),
    )
    SW.write_json(os.path.join(args.outdir, "recipe.json"), recipe)
    runtime_forecast = _runtime_gate(args, ell=ell, cap=cap, python=python)
    if args.runtime_gate_only:
        payload = dict(
            status="SFM_B1_RUNTIME_GATE_ONLY_COMPLETE",
            source=source, gpu=gpu, authentication=authentication,
            preflight=preflight, recipe=recipe, runtime_forecast=runtime_forecast,
            wall_seconds=time.perf_counter() - started,
        )
        SW.write_json(os.path.join(args.outdir, "RUNTIME_GATE_ONLY_COMPLETE.json"), payload)
        return payload

    training_jobs = []
    for arm in arms:
        training_jobs.append(([
            python, os.path.join(SW.HERE, "sfm_b1_expand.py"),
            "--checkpoint", args.checkpoint, "--outdir", os.path.join(args.outdir, "arms", arm.name),
            "--custom-name", arm.name, "--selector", "margin", "--alpha", str(arm.alpha),
            "--optimizer-steps", str(OPTIMIZER_CHUNKS),
            "--inner-epochs", str(arm.inner_epochs), "--lr", str(SWEEP_LR), "--sanity-M", "10",
            "--ell", str(ell), "--cap", str(cap),
            "--rounds", str(args.rounds), "--device", "cuda:0", "--verifier-workers", str(args.workers),
            "--scene-profile", args.scene_profile,
        ], f"train_{arm.name}"))
    logs = list(runtime_forecast["logs"])
    logs.extend(_pairs(training_jobs, os.path.join(args.outdir, "logs")))

    evaluation_jobs = []
    for arm in arms:
        evaluation_jobs.append(([
            python, os.path.join(SW.HERE, "sfm_b1_curve_eval.py"), "run",
            "--checkpoint-dir", os.path.join(args.outdir, "arms", arm.name),
            "--scene-profile", args.scene_profile,
            "--outdir", os.path.join(args.outdir, "curves", arm.name),
            "--rounds", f"0:{args.rounds}", "--device", "cuda:0", "--workers", str(args.workers),
            "--tune-M", str(args.tune_M), "--screen-M", str(args.screen_M),
        ], f"curve_{arm.name}"))
    logs.extend(_pairs(evaluation_jobs, os.path.join(args.outdir, "logs")))

    candidates, table = [], []
    for arm in arms:
        curve_dir = os.path.join(args.outdir, "curves", arm.name)
        with open(os.path.join(curve_dir, "COMPLETE.json")) as stream:
            complete = json.load(stream)
        if complete.get("status") != "SFM_B1_CURVE_EVAL_COMPLETE":
            raise RuntimeError(f"curve evaluation incomplete: {arm.name}")
        record = _selected_record(curve_dir, complete["best_screening"]["round"])
        candidates.append((screening_key(record, arm), arm, record))
        table.append(_table_row(arm, record))
    _, winner, winning_record = min(candidates, key=lambda value: value[0])
    table_path = os.path.join(args.outdir, "screening_table.csv")
    with open(table_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)

    winning_round = int(winning_record["round"])
    winning_temperatures = dict(winning_record["temperature_by_gamma"])
    _plot_arm_comparison(
        args.outdir, arms, winner=winner, winning_round=winning_round,
    )
    confirmation_dir = os.path.join(args.outdir, "confirmation")
    checkpoint = os.path.join(args.outdir, "arms", winner.name, f"round_{winning_round:02d}.pt")
    logs.extend(SW.run_parallel([("1", [
        python, os.path.join(SW.HERE, "sfm_b1_curve_eval.py"), "confirm",
        "--checkpoint", checkpoint, "--round", str(winning_round),
        "--temperature-by-gamma", json.dumps(winning_temperatures, sort_keys=True),
        "--scene-profile", args.scene_profile,
        "--outdir", confirmation_dir, "--device", "cuda:0", "--workers", str(args.workers),
        "--M", str(args.confirm_M),
    ], "final_confirmation")], os.path.join(args.outdir, "logs")))
    with open(os.path.join(confirmation_dir, "COMPLETE.json")) as stream:
        confirmation = json.load(stream)
    if confirmation.get("status") != "SFM_B1_FINAL_CONFIRMATION_COMPLETE":
        raise RuntimeError("final confirmation is incomplete")
    final = dict(
        status="SFM_B1_ALPHA_INNER_EPOCH_SWEEP_COMPLETE", source=source, gpu=gpu,
        checkpoint=recipe["checkpoint"], checkpoint_sha256=recipe["checkpoint_sha256"],
        scene_profile=args.scene_profile, arms=[arm.__dict__ | {"name": arm.name} for arm in arms],
        winner=dict(arm=winner.name, alpha=winner.alpha,
                    optimizer_chunks=OPTIMIZER_CHUNKS, inner_epochs=winner.inner_epochs,
                    round=winning_round, temperature_by_gamma=winning_temperatures,
                    checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=SW.sha256_file(checkpoint)),
        screening_table=os.path.abspath(table_path), confirmation=confirmation,
        comparison_plot={
            suffix: dict(
                path=os.path.abspath(os.path.join(args.outdir, f"arm_comparison.{suffix}")),
                sha256=SW.sha256_file(os.path.join(args.outdir, f"arm_comparison.{suffix}")),
            ) for suffix in ("png", "pdf")
        },
        seed_banks={key: value for key, value in seeds.items() if key != "payload"},
        authentication=authentication, preflight=preflight, logs=logs,
        runtime_forecast=runtime_forecast,
        wall_seconds=time.perf_counter() - started,
        scientific_boundary=("per-gamma temperatures are chosen jointly only on M10 after training; "
                             "M50 selects arm/round only, "
                             "and the declared M100 bank is untouched until final confirmation; "
                             "selected-temperature curves are deployment tuning while dotted "
                             "temperature-1 curves measure the canonical generator"),
    )
    SW.write_json(os.path.join(args.outdir, "SWEEP_COMPLETE.json"), final)
    return final


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--scene-profile", default="double_density_velocity_ood",
                        choices=("double_density_velocity_ood", "density_ood", "requested_ood"))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tune-M", type=int, default=10)
    parser.add_argument("--screen-M", type=int, default=50)
    parser.add_argument("--confirm-M", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--runtime-gate-only", action="store_true")
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
