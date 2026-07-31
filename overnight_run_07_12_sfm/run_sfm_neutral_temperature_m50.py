#!/usr/bin/env python3
"""Validation-select one global raw temperature, then run disjoint M50.

This is a post-training evaluator.  It never mutates the four neutral-teacher
training runs.  Temperature and checkpoint selection use only the completed
runs' M20 screen plus a fresh M10 validation bank.  The final M50 scenario and
noise bank are not read until one expanded checkpoint/temperature and one
pretrained temperature are locked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grid_policy_sfm as GPS
import run_sfm_b1_offline_eval_funnel as FUNNEL
import sfm_b1_offline_eval as EVAL
import sfm_kazuki as KZ
import sfm_protocol as SP
import sfm_scene as SS


HERE = Path(__file__).resolve().parent
STATUS = "SFM_NEUTRAL_TEMPERATURE_DISJOINT_M50_COMPLETE"
METRICS = ("CR", "Validity", "clearance", "time_to_goal")
LOWER_IS_BETTER = {"CR", "time_to_goal"}


def _read(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _write(path: Path, value) -> None:
    FUNNEL.write_json(path, value)


def _wait_for_deliveries(root: Path, arm_names: list[str], poll: int) -> list[dict]:
    paths = [root / name / "DELIVERY_COMPLETE.json" for name in arm_names]
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            break
        print(f"WAITING deliveries {len(paths) - len(missing)}/{len(paths)}", flush=True)
        time.sleep(int(poll))
    payloads = [_read(path) for path in paths]
    for path, payload in zip(paths, payloads):
        if payload.get("status") != "SFM_B1_NEUTRAL_MULTIROUND_COMPLETE":
            raise RuntimeError(f"invalid delivery: {path}")
    checkpoints = {payload["checkpoint_sha256"] for payload in payloads}
    if len(checkpoints) != 1:
        raise RuntimeError("arms do not share one pretrained checkpoint")
    return payloads


def _validate_banks(
    payloads: list[dict], *, screen_ep0: int, screen_M: int,
    validation_ep0: int, validation_M: int, final_ep0: int,
    expected_final_round: int,
) -> set[int]:
    training_scenarios = set()
    for payload in payloads:
        screen = payload.get("disjoint_raw_evaluation", {})
        if (
            int(screen.get("ep0", -1)) != int(screen_ep0)
            or int(screen.get("M_per_gamma", -1)) != int(screen_M)
        ):
            raise RuntimeError("delivery screening bank differs from declaration")
        if int(payload.get("rounds", -1)) != int(expected_final_round):
            raise RuntimeError("delivery does not reach the expected final round")
        arm_scenarios = set()
        for marker in payload.get("round_records", []):
            record = _read(Path(marker))
            arm_scenarios.update(map(int, record.get("scenarios", ())))
        expected_scenarios = 2 * int(payload.get(
            "rounds_run_this_invocation", payload["rounds"]
        ))
        if len(arm_scenarios) != expected_scenarios:
            raise RuntimeError("training scenario lineage is incomplete")
        if training_scenarios and arm_scenarios != training_scenarios:
            raise RuntimeError("arms used different training scenarios")
        training_scenarios = arm_scenarios
    banks = {
        "screen": set(range(int(screen_ep0), int(screen_ep0) + int(screen_M))),
        "validation": set(range(
            int(validation_ep0), int(validation_ep0) + int(validation_M)
        )),
        "final": set(range(int(final_ep0), int(final_ep0) + 50)),
    }
    names = list(banks)
    for index, name in enumerate(names):
        if banks[name] & training_scenarios:
            raise RuntimeError(f"{name} bank overlaps training scenarios")
        for other in names[index + 1:]:
            if banks[name] & banks[other]:
                raise RuntimeError(f"{name} and {other} banks overlap")
    return training_scenarios


def _mean(value: dict) -> float:
    result = value.get("mean")
    return float("nan") if result is None else float(result)


def _pooled(source: dict) -> dict:
    return {
        "SR": float(source["SR"]),
        "CR": float(source["CR"]),
        "timeout": float(source["timeout"]),
        "Validity": _mean(source["Validity"]),
        "clearance": _mean(source["successful_clearance"]),
        "time_to_goal": _mean(source["successful_time_to_goal"]),
    }


def _record_from_cell(payload: dict, *, method: str, temperature: float) -> dict:
    record = payload["records"][0]
    return {
        "method": method,
        "round": int(record["round"]),
        "checkpoint": record["cell"]["checkpoint"],
        "checkpoint_sha256": record["cell"]["checkpoint_sha256"],
        "temperature": float(temperature),
        "pooled": _pooled(record["cell"]["summary"]["pooled"]),
        "per_gamma": {
            gamma: _pooled(cell)
            for gamma, cell in record["cell"]["summary"]["per_gamma"].items()
        },
        "metrics_json": payload["metrics_json"] if "metrics_json" in payload else None,
    }


def _screen_rows(root: Path, arm_names: list[str], payloads: list[dict]) -> tuple[list[dict], dict]:
    rows, r0 = [], None
    for arm_name, delivery in zip(arm_names, payloads):
        for record in delivery["disjoint_raw_evaluation"]["records"]:
            pooled = _pooled(record["pooled"])
            round_index = int(record["round"])
            row = {
                "arm": arm_name,
                "round": round_index,
                "checkpoint": str(Path(record.get(
                    "checkpoint",
                    root / arm_name / "checkpoints" / f"round_{round_index:02d}.pt",
                )).resolve()),
                "pooled": pooled,
            }
            if round_index == 0:
                r0 = r0 or row
            else:
                rows.append(row)
    if r0 is None:
        raise RuntimeError("completed runs contain no r0 screening record")
    return rows, r0


def _screen_key(row: dict, r0: dict) -> tuple:
    value, base = row["pooled"], r0["pooled"]
    if any(not math.isfinite(float(value[key])) for key in METRICS):
        return (1, float("inf"), float("inf"), float("inf"), float("inf"),
                float("inf"), row["round"], row["arm"])
    wins = (
        int(value["CR"] < base["CR"])
        + int(value["Validity"] > base["Validity"])
        + int(value["clearance"] > base["clearance"])
        + int(value["time_to_goal"] < base["time_to_goal"])
    )
    return (
        0,
        -wins,
        value["CR"],
        -value["Validity"],
        -value["clearance"],
        value["time_to_goal"],
        -value["SR"],
        row["round"],
        row["arm"],
    )


def _shortlist(rows: list[dict], r0: dict, arm_names: list[str]) -> list[dict]:
    selected = []
    for arm in arm_names:
        values = [row for row in rows if row["arm"] == arm]
        if not values:
            raise RuntimeError(f"no screening rows for {arm}")
        selected.append(min(values, key=lambda row: _screen_key(row, r0)))
    return selected


def _run_jobs(jobs: list[dict], *, gpus: list[int], workers: int, log_dir: Path) -> list[dict]:
    queues = [[] for _ in gpus]
    for index, job in enumerate(jobs):
        queues[index % len(gpus)].append(job)

    def consume(gpu: int, cpu_start: int, values: list[dict]) -> list[dict]:
        completed = []
        for job in values:
            completed.append(FUNNEL.run_job(
                job["name"],
                job["command"],
                gpu_index=gpu,
                cpu_start=cpu_start,
                cpu_count=workers,
                log_dir=log_dir,
            ))
            print(f"COMPLETE {job['name']} gpu={gpu}", flush=True)
        return completed

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(consume, gpu, 16 + index * 80, queue)
            for index, (gpu, queue) in enumerate(zip(gpus, queues))
        ]
        return [item for future in futures for item in future.result()]


def _raw_command(
    checkpoint: str,
    *,
    round_index: int,
    temperature: float,
    ep0: int,
    noise_seed: int,
    M: int,
    workers: int,
    output: Path,
    cache: Path,
) -> list[str]:
    return FUNNEL.evaluator_command(
        [checkpoint],
        [f"r{round_index}"],
        scene_profile="double_density_velocity_ood",
        ep0=ep0,
        noise_seed=noise_seed,
        m_per_gamma=M,
        workers=workers,
        cache_dir=cache,
        output_dir=output,
        temperature=temperature,
    )


def _compact_kazuki(rollout: dict, episode: int, gamma: float) -> dict:
    status = (
        "success" if rollout["success"]
        else "collision" if rollout["collision"]
        else "timeout"
    )
    steps = int(rollout["steps"])
    return {
        "episode": int(episode),
        "gamma": float(gamma),
        "status": status,
        "success": bool(rollout["success"]),
        "collision": bool(rollout["collision"]),
        "timeout": status == "timeout",
        "steps": steps,
        "time_to_goal": steps * SS.DT if rollout["success"] else None,
        "min_clearance": float(rollout["min_clear"]),
        "successful_clearance": (
            float(rollout["min_clear"]) if rollout["success"] else None
        ),
        "states": np.asarray(rollout["states"], np.float32),
        "controls": np.asarray(rollout["controls"], np.float32),
        "ped_xy": np.asarray(rollout["peds"], np.float32),
        "ped_vel": np.asarray(rollout["ped_vels"], np.float32),
    }


def run_kazuki(args) -> dict:
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=args.device)
    policy.eval()
    config = KZ.KazukiConfig(safe_coefs=(0.3,), goal_coef=0.5).validate()
    environment = SS.scene_profile("double_density_velocity_ood")
    rows = []
    for gamma in SP.GAMMAS:
        for episode in range(int(args.ep0), int(args.ep0) + int(args.M)):
            rollout = KZ.kazuki_sfm_deploy(
                policy,
                episode,
                gamma,
                cfg=config,
                n_ped=environment["n_ped"],
                T=SP.T,
                device=args.device,
                ped_speed_range=tuple(environment["ped_speed_range"]),
                sample_seed=700_000,
                collect_diagnostics=False,
            )
            rows.append(_compact_kazuki(rollout, episode, gamma))
    del policy
    import multiprocessing as mp
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=mp.get_context("spawn")
    ) as executor:
        compact = EVAL._attach_validity(rows, executor)
    summary = EVAL.summarize(compact, seed=int(args.ep0) + 700)
    EVAL._assert_zero_verifier_errors(summary)
    result = {
        "status": "SFM_LOCKED_KAZUKI_EXACT_VALIDITY_COMPLETE",
        "method": "locked Kazuki generate-guide-refine",
        "safe_coef": 0.3,
        "goal_coef": 0.5,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": FUNNEL.sha256_file(Path(args.checkpoint)),
        "scene_profile": "double_density_velocity_ood",
        "ep0": int(args.ep0),
        "M_per_gamma": int(args.M),
        "summary": summary,
        "rows": compact,
    }
    _write(Path(args.output), result)
    return result


def _kazuki_record(payload: dict) -> dict:
    return {
        "method": "kazuki_locked",
        "round": 0,
        "checkpoint": payload["checkpoint"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "temperature": None,
        "pooled": _pooled(payload["summary"]["pooled"]),
        "per_gamma": {
            gamma: _pooled(cell)
            for gamma, cell in payload["summary"]["per_gamma"].items()
        },
    }


def _envelope(pretrained: list[dict], kazuki: dict) -> dict:
    references = [*pretrained, kazuki]
    return {
        metric: (
            min(row["pooled"][metric] for row in references)
            if metric in LOWER_IS_BETTER
            else max(row["pooled"][metric] for row in references)
        )
        for metric in METRICS
    }


def _shortfalls(row: dict, target: dict) -> dict:
    value = row["pooled"]
    result = {}
    for metric in METRICS:
        if not math.isfinite(float(value[metric])):
            result[metric] = float("inf")
            continue
        scale = max(abs(float(target[metric])), .02)
        if metric in LOWER_IS_BETTER:
            result[metric] = max(0.0, value[metric] - target[metric]) / scale
        else:
            result[metric] = max(0.0, target[metric] - value[metric]) / scale
    return result


def _liveness_contract(pretrained: list[dict], kazuki: dict) -> dict:
    references = [*pretrained, kazuki]
    return {
        "minimum_SR": max(
            0.0, min(row["pooled"]["SR"] for row in references) - .05
        ),
        "maximum_timeout": min(
            1.0, max(row["pooled"]["timeout"] for row in references) + .05
        ),
        "every_gamma_has_success": True,
    }


def _liveness_eligible(row: dict, contract: dict) -> bool:
    if (
        row["pooled"]["SR"] < contract["minimum_SR"]
        or row["pooled"]["timeout"] > contract["maximum_timeout"]
    ):
        return False
    return all(
        math.isfinite(float(cell["clearance"]))
        and math.isfinite(float(cell["time_to_goal"]))
        for cell in row["per_gamma"].values()
    )


def _trend(record: dict) -> dict:
    rows = [record["per_gamma"][str(gamma)] for gamma in SP.GAMMAS]
    tests = {
        "CR_low_gamma_not_higher": [
            rows[i]["CR"] <= rows[i + 1]["CR"] + .10
            for i in range(len(rows) - 1)
        ],
        "Validity_rises_with_gamma": [
            rows[i]["Validity"] <= rows[i + 1]["Validity"] + .10
            for i in range(len(rows) - 1)
        ],
        "clearance_falls_with_gamma": [
            rows[i]["clearance"] >= rows[i + 1]["clearance"] - .02
            for i in range(len(rows) - 1)
        ],
        "time_falls_with_gamma": [
            rows[i]["time_to_goal"] >= rows[i + 1]["time_to_goal"] - 1.0
            for i in range(len(rows) - 1)
        ],
    }
    fractions = {
        name: sum(values) / len(values) for name, values in tests.items()
    }
    return {
        "adjacent_pair_fractions": fractions,
        "mean_fraction": float(np.mean(list(fractions.values()))),
        "tolerances": {"rate": .10, "clearance_m": .02, "time_s": 1.0},
    }


def _selection_key(row: dict, target: dict, liveness: dict) -> tuple:
    shortfall = _shortfalls(row, target)
    trend = _trend(row)
    return (
        0 if _liveness_eligible(row, liveness) else 1,
        max(shortfall.values()),
        sum(shortfall.values()),
        -trend["mean_fraction"],
        -row["pooled"]["SR"],
        row["pooled"]["timeout"],
        row["round"],
        row["method"],
        row["temperature"],
    )


def _render(records: list[dict], output: Path) -> None:
    specs = (
        ("CR", "Collision rate"),
        ("Validity", "Validity"),
        ("clearance", "Min. clearance [m]"),
        ("time_to_goal", "Time-to-goal [s]"),
    )
    colors = {
        "pretrained": "#7f7f7f",
        "expanded": "#0072B2",
        "kazuki_locked": "#CC79A7",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.6, 10.8))
    gammas = np.asarray(SP.GAMMAS, float)
    for axis, (metric, title) in zip(axes.flat, specs):
        for record in records:
            values = [
                record["per_gamma"][str(gamma)][metric] for gamma in SP.GAMMAS
            ]
            label = record["method"]
            if record["temperature"] is not None:
                label += f" (temp={record['temperature']:g})"
            axis.plot(
                gammas, values, marker="o", lw=2.4,
                color=colors[record["method"]], label=label,
            )
        axis.set_title(title)
        axis.set_xlabel(r"$\gamma$")
        axis.grid(alpha=.25)
        if metric in {"CR", "Validity"}:
            axis.set_ylim(-.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args) -> dict:
    started = datetime.now(timezone.utc)
    training_root = Path(args.training_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    arm_names = args.arm_names.split(",")
    temperatures = [float(item) for item in args.temperatures.split(",")]
    if 1.0 not in temperatures or any(item <= 0 or item > 1 for item in temperatures):
        raise ValueError("temperatures must be in (0,1] and include 1")
    deliveries = _wait_for_deliveries(training_root, arm_names, args.poll_seconds)
    training_scenarios = _validate_banks(
        deliveries,
        screen_ep0=args.screen_ep0,
        screen_M=args.screen_M,
        validation_ep0=args.validation_ep0,
        validation_M=args.validation_M,
        final_ep0=args.final_ep0,
        expected_final_round=args.expected_final_round,
    )
    screen_rows, screen_r0 = _screen_rows(
        training_root, arm_names, deliveries
    )
    shortlist = _shortlist(screen_rows, screen_r0, arm_names)
    pretrained = str(Path(deliveries[0]["checkpoint"]).resolve())

    validation_root = output / "validation_m10"
    cache = validation_root / "cache"
    jobs = []
    job_meta = {}
    for temperature in temperatures:
        name = f"pretrained_temp{temperature:g}".replace(".", "p")
        out = validation_root / name
        jobs.append({
            "name": name,
            "command": _raw_command(
                pretrained, round_index=0, temperature=temperature,
                ep0=args.validation_ep0, noise_seed=args.validation_noise_seed,
                M=args.validation_M, workers=args.workers,
                output=out, cache=cache,
            ),
        })
        job_meta[name] = ("pretrained", 0, temperature, out)
    for candidate in shortlist:
        for temperature in temperatures:
            name = (
                f"{candidate['arm']}_r{candidate['round']}_temp{temperature:g}"
            ).replace(".", "p")
            out = validation_root / name
            jobs.append({
                "name": name,
                "command": _raw_command(
                    candidate["checkpoint"], round_index=candidate["round"],
                    temperature=temperature, ep0=args.validation_ep0,
                    noise_seed=args.validation_noise_seed,
                    M=args.validation_M, workers=args.workers,
                    output=out, cache=cache,
                ),
            })
            job_meta[name] = (
                candidate["arm"], candidate["round"], temperature, out
            )
    _run_jobs(
        jobs, gpus=args.gpus, workers=args.workers,
        log_dir=output / "logs",
    )

    validation_records = []
    for name, (method, round_index, temperature, out) in job_meta.items():
        payload = _read(out / f"raw_m{args.validation_M}_offline_metrics.json")
        payload["metrics_json"] = str(
            out / f"raw_m{args.validation_M}_offline_metrics.json"
        )
        validation_records.append(_record_from_cell(
            payload, method=method, temperature=temperature
        ))

    kazuki_validation_path = validation_root / "kazuki_locked.json"
    kazuki_command = [
        sys.executable, str(Path(__file__).resolve()), "kazuki",
        "--checkpoint", pretrained,
        "--ep0", str(args.validation_ep0),
        "--M", str(args.validation_M),
        "--workers", str(args.workers),
        "--device", "cuda:0",
        "--output", str(kazuki_validation_path),
    ]
    FUNNEL.run_job(
        "kazuki_validation",
        kazuki_command,
        gpu_index=args.gpus[0],
        cpu_start=16,
        cpu_count=args.workers,
        log_dir=output / "logs",
    )
    kazuki_validation = _kazuki_record(_read(kazuki_validation_path))
    pretrained_records = [
        row for row in validation_records if row["method"] == "pretrained"
    ]
    expanded_records = [
        row for row in validation_records if row["method"] != "pretrained"
    ]
    target = _envelope(pretrained_records, kazuki_validation)
    liveness = _liveness_contract(pretrained_records, kazuki_validation)
    selected_expanded = min(
        expanded_records,
        key=lambda row: _selection_key(row, target, liveness),
    )
    selected_pretrained = min(
        pretrained_records,
        key=lambda row: _selection_key(row, target, liveness),
    )
    selection = {
        "status": "LOCKED_BEFORE_DISJOINT_M50",
        "rule": (
            "minimize maximum then total normalized shortfall from the "
            "metric-wise envelope of all pretrained temperatures and locked "
            "Kazuki; break ties by gamma-trend score, SR, timeout, round, name"
        ),
        "global_temperature_only": True,
        "per_gamma_temperature_forbidden": True,
        "target_envelope": target,
        "liveness_contract": liveness,
        "selected_expanded": selected_expanded,
        "selected_pretrained": selected_pretrained,
        "expanded_shortfalls": _shortfalls(selected_expanded, target),
        "expanded_point_estimate_four_metric_gate": (
            _liveness_eligible(selected_expanded, liveness)
            and all(
                value == 0
                for value in _shortfalls(selected_expanded, target).values()
            )
        ),
        "expanded_gamma_trend": _trend(selected_expanded),
    }
    _write(output / "SELECTION_LOCKED.json", selection)

    final_root = output / "disjoint_m50"
    final_cache = final_root / "cache"
    final_jobs = []
    final_meta = {}
    for method, record in (
        ("pretrained", selected_pretrained),
        ("expanded", selected_expanded),
    ):
        out = final_root / method
        final_jobs.append({
            "name": f"final_{method}",
            "command": _raw_command(
                record["checkpoint"], round_index=record["round"],
                temperature=record["temperature"], ep0=args.final_ep0,
                noise_seed=args.final_noise_seed, M=50,
                workers=args.workers, output=out, cache=final_cache,
            ),
        })
        final_meta[method] = (record, out)
    final_kazuki = final_root / "kazuki_locked.json"
    final_jobs.append({
        "name": "final_kazuki",
        "command": [
            sys.executable, str(Path(__file__).resolve()), "kazuki",
            "--checkpoint", pretrained,
            "--ep0", str(args.final_ep0),
            "--M", "50",
            "--workers", str(args.workers),
            "--device", "cuda:0",
            "--output", str(final_kazuki),
        ],
    })
    _run_jobs(
        final_jobs, gpus=args.gpus, workers=args.workers,
        log_dir=output / "logs",
    )

    final_records = []
    for method, (record, out) in final_meta.items():
        payload = _read(out / "raw_m50_offline_metrics.json")
        final_records.append(_record_from_cell(
            payload, method=method, temperature=record["temperature"]
        ))
    final_records.append(_kazuki_record(_read(final_kazuki)))
    expanded_final = next(row for row in final_records if row["method"] == "expanded")
    references = [
        row for row in final_records if row["method"] != "expanded"
    ]
    final_target = _envelope(
        [row for row in references if row["method"] == "pretrained"],
        next(row for row in references if row["method"] == "kazuki_locked"),
    )
    final_shortfall = _shortfalls(expanded_final, final_target)
    _render(final_records, final_root / "four_metric_per_gamma.png")
    result = {
        "status": STATUS,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        "training_root": str(training_root),
        "training_scenario_ids": sorted(training_scenarios),
        "training_deliveries": [
            str(training_root / name / "DELIVERY_COMPLETE.json")
            for name in arm_names
        ],
        "banks": {
            "training_screen": {
                "M_per_gamma": args.screen_M, "ep0": args.screen_ep0
            },
            "temperature_validation": {
                "M_per_gamma": args.validation_M,
                "ep0": args.validation_ep0,
                "noise_seed": args.validation_noise_seed,
            },
            "disjoint_confirmation": {
                "M_per_gamma": 50,
                "ep0": args.final_ep0,
                "noise_seed": args.final_noise_seed,
            },
        },
        "temperatures": temperatures,
        "shortlist": shortlist,
        "selection": selection,
        "final_records": final_records,
        "final_target_envelope": final_target,
        "final_expanded_shortfalls": final_shortfall,
        "final_point_estimate_four_metric_gate": all(
            value == 0 for value in final_shortfall.values()
        ),
        "scientific_win_status": (
            "PENDING_PAIRED_SCENARIO_CLUSTER_CI; point estimates alone are "
            "not a win claim"
        ),
        "final_gamma_trends": {
            row["method"]: _trend(row) for row in final_records
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
    }
    _write(output / "DELIVERY_COMPLETE.json", result)
    print(json.dumps({
        "status": STATUS,
        "point_estimate_four_metric_gate": (
            result["final_point_estimate_four_metric_gate"]
        ),
        "expanded": expanded_final["pooled"],
        "delivery": str(output / "DELIVERY_COMPLETE.json"),
    }, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    full = sub.add_parser("run")
    full.add_argument("--training-root", required=True)
    full.add_argument("--output-dir", required=True)
    full.add_argument(
        "--arm-names",
        default="lr1em5_s01,lr1em5_s04,lr3em5_s01,lr3em5_s04",
    )
    full.add_argument("--temperatures", default="0.55,0.7,0.85,1.0")
    full.add_argument("--gpus", type=int, nargs="+", default=[1, 3])
    full.add_argument("--workers", type=int, default=32)
    full.add_argument("--poll-seconds", type=int, default=60)
    full.add_argument("--screen-ep0", type=int, default=270_000)
    full.add_argument("--screen-M", type=int, default=20)
    full.add_argument("--validation-ep0", type=int, default=460_000)
    full.add_argument("--validation-M", type=int, default=10)
    full.add_argument("--validation-noise-seed", type=int, default=2_026_073_4)
    full.add_argument("--final-ep0", type=int, default=470_000)
    full.add_argument("--final-noise-seed", type=int, default=2_026_073_5)
    full.add_argument("--expected-final-round", type=int, default=50)

    kazuki = sub.add_parser("kazuki")
    kazuki.add_argument("--checkpoint", required=True)
    kazuki.add_argument("--ep0", type=int, required=True)
    kazuki.add_argument("--M", type=int, required=True)
    kazuki.add_argument("--workers", type=int, default=32)
    kazuki.add_argument("--device", default="cuda:0")
    kazuki.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "kazuki":
        run_kazuki(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
