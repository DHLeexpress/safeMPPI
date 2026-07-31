#!/usr/bin/env python3
"""Calibrate a per-gamma raw temperature schedule, then reconfirm once.

The preceding global-temperature M50 is explicitly reclassified as the
calibration bank.  A seven-value schedule is selected independently for the
pretrained and expanded policies, hashed, and then evaluated on one fresh
disjoint M50 bank.  Locked Kazuki is never retuned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

import run_sfm_neutral_temperature_m50 as BASE
import sfm_protocol as SP


STATUS = "SFM_NEUTRAL_GAMMA_TEMPERATURE_M50_COMPLETE"


def _wait(path: Path, poll: int) -> dict:
    while not path.is_file():
        print(f"WAITING {path}", flush=True)
        time.sleep(int(poll))
    return BASE._read(path)


def _point(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty evaluation rows")
    successes = [row for row in rows if row["success"]]
    return {
        "SR": float(np.mean([row["success"] for row in rows])),
        "CR": float(np.mean([row["collision"] for row in rows])),
        "timeout": float(np.mean([row["timeout"] for row in rows])),
        "Validity": float(np.mean([row["validity"] for row in rows])),
        "clearance": (
            float(np.mean([row["successful_clearance"] for row in successes]))
            if successes else float("nan")
        ),
        "time_to_goal": (
            float(np.mean([row["time_to_goal"] for row in successes]))
            if successes else float("nan")
        ),
    }


def _candidate(method: str, schedule: tuple[float, ...], payloads: dict[float, dict],
               checkpoint: str, round_index: int) -> dict:
    rows, per_gamma = [], {}
    for gamma, temperature in zip(SP.GAMMAS, schedule):
        payload = payloads[float(temperature)]
        source = payload["records"][0]["cell"]["rows"]
        selected = [row for row in source if float(row["gamma"]) == float(gamma)]
        if len(selected) != 50:
            raise RuntimeError("calibration cell lacks M50/gamma rows")
        rows.extend(selected)
        per_gamma[str(gamma)] = _point(selected)
    return {
        "method": method,
        "round": int(round_index),
        "checkpoint": checkpoint,
        "temperature": None,
        "temperature_by_gamma": list(schedule),
        "pooled": _point(rows),
        "per_gamma": per_gamma,
    }


def _enumerate(method: str, payloads: dict[float, dict], temperatures: list[float],
               checkpoint: str, round_index: int):
    for schedule in itertools.product(temperatures, repeat=len(SP.GAMMAS)):
        yield _candidate(method, schedule, payloads, checkpoint, round_index)


def _single_reference_target(reference: dict) -> dict:
    return {metric: reference["pooled"][metric] for metric in BASE.METRICS}


def _pick(candidates, *, target: dict, liveness: dict) -> dict:
    def key(row):
        trend = BASE._trend(row)
        shortfall = BASE._shortfalls(row, target)
        return (
            0 if BASE._liveness_eligible(row, liveness) else 1,
            0 if trend["mean_fraction"] >= .75 else 1,
            max(shortfall.values()),
            sum(shortfall.values()),
            -trend["mean_fraction"],
            -row["pooled"]["SR"],
            row["pooled"]["timeout"],
            tuple(row["temperature_by_gamma"]),
        )
    return min(
        candidates,
        key=key,
    )


def _schedule_sha(record: dict) -> str:
    value = json.dumps({
        "checkpoint_sha256": record["checkpoint_sha256"],
        "round": record["round"],
        "temperature_by_gamma": record["temperature_by_gamma"],
        "gammas": list(map(float, SP.GAMMAS)),
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _final_objective_gates(final_records: list[dict], comparisons: dict) -> dict:
    by_method = {row["method"]: row for row in final_records}
    pretrained = by_method["pretrained"]
    expanded = by_method["expanded"]
    kazuki = by_method["kazuki_locked"]
    liveness_contract = BASE._liveness_contract([pretrained], kazuki)
    expanded_trend = BASE._trend(expanded)
    paired_ci_clean = all(map(_ci_win, comparisons.values()))
    liveness_eligible = BASE._liveness_eligible(expanded, liveness_contract)
    gamma_trend_eligible = expanded_trend["mean_fraction"] >= .75
    return {
        "paired_ci_clean_four_metric_win": paired_ci_clean,
        "final_liveness_contract": liveness_contract,
        "final_liveness_eligible": liveness_eligible,
        "final_gamma_trend_eligible": gamma_trend_eligible,
        "objective_achieved": (
            paired_ci_clean and liveness_eligible and gamma_trend_eligible
        ),
    }


def _raw_rows(payload: dict) -> list[dict]:
    return payload["records"][0]["cell"]["rows"]


def _metric(rows: list[dict], name: str) -> float:
    point = _point(rows)
    return float(point[name])


def _paired_cluster_ci(expanded: list[dict], reference: list[dict], *, seed: int,
                       draws: int = 5_000) -> dict:
    exp_by_episode, ref_by_episode = {}, {}
    for row in expanded:
        exp_by_episode.setdefault(int(row["episode"]), []).append(row)
    for row in reference:
        ref_by_episode.setdefault(int(row["episode"]), []).append(row)
    episodes = sorted(set(exp_by_episode) & set(ref_by_episode))
    if len(episodes) != 50 or any(
        len(exp_by_episode[key]) != 7 or len(ref_by_episode[key]) != 7
        for key in episodes
    ):
        raise RuntimeError("paired CI requires 50 complete scenario clusters")
    generator = np.random.default_rng(int(seed))
    samples = {metric: [] for metric in BASE.METRICS}
    for _ in range(int(draws)):
        chosen = generator.choice(episodes, size=len(episodes), replace=True)
        exp_rows = [row for key in chosen for row in exp_by_episode[int(key)]]
        ref_rows = [row for key in chosen for row in ref_by_episode[int(key)]]
        for metric in BASE.METRICS:
            samples[metric].append(
                _metric(exp_rows, metric) - _metric(ref_rows, metric)
            )
    result = {}
    for metric, values in samples.items():
        values = np.asarray(values, float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            result[metric] = {"mean_delta": None, "paired_cluster_95": [None, None]}
        else:
            result[metric] = {
                "mean_delta": float(np.mean(finite)),
                "paired_cluster_95": list(map(
                    float, np.quantile(finite, [.025, .975])
                )),
                "desired_sign": "negative" if metric in BASE.LOWER_IS_BETTER else "positive",
            }
    return result


def _ci_win(value: dict) -> bool:
    for metric, row in value.items():
        low, high = row["paired_cluster_95"]
        if low is None:
            return False
        if metric in BASE.LOWER_IS_BETTER:
            if not high < 0.0:
                return False
        elif not low > 0.0:
            return False
    return True


def run(args) -> dict:
    started = datetime.now(timezone.utc)
    initial_path = Path(args.initial_delivery).resolve()
    initial = _wait(initial_path, args.poll_seconds)
    if initial.get("status") != BASE.STATUS:
        raise RuntimeError("invalid global-temperature delivery")
    calibration_bank = initial["banks"]["disjoint_confirmation"]
    if int(calibration_bank["M_per_gamma"]) != 50:
        raise RuntimeError("initial result is not M50")
    final_ids = set(range(int(args.final_ep0), int(args.final_ep0) + 50))
    calibration_ids = set(range(
        int(calibration_bank["ep0"]), int(calibration_bank["ep0"]) + 50
    ))
    if final_ids & calibration_ids:
        raise RuntimeError("fresh confirmation overlaps calibration M50")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    temperatures = list(map(float, args.temperatures.split(",")))
    if temperatures != [0.55, 0.7, 0.85, 1.0]:
        raise ValueError("calibration grid changed")
    selected = initial["selection"]
    methods = {
        "pretrained": selected["selected_pretrained"],
        "expanded": selected["selected_expanded"],
    }

    calibration_root = output / "calibration_m50"
    jobs, metadata = [], {}
    for method, record in methods.items():
        for temperature in temperatures:
            name = f"{method}_temp{temperature:g}".replace(".", "p")
            out = calibration_root / name
            jobs.append({
                "name": name,
                "command": BASE._raw_command(
                    record["checkpoint"], round_index=record["round"],
                    temperature=temperature,
                    ep0=int(calibration_bank["ep0"]),
                    noise_seed=int(calibration_bank["noise_seed"]), M=50,
                    workers=args.workers, output=out,
                    cache=calibration_root / "cache",
                ),
            })
            metadata[name] = (method, temperature, out)
    BASE._run_jobs(
        jobs, gpus=args.gpus, workers=args.workers, log_dir=output / "logs"
    )
    cells = {method: {} for method in methods}
    for method, temperature, out in metadata.values():
        cells[method][temperature] = BASE._read(
            out / "raw_m50_offline_metrics.json"
        )

    kazuki_path = initial_path.parent / "disjoint_m50" / "kazuki_locked.json"
    kazuki_payload = BASE._read(kazuki_path)
    kazuki = BASE._kazuki_record(kazuki_payload)
    kazuki_liveness = BASE._liveness_contract([], kazuki)
    pretrained = _pick(
        _enumerate(
            "pretrained", cells["pretrained"], temperatures,
            methods["pretrained"]["checkpoint"], methods["pretrained"]["round"],
        ),
        target=_single_reference_target(kazuki),
        liveness=kazuki_liveness,
    )
    pretrained["checkpoint_sha256"] = methods["pretrained"]["checkpoint_sha256"]
    target = BASE._envelope([pretrained], kazuki)
    liveness = BASE._liveness_contract([pretrained], kazuki)
    expanded = _pick(
        _enumerate(
            "expanded", cells["expanded"], temperatures,
            methods["expanded"]["checkpoint"], methods["expanded"]["round"],
        ),
        target=target,
        liveness=liveness,
    )
    expanded["checkpoint_sha256"] = methods["expanded"]["checkpoint_sha256"]
    lock = {
        "status": "GAMMA_TEMPERATURE_SCHEDULE_LOCKED",
        "calibration_bank_role": (
            "the former global-temperature M50 is intentionally reused and "
            "therefore reclassified as calibration, not confirmation"
        ),
        "temperature_grid": temperatures,
        "pretrained": pretrained,
        "expanded": expanded,
        "kazuki": kazuki,
        "target_envelope": target,
        "liveness_contract": liveness,
        "gamma_trend_gate": {
            "minimum_adjacent_pair_mean_fraction": .75,
            "pretrained": BASE._trend(pretrained),
            "expanded": BASE._trend(expanded),
        },
        "expanded_shortfalls": BASE._shortfalls(expanded, target),
        "schedule_sha256": {
            "pretrained": _schedule_sha(pretrained),
            "expanded": _schedule_sha(expanded),
        },
    }
    BASE._write(output / "SCHEDULE_LOCKED.json", lock)

    final_root = output / "fresh_disjoint_m50"
    final_jobs, final_meta = [], {}
    for method, record in (("pretrained", pretrained), ("expanded", expanded)):
        out = final_root / method
        final_jobs.append({
            "name": f"fresh_{method}",
            "command": BASE.FUNNEL.evaluator_command(
                [record["checkpoint"]], [f"r{record['round']}"],
                scene_profile="double_density_velocity_ood",
                ep0=args.final_ep0, noise_seed=args.final_noise_seed,
                m_per_gamma=50, workers=args.workers,
                cache_dir=final_root / "cache", output_dir=out,
                temperature=1.0,
                temperature_by_gamma=record["temperature_by_gamma"],
            ),
        })
        final_meta[method] = out
    final_kazuki = final_root / "kazuki_locked.json"
    final_jobs.append({
        "name": "fresh_kazuki",
        "command": [
            sys.executable, str(Path(BASE.__file__).resolve()), "kazuki",
            "--checkpoint", methods["pretrained"]["checkpoint"],
            "--ep0", str(args.final_ep0), "--M", "50",
            "--workers", str(args.workers), "--device", "cuda:0",
            "--output", str(final_kazuki),
        ],
    })
    BASE._run_jobs(
        final_jobs, gpus=args.gpus, workers=args.workers,
        log_dir=output / "logs",
    )
    raw_payloads = {
        method: BASE._read(path / "raw_m50_offline_metrics.json")
        for method, path in final_meta.items()
    }
    final_kazuki_payload = BASE._read(final_kazuki)
    final_records = [
        BASE._record_from_cell(
            raw_payloads[method], method=method, temperature=1.0
        )
        for method in ("pretrained", "expanded")
    ]
    for row, locked in zip(final_records, (pretrained, expanded)):
        row["temperature"] = None
        row["temperature_by_gamma"] = locked["temperature_by_gamma"]
    final_records.append(BASE._kazuki_record(final_kazuki_payload))
    BASE._render(final_records, final_root / "four_metric_per_gamma.png")
    exp_rows = _raw_rows(raw_payloads["expanded"])
    comparisons = {
        "expanded_minus_pretrained": _paired_cluster_ci(
            exp_rows, _raw_rows(raw_payloads["pretrained"]),
            seed=args.final_noise_seed + 1,
        ),
        "expanded_minus_kazuki": _paired_cluster_ci(
            exp_rows, final_kazuki_payload["rows"],
            seed=args.final_noise_seed + 2,
        ),
    }
    objective_gates = _final_objective_gates(final_records, comparisons)
    result = {
        "status": STATUS,
        "initial_delivery": str(initial_path),
        "calibration_bank": calibration_bank,
        "fresh_confirmation_bank": {
            "M_per_gamma": 50,
            "ep0": args.final_ep0,
            "noise_seed": args.final_noise_seed,
        },
        "lock": lock,
        "final_records": final_records,
        "paired_cluster_differences": comparisons,
        "ci_clean_four_metric_win": objective_gates[
            "paired_ci_clean_four_metric_win"
        ],
        **objective_gates,
        "gamma_trends": {
            row["method"]: BASE._trend(row) for row in final_records
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    BASE._write(output / "DELIVERY_COMPLETE.json", result)
    print(json.dumps({
        "status": STATUS,
        "ci_clean_four_metric_win": result["ci_clean_four_metric_win"],
        "objective_achieved": result["objective_achieved"],
        "delivery": str(output / "DELIVERY_COMPLETE.json"),
    }, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-delivery", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperatures", default="0.55,0.7,0.85,1.0")
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--final-ep0", type=int, default=480_000)
    parser.add_argument("--final-noise-seed", type=int, default=2_026_073_6)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
