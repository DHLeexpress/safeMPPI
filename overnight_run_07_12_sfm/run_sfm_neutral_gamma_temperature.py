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
import subprocess
import sys
import time

import numpy as np

import run_sfm_neutral_temperature_m50 as BASE
import sfm_protocol as SP


STATUS = "SFM_NEUTRAL_GAMMA_TEMPERATURE_M50_COMPLETE"


def _gpu_inventory(indices: list[int]) -> dict:
    output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError("unexpected nvidia-smi inventory row")
        rows.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "driver_version": fields[3], "memory_total_MiB": int(fields[4]),
        })
    by_index = {row["index"]: row for row in rows}
    if any(int(index) not in by_index for index in indices):
        raise RuntimeError("requested physical GPU is absent")
    return {"requested_indices": list(map(int, indices)), "devices": rows}


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
            0 if BASE._trend_eligible(row) else 1,
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
    gamma_trend_eligible = BASE._trend_eligible(expanded)
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


def _rows_sha256(payload: dict) -> str:
    encoded = json.dumps(
        _raw_rows(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_value_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reuse_global_temperature_cells(
    initial_path: Path,
    initial: dict,
    methods: dict[str, dict],
) -> tuple[dict[str, dict[float, dict]], dict]:
    bank = initial["banks"]["disjoint_confirmation"]
    cells = {method: {} for method in methods}
    records = {}
    frozen_records = {
        row["method"]: row for row in initial.get("final_records", ())
    }
    for method in ("pretrained", "expanded"):
        selected = methods[method]
        temperature = float(selected["temperature"])
        reference_path = (
            initial_path.parent / "disjoint_m50" / method
            / "raw_m50_offline_metrics.json"
        )
        payload = BASE._read(reference_path)
        record = payload["records"][0]
        rows = _raw_rows(payload)
        expected_ids = list(range(
            int(bank["ep0"]), int(bank["ep0"]) + int(bank["M_per_gamma"])
        ))
        noise = payload.get("noise_bank", {})
        per_gamma_ids = {
            float(gamma): sorted(
                int(row["episode"]) for row in rows
                if float(row["gamma"]) == float(gamma)
            )
            for gamma in SP.GAMMAS
        }
        observed_schedule = (
            payload.get("temperature_by_gamma")
            or noise.get("temperature_by_gamma")
            or [float(payload["temperature"])] * len(SP.GAMMAS)
        )
        if (
            payload.get("scene_profile") != "double_density_velocity_ood"
            or record["cell"].get("scene_profile")
            != "double_density_velocity_ood"
            or int(payload["bank"]["ep0"]) != int(bank["ep0"])
            or int(payload["bank"]["M_per_gamma"]) != int(bank["M_per_gamma"])
            or payload["bank"].get("scenario_ids") != expected_ids
            or int(payload["noise_bank"]["seed"]) != int(bank["noise_seed"])
            or noise.get("gammas") != list(map(float, SP.GAMMAS))
            or int(noise.get("NFE", -1)) != 8
            or noise.get("dtype") != "float32"
            or noise.get("shape") != [
                len(SP.GAMMAS), int(bank["M_per_gamma"]), 180, 20
            ]
            or not isinstance(noise.get("sha256"), str)
            or len(noise["sha256"]) != 64
            or float(payload["temperature"]) != temperature
            or observed_schedule != [temperature] * len(SP.GAMMAS)
            or int(record["round"]) != int(selected["round"])
            or record["cell"]["checkpoint_sha256"]
            != selected["checkpoint_sha256"]
            or len(rows) != int(bank["M_per_gamma"]) * len(SP.GAMMAS)
            or any(ids != expected_ids for ids in per_gamma_ids.values())
        ):
            raise RuntimeError(
                f"global-temperature reference contract failed for {method}"
            )
        observed = BASE._record_from_cell(
            payload, method=method, temperature=temperature,
        )
        frozen = frozen_records.get(method)
        if frozen is None or any(
            observed[key] != frozen[key]
            for key in (
                "round", "checkpoint", "checkpoint_sha256", "temperature",
                "pooled", "per_gamma",
            )
        ):
            raise RuntimeError(
                f"global-temperature sidecar changed after delivery for {method}"
            )
        cells[method][temperature] = payload
        records[method] = {
            "temperature": temperature,
            "reference": str(reference_path),
            "reference_file_sha256": BASE.FUNNEL.sha256_file(reference_path),
            "reference_rows_sha256": _rows_sha256(payload),
            "noise_bank_sha256": noise["sha256"],
            "cell_key": record["cell"]["cell_key"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "rerun": False,
        }
    if (
        records["pretrained"]["noise_bank_sha256"]
        != records["expanded"]["noise_bank_sha256"]
    ):
        raise RuntimeError(
            "reused pretrained and expanded cells do not share CRN bytes"
        )
    reuse = {
        "status": "GLOBAL_TEMPERATURE_CALIBRATION_CELLS_REUSED",
        "reason": (
            "the locked global-temperature M50 cells are already the exact "
            "calibration-bank observations; re-running a GPU rollout can "
            "perturb borderline trajectories and is neither required nor "
            "scientifically preferable"
        ),
        "records": records,
    }
    return cells, reuse


def _validate_prior_calibration_cell(
    path: Path, payload: dict, *, method: str, temperature: float,
    selected: dict, bank: dict,
) -> dict:
    record = payload["records"][0]
    rows = _raw_rows(payload)
    expected_ids = list(range(
        int(bank["ep0"]), int(bank["ep0"]) + int(bank["M_per_gamma"])
    ))
    noise = payload.get("noise_bank", {})
    per_gamma_ids = {
        float(gamma): sorted(
            int(row["episode"]) for row in rows
            if float(row["gamma"]) == float(gamma)
        )
        for gamma in SP.GAMMAS
    }
    if (
        payload.get("scene_profile") != "double_density_velocity_ood"
        or record["cell"].get("scene_profile")
        != "double_density_velocity_ood"
        or int(payload["bank"]["ep0"]) != int(bank["ep0"])
        or int(payload["bank"]["M_per_gamma"]) != int(bank["M_per_gamma"])
        or payload["bank"].get("scenario_ids") != expected_ids
        or int(noise.get("seed", -1)) != int(bank["noise_seed"])
        or noise.get("gammas") != list(map(float, SP.GAMMAS))
        or int(noise.get("NFE", -1)) != 8
        or noise.get("dtype") != "float32"
        or noise.get("shape") != [
            len(SP.GAMMAS), int(bank["M_per_gamma"]), 180, 20
        ]
        or not isinstance(noise.get("sha256"), str)
        or len(noise["sha256"]) != 64
        or float(payload["temperature"]) != float(temperature)
        or payload.get("temperature_by_gamma")
        != [float(temperature)] * len(SP.GAMMAS)
        or int(record["round"]) != int(selected["round"])
        or record["cell"]["checkpoint_sha256"]
        != selected["checkpoint_sha256"]
        or len(rows) != int(bank["M_per_gamma"]) * len(SP.GAMMAS)
        or any(ids != expected_ids for ids in per_gamma_ids.values())
    ):
        raise RuntimeError(
            f"prior calibration contract failed for {method} temp={temperature:g}"
        )
    return {
        "method": method,
        "temperature": float(temperature),
        "reference": str(path),
        "reference_file_sha256": BASE.FUNNEL.sha256_file(path),
        "reference_rows_sha256": _rows_sha256(payload),
        "noise_bank_sha256": noise["sha256"],
        "cell_key": record["cell"]["cell_key"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
    }


def _reuse_prior_calibration_cells(
    root: Path, methods: dict[str, dict], bank: dict,
    temperatures: list[float], cells: dict[str, dict[float, dict]],
) -> dict:
    records = []
    for method, selected in methods.items():
        for temperature in temperatures:
            if float(temperature) in cells[method]:
                continue
            name = f"{method}_temp{temperature:g}".replace(".", "p")
            path = root / "calibration_m50" / name / "raw_m50_offline_metrics.json"
            if not path.is_file():
                continue
            payload, file_sha = BASE._read_hashed(path)
            provenance = _validate_prior_calibration_cell(
                path, payload, method=method, temperature=temperature,
                selected=selected, bank=bank,
            )
            if provenance["reference_file_sha256"] != file_sha:
                raise RuntimeError("calibration bytes changed while being read")
            cells[method][float(temperature)] = payload
            records.append(provenance)

    for temperature in temperatures:
        if all(float(temperature) in cells[method] for method in methods):
            hashes = {
                cells[method][float(temperature)]["noise_bank"]["sha256"]
                for method in methods
            }
            if len(hashes) != 1:
                raise RuntimeError(
                    f"reused temp={temperature:g} cells do not share CRN bytes"
                )
    return {
        "status": "PRIOR_CALIBRATION_CELLS_CONTENT_AUTHENTICATED",
        "root": str(root),
        "records": records,
        "missing_cells_will_be_run": [
            {"method": method, "temperature": float(temperature)}
            for method in methods for temperature in temperatures
            if float(temperature) not in cells[method]
        ],
    }


def _calibration_crn_sha(
    cells: dict[str, dict[float, dict]], temperatures: list[float],
    methods: dict[str, dict],
) -> str:
    missing = [
        (method, float(temperature))
        for method in ("pretrained", "expanded")
        for temperature in temperatures
        if float(temperature) not in cells[method]
    ]
    if missing:
        raise RuntimeError(f"calibration grid is incomplete: {missing}")
    hashes = {
        cells[method][float(temperature)]["noise_bank"]["sha256"]
        for method in ("pretrained", "expanded")
        for temperature in temperatures
    }
    if len(hashes) != 1:
        raise RuntimeError("calibration temperatures do not share one CRN bank")
    for method, selected in methods.items():
        for temperature in temperatures:
            observed = cells[method][float(temperature)]["records"][0][
                "cell"
            ]["checkpoint_sha256"]
            if observed != selected["checkpoint_sha256"]:
                raise RuntimeError("calibration cell checkpoint digest changed")
    return next(iter(hashes))


def _authenticate_method_checkpoints(methods: dict[str, dict]) -> None:
    for method, selected in methods.items():
        path = Path(selected["checkpoint"]).resolve()
        if BASE.FUNNEL.sha256_file(path) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"locked {method} checkpoint digest mismatch")


def _validate_fresh_raw_cell(
    payload: dict, *, locked: dict, ep0: int, noise_seed: int,
) -> str:
    record = payload["records"][0]
    rows = _raw_rows(payload)
    expected_ids = list(range(int(ep0), int(ep0) + 50))
    noise = payload.get("noise_bank", {})
    per_gamma_ids = {
        float(gamma): sorted(
            int(row["episode"]) for row in rows
            if float(row["gamma"]) == float(gamma)
        )
        for gamma in SP.GAMMAS
    }
    if (
        payload.get("scene_profile") != "double_density_velocity_ood"
        or int(payload["bank"]["ep0"]) != int(ep0)
        or int(payload["bank"]["M_per_gamma"]) != 50
        or payload["bank"].get("scenario_ids") != expected_ids
        or int(noise.get("seed", -1)) != int(noise_seed)
        or noise.get("gammas") != list(map(float, SP.GAMMAS))
        or int(noise.get("NFE", -1)) != 8
        or noise.get("dtype") != "float32"
        or noise.get("shape") != [len(SP.GAMMAS), 50, 180, 20]
        or not isinstance(noise.get("sha256"), str)
        or len(noise["sha256"]) != 64
        or payload.get("temperature_by_gamma")
        != list(map(float, locked["temperature_by_gamma"]))
        or int(record["round"]) != int(locked["round"])
        or record["cell"]["checkpoint_sha256"]
        != locked["checkpoint_sha256"]
        or len(rows) != 50 * len(SP.GAMMAS)
        or any(ids != expected_ids for ids in per_gamma_ids.values())
    ):
        raise RuntimeError("fresh raw confirmation contract changed")
    return noise["sha256"]


def _validate_fresh_kazuki(
    payload: dict, *, pretrained: dict, ep0: int,
) -> None:
    expected_ids = list(range(int(ep0), int(ep0) + 50))
    rows = payload.get("rows", ())
    per_gamma_ids = {
        float(gamma): sorted(
            int(row["episode"]) for row in rows
            if float(row["gamma"]) == float(gamma)
        )
        for gamma in SP.GAMMAS
    }
    if (
        payload.get("scene_profile") != "double_density_velocity_ood"
        or int(payload.get("ep0", -1)) != int(ep0)
        or int(payload.get("M_per_gamma", -1)) != 50
        or payload.get("checkpoint_sha256")
        != pretrained["checkpoint_sha256"]
        or len(rows) != 50 * len(SP.GAMMAS)
        or any(ids != expected_ids for ids in per_gamma_ids.values())
    ):
        raise RuntimeError("fresh Kazuki confirmation contract changed")


def _validate_calibration_kazuki(
    payload: dict, *, initial: dict, bank: dict, pretrained: dict,
) -> dict:
    frozen = next(
        (row for row in initial.get("final_records", ())
         if row.get("method") == "kazuki_locked"),
        None,
    )
    observed = BASE._kazuki_record(payload)
    if frozen is None or any(
        observed[key] != frozen[key]
        for key in (
            "round", "checkpoint", "checkpoint_sha256", "temperature",
            "pooled", "per_gamma",
        )
    ):
        raise RuntimeError("calibration Kazuki sidecar changed after delivery")
    _validate_fresh_kazuki(
        payload, pretrained=pretrained, ep0=int(bank["ep0"]),
    )
    if int(bank["M_per_gamma"]) != 50:
        raise RuntimeError("Kazuki calibration bank is not M50")
    return observed


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
    source_commit = BASE._source_gate(args.expected_source_commit)
    gpu_provenance = _gpu_inventory(args.gpus)
    initial_path = Path(args.initial_delivery).resolve()
    _wait(initial_path, args.poll_seconds)
    initial, initial_sha256 = BASE._read_hashed(initial_path)
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
    _authenticate_method_checkpoints(methods)
    cells, reuse = _reuse_global_temperature_cells(
        initial_path, initial, methods
    )
    BASE._write(output / "GLOBAL_TEMPERATURE_REUSE.json", reuse)
    prior_reuse = {
        "status": "PRIOR_CALIBRATION_REUSE_NOT_REQUESTED", "records": [],
    }
    if args.reuse_calibration_root:
        prior_reuse = _reuse_prior_calibration_cells(
            Path(args.reuse_calibration_root).resolve(), methods,
            calibration_bank, temperatures, cells,
        )
    BASE._write(output / "PRIOR_CALIBRATION_REUSE.json", prior_reuse)

    calibration_root = output / "calibration_m50"
    jobs, metadata = [], {}
    for method, record in methods.items():
        for temperature in temperatures:
            if temperature in cells[method]:
                continue
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
    for method, temperature, out in metadata.values():
        cells[method][temperature] = BASE._read(
            out / "raw_m50_offline_metrics.json"
        )
    calibration_noise_sha256 = _calibration_crn_sha(
        cells, temperatures, methods,
    )
    _authenticate_method_checkpoints(methods)

    kazuki_path = initial_path.parent / "disjoint_m50" / "kazuki_locked.json"
    kazuki_payload = BASE._read(kazuki_path)
    kazuki = _validate_calibration_kazuki(
        kazuki_payload, initial=initial, bank=calibration_bank,
        pretrained=methods["pretrained"],
    )
    kazuki_calibration_reference = {
        "path": str(kazuki_path),
        "file_sha256": BASE.FUNNEL.sha256_file(kazuki_path),
        "rows_sha256": _json_value_sha256(kazuki_payload["rows"]),
    }
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
        "analysis_source_commit": source_commit,
        "initial_delivery": str(initial_path),
        "initial_delivery_sha256": initial_sha256,
        "temperature_grid": temperatures,
        "calibration_noise_bank_sha256": calibration_noise_sha256,
        "pretrained": pretrained,
        "expanded": expanded,
        "kazuki": kazuki,
        "kazuki_calibration_reference": kazuki_calibration_reference,
        "target_envelope": target,
        "liveness_contract": liveness,
        "gamma_trend_gate": {
            "minimum_each_adjacent_pair_family_fraction": (
                BASE.MIN_TREND_FAMILY_FRACTION
            ),
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
    _authenticate_method_checkpoints(methods)
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
            "--expected-source-commit", source_commit,
        ],
    })
    confirmation_gpu = int(args.gpus[0])
    BASE._run_jobs(
        final_jobs, gpus=[confirmation_gpu], workers=args.workers,
        log_dir=output / "logs",
    )
    raw_payloads = {
        method: BASE._read(path / "raw_m50_offline_metrics.json")
        for method, path in final_meta.items()
    }
    final_kazuki_payload = BASE._read(final_kazuki)
    _authenticate_method_checkpoints(methods)
    fresh_noise_hashes = {
        _validate_fresh_raw_cell(
            raw_payloads[method], locked=locked,
            ep0=args.final_ep0, noise_seed=args.final_noise_seed,
        )
        for method, locked in (("pretrained", pretrained), ("expanded", expanded))
    }
    if len(fresh_noise_hashes) != 1:
        raise RuntimeError("fresh raw methods do not share one CRN bank")
    _validate_fresh_kazuki(
        final_kazuki_payload, pretrained=methods["pretrained"],
        ep0=args.final_ep0,
    )
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
    plot_path = final_root / "four_metric_per_gamma.png"
    BASE._render(final_records, plot_path)
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
    if BASE.FUNNEL.sha256_file(initial_path) != initial_sha256:
        raise RuntimeError("initial delivery changed during gamma calibration")
    BASE._source_gate(source_commit)
    fresh_artifacts = {}
    for method, path in final_meta.items():
        metrics_path = path / "raw_m50_offline_metrics.json"
        payload = raw_payloads[method]
        fresh_artifacts[method] = {
            "path": str(metrics_path),
            "file_sha256": BASE.FUNNEL.sha256_file(metrics_path),
            "rows_sha256": _rows_sha256(payload),
            "noise_bank_sha256": payload["noise_bank"]["sha256"],
            "checkpoint_sha256": payload["records"][0]["cell"][
                "checkpoint_sha256"
            ],
        }
    fresh_artifacts["kazuki_locked"] = {
        "path": str(final_kazuki),
        "file_sha256": BASE.FUNNEL.sha256_file(final_kazuki),
        "rows_sha256": _json_value_sha256(final_kazuki_payload["rows"]),
        "checkpoint_sha256": final_kazuki_payload["checkpoint_sha256"],
    }
    fresh_artifacts["plots"] = {
        str(path): BASE.FUNNEL.sha256_file(path)
        for path in (plot_path, plot_path.with_suffix(".pdf"))
    }
    result = {
        "status": STATUS,
        "source_commit": source_commit,
        "initial_delivery": str(initial_path),
        "initial_delivery_sha256": initial_sha256,
        "global_temperature_calibration_reuse": reuse,
        "prior_calibration_reuse": prior_reuse,
        "gpu_provenance": gpu_provenance,
        "calibration_bank": calibration_bank,
        "fresh_confirmation_bank": {
            "M_per_gamma": 50,
            "ep0": args.final_ep0,
            "noise_seed": args.final_noise_seed,
            "single_physical_gpu_index": confirmation_gpu,
            "execution": "sequential to avoid cross-GPU comparison noise",
        },
        "fresh_confirmation_artifacts": fresh_artifacts,
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
    parser.add_argument("--reuse-calibration-root")
    parser.add_argument("--expected-source-commit")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
