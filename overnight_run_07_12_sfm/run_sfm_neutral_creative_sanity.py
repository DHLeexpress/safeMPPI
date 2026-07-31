#!/usr/bin/env python3
"""Fail-closed causal sanity stages after the neutral r100 follow-up fails.

The stages are intentionally sequential and never combined:

* B: no-training, disjoint-M10 post-D+ versus post-D0 audit;
* A: five fresh rounds with only the nontrap/progress-gated margin selector;
* C: three, then at most five, fresh rounds with only E_g unfrozen at 0.1x.

Stage B can stop the coordinator before A when it clearly identifies D0
overwrite.  Stage C is reached only after A fails its declared gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

import run_sfm_neutral_autonomous_followup as AUTO
import run_sfm_neutral_gamma_temperature as GAMMA
import run_sfm_neutral_temperature_m50 as GLOBAL
import sfm_b1_full_episode_audit as FA
import sfm_b1_neutral_multiround as TRAIN


HERE = Path(__file__).resolve().parent
STATUS = "SFM_NEUTRAL_CREATIVE_SANITY_COMPLETE"
STAGE_B_STATUS = "SFM_NEUTRAL_STAGE_B_POSTPHASE_AUDIT_COMPLETE"
STAGE_A_STATUS = "SFM_NEUTRAL_STAGE_A_SELECTOR_SANITY_COMPLETE"
STAGE_C_STATUS = "SFM_NEUTRAL_STAGE_C_ENCODER_SANITY_COMPLETE"
DEFAULT_AUDIT_ROUNDS = (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _read(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _write(path: Path, payload: dict) -> None:
    GLOBAL._write(path, payload)


def _write_final(output: Path, result: dict, source: dict) -> None:
    finished = FA._source()
    if (
        not finished["tracked_worktree_clean"]
        or finished["commit"] != source["commit"]
    ):
        raise RuntimeError("creative source worktree changed during execution")
    result.setdefault("source", source)
    _write(output / "DELIVERY_COMPLETE.json", result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}


def _wait_trigger(path: Path, poll_seconds: int) -> dict:
    while not path.is_file():
        print(f"WAITING {path}", flush=True)
        time.sleep(int(poll_seconds))
    return _read(path)


def _validate_trigger(
    path: Path, trigger: dict, *, expected_source: str,
) -> dict:
    if (
        trigger.get("status") != AUTO.CREATIVE_TRIGGER_STATUS
        or trigger.get("action") != "CREATIVE_SANITY_REQUIRED"
    ):
        raise RuntimeError("creative coordinator received a non-trigger marker")
    if trigger.get("source_commit") != expected_source:
        raise RuntimeError("creative trigger was produced by an unexpected source")
    delivery_path = Path(trigger["autonomous_delivery"]).resolve()
    if _sha256(delivery_path) != trigger.get("autonomous_delivery_sha256"):
        raise RuntimeError("autonomous delivery digest mismatch")
    delivery = _read(delivery_path)
    if (
        delivery.get("status") != AUTO.STATUS
        or delivery.get("action") != "CREATIVE_SANITY_REQUIRED"
        or delivery.get("objective_achieved") is not False
    ):
        raise RuntimeError("autonomous delivery does not authorize creative sanity")
    if delivery.get("source_commit") != trigger.get("source_commit"):
        raise RuntimeError("trigger/autonomous source commit mismatch")
    for key in (
        "selected_arm",
        "r100_training_delivery", "r100_training_delivery_sha256",
        "r100_global_delivery", "r100_global_delivery_sha256",
        "r100_gamma_delivery", "r100_gamma_delivery_sha256",
    ):
        if trigger.get(key) != delivery.get(key):
            raise RuntimeError(f"trigger/autonomous mismatch: {key}")
    for key in (
        "r100_training_delivery", "r100_global_delivery",
        "r100_gamma_delivery",
    ):
        referenced = Path(trigger[key]).resolve()
        if _sha256(referenced) != trigger[f"{key}_sha256"]:
            raise RuntimeError(f"referenced delivery digest mismatch: {key}")
    gamma = _read(Path(trigger["r100_gamma_delivery"]).resolve())
    if (
        gamma.get("status") != GAMMA.STATUS
        or gamma.get("objective_achieved") is not False
    ):
        raise RuntimeError("r100 gamma result does not require creative sanity")
    return {"delivery": delivery, "gamma": gamma, "path": delivery_path}


def _delivery_chain(final_path: Path) -> list[tuple[Path, dict]]:
    chain = []
    path = final_path.resolve()
    seen = set()
    while True:
        if path in seen:
            raise RuntimeError("resume delivery chain contains a cycle")
        seen.add(path)
        payload = _read(path)
        if payload.get("status") != TRAIN.STATUS:
            raise RuntimeError(f"invalid neutral training delivery: {path}")
        chain.append((path, payload))
        resume = payload.get("resume")
        if not resume:
            break
        path = Path(resume["delivery"]).resolve()
        if _sha256(path) != resume.get("delivery_sha256"):
            raise RuntimeError("resume delivery digest mismatch")
    chain.reverse()
    checkpoint_hashes = {payload["checkpoint_sha256"] for _, payload in chain}
    if len(checkpoint_hashes) != 1:
        raise RuntimeError("training chain changed pretrained checkpoint")
    return chain


def _round_catalog(chain: list[tuple[Path, dict]]) -> dict[int, dict]:
    frozen_refs = {}
    for _, delivery in chain:
        candidates = list(delivery.get("round_record_refs", ()))
        candidates.extend(delivery.get("resume", {}).get("round_record_refs", ()))
        for ref in candidates:
            key = str(Path(ref["path"]).resolve())
            if key in frozen_refs and frozen_refs[key] != ref:
                raise RuntimeError("conflicting frozen round reference")
            frozen_refs[key] = ref
    catalog = {}
    for _, delivery in chain:
        for marker_path in delivery.get("round_records", ()):
            marker_path = Path(marker_path).resolve()
            ref = frozen_refs.get(str(marker_path))
            if ref is None or _sha256(marker_path) != ref.get("sha256"):
                raise RuntimeError("round marker lacks an authenticated snapshot")
            marker = _read(marker_path)
            round_i = int(marker.get("round", -1))
            if marker.get("status") != TRAIN.ROUND_STATUS or round_i in catalog:
                raise RuntimeError("invalid or duplicate round marker")
            checkpoint = Path(marker["checkpoint"]).resolve()
            if (
                str(checkpoint) != str(Path(ref["post_D0"]).resolve())
                or _sha256(checkpoint) != ref.get("post_D0_sha256")
                or ref.get("post_D0_sha256") != marker.get("checkpoint_sha256")
            ):
                raise RuntimeError("post-D0 checkpoint digest mismatch")
            post_positive = (
                checkpoint.parent / f"round_{round_i:02d}_post_positive.pt"
            )
            payload = torch.load(
                post_positive, map_location="cpu", weights_only=False,
            )
            if (
                str(post_positive) != str(Path(ref["post_Dplus"]).resolve())
                or _sha256(post_positive) != ref.get("post_Dplus_sha256")
                or int(payload.get("round", -1)) != round_i
                or payload.get("phase") != "post_Dplus"
            ):
                raise RuntimeError("post-D+ checkpoint metadata mismatch")
            catalog[round_i] = {
                "round": round_i,
                "marker": str(marker_path),
                "marker_sha256": _sha256(marker_path),
                "post_positive": str(post_positive),
                "post_positive_sha256": _sha256(post_positive),
                "post_D0": str(checkpoint),
                "post_D0_sha256": marker["checkpoint_sha256"],
                "scenarios": list(map(int, marker["scenarios"])),
                "record": marker,
            }
    rounds = sorted(catalog)
    if not rounds or rounds != list(range(1, rounds[-1] + 1)):
        raise RuntimeError("round lineage is not contiguous from one")
    return catalog


def _bank_from(ep0: int, M: int, role: str) -> dict:
    return {
        "role": str(role),
        "ep0": int(ep0),
        "M_per_gamma": int(M),
        "scenario_ids": list(range(int(ep0), int(ep0) + int(M))),
    }


def _known_banks(chain, gamma_payload: dict) -> list[dict]:
    banks = []
    for _, delivery in chain:
        source = delivery.get("disjoint_raw_evaluation", {})
        if "ep0" in source and "M_per_gamma" in source:
            banks.append(_bank_from(
                source["ep0"], source["M_per_gamma"], "prior_training_screen",
            ))

    def visit(value, path="gamma"):
        if isinstance(value, dict):
            if "ep0" in value and "M_per_gamma" in value:
                banks.append(_bank_from(
                    value["ep0"], value["M_per_gamma"], path,
                ))
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(gamma_payload)
    initial_path = gamma_payload.get("initial_delivery")
    if initial_path:
        visit(_read(Path(initial_path).resolve()), "gamma.initial_delivery")
    unique = {}
    for bank in banks:
        key = (bank["ep0"], bank["M_per_gamma"])
        unique.setdefault(key, bank)
    return list(unique.values())


def _validate_new_banks(new_banks, known_banks, training_scenarios) -> None:
    for index, bank in enumerate(new_banks):
        values = set(bank["scenario_ids"])
        if values & set(training_scenarios):
            raise RuntimeError(f"{bank['role']} overlaps training scenarios")
        for other in [*known_banks, *new_banks[index + 1:]]:
            if values & set(other["scenario_ids"]):
                raise RuntimeError(
                    f"{bank['role']} overlaps {other['role']}"
                )


def _pooled(payload: dict) -> dict:
    return GLOBAL._pooled(payload["records"][0]["cell"]["summary"]["pooled"])


def _per_gamma(payload: dict) -> dict:
    return {
        gamma: GLOBAL._pooled(cell)
        for gamma, cell in payload["records"][0]["cell"][
            "summary"
        ]["per_gamma"].items()
    }


def _evaluate_specs(specs, *, root, bank, noise_seed, gpus, workers):
    cache = root / "cache"
    jobs, lookup = [], {}
    for spec in specs:
        name = str(spec["name"])
        out = root / name
        command = GLOBAL._raw_command(
            spec["checkpoint"], round_index=int(spec["round"]),
            temperature=1.0, ep0=bank["ep0"],
            noise_seed=int(noise_seed), M=bank["M_per_gamma"],
            workers=int(workers), output=out, cache=cache,
        )
        jobs.append({"name": name, "command": command})
        lookup[name] = (spec, out)
    GLOBAL._run_jobs(
        jobs, gpus=list(map(int, gpus)), workers=int(workers),
        log_dir=root / "logs",
    )
    rows = []
    for name, (spec, out) in lookup.items():
        metrics = out / f"raw_m{bank['M_per_gamma']}_offline_metrics.json"
        payload = _read(metrics)
        rows.append({
            **spec,
            "checkpoint_sha256": _sha256(Path(spec["checkpoint"])),
            "metrics_json": str(metrics),
            "metrics_sha256": _sha256(metrics),
            "pooled": _pooled(payload),
            "per_gamma": _per_gamma(payload),
        })
    return rows


def _d0_gate(rows: list[dict]) -> dict:
    by_round = {}
    for row in rows:
        by_round.setdefault(int(row["round"]), {})[row["phase"]] = row
    comparisons = []
    for round_i in sorted(by_round):
        phases = by_round[round_i]
        if set(phases) != {"post_Dplus", "post_D0"}:
            raise RuntimeError("stage-B phase pair is incomplete")
        positive, neutral = phases["post_Dplus"], phases["post_D0"]
        delta = {
            key: neutral["pooled"][key] - positive["pooled"][key]
            for key in ("SR", "CR", "timeout", "Validity")
        }
        liveness_harm = delta["SR"] <= -0.05 or delta["timeout"] >= 0.05
        no_material_safety_compensation = (
            delta["CR"] >= -0.03 and delta["Validity"] <= 0.03
        )
        comparisons.append({
            "round": round_i,
            "post_D0_minus_post_Dplus": delta,
            "clear_overwrite_cell": bool(
                liveness_harm and no_material_safety_compensation
            ),
        })
    tail = comparisons[-min(3, len(comparisons)):]
    required = max(1, math.ceil(2 * len(tail) / 3))
    clear = sum(row["clear_overwrite_cell"] for row in tail) >= required
    return {
        "rule": (
            "D0 clearly overwrites D+ only when >=2/3 of the final three "
            "audited rounds lose >=5pp SR or gain >=5pp timeout, without "
            ">3pp CR or Validity compensation"
        ),
        "comparisons": comparisons,
        "tail_rounds": [row["round"] for row in tail],
        "required_tail_cells": required,
        "clear_D0_overwrite": bool(clear),
        "retain_D0_unless_clear": True,
        "stage_A_indicated": bool(not clear),
    }


def _training_command(
    *, checkpoint, output, name, rounds, scenario_ep0, eval_ep0,
    eval_rounds, lr, inner_steps, ell, gp_cap, selector,
    encoder_lr_ratio, workers, noise_seed, sample_seed, audit_seed,
    train_seed, probe_seed, neutral_replay=True, resume=None,
    locked_eval=None, eval_M=10,
):
    command = [
        sys.executable, str(HERE / "sfm_b1_neutral_multiround.py"),
        "--checkpoint", str(checkpoint),
        "--output-root", str(output),
        "--name", str(name),
        "--rounds", str(int(rounds)),
        "--scenario-ep0", str(int(scenario_ep0)),
        "--eval-ep0", str(int(eval_ep0)),
        "--eval-M", str(int(eval_M)),
        "--eval-rounds", ",".join(map(str, eval_rounds)),
        "--lr", str(float(lr)),
        "--inner-steps", str(int(inner_steps)),
        "--ell", str(float(ell)),
        "--gp-cap", str(int(gp_cap)),
        "--selector", str(selector),
        "--encoder-lr-ratio", str(float(encoder_lr_ratio)),
        "--noise-seed", str(int(noise_seed)),
        "--sample-seed", str(int(sample_seed)),
        "--audit-seed", str(int(audit_seed)),
        "--train-seed", str(int(train_seed)),
        "--probe-seed", str(int(probe_seed)),
        "--device", "cuda:0",
        "--workers", str(int(workers)),
    ]
    if resume is not None:
        command.extend(["--resume-run-root", str(resume)])
    if locked_eval is not None:
        command.extend([
            "--locked-eval-checkpoint", str(locked_eval["checkpoint"]),
            "--locked-eval-round", str(int(locked_eval["round"])),
            "--locked-eval-sha256", str(locked_eval["checkpoint_sha256"]),
        ])
    if not neutral_replay:
        command.append("--no-neutral-replay")
    return command


def _run(
    command, *, log: Path, gpu: int | None, cpu_range: str | None,
) -> None:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONPATH"] = str(HERE)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        launched = list(command)
        if cpu_range is not None:
            launched = ["taskset", "-c", cpu_range, *launched]
        completed = subprocess.run(
            launched,
            cwd=HERE, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"job failed ({completed.returncode}): {log}")


def _confirm_candidate(
    *, training_run: Path, arm_name: str, output: Path, ep0: int,
    noise_seed: int, gpus, workers: int, selected: dict,
) -> dict:
    """Calibrate temperature, then apply the existing fresh-M50 objective."""
    training_delivery = _read(training_run / "DELIVERY_COMPLETE.json")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
    ).strip()
    training_screen = training_delivery["disjoint_raw_evaluation"]
    input_root = output / "training_input"
    prepared_arm = input_root / arm_name
    prepared_arm.mkdir(parents=True)
    selected_round = int(selected["round"])
    selected_sha = str(selected["checkpoint_sha256"])
    filtered_records = [
        record for record in training_screen["records"]
        if int(record["round"]) in (0, selected_round)
    ]
    if len(filtered_records) != 2:
        raise RuntimeError("confirmation input must contain r0 and one winner")
    winner = next(
        record for record in filtered_records
        if int(record["round"]) == selected_round
    )
    if winner["checkpoint_sha256"] != selected_sha:
        raise RuntimeError("qualification winner checkpoint digest changed")
    prepared_delivery = dict(training_delivery)
    prepared_delivery["disjoint_raw_evaluation"] = {
        **training_screen,
        "records": filtered_records,
    }
    prepared_delivery["confirmation_filter"] = {
        "source_delivery": str(
            (training_run / "DELIVERY_COMPLETE.json").resolve()
        ),
        "source_delivery_sha256": _sha256(
            training_run / "DELIVERY_COMPLETE.json"
        ),
        "selected_round": selected_round,
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected_sha,
    }
    prepared_delivery_path = prepared_arm / "DELIVERY_COMPLETE.json"
    _write(prepared_delivery_path, prepared_delivery)
    global_root = output / "global_temperature"
    global_command = [
        sys.executable, str(HERE / "run_sfm_neutral_temperature_m50.py"),
        "run",
        "--training-root", str(input_root),
        "--output-dir", str(global_root),
        "--arm-names", str(arm_name),
        "--temperatures", "0.55,0.7,0.85,1.0",
        "--gpus", *map(str, gpus),
        "--workers", str(int(workers)),
        "--screen-ep0", str(int(training_screen["ep0"])),
        "--screen-M", str(int(training_screen["M_per_gamma"])),
        "--validation-ep0", str(int(ep0)),
        "--validation-M", "10",
        "--validation-noise-seed", str(int(noise_seed)),
        "--final-ep0", str(int(ep0) + 1_000),
        "--final-noise-seed", str(int(noise_seed) + 1),
        "--expected-final-round", str(int(training_delivery["rounds"])),
        "--expected-source-commit", source_commit,
    ]
    _run(
        global_command, log=output / "logs" / "global_temperature.log",
        gpu=None, cpu_range=None,
    )
    global_delivery = _read(global_root / "DELIVERY_COMPLETE.json")
    locked = global_delivery["selection"]["selected_expanded"]
    if (
        int(locked["round"]) != selected_round
        or locked["checkpoint_sha256"] != selected_sha
    ):
        raise RuntimeError("global confirmation changed the qualified checkpoint")
    gamma_root = output / "gamma_temperature"
    gamma_command = [
        sys.executable, str(HERE / "run_sfm_neutral_gamma_temperature.py"),
        "--initial-delivery", str(global_root / "DELIVERY_COMPLETE.json"),
        "--output-dir", str(gamma_root),
        "--temperatures", "0.55,0.7,0.85,1.0",
        "--gpus", *map(str, gpus),
        "--workers", str(int(workers)),
        "--final-ep0", str(int(ep0) + 2_000),
        "--final-noise-seed", str(int(noise_seed) + 2),
        "--expected-source-commit", source_commit,
    ]
    _run(
        gamma_command, log=output / "logs" / "gamma_temperature.log",
        gpu=None, cpu_range=None,
    )
    global_path = global_root / "DELIVERY_COMPLETE.json"
    gamma_path = gamma_root / "DELIVERY_COMPLETE.json"
    gamma = _read(gamma_path)
    if gamma.get("status") != GAMMA.STATUS:
        raise RuntimeError("creative candidate confirmation is incomplete")
    return {
        "prepared_training_delivery": str(prepared_delivery_path),
        "prepared_training_delivery_sha256": _sha256(prepared_delivery_path),
        "global_delivery": str(global_path),
        "global_delivery_sha256": _sha256(global_path),
        "gamma_delivery": str(gamma_path),
        "gamma_delivery_sha256": _sha256(gamma_path),
        "ci_clean_four_metric_win": bool(
            gamma.get("ci_clean_four_metric_win")
        ),
        "objective_achieved": bool(gamma.get("objective_achieved")),
        "final_liveness_eligible": bool(
            gamma.get("final_liveness_eligible")
        ),
        "final_gamma_trend_eligible": bool(
            gamma.get("final_gamma_trend_eligible")
        ),
    }


def _expanded_lock(
    gamma_delivery: str, *, name: str, training_run: Path | None = None,
) -> dict:
    path = Path(gamma_delivery).resolve()
    payload = _read(path)
    if payload.get("status") != GAMMA.STATUS:
        raise RuntimeError("invalid gamma-temperature candidate delivery")
    record = next(
        row for row in payload["final_records"]
        if row["method"] == "expanded"
    )
    return {
        "name": str(name),
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "round": int(record["round"]),
        "temperature_by_gamma": record["temperature_by_gamma"],
        "gamma_delivery": str(path),
        "gamma_delivery_sha256": _sha256(path),
        "training_run": (
            None if training_run is None else str(training_run.resolve())
        ),
    }


def _common_best_available(
    locks, *, output: Path, ep0: int, noise_seed: int, gpus, workers: int,
) -> dict:
    """Rank failed-objective candidates on one common M50 bank."""
    cache = output / "cache"
    jobs, destinations = [], {}
    for lock in locks:
        name = lock["name"]
        destination = output / name
        jobs.append({
            "name": f"common_{name}",
            "command": GLOBAL.FUNNEL.evaluator_command(
                [lock["checkpoint"]], [f"r{lock['round']}"],
                scene_profile="double_density_velocity_ood",
                ep0=int(ep0), noise_seed=int(noise_seed),
                m_per_gamma=50, workers=int(workers),
                cache_dir=cache, output_dir=destination,
                temperature=1.0,
                temperature_by_gamma=lock["temperature_by_gamma"],
            ),
        })
        destinations[name] = (lock, destination)
    GLOBAL._run_jobs(
        jobs, gpus=[int(gpus[0])], workers=int(workers),
        log_dir=output / "logs",
    )
    records = []
    for name, (lock, destination) in destinations.items():
        metrics = destination / "raw_m50_offline_metrics.json"
        row = GLOBAL._record_from_cell(
            _read(metrics), method=name, temperature=1.0,
        )
        row["temperature"] = None
        row["temperature_by_gamma"] = lock["temperature_by_gamma"]
        row["metrics_json"] = str(metrics)
        row["metrics_sha256"] = _sha256(metrics)
        records.append(row)

    baseline = next(
        row for row in records if row["method"] == "r100_baseline"
    )
    liveness = GLOBAL._liveness_contract([baseline], baseline)
    for row in records:
        row["liveness_eligible"] = GLOBAL._liveness_eligible(row, liveness)
        row["gamma_trend_eligible"] = GLOBAL._trend_eligible(row)

    def rank(row):
        value = row["pooled"]
        return (
            value["CR"], -value["Validity"], -value["clearance"],
            value["time_to_goal"], -value["SR"], row["method"],
        )

    eligible = [
        row for row in records
        if row["liveness_eligible"] and row["gamma_trend_eligible"]
    ]
    selected = None if not eligible else min(eligible, key=rank)
    result = {
        "status": (
            "SFM_NEUTRAL_CREATIVE_COMMON_M50_COMPLETE"
            if selected is not None
            else "SFM_NEUTRAL_CREATIVE_COMMON_M50_NO_ELIGIBLE_POLICY"
        ),
        "selection_scope": (
            "best available only among policies within 5pp SR/timeout of the "
            "r100 baseline and passing every gamma-trend family; then "
            "safety-first lexicographic ranking, not a four-metric win claim"
        ),
        "single_physical_gpu_index": int(gpus[0]),
        "liveness_contract": liveness,
        "bank": _bank_from(ep0, 50, "creative_common_best_M50"),
        "noise_seed": int(noise_seed),
        "locks": locks,
        "records": records,
        "selected": selected,
        "selected_lock": (
            None if selected is None else next(
                lock for lock in locks if lock["name"] == selected["method"]
            )
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(output / "COMMON_BEST_COMPLETE.json", result)
    return result


def _confirm_full_run(
    *, training_run: Path, output: Path, ep0: int, noise_seed: int,
    gpus, workers: int,
) -> dict:
    delivery = _read(training_run / "DELIVERY_COMPLETE.json")
    arm_name = str(delivery["config"]["name"])
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
    ).strip()
    global_root = output / "global_temperature"
    global_command = [
        sys.executable, str(HERE / "run_sfm_neutral_temperature_m50.py"),
        "run", "--training-root", str(training_run.parent),
        "--output-dir", str(global_root), "--arm-names", arm_name,
        "--temperatures", "0.55,0.7,0.85,1.0",
        "--gpus", *map(str, gpus), "--workers", str(int(workers)),
        "--screen-ep0", str(delivery["disjoint_raw_evaluation"]["ep0"]),
        "--screen-M", str(delivery["disjoint_raw_evaluation"]["M_per_gamma"]),
        "--validation-ep0", str(int(ep0)), "--validation-M", "10",
        "--validation-noise-seed", str(int(noise_seed)),
        "--final-ep0", str(int(ep0) + 1_000),
        "--final-noise-seed", str(int(noise_seed) + 1),
        "--expected-final-round", str(int(delivery["rounds"])),
        "--expected-source-commit", source_commit,
    ]
    _run(
        global_command, log=output / "logs" / "global_temperature.log",
        gpu=None, cpu_range=None,
    )
    gamma_root = output / "gamma_temperature"
    gamma_command = [
        sys.executable, str(HERE / "run_sfm_neutral_gamma_temperature.py"),
        "--initial-delivery", str(global_root / "DELIVERY_COMPLETE.json"),
        "--output-dir", str(gamma_root),
        "--temperatures", "0.55,0.7,0.85,1.0",
        "--gpus", *map(str, gpus), "--workers", str(int(workers)),
        "--final-ep0", str(int(ep0) + 2_000),
        "--final-noise-seed", str(int(noise_seed) + 2),
        "--expected-source-commit", source_commit,
    ]
    _run(
        gamma_command, log=output / "logs" / "gamma_temperature.log",
        gpu=None, cpu_range=None,
    )
    global_path = global_root / "DELIVERY_COMPLETE.json"
    gamma_path = gamma_root / "DELIVERY_COMPLETE.json"
    gamma = _read(gamma_path)
    return {
        "global_delivery": str(global_path),
        "global_delivery_sha256": _sha256(global_path),
        "gamma_delivery": str(gamma_path),
        "gamma_delivery_sha256": _sha256(gamma_path),
        "objective_achieved": bool(gamma.get("objective_achieved")),
        "final_liveness_eligible": bool(gamma.get("final_liveness_eligible")),
        "final_gamma_trend_eligible": bool(
            gamma.get("final_gamma_trend_eligible")
        ),
    }


def _extend_common_winner_to_r100(
    common: dict, *, output: Path, eval_ep0: int, confirm_ep0: int,
    noise_seed: int, gpus, workers: int,
) -> dict | None:
    lock = common.get("selected_lock")
    if lock is None or lock["name"] == "r100_baseline":
        return None
    source_run = Path(lock["training_run"]).resolve()
    source_delivery = _read(source_run / "DELIVERY_COMPLETE.json")
    cfg = source_delivery["config"]
    marker_paths = [
        ref["path"]
        for ref in source_delivery.get("resume", {}).get(
            "round_record_refs", ()
        )
    ]
    marker_paths.extend(source_delivery["round_records"])
    scenario_ids = [
        int(scenario)
        for marker_path in marker_paths
        for scenario in _read(Path(marker_path))["scenarios"]
    ]
    full_parent = output / "full_r100_training"
    arm_name = f"{cfg['name']}_full_r100"
    full_run = full_parent / arm_name
    resume_round = int(source_delivery["rounds"])
    eval_rounds = sorted({
        0, resume_round, int(lock["round"]), *range(10, 101, 10),
    })
    command = _training_command(
        checkpoint=source_delivery["checkpoint"], output=full_run,
        name=arm_name, rounds=100, scenario_ep0=min(scenario_ids),
        eval_ep0=int(eval_ep0), eval_rounds=eval_rounds,
        lr=cfg["lr"], inner_steps=cfg["inner_steps"], ell=cfg["ell"],
        gp_cap=cfg["gp_cap"], selector=cfg["selector"],
        encoder_lr_ratio=cfg["encoder_lr_ratio"], workers=int(workers),
        noise_seed=int(noise_seed), sample_seed=cfg["sample_seed"],
        audit_seed=cfg["audit_seed"], train_seed=cfg["train_seed"],
        probe_seed=cfg["probe_seed"],
        neutral_replay=cfg.get("neutral_replay", True),
        resume=source_run,
        locked_eval=(None if int(lock["round"]) == resume_round else lock),
        eval_M=20,
    )
    training_gpu = (
        int(gpus[-1]) if float(cfg["encoder_lr_ratio"]) > 0 else int(gpus[0])
    )
    _run(
        command, log=output / "logs" / "full_r100_training.log",
        gpu=training_gpu,
        cpu_range="96-159" if training_gpu == int(gpus[-1]) else "16-79",
    )
    full_delivery = full_run / "DELIVERY_COMPLETE.json"
    confirmation = _confirm_full_run(
        training_run=full_run, output=output / "full_r100_confirmation",
        ep0=int(confirm_ep0), noise_seed=int(noise_seed) + 100,
        gpus=gpus, workers=int(workers),
    )
    return {
        "source_lock": lock,
        "training_gpu": training_gpu,
        "training_delivery": str(full_delivery),
        "training_delivery_sha256": _sha256(full_delivery),
        "confirmation": confirmation,
        "objective_achieved": confirmation["objective_achieved"],
    }


def _delivery_eval_rows(delivery: dict) -> list[dict]:
    source = _read(Path(delivery["disjoint_raw_evaluation"]["file"]))
    rows = []
    for row in source["records"]:
        rows.append({
            "name": row["name"],
            "round": int(row["round"]),
            "phase": row["phase"],
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "pooled": GLOBAL._pooled(row["cell"]["summary"]["pooled"]),
            "per_gamma": {
                gamma: GLOBAL._pooled(cell)
                for gamma, cell in row["cell"]["summary"][
                    "per_gamma"
                ].items()
            },
        })
    return rows


def _trend_ok(row: dict) -> bool:
    return GLOBAL._trend_eligible(row)


def _selector_gate(
    candidate_rows, controls, r100_reference=None, *, diagnostic_rounds=None,
) -> dict:
    control_by_round = {int(row["round"]): row for row in controls}
    candidate_by_round = {
        int(row["round"]): row for row in candidate_rows
    }
    decisions = []
    for row in candidate_rows:
        if int(row["round"]) == 0:
            continue
        control = control_by_round[int(row["round"])]
        value, base = row["pooled"], control["pooled"]
        safe = (
            value["CR"] <= base["CR"] + .03
            and value["Validity"] >= base["Validity"] - .03
        )
        live = (
            value["SR"] >= base["SR"] + .05
            or value["timeout"] <= base["timeout"] - .05
        )
        reference = None if r100_reference is None else r100_reference["pooled"]
        noninferior_to_r100 = None if reference is None else (
            value["SR"] >= reference["SR"] - .03
            and value["timeout"] <= reference["timeout"] + .03
            and value["CR"] <= reference["CR"] + .03
            and value["Validity"] >= reference["Validity"] - .03
        )
        diagnostic_eligible = (
            diagnostic_rounds is None
            or int(row["round"]) in set(map(int, diagnostic_rounds))
        )
        decisions.append({
            "round": int(row["round"]),
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "pooled": value,
            "safety_noninferior_to_matched_margin": bool(safe),
            "liveness_improved_over_matched_margin": bool(live),
            "noninferior_to_r100_reference": bool(noninferior_to_r100),
            "diagnostic_round_eligible": bool(diagnostic_eligible),
            "gamma_trend_pass": _trend_ok(row),
            "eligible": bool(
                safe and live and diagnostic_eligible and _trend_ok(row)
            ),
        })
    eligible = [row for row in decisions if row["eligible"]]
    selected = None if not eligible else min(
        eligible,
        key=lambda item: (
            -candidate_by_round[item["round"]]["pooled"]["SR"],
            candidate_by_round[item["round"]]["pooled"]["CR"],
            item["round"],
        ),
    )
    return {
        "rule": (
            "promote only if SR improves >=5pp or timeout falls >=5pp vs "
            "the matched equal-dose margin round, while CR/Validity are within "
            "3pp and every gamma-trend family is >=0.75; r100 is diagnostic "
            "only until the mechanism reaches an equal dose"
        ),
        "r100_reference": r100_reference,
        "decisions": decisions,
        "passed": selected is not None,
        "selected": selected,
    }


def _control_gp_fields(marker: dict) -> dict:
    gather = marker["gather"]
    if "acquisition" in gather:
        return {
            "uplift": float(gather["acquisition"]["uplift"]),
            "effective_rank": (
                None if "gp_diagnostics" not in gather else float(
                    gather["gp_diagnostics"]["kernel_effective_rank"]
                )
            ),
            "source": "round_marker",
        }
    trace_path = Path(gather["trace_path"]).resolve()
    if _sha256(trace_path) != gather.get("trace_sha256"):
        raise RuntimeError("legacy control gather trace digest mismatch")
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    if trace.get("status") != TRAIN.RA.STATUS:
        raise RuntimeError("invalid legacy control gather trace")
    acquisition = trace.get("protocol", {}).get("acquisition")
    if acquisition is None or "uplift" not in acquisition:
        raise RuntimeError("legacy control trace lacks acquisition diagnostics")
    return {
        "uplift": float(acquisition["uplift"]),
        "effective_rank": None,
        "source": "authenticated_legacy_trace",
    }


def _encoder_stage_gate(
    delivery, candidate_rows, control_rows, control_markers,
) -> dict:
    controls = {int(row["round"]): row for row in control_rows}
    markers = [_read(Path(path)) for path in delivery["round_records"]]
    diagnostics = []
    for marker in markers:
        round_i = int(marker["round"])
        candidate = next(
            row for row in candidate_rows if int(row["round"]) == round_i
        )
        control = controls[round_i]
        encoder = marker["encoder_diagnostics"]
        cumulative = encoder["cumulative_from_reference"]
        gather = marker["gather"]
        control_marker = control_markers[round_i]
        control_gp = _control_gp_fields(control_marker)
        candidate_rank = float(
            gather["gp_diagnostics"]["kernel_effective_rank"]
        )
        control_rank = control_gp["effective_rank"]
        unsafe = (
            cumulative["token_cosine"] < .98
            or marker["paired_trigger_probe"]["Dplus_increment"][
                "Dplus_regressed"
            ] > 0
            or candidate["pooled"]["CR"] - control["pooled"]["CR"] > .03
        )
        signal = (
            candidate["pooled"]["Validity"]
            - control["pooled"]["Validity"] >= .01
            or gather["acquisition"]["uplift"] - control_gp["uplift"] >= .002
            or (
                control_rank is not None
                and candidate_rank - control_rank >= 1.0
            )
        )
        diagnostics.append({
            "round": round_i,
            "token_cosine": encoder["token_cosine"],
            "encoder_relative_drift": encoder["relative_parameter_drift"],
            "cumulative_token_cosine": cumulative["token_cosine"],
            "cumulative_token_rms_change": cumulative["token_rms_change"],
            "cumulative_encoder_relative_drift": cumulative[
                "relative_parameter_drift"
            ],
            "encoder_gradient_norm_Dplus": marker["updates"]["Dplus"][
                "encoder_gradient_norms"
            ],
            "encoder_gradient_norm_D0": marker["updates"]["D0"][
                "encoder_gradient_norms"
            ],
            "Dplus_regressed": marker["paired_trigger_probe"][
                "Dplus_increment"
            ]["Dplus_regressed"],
            "gp_effective_rank": gather["gp_diagnostics"][
                "kernel_effective_rank"
            ],
            "uncertainty_uplift": gather["acquisition"]["uplift"],
            "frozen_gp_effective_rank": control_rank,
            "frozen_uncertainty_uplift": control_gp["uplift"],
            "frozen_gp_diagnostic_source": control_gp["source"],
            "delta_gp_effective_rank_vs_frozen": (
                None if control_rank is None else candidate_rank - control_rank
            ),
            "delta_uncertainty_uplift_vs_frozen": (
                gather["acquisition"]["uplift"]
                - control_gp["uplift"]
            ),
            "delta_CR_vs_frozen": (
                candidate["pooled"]["CR"] - control["pooled"]["CR"]
            ),
            "delta_Validity_vs_frozen": (
                candidate["pooled"]["Validity"]
                - control["pooled"]["Validity"]
            ),
            "unsafe": bool(unsafe),
            "signal": bool(signal),
            "eligible": bool(not unsafe and signal),
        })
    eligible_rounds = [row["round"] for row in diagnostics if row["eligible"]]
    return {
        "rule": (
            "stop on cumulative E_g token cosine <0.98, any D+ regression, or CR >3pp "
            "above matched frozen control; continue only with >=1pp Validity "
            "or matched GP uplift/rank improvement"
        ),
        "diagnostics": diagnostics,
        "unsafe": bool(not eligible_rounds),
        "signal": bool(any(row["signal"] for row in diagnostics)),
        "eligible_rounds": eligible_rounds,
        "continue_to_round5": bool(
            diagnostics and diagnostics[-1]["eligible"]
        ),
    }


def _base_specs(catalog, rounds, prefix):
    return [{
        "name": f"{prefix}_r{round_i}",
        "phase": "post_D0",
        "round": int(round_i),
        "checkpoint": catalog[int(round_i)]["post_D0"],
    } for round_i in rounds]


def run(args) -> dict:
    started = datetime.now(timezone.utc)
    trigger_path = Path(args.trigger).resolve()
    trigger = _wait_trigger(trigger_path, args.poll_seconds)
    authenticated = _validate_trigger(
        trigger_path, trigger,
        expected_source=args.expected_trigger_source,
    )
    source = FA._source()
    if not source["tracked_worktree_clean"]:
        raise RuntimeError("creative sanity requires a clean frozen worktree")
    if source["commit"] != args.expected_trigger_source:
        raise RuntimeError("creative sanity source does not match its trigger")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)

    chain = _delivery_chain(Path(trigger["r100_training_delivery"]))
    catalog = _round_catalog(chain)
    selected_arm = str(trigger["selected_arm"])
    final_delivery = chain[-1][1]
    if chain[0][1]["config"]["name"] != selected_arm:
        raise RuntimeError("selected arm does not match r100 lineage")
    audit_rounds = tuple(sorted({
        int(value) for value in str(args.audit_rounds).split(",") if value
    }))
    if not audit_rounds or audit_rounds[-1] != max(catalog):
        raise ValueError("audit rounds must include the final lineage round")
    if any(value not in catalog for value in audit_rounds):
        raise ValueError("audit round is absent from the lineage")

    training_scenarios = {
        scenario
        for record in catalog.values() for scenario in record["scenarios"]
    }
    banks = [
        _bank_from(args.stage_b_ep0, 10, "stage_B_disjoint_M10"),
        _bank_from(args.stage_a_ep0, 10, "stage_A_disjoint_M10"),
        _bank_from(args.stage_c3_ep0, 10, "stage_C3_disjoint_M10"),
        _bank_from(args.stage_c5_ep0, 10, "stage_C5_disjoint_M10"),
        _bank_from(
            args.stage_b_causal_ep0, 10,
            "stage_B_Dplus_only_disjoint_M10",
        ),
    ]
    confirmation_banks = []
    for label, ep0 in (
        ("B_Dplus_only", args.stage_b_confirm_ep0),
        ("A_selector", args.stage_a_confirm_ep0),
        ("C_encoder", args.stage_c_confirm_ep0),
    ):
        confirmation_banks.extend([
            _bank_from(ep0, 10, f"{label}_temperature_validation_M10"),
            _bank_from(ep0 + 1_000, 50, f"{label}_global_M50"),
            _bank_from(ep0 + 2_000, 50, f"{label}_gamma_fresh_M50"),
        ])
    confirmation_banks.append(_bank_from(
        args.common_best_ep0, 50, "creative_common_best_M50",
    ))
    confirmation_banks.extend([
        _bank_from(args.full_eval_ep0, 20, "creative_full_r100_screen_M20"),
        _bank_from(args.full_confirm_ep0, 10, "creative_full_temperature_M10"),
        _bank_from(
            args.full_confirm_ep0 + 1_000, 50,
            "creative_full_global_M50",
        ),
        _bank_from(
            args.full_confirm_ep0 + 2_000, 50,
            "creative_full_gamma_fresh_M50",
        ),
    ])
    known_banks = _known_banks(chain, authenticated["gamma"])
    _validate_new_banks(
        [*banks, *confirmation_banks], known_banks, training_scenarios,
    )
    output.mkdir(parents=True)
    baseline_lock = _expanded_lock(
        trigger["r100_gamma_delivery"], name="r100_baseline",
        training_run=Path(trigger["r100_training_delivery"]).resolve().parent,
    )
    creative_locks = []
    provenance = {
        "status": "SFM_NEUTRAL_CREATIVE_SANITY_PREREGISTERED",
        "source": source,
        "trigger": str(trigger_path),
        "trigger_sha256": _sha256(trigger_path),
        "autonomous_delivery": str(authenticated["path"]),
        "autonomous_delivery_sha256": _sha256(authenticated["path"]),
        "selected_arm": selected_arm,
        "lineage_catalog": {
            str(round_i): {
                key: value for key, value in item.items() if key != "record"
            }
            for round_i, item in catalog.items()
        },
        "pretrained_checkpoint": final_delivery["checkpoint"],
        "pretrained_checkpoint_sha256": final_delivery["checkpoint_sha256"],
        "r100_training_delivery": trigger["r100_training_delivery"],
        "r100_training_delivery_sha256": _sha256(
            Path(trigger["r100_training_delivery"])
        ),
        "audit_rounds": list(audit_rounds),
        "known_banks": known_banks,
        "new_banks": [*banks, *confirmation_banks],
        "no_combined_arm": True,
        "stage_order": [
            "B_postphase_audit",
            "B_Dplus_only_if_indicated",
            "A_selector",
            "C_encoder",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(output / "PREREGISTRATION.json", provenance)

    stage_b_root = output / "stage_B_postphase_audit"
    stage_b_specs = []
    for round_i in audit_rounds:
        item = catalog[round_i]
        stage_b_specs.extend([
            {
                "name": f"r{round_i}_post_Dplus", "round": round_i,
                "phase": "post_Dplus", "checkpoint": item["post_positive"],
            },
            {
                "name": f"r{round_i}_post_D0", "round": round_i,
                "phase": "post_D0", "checkpoint": item["post_D0"],
            },
        ])
    stage_b_rows = _evaluate_specs(
        stage_b_specs, root=stage_b_root, bank=banks[0],
        noise_seed=args.stage_b_noise_seed, gpus=[args.gpus[0]],
        workers=args.workers,
    )
    stage_b_gate = _d0_gate(stage_b_rows)
    stage_b = {
        "status": STAGE_B_STATUS,
        "zero_new_training": True,
        "bank": banks[0],
        "noise_seed": int(args.stage_b_noise_seed),
        "rows": stage_b_rows,
        "gate": stage_b_gate,
    }
    _write(stage_b_root / "STAGE_COMPLETE.json", stage_b)
    cfg = final_delivery["config"]
    scenario_ep0 = min(training_scenarios)
    stage_b_causal = None
    if stage_b_gate["clear_D0_overwrite"]:
        dplus_root = output / "stage_B_Dplus_only"
        command = _training_command(
            checkpoint=final_delivery["checkpoint"], output=dplus_root,
            name=f"{selected_arm}_Dplus_only", rounds=5,
            scenario_ep0=scenario_ep0, eval_ep0=banks[4]["ep0"],
            eval_rounds=(0, 1, 2, 3, 4, 5), lr=cfg["lr"],
            inner_steps=cfg["inner_steps"], ell=cfg["ell"],
            gp_cap=cfg["gp_cap"], selector="margin",
            encoder_lr_ratio=0.0, neutral_replay=False,
            workers=args.workers, noise_seed=args.stage_b_causal_noise_seed,
            sample_seed=cfg["sample_seed"], audit_seed=cfg["audit_seed"],
            train_seed=cfg["train_seed"], probe_seed=cfg["probe_seed"],
        )
        _run(
            command, log=output / "logs" / "stage_B_Dplus_only.log",
            gpu=args.gpus[0], cpu_range="16-79",
        )
        dplus_delivery = _read(dplus_root / "DELIVERY_COMPLETE.json")
        if dplus_delivery["config"].get("neutral_replay") is not False:
            raise RuntimeError("stage B causal arm replayed D0")
        dplus_rows = _delivery_eval_rows(dplus_delivery)
        dplus_controls = _evaluate_specs(
            _base_specs(catalog, range(1, 6), "margin_control"),
            root=dplus_root / "matched_margin_control", bank=banks[4],
            noise_seed=args.stage_b_causal_noise_seed,
            gpus=[args.gpus[0]], workers=args.workers,
        )
        dplus_r100 = _evaluate_specs([{
            "name": "r100_reference", "round": 100,
            "phase": "post_D0", "checkpoint": catalog[100]["post_D0"],
        }], root=dplus_root / "r100_reference", bank=banks[4],
            noise_seed=args.stage_b_causal_noise_seed,
            gpus=[args.gpus[0]], workers=args.workers)[0]
        dplus_gate = _selector_gate(
            dplus_rows, dplus_controls, dplus_r100,
        )
        stage_b_causal = {
            "status": "SFM_NEUTRAL_STAGE_B_DPLUS_ONLY_COMPLETE",
            "single_change": "collect/audit D0 but omit D0 replay",
            "training_delivery": _ref(
                dplus_root / "DELIVERY_COMPLETE.json"
            ),
            "bank": banks[4],
            "rows": dplus_rows,
            "matched_margin_controls": dplus_controls,
            "qualification_gate": dplus_gate,
            "confirmation": None,
        }
        if dplus_gate["passed"]:
            stage_b_causal["confirmation"] = _confirm_candidate(
                training_run=dplus_root,
                arm_name=dplus_delivery["config"]["name"],
                output=dplus_root / "confirmation",
                ep0=args.stage_b_confirm_ep0,
                noise_seed=args.stage_b_confirm_noise_seed,
                gpus=args.gpus, workers=args.workers,
                selected=dplus_gate["selected"],
            )
            creative_locks.append(_expanded_lock(
                stage_b_causal["confirmation"]["gamma_delivery"],
                name="Dplus_only",
                training_run=dplus_root,
            ))
        _write(dplus_root / "STAGE_COMPLETE.json", stage_b_causal)
        if (
            stage_b_causal["confirmation"] is not None
            and stage_b_causal["confirmation"]["objective_achieved"]
        ):
            result = {
                "status": STATUS,
                "action": "STOP_GOAL_ACHIEVED_BY_DPLUS_ONLY",
                "objective_achieved": True,
                "stages_completed": ["B", "B_Dplus_only"],
                "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
                "stage_B_Dplus_only": _ref(
                    dplus_root / "STAGE_COMPLETE.json"
                ),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_final(output, result, source)
            return result

    stage_a_root = output / "stage_A_progress_selector"
    command = _training_command(
        checkpoint=final_delivery["checkpoint"], output=stage_a_root,
        name=f"{selected_arm}_progress_gated", rounds=5,
        scenario_ep0=scenario_ep0, eval_ep0=banks[1]["ep0"],
        eval_rounds=(0, 1, 2, 3, 4, 5), lr=cfg["lr"],
        inner_steps=cfg["inner_steps"], ell=cfg["ell"],
        gp_cap=cfg["gp_cap"], selector="progress_gated_margin",
        encoder_lr_ratio=0.0, workers=args.workers,
        noise_seed=args.stage_a_noise_seed,
        sample_seed=cfg["sample_seed"], audit_seed=cfg["audit_seed"],
        train_seed=cfg["train_seed"], probe_seed=cfg["probe_seed"],
    )
    _run(
        command, log=output / "logs" / "stage_A.log",
        gpu=args.gpus[0], cpu_range="16-79",
    )
    stage_a_delivery = _read(stage_a_root / "DELIVERY_COMPLETE.json")
    if (
        stage_a_delivery["config"]["selector"] != "progress_gated_margin"
        or stage_a_delivery["config"]["encoder_lr_ratio"] != 0.0
    ):
        raise RuntimeError("stage A combined more than the selector change")
    stage_a_rows = _delivery_eval_rows(stage_a_delivery)
    controls_a = _evaluate_specs(
        _base_specs(catalog, range(1, 6), "margin_control"),
        root=stage_a_root / "matched_margin_control",
        bank=banks[1], noise_seed=args.stage_a_noise_seed,
        gpus=[args.gpus[0]], workers=args.workers,
    )
    r100_a = _evaluate_specs([
        {
            "name": "r100_reference", "round": 100,
            "phase": "post_D0", "checkpoint": catalog[100]["post_D0"],
        }
    ], root=stage_a_root / "r100_reference", bank=banks[1],
        noise_seed=args.stage_a_noise_seed, gpus=[args.gpus[0]],
        workers=args.workers)[0]
    stage_a_gate = _selector_gate(stage_a_rows, controls_a, r100_a)
    stage_a = {
        "status": STAGE_A_STATUS,
        "single_change": "progress_gated_margin selector",
        "training_delivery": _ref(stage_a_root / "DELIVERY_COMPLETE.json"),
        "bank": banks[1],
        "rows": stage_a_rows,
        "matched_margin_controls": controls_a,
        "qualification_gate": stage_a_gate,
        "confirmation": None,
    }
    if stage_a_gate["passed"]:
        stage_a["confirmation"] = _confirm_candidate(
            training_run=stage_a_root,
            arm_name=stage_a_delivery["config"]["name"],
            output=stage_a_root / "confirmation",
            ep0=args.stage_a_confirm_ep0,
            noise_seed=args.stage_a_confirm_noise_seed,
            gpus=args.gpus, workers=args.workers,
            selected=stage_a_gate["selected"],
        )
        creative_locks.append(_expanded_lock(
            stage_a["confirmation"]["gamma_delivery"],
            name="progress_gated_margin",
            training_run=stage_a_root,
        ))
    _write(stage_a_root / "STAGE_COMPLETE.json", stage_a)
    if (
        stage_a["confirmation"] is not None
        and stage_a["confirmation"]["objective_achieved"]
    ):
        result = {
            "status": STATUS,
            "action": "STOP_GOAL_ACHIEVED_BY_STAGE_A",
            "objective_achieved": True,
            "stages_completed": ["B", "A"],
            "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
            "stage_B_Dplus_only": (
                None if stage_b_causal is None else _ref(
                    output / "stage_B_Dplus_only" / "STAGE_COMPLETE.json"
                )
            ),
            "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_final(output, result, source)
        return result

    stage_c3_root = output / "stage_C_encoder_unfreeze_r3"
    command = _training_command(
        checkpoint=final_delivery["checkpoint"], output=stage_c3_root,
        name=f"{selected_arm}_encoder01x", rounds=3,
        scenario_ep0=scenario_ep0, eval_ep0=banks[2]["ep0"],
        eval_rounds=(0, 1, 2, 3), lr=cfg["lr"],
        inner_steps=cfg["inner_steps"], ell=cfg["ell"],
        gp_cap=cfg["gp_cap"], selector="margin",
        encoder_lr_ratio=0.1, workers=args.workers,
        noise_seed=args.stage_c3_noise_seed,
        sample_seed=cfg["sample_seed"], audit_seed=cfg["audit_seed"],
        train_seed=cfg["train_seed"], probe_seed=cfg["probe_seed"],
    )
    _run(
        command, log=output / "logs" / "stage_C3.log",
        gpu=args.gpus[-1], cpu_range="96-159",
    )
    c3_delivery = _read(stage_c3_root / "DELIVERY_COMPLETE.json")
    if (
        c3_delivery["config"]["selector"] != "margin"
        or c3_delivery["config"]["encoder_lr_ratio"] != 0.1
    ):
        raise RuntimeError("stage C combined more than encoder unfreezing")
    c3_rows = _delivery_eval_rows(c3_delivery)
    c3_controls = _evaluate_specs(
        _base_specs(catalog, range(1, 4), "frozen_control"),
        root=stage_c3_root / "matched_frozen_control", bank=banks[2],
        noise_seed=args.stage_c3_noise_seed, gpus=[args.gpus[-1]],
        workers=args.workers,
    )
    c3_r100 = _evaluate_specs([{
        "name": "r100_reference", "round": 100,
        "phase": "post_D0", "checkpoint": catalog[100]["post_D0"],
    }], root=stage_c3_root / "r100_reference", bank=banks[2],
        noise_seed=args.stage_c3_noise_seed, gpus=[args.gpus[-1]],
        workers=args.workers)[0]
    c3_gate = _encoder_stage_gate(
        c3_delivery, c3_rows, c3_controls,
        {round_i: catalog[round_i]["record"] for round_i in range(1, 4)},
    )
    selector_c3 = _selector_gate(
        c3_rows, c3_controls, c3_r100,
        diagnostic_rounds=c3_gate["eligible_rounds"],
    )
    c3 = {
        "status": STAGE_C_STATUS,
        "phase": "rounds_1_to_3",
        "single_change": "enc_grid lr = 0.1 * trunk lr",
        "training_delivery": _ref(stage_c3_root / "DELIVERY_COMPLETE.json"),
        "bank": banks[2],
        "rows": c3_rows,
        "matched_frozen_controls": c3_controls,
        "gate": c3_gate,
        "promotion_gate": selector_c3,
        "confirmation": None,
    }
    if selector_c3["passed"]:
        c3["confirmation"] = _confirm_candidate(
            training_run=stage_c3_root,
            arm_name=c3_delivery["config"]["name"],
            output=stage_c3_root / "confirmation",
            ep0=args.stage_c_confirm_ep0,
            noise_seed=args.stage_c_confirm_noise_seed,
            gpus=args.gpus, workers=args.workers,
            selected=selector_c3["selected"],
        )
        creative_locks.append(_expanded_lock(
            c3["confirmation"]["gamma_delivery"],
            name="encoder_unfreeze_01x_r3",
            training_run=stage_c3_root,
        ))
    _write(stage_c3_root / "STAGE_COMPLETE.json", c3)
    if (
        c3["confirmation"] is not None
        and c3["confirmation"]["objective_achieved"]
    ):
        result = {
            "status": STATUS,
            "action": "STOP_GOAL_ACHIEVED_BY_STAGE_C3",
            "objective_achieved": True,
            "stages_completed": ["B", "A", "C3"],
            "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
            "stage_B_Dplus_only": (
                None if stage_b_causal is None else _ref(
                    output / "stage_B_Dplus_only" / "STAGE_COMPLETE.json"
                )
            ),
            "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
            "stage_C3": _ref(stage_c3_root / "STAGE_COMPLETE.json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_final(output, result, source)
        return result
    if not c3_gate["continue_to_round5"]:
        common = _common_best_available(
            [baseline_lock, *creative_locks],
            output=output / "common_best_M50",
            ep0=args.common_best_ep0,
            noise_seed=args.common_best_noise_seed,
            gpus=args.gpus, workers=args.workers,
        )
        extension = _extend_common_winner_to_r100(
            common, output=output / "selected_creative_full",
            eval_ep0=args.full_eval_ep0,
            confirm_ep0=args.full_confirm_ep0,
            noise_seed=args.full_noise_seed,
            gpus=args.gpus, workers=args.workers,
        )
        achieved = bool(
            extension is not None and extension["objective_achieved"]
        )
        result = {
            "status": STATUS,
            "action": (
                "STOP_GOAL_ACHIEVED_BY_FULL_CREATIVE"
                if achieved else "STOP_NO_VALIDATED_CREATIVE_ARM"
            ),
            "objective_achieved": achieved,
            "stages_completed": ["B", "A", "C3"],
            "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
            "stage_B_Dplus_only": (
                None if stage_b_causal is None else _ref(
                    output / "stage_B_Dplus_only" / "STAGE_COMPLETE.json"
                )
            ),
            "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
            "stage_C3": _ref(stage_c3_root / "STAGE_COMPLETE.json"),
            "best_available_common_M50": _ref(
                output / "common_best_M50" / "COMMON_BEST_COMPLETE.json"
            ),
            "best_available": common["selected"],
            "full_creative_continuation": extension,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_final(output, result, source)
        return result

    stage_c5_root = output / "stage_C_encoder_unfreeze_r5"
    command = _training_command(
        checkpoint=final_delivery["checkpoint"], output=stage_c5_root,
        name=f"{selected_arm}_encoder01x", rounds=5,
        scenario_ep0=scenario_ep0, eval_ep0=banks[3]["ep0"],
        eval_rounds=(0, 3, 4, 5), lr=cfg["lr"],
        inner_steps=cfg["inner_steps"], ell=cfg["ell"],
        gp_cap=cfg["gp_cap"], selector="margin",
        encoder_lr_ratio=0.1, workers=args.workers,
        noise_seed=args.stage_c5_noise_seed, resume=stage_c3_root,
        sample_seed=cfg["sample_seed"], audit_seed=cfg["audit_seed"],
        train_seed=cfg["train_seed"], probe_seed=cfg["probe_seed"],
    )
    _run(
        command, log=output / "logs" / "stage_C5.log",
        gpu=args.gpus[-1], cpu_range="96-159",
    )
    c5_delivery = _read(stage_c5_root / "DELIVERY_COMPLETE.json")
    c5_rows = _delivery_eval_rows(c5_delivery)
    c5_controls = _evaluate_specs(
        _base_specs(catalog, (3, 4, 5), "frozen_control"),
        root=stage_c5_root / "matched_frozen_control", bank=banks[3],
        noise_seed=args.stage_c5_noise_seed, gpus=[args.gpus[-1]],
        workers=args.workers,
    )
    r100_c5 = _evaluate_specs([{
        "name": "r100_reference", "round": 100,
        "phase": "post_D0", "checkpoint": catalog[100]["post_D0"],
    }], root=stage_c5_root / "r100_reference", bank=banks[3],
        noise_seed=args.stage_c5_noise_seed, gpus=[args.gpus[-1]],
        workers=args.workers)[0]
    c5_increment_gate = _encoder_stage_gate(
        c5_delivery, c5_rows, c5_controls,
        {round_i: catalog[round_i]["record"] for round_i in (4, 5)},
    )
    combined_diagnostics = [
        *c3_gate["diagnostics"], *c5_increment_gate["diagnostics"],
    ]
    c5_gate = {
        "rule": c5_increment_gate["rule"],
        "diagnostics": combined_diagnostics,
        "unsafe": bool(not any(row["eligible"] for row in combined_diagnostics)),
        "signal": bool(any(row["signal"] for row in combined_diagnostics)),
        "eligible_rounds": [
            row["round"] for row in combined_diagnostics if row["eligible"]
        ],
    }
    c5_gate["continue_to_round5"] = bool(
        not c5_gate["unsafe"] and c5_gate["signal"]
    )
    selector_c5 = _selector_gate(
        c5_rows, c5_controls, r100_c5,
        diagnostic_rounds=c5_gate["eligible_rounds"],
    )
    passed = bool(selector_c5["passed"])
    c5 = {
        "status": STAGE_C_STATUS,
        "phase": "rounds_4_to_5_after_authenticated_r3_resume",
        "single_change": "enc_grid lr = 0.1 * trunk lr",
        "training_delivery": _ref(stage_c5_root / "DELIVERY_COMPLETE.json"),
        "bank": banks[3],
        "rows": c5_rows,
        "matched_frozen_controls": c5_controls,
        "diagnostic_gate": c5_gate,
        "promotion_gate": selector_c5,
        "passed": passed,
        "confirmation": None,
    }
    if passed:
        c5["confirmation"] = _confirm_candidate(
            training_run=stage_c5_root,
            arm_name=c5_delivery["config"]["name"],
            output=stage_c5_root / "confirmation",
            ep0=args.stage_c_confirm_ep0,
            noise_seed=args.stage_c_confirm_noise_seed,
            gpus=args.gpus, workers=args.workers,
            selected=selector_c5["selected"],
        )
        creative_locks.append(_expanded_lock(
            c5["confirmation"]["gamma_delivery"],
            name="encoder_unfreeze_01x",
            training_run=stage_c5_root,
        ))
    _write(stage_c5_root / "STAGE_COMPLETE.json", c5)
    objective_achieved = bool(
        c5["confirmation"] is not None
        and c5["confirmation"]["objective_achieved"]
    )
    common = None
    extension = None
    if not objective_achieved:
        common = _common_best_available(
            [baseline_lock, *creative_locks],
            output=output / "common_best_M50",
            ep0=args.common_best_ep0,
            noise_seed=args.common_best_noise_seed,
            gpus=args.gpus, workers=args.workers,
        )
        extension = _extend_common_winner_to_r100(
            common, output=output / "selected_creative_full",
            eval_ep0=args.full_eval_ep0,
            confirm_ep0=args.full_confirm_ep0,
            noise_seed=args.full_noise_seed,
            gpus=args.gpus, workers=args.workers,
        )
        objective_achieved = bool(
            extension is not None and extension["objective_achieved"]
        )
    result = {
        "status": STATUS,
        "action": (
            "STOP_GOAL_ACHIEVED_BY_STAGE_C"
            if objective_achieved else "STOP_NO_VALIDATED_CREATIVE_ARM"
        ),
        "stages_completed": ["B", "A", "C3", "C5"],
        "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
        "stage_B_Dplus_only": (
            None if stage_b_causal is None else _ref(
                output / "stage_B_Dplus_only" / "STAGE_COMPLETE.json"
            )
        ),
        "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
        "stage_C3": _ref(stage_c3_root / "STAGE_COMPLETE.json"),
        "stage_C5": _ref(stage_c5_root / "STAGE_COMPLETE.json"),
        "M10_qualified": bool(passed),
        "objective_achieved": objective_achieved,
        "full_creative_continuation": extension,
        "best_available_common_M50": (
            None if common is None else _ref(
                output / "common_best_M50" / "COMMON_BEST_COMPLETE.json"
            )
        ),
        "best_available": None if common is None else common["selected"],
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_final(output, result, source)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--expected-trigger-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpus", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--audit-rounds",
        default=",".join(map(str, DEFAULT_AUDIT_ROUNDS)),
    )
    parser.add_argument("--stage-b-ep0", type=int, default=530_000)
    parser.add_argument("--stage-a-ep0", type=int, default=540_000)
    parser.add_argument("--stage-c3-ep0", type=int, default=550_000)
    parser.add_argument("--stage-c5-ep0", type=int, default=560_000)
    parser.add_argument("--stage-b-causal-ep0", type=int, default=535_000)
    parser.add_argument("--stage-b-confirm-ep0", type=int, default=570_000)
    parser.add_argument("--stage-a-confirm-ep0", type=int, default=580_000)
    parser.add_argument("--stage-c-confirm-ep0", type=int, default=590_000)
    parser.add_argument("--common-best-ep0", type=int, default=600_000)
    parser.add_argument("--full-eval-ep0", type=int, default=610_000)
    parser.add_argument("--full-confirm-ep0", type=int, default=620_000)
    parser.add_argument("--stage-b-noise-seed", type=int, default=2_026_074_1)
    parser.add_argument("--stage-a-noise-seed", type=int, default=2_026_074_2)
    parser.add_argument("--stage-c3-noise-seed", type=int, default=2_026_074_3)
    parser.add_argument("--stage-c5-noise-seed", type=int, default=2_026_074_4)
    parser.add_argument(
        "--stage-b-causal-noise-seed", type=int, default=2_026_074_5,
    )
    parser.add_argument(
        "--stage-b-confirm-noise-seed", type=int, default=2_026_074_6,
    )
    parser.add_argument(
        "--stage-a-confirm-noise-seed", type=int, default=2_026_074_7,
    )
    parser.add_argument(
        "--stage-c-confirm-noise-seed", type=int, default=2_026_074_8,
    )
    parser.add_argument(
        "--common-best-noise-seed", type=int, default=2_026_074_9,
    )
    parser.add_argument("--full-noise-seed", type=int, default=2_026_075_0)
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))
