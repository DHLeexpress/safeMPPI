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
EXPECTED_TRIGGER_SOURCE = "9c0ec8ac657e8711e368af4bae4cf9b63328de6b"
DEFAULT_TRIGGER = Path(
    "/data3/research1/sfm_neutral_autonomous_9c0ec8a/"
    "CREATIVE_SANITY_REQUIRED.json"
)
DEFAULT_AUDIT_ROUNDS = (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _read(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _write(path: Path, payload: dict) -> None:
    GLOBAL._write(path, payload)


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
    path: Path, trigger: dict, *, expected_source=EXPECTED_TRIGGER_SOURCE,
) -> dict:
    if (
        trigger.get("status") != AUTO.CREATIVE_TRIGGER_STATUS
        or trigger.get("action") != "CREATIVE_SANITY_REQUIRED"
    ):
        raise RuntimeError("creative coordinator received a non-trigger marker")
    if expected_source and trigger.get("source_commit") != expected_source:
        raise RuntimeError("creative trigger was produced by an unexpected source")
    delivery_path = Path(trigger["autonomous_delivery"]).resolve()
    if _sha256(delivery_path) != trigger.get("autonomous_delivery_sha256"):
        raise RuntimeError("autonomous delivery digest mismatch")
    delivery = _read(delivery_path)
    if (
        delivery.get("status") != AUTO.STATUS
        or delivery.get("action") != "CREATIVE_SANITY_REQUIRED"
        or delivery.get("ci_clean_four_metric_win") is not False
    ):
        raise RuntimeError("autonomous delivery does not authorize creative sanity")
    if delivery.get("source_commit") != trigger.get("source_commit"):
        raise RuntimeError("trigger/autonomous source commit mismatch")
    for key in (
        "selected_arm", "r100_training_delivery", "r100_gamma_delivery",
    ):
        if trigger.get(key) != delivery.get(key):
            raise RuntimeError(f"trigger/autonomous mismatch: {key}")
    gamma = _read(Path(trigger["r100_gamma_delivery"]).resolve())
    if gamma.get("status") != GAMMA.STATUS or gamma.get(
        "ci_clean_four_metric_win"
    ) is not False:
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
    catalog = {}
    for _, delivery in chain:
        for marker_path in delivery.get("round_records", ()):
            marker_path = Path(marker_path).resolve()
            marker = _read(marker_path)
            round_i = int(marker.get("round", -1))
            if marker.get("status") != TRAIN.ROUND_STATUS or round_i in catalog:
                raise RuntimeError("invalid or duplicate round marker")
            checkpoint = Path(marker["checkpoint"]).resolve()
            if _sha256(checkpoint) != marker.get("checkpoint_sha256"):
                raise RuntimeError("post-D0 checkpoint digest mismatch")
            post_positive = (
                checkpoint.parent / f"round_{round_i:02d}_post_positive.pt"
            )
            payload = torch.load(
                post_positive, map_location="cpu", weights_only=False,
            )
            if (
                int(payload.get("round", -1)) != round_i
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
    train_seed, probe_seed, resume=None,
):
    command = [
        sys.executable, str(HERE / "sfm_b1_neutral_multiround.py"),
        "--checkpoint", str(checkpoint),
        "--output-root", str(output),
        "--name", str(name),
        "--rounds", str(int(rounds)),
        "--scenario-ep0", str(int(scenario_ep0)),
        "--eval-ep0", str(int(eval_ep0)),
        "--eval-M", "10",
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
    return command


def _run(command, *, log: Path, gpu: int, cpu_range: str) -> None:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    environment["PYTHONPATH"] = str(HERE)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            ["taskset", "-c", cpu_range, *command],
            cwd=HERE, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"job failed ({completed.returncode}): {log}")


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
    return GLOBAL._trend(row)["mean_fraction"] >= 0.75


def _selector_gate(candidate_rows, controls, r100_reference) -> dict:
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
        decisions.append({
            "round": int(row["round"]),
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "pooled": value,
            "safety_noninferior_to_matched_margin": bool(safe),
            "liveness_improved_over_matched_margin": bool(live),
            "gamma_trend_pass": _trend_ok(row),
            "eligible": bool(safe and live and _trend_ok(row)),
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
            "the matched margin round, while CR/Validity are within 3pp and "
            "the gamma trend score is >=0.75"
        ),
        "r100_reference": r100_reference,
        "decisions": decisions,
        "passed": selected is not None,
        "selected": selected,
    }


def _encoder_stage_gate(delivery, candidate_rows, control_rows) -> dict:
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
        gather = marker["gather"]
        diagnostics.append({
            "round": round_i,
            "token_cosine": encoder["token_cosine"],
            "encoder_relative_drift": encoder["relative_parameter_drift"],
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
            "delta_CR_vs_frozen": (
                candidate["pooled"]["CR"] - control["pooled"]["CR"]
            ),
            "delta_Validity_vs_frozen": (
                candidate["pooled"]["Validity"]
                - control["pooled"]["Validity"]
            ),
        })
    unsafe = any(
        row["token_cosine"] < .98
        or row["Dplus_regressed"] > 0
        or row["delta_CR_vs_frozen"] > .03
        for row in diagnostics
    )
    signal = any(
        row["delta_Validity_vs_frozen"] >= .01
        or row["uncertainty_uplift"] >= .005
        for row in diagnostics
    )
    return {
        "rule": (
            "stop on E_g token cosine <0.98, any D+ regression, or CR >3pp "
            "above matched frozen control; continue only with >=1pp Validity "
            "or >=0.005 uncertainty-uplift signal"
        ),
        "diagnostics": diagnostics,
        "unsafe": bool(unsafe),
        "signal": bool(signal),
        "continue_to_round5": bool(not unsafe and signal),
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
    ]
    known_banks = _known_banks(chain, authenticated["gamma"])
    _validate_new_banks(banks, known_banks, training_scenarios)
    output.mkdir(parents=True)
    provenance = {
        "status": "SFM_NEUTRAL_CREATIVE_SANITY_PREREGISTERED",
        "source": source,
        "trigger": str(trigger_path),
        "trigger_sha256": _sha256(trigger_path),
        "autonomous_delivery": str(authenticated["path"]),
        "autonomous_delivery_sha256": _sha256(authenticated["path"]),
        "selected_arm": selected_arm,
        "pretrained_checkpoint": final_delivery["checkpoint"],
        "pretrained_checkpoint_sha256": final_delivery["checkpoint_sha256"],
        "r100_training_delivery": trigger["r100_training_delivery"],
        "r100_training_delivery_sha256": _sha256(
            Path(trigger["r100_training_delivery"])
        ),
        "audit_rounds": list(audit_rounds),
        "known_banks": known_banks,
        "new_banks": banks,
        "no_combined_arm": True,
        "stage_order": ["B_postphase_audit", "A_selector", "C_encoder"],
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
        noise_seed=args.stage_b_noise_seed, gpus=args.gpus,
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
    if stage_b_gate["clear_D0_overwrite"]:
        result = {
            "status": STATUS,
            "action": "STOP_DPLUS_ONLY_CAUSAL_ARM_REQUIRED",
            "reason": (
                "disjoint evidence identifies D0 overwrite; changing the "
                "selector or encoder now would confound that mechanism"
            ),
            "stages_completed": ["B"],
            "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write(output / "DELIVERY_COMPLETE.json", result)
        return result

    cfg = final_delivery["config"]
    scenario_ep0 = min(training_scenarios)
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
        gpus=args.gpus, workers=args.workers,
    )
    r100_a = _evaluate_specs([
        {
            "name": "r100_reference", "round": 100,
            "phase": "post_D0", "checkpoint": catalog[100]["post_D0"],
        }
    ], root=stage_a_root / "r100_reference", bank=banks[1],
        noise_seed=args.stage_a_noise_seed, gpus=args.gpus,
        workers=args.workers)[0]
    stage_a_gate = _selector_gate(stage_a_rows, controls_a, r100_a)
    stage_a = {
        "status": STAGE_A_STATUS,
        "single_change": "progress_gated_margin selector",
        "training_delivery": _ref(stage_a_root / "DELIVERY_COMPLETE.json"),
        "bank": banks[1],
        "rows": stage_a_rows,
        "matched_margin_controls": controls_a,
        "gate": stage_a_gate,
    }
    _write(stage_a_root / "STAGE_COMPLETE.json", stage_a)
    if stage_a_gate["passed"]:
        result = {
            "status": STATUS,
            "action": "PROMOTE_STAGE_A_CANDIDATE",
            "stages_completed": ["B", "A"],
            "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
            "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
            "selected": stage_a_gate["selected"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write(output / "DELIVERY_COMPLETE.json", result)
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
        noise_seed=args.stage_c3_noise_seed, gpus=args.gpus,
        workers=args.workers,
    )
    c3_gate = _encoder_stage_gate(c3_delivery, c3_rows, c3_controls)
    c3 = {
        "status": STAGE_C_STATUS,
        "phase": "rounds_1_to_3",
        "single_change": "enc_grid lr = 0.1 * trunk lr",
        "training_delivery": _ref(stage_c3_root / "DELIVERY_COMPLETE.json"),
        "bank": banks[2],
        "rows": c3_rows,
        "matched_frozen_controls": c3_controls,
        "gate": c3_gate,
    }
    _write(stage_c3_root / "STAGE_COMPLETE.json", c3)
    if not c3_gate["continue_to_round5"]:
        result = {
            "status": STATUS,
            "action": "STOP_NO_CREDIBLE_CREATIVE_ARM",
            "stages_completed": ["B", "A", "C3"],
            "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
            "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
            "stage_C3": _ref(stage_c3_root / "STAGE_COMPLETE.json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write(output / "DELIVERY_COMPLETE.json", result)
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
        noise_seed=args.stage_c5_noise_seed, gpus=args.gpus,
        workers=args.workers,
    )
    r100_c5 = _evaluate_specs([{
        "name": "r100_reference", "round": 100,
        "phase": "post_D0", "checkpoint": catalog[100]["post_D0"],
    }], root=stage_c5_root / "r100_reference", bank=banks[3],
        noise_seed=args.stage_c5_noise_seed, gpus=args.gpus,
        workers=args.workers)[0]
    c5_gate = _encoder_stage_gate(c5_delivery, c5_rows, c5_controls)
    selector_c5 = _selector_gate(c5_rows, c5_controls, r100_c5)
    passed = bool(not c5_gate["unsafe"] and selector_c5["passed"])
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
    }
    _write(stage_c5_root / "STAGE_COMPLETE.json", c5)
    result = {
        "status": STATUS,
        "action": (
            "PROMOTE_STAGE_C_CANDIDATE"
            if passed else "STOP_NO_CREDIBLE_CREATIVE_ARM"
        ),
        "stages_completed": ["B", "A", "C3", "C5"],
        "stage_B": _ref(stage_b_root / "STAGE_COMPLETE.json"),
        "stage_A": _ref(stage_a_root / "STAGE_COMPLETE.json"),
        "stage_C3": _ref(stage_c3_root / "STAGE_COMPLETE.json"),
        "stage_C5": _ref(stage_c5_root / "STAGE_COMPLETE.json"),
        "selected": selector_c5["selected"] if passed else None,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(output / "DELIVERY_COMPLETE.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", default=str(DEFAULT_TRIGGER))
    parser.add_argument(
        "--expected-trigger-source", default=EXPECTED_TRIGGER_SOURCE,
    )
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
    parser.add_argument("--stage-b-noise-seed", type=int, default=2_026_074_1)
    parser.add_argument("--stage-a-noise-seed", type=int, default=2_026_074_2)
    parser.add_argument("--stage-c3-noise-seed", type=int, default=2_026_074_3)
    parser.add_argument("--stage-c5-noise-seed", type=int, default=2_026_074_4)
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))
