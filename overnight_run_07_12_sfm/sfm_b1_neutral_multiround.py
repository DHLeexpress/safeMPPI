"""Small-lineage multi-round SFM expansion with paired trigger probes.

This additive study keeps the existing max-margin collector semantics while
changing only the experiment schedule:

* two scenario seeds x all seven gammas per macro-round;
* previous-round executed D+ only in the RBF GP;
* whole-D+ followed by isolated whole-D0 CFM replay;
* one occurrence of every row per inner pass and one Adam step per pass;
* fixed-context, fixed-latent exact-H10 probes before and after each phase.

D0 retains its exact verifier label y=0.  It is a separate teacher population
and never enters D+, the RBF GP, or the ordinary replay population.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_teacher_sanity as NS
import sfm_b1_offline_eval as OE
import sfm_b1_offline_store as OS
import sfm_b1_r2_alpha_replay as R2
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


STATUS = "SFM_B1_NEUTRAL_MULTIROUND_COMPLETE"
ROUND_STATUS = "SFM_B1_NEUTRAL_MULTIROUND_ROUND_COMPLETE"
PROBE_STATUS = "SFM_B1_PAIRED_TRIGGER_PROBE_COMPLETE"
DEFAULT_SCENARIO_EP0 = 260_000
DEFAULT_EVAL_EP0 = 270_000
DEFAULT_PROBE_PER_GAMMA = 4
DEFAULT_GP_CAP = 512
DEFAULT_LR = 3.0e-5
DEFAULT_INNER_STEPS = 4


@dataclass(frozen=True)
class StudyConfig:
    name: str
    rounds: int = 2
    scenarios_per_round: int = 2
    lr: float = DEFAULT_LR
    inner_steps: int = DEFAULT_INNER_STEPS
    batch: int = 128
    K: int = 16
    B: int = 4
    H: int = 10
    T: int = 180
    nfe: int = 8
    temp: float = 1.0
    phi_s: float = 0.9
    ell: float = RA.DEFAULT_ELL
    gp_cap: int = DEFAULT_GP_CAP
    gp_lambda: float = 1.0e-2
    ess_target: float = 0.5
    selector: str = "margin"
    encoder_lr_ratio: float = 0.0
    neutral_replay: bool = True
    alpha: float = 0.0
    scene_profile: str = "double_density_velocity_ood"
    sample_seed: int = 700_000
    audit_seed: int = 2_026_073_0
    train_seed: int = 2_026_073_1
    probe_seed: int = 2_026_073_2

    def validate(self):
        if int(self.rounds) < 1:
            raise ValueError("rounds must be positive")
        if int(self.scenarios_per_round) != 2:
            raise ValueError("this study requires exactly two scenarios/round")
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError("lr must be finite and positive")
        if int(self.inner_steps) < 1:
            raise ValueError("inner_steps must be positive")
        if int(self.batch) != 128:
            raise ValueError("canonical microbatch size is 128")
        if (
            int(self.K) != 16
            or int(self.B) != 4
            or int(self.H) != 10
            or int(self.T) != 180
        ):
            raise ValueError("K/B/H/T protocol changed")
        if self.selector not in ("margin", "progress_gated_margin"):
            raise ValueError("unsupported neutral-study selector")
        if float(self.encoder_lr_ratio) not in (0.0, 0.1):
            raise ValueError("encoder_lr_ratio must be 0 (frozen) or 0.1")
        if not isinstance(self.neutral_replay, bool):
            raise ValueError("neutral_replay must be boolean")
        if float(self.alpha) != 0.0:
            raise ValueError("this study is pinned to alpha=0")
        if self.scene_profile != "double_density_velocity_ood":
            raise ValueError("scene profile changed")
        return self


class _NeutralHolder:
    def __init__(self, round_i):
        self.round_i = int(round_i)
        self.contexts = []
        self.windows = []


def _write_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _sha256_jsonable(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return {
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "bytes": hashlib.sha256(item.tobytes()).hexdigest(),
            }
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {
                str(key): convert(item[key])
                for key in sorted(item, key=str)
            }
        if isinstance(item, (list, tuple)):
            return [convert(row) for row in item]
        return item

    encoded = json.dumps(
        convert(value), sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compact_mass(accounting):
    return {
        "total": float(accounting["total"]),
        "gamma": {
            str(key): float(value)
            for key, value in accounting["gamma"].items()
        },
        "cells": len(accounting["cells"]),
        "contexts": len(accounting["contexts"]),
    }


def _neutral_records(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != "SFM_B1_NEUTRAL_ROUND_COMPLETE":
        raise ValueError("not an authenticated neutral-round payload")
    holder = _NeutralHolder(payload["round"])
    records = []
    for expected, source in enumerate(payload["records"]):
        result = source.get("verifier_result", {})
        if (
            int(source["neutral_id"]) != expected
            or source["population"] != "D0"
            or source["semantic_label"] != "neutral"
            or int(source["verifier_y"]) != 0
            or not result.get("resolved")
            or int(result.get("y", -1)) != 0
            or not result.get("full_h")
            or int(result.get("terminal_step", -1)) != SP.H
            or source["train_eligible"]
            or source["replay_default"]
            or source["gp_eligible"]
        ):
            raise RuntimeError("D0 semantics changed")
        context_id = len(holder.contexts)
        holder.contexts.append({
            "context_id": context_id,
            "round": int(payload["round"]),
            "scenario_id": int(source["scenario_id"]),
            "gamma": float(source["gamma"]),
            "step": int(source["step"]),
            "state": np.asarray(source["state"], np.float32),
            "hp10": np.asarray(source["hp10"], np.float32),
            "low5": np.asarray(source["low5"], np.float32),
            "hist": np.asarray(source["hist"], np.float32),
            "ped_xy": np.asarray(source["ped_xy"], np.float32),
            "ped_vel": np.asarray(source["ped_vel"], np.float32),
        })
        row = {
            "window_id": expected,
            "query_id": expected,
            "context_id": context_id,
            "controls": np.asarray(source["controls"], np.float32),
            "x0": np.asarray(source["x0"], np.float32),
            "y": 0,
            "semantic_label": "neutral",
        }
        holder.windows.append(row)
        records.append((holder, row))
    if len(records) != int(payload["summary"]["D0"]):
        raise RuntimeError("D0 payload count mismatch")
    return holder, records


def _identity(holder, row):
    return int(holder.round_i), int(row["query_id"])


def _population_update(
    policy,
    optimizer,
    records,
    *,
    population,
    inner_steps,
    batch,
    device,
    seed,
):
    if not records:
        raise ValueError(f"{population} replay requires nonempty support")
    mass, accounting = BS.hierarchy_mass(records)
    encoder_before = BS.module_sha256(policy.enc_grid)
    expected = {_identity(holder, row) for holder, row in records}
    losses = []
    encoder_gradient_norms = []
    exposure_hashes = []
    policy.train()
    for inner in range(int(inner_steps)):
        optimizer.zero_grad(set_to_none=True)
        loss, visited = NS._objective(
            policy,
            records,
            mass,
            batch=int(batch),
            device=device,
            seed=int(seed) + inner * 1_000_003,
            backward=True,
        )
        identities = list(map(tuple, visited))
        if len(identities) != len(expected) or set(identities) != expected:
            raise RuntimeError(
                f"{population} pass duplicated or omitted a sample"
            )
        squared = torch.zeros((), dtype=torch.float64)
        for parameter in policy.enc_grid.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().to(
                    dtype=torch.float64,
                ).square().sum().cpu()
        encoder_gradient_norms.append(float(squared.sqrt()))
        optimizer.step()
        losses.append(float(loss))
        exposure_hashes.append(_sha256_jsonable(identities))
    policy.eval()
    encoder_after = BS.module_sha256(policy.enc_grid)
    if (
        not any(parameter.requires_grad for parameter in policy.enc_grid.parameters())
        and encoder_after != encoder_before
    ):
        raise RuntimeError("visual encoder changed during replay")
    return {
        "population": str(population),
        "records": len(records),
        "inner_steps": int(inner_steps),
        "optimizer_steps": int(inner_steps),
        "sample_exposures": len(records) * int(inner_steps),
        "exact_once_per_inner_step": True,
        "exposure_identity_sha256": exposure_hashes,
        "losses": losses,
        "encoder_gradient_norms": encoder_gradient_norms,
        "mass": _compact_mass(accounting),
        "encoder_sha_before": encoder_before,
        "encoder_sha_after": encoder_after,
    }


def _skipped_population_update(records, *, population):
    _, accounting = BS.hierarchy_mass(records)
    return {
        "population": str(population),
        "records": len(records),
        "inner_steps": 0,
        "optimizer_steps": 0,
        "sample_exposures": 0,
        "exact_once_per_inner_step": None,
        "exposure_identity_sha256": [],
        "losses": [],
        "encoder_gradient_norms": [],
        "mass": _compact_mass(accounting),
        "skipped": True,
        "reason": "D0 collected for causal audit but neutral replay disabled",
    }


@torch.no_grad()
def _encoder_probe(policy, records, *, device, limit=256):
    """Return deterministic E_g tokens for a fixed record prefix."""
    ordered = sorted(records, key=lambda item: _identity(*item))[:int(limit)]
    if not ordered:
        raise ValueError("encoder probe requires records")
    grid, _, _, _ = BS._tensor_batch(ordered, device)
    was_training = policy.training
    policy.eval()
    token = policy.enc_grid(grid.float()).detach().cpu()
    policy.train(was_training)
    return token


def _encoder_probe_comparison(before, after):
    before = torch.as_tensor(before, dtype=torch.float64).reshape(-1)
    after = torch.as_tensor(after, dtype=torch.float64).reshape(-1)
    if tuple(before.shape) != tuple(after.shape):
        raise ValueError("encoder probe shapes changed")
    cosine = float(torch.dot(before, after) / (
        before.norm() * after.norm()
    ).clamp_min(1.0e-12))
    return {
        "token_cosine": cosine,
        "token_rms_change": float((after - before).square().mean().sqrt()),
        "values": int(before.numel()),
    }


@torch.no_grad()
def _encoder_probe_from_grid(policy, grid, *, device):
    was_training = policy.training
    policy.eval()
    token = policy.enc_grid(
        torch.as_tensor(grid, dtype=torch.float32, device=device)
    ).detach().cpu()
    policy.train(was_training)
    return token


def _create_encoder_reference(
    path, policy, records, *, device, anchor_round, checkpoint_sha256,
    limit=256,
):
    ordered = sorted(records, key=lambda item: _identity(*item))[:int(limit)]
    if not ordered:
        raise ValueError("encoder reference requires records")
    grid, _, _, _ = BS._tensor_batch(ordered, device)
    grid = grid.detach().cpu()
    payload = {
        "status": "SFM_B1_ENCODER_REFERENCE_PROBE",
        "anchor_round": int(anchor_round),
        "checkpoint_sha256": str(checkpoint_sha256),
        "grid": grid,
        "token": _encoder_probe_from_grid(policy, grid, device=device),
        "encoder_snapshot": R2._module_snapshot(policy)["E_g"],
        "records": len(ordered),
    }
    torch.save(payload, path)
    return payload


def _load_encoder_reference(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != "SFM_B1_ENCODER_REFERENCE_PROBE":
        raise RuntimeError("invalid encoder reference probe")
    if not all(key in payload for key in ("grid", "token", "encoder_snapshot")):
        raise RuntimeError("incomplete encoder reference probe")
    return payload


def _encoder_reference_comparison(policy, reference, *, device):
    current_token = _encoder_probe_from_grid(
        policy, reference["grid"], device=device,
    )
    current_snapshot = R2._module_snapshot(policy)["E_g"]
    return {
        **_encoder_probe_comparison(reference["token"], current_token),
        "relative_parameter_drift": R2._module_relative_drift(
            {"E_g": reference["encoder_snapshot"]},
            {"E_g": current_snapshot},
        )["E_g"],
        "anchor_round": int(reference["anchor_round"]),
        "records": int(reference["records"]),
    }


def _gradient_diagnostics(
    policy, positive_records, neutral_records, *, batch, device, seed,
):
    positive = NS._gradient(
        policy,
        positive_records,
        batch=batch,
        device=device,
        seed=seed,
    )
    neutral = NS._gradient(
        policy,
        neutral_records,
        batch=batch,
        device=device,
        seed=seed,
    )
    return {
        "Dplus_norm": BS._gradient_norm(positive),
        "D0_norm": BS._gradient_norm(neutral),
        "Dplus_D0_cosine": NS._cosine(positive, neutral),
    }


def _anchor_catalog(trace_path, *, per_gamma, seed, output_path):
    payload = torch.load(
        trace_path, map_location="cpu", weights_only=False,
    )
    candidates = []
    for trace in payload["traces"]:
        if (
            trace.get("repair_trigger") is None
            or trace.get("executed_controls") is None
            or len(trace.get("all_K", ())) != 16
            or len(trace.get("selected_ids", ())) != 4
        ):
            continue
        all_x0 = np.stack([
            np.asarray(row["x0"], np.float32)
            for row in trace["all_K"]
        ])
        target_x0 = np.asarray(trace["executed_x0"], np.float32)
        latent_distance = np.linalg.norm(
            all_x0 - target_x0[None], axis=1,
        )
        target_latent_id = int(np.argmin(latent_distance))
        if float(latent_distance[target_latent_id]) > 1.0e-7:
            raise RuntimeError(
                "executed target x0 is absent from the original K bank"
            )
        base_lookup = {
            int(row["candidate_id"]): row
            for row in trace["query_rows"]
        }
        original_B_admissible = 0
        original_B_exact_positive = 0
        for candidate_id in map(int, trace["selected_ids"]):
            row = base_lookup[candidate_id]
            result = row["result"]
            margin, _, _ = BC.nominal_hp_margin(
                trace["state"],
                row["controls"][0],
                trace["ped_xy"],
                trace["gamma"],
            )
            original_B_exact_positive += int(result["y"])
            original_B_admissible += int(
                int(result["y"]) == 1 and margin >= -1.0e-9
            )
        original_controls = np.stack([
            np.asarray(row["controls"], np.float32)
            for row in trace["all_K"]
        ])
        anchor = {
            "anchor_id": len(candidates),
            "round": int(trace["round"]),
            "scenario_id": int(trace["scenario_id"]),
            "gamma": float(trace["gamma"]),
            "step": int(trace["step"]),
            "trigger": str(trace["repair_trigger"]),
            "population": (
                "D0" if trace["neutral_execution"] else "Dplus"
            ),
            "execution_source": str(trace["execution_source"]),
            "state": np.asarray(trace["state"], np.float32),
            "hp10": None,
            "low5": None,
            "hist": None,
            "ped_xy": np.asarray(trace["ped_xy"], np.float32),
            "ped_vel": np.asarray(trace["ped_vel"], np.float32),
            "target_controls": np.asarray(
                trace["executed_controls"], np.float32,
            ),
            "target_x0": target_x0,
            "target_latent_id": target_latent_id,
            "K_x0": all_x0,
            "original_K_controls": original_controls,
            "original_B_ids": list(map(int, trace["selected_ids"])),
            "trace_B_exact_positive": original_B_exact_positive,
            "trace_B_admissible": original_B_admissible,
            "target_verifier_y": int(
                trace["executed_result"]["y"]
            ),
        }
        expected_y = 0 if anchor["population"] == "D0" else 1
        if anchor["target_verifier_y"] != expected_y:
            raise RuntimeError(
                "trigger target population disagrees with exact verifier y"
            )
        candidates.append(anchor)

    # Context tensors are authoritative in the training stores, not rebuilt
    # from floating-point state.
    gather_dir = os.path.dirname(trace_path)
    executed = OS.ExecutedRoundShard.load(
        os.path.join(gather_dir, "executed_round.pt")
    )
    neutral_payload = torch.load(
        os.path.join(gather_dir, "neutral_round.pt"),
        map_location="cpu",
        weights_only=False,
    )
    contexts = {}
    for context in executed.contexts:
        key = (
            int(context["scenario_id"]),
            round(float(context["gamma"]), 8),
            int(context["step"]),
        )
        contexts[key] = context
    for row in neutral_payload["records"]:
        key = (
            int(row["scenario_id"]),
            round(float(row["gamma"]), 8),
            int(row["step"]),
        )
        contexts[key] = row
    for anchor in candidates:
        key = (
            anchor["scenario_id"],
            round(anchor["gamma"], 8),
            anchor["step"],
        )
        context = contexts.get(key)
        if context is None:
            raise RuntimeError(f"trigger anchor has no stored context: {key}")
        anchor["hp10"] = np.asarray(context["hp10"], np.float32)
        anchor["low5"] = np.asarray(context["low5"], np.float32)
        anchor["hist"] = np.asarray(context["hist"], np.float32)

    grouped = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for anchor in candidates:
        grouped[round(anchor["gamma"], 8)][
            int(anchor["scenario_id"])
        ][anchor["population"]].append(anchor)
    generator = np.random.default_rng(int(seed))
    selected = []
    for gamma in map(float, SP.GAMMAS):
        key = round(gamma, 8)
        values = []
        for scenario_id in sorted(grouped[key]):
            populations = grouped[key][scenario_id]
            for population in ("D0", "Dplus"):
                rows = populations[population]
                if rows:
                    index = int(generator.integers(0, len(rows)))
                    values.append(rows[index])
        values = values[:int(per_gamma)]
        if len(values) < int(per_gamma):
            used = {id(row) for row in values}
            remainder = [
                row
                for scenario_id in sorted(grouped[key])
                for population in ("D0", "Dplus")
                for row in grouped[key][scenario_id][population]
                if id(row) not in used
            ]
            remainder = [
                remainder[index]
                for index in generator.permutation(len(remainder))
            ]
            values.extend(remainder[:int(per_gamma) - len(values)])
        selected.extend(values)
    for anchor_id, anchor in enumerate(selected):
        anchor["anchor_id"] = anchor_id
    catalog = {
        "status": "SFM_B1_TRIGGER_ANCHOR_CATALOG_COMPLETE",
        "round": int(payload["round"]),
        "requested_per_gamma": int(per_gamma),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_per_gamma": {
            str(gamma): sum(
                round(row["gamma"], 8) == round(float(gamma), 8)
                for row in selected
            )
            for gamma in SP.GAMMAS
        },
        "selected_per_scenario_gamma": {
            f"{scenario_id}:{gamma}": sum(
                row["scenario_id"] == int(scenario_id)
                and round(row["gamma"], 8) == round(float(gamma), 8)
                for row in selected
            )
            for scenario_id in sorted({
                row["scenario_id"] for row in candidates
            })
            for gamma in SP.GAMMAS
        },
        "anchors": selected,
    }
    temporary = output_path + ".tmp"
    torch.save(catalog, temporary)
    os.replace(temporary, output_path)
    _write_json(output_path + ".COMPLETE.json", {
        "status": catalog["status"],
        "round": catalog["round"],
        "file": os.path.abspath(output_path),
        "sha256": FA._sha256_file(output_path),
        "candidate_count": catalog["candidate_count"],
        "selected_count": catalog["selected_count"],
        "selected_per_gamma": catalog["selected_per_gamma"],
        "selected_per_scenario_gamma": (
            catalog["selected_per_scenario_gamma"]
        ),
    })
    return selected


def _goal_progress(state, controls):
    segment = SM.rollout_positions(state, controls)
    initial = float(np.linalg.norm(segment[0] - SS.GOAL))
    one = float(np.linalg.norm(segment[1] - SS.GOAL))
    final = float(np.linalg.norm(segment[-1] - SS.GOAL))
    return initial - one, initial - final


def _predicted_clearance(state, controls, ped_xy, ped_vel):
    robot = SM.rollout_positions(state, controls)
    peds = SM.predict_pedestrians(ped_xy, ped_vel, len(controls))
    if peds.shape[1] == 0:
        return float("inf")
    return float(
        np.linalg.norm(robot[:, None] - peds, axis=2).min() - SS.R_PED
    )


def _probe_checkpoint(
    checkpoint,
    anchors,
    *,
    phase,
    device,
    executor,
    batch,
    selector="margin",
):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    generated = []
    features = []
    reconstruction = []
    for start in range(0, len(anchors), int(batch)):
        values = anchors[start:start + int(batch)]
        hp10 = torch.as_tensor(
            np.stack([row["hp10"] for row in values]), device=device,
        ).float()
        low5 = torch.as_tensor(
            np.stack([row["low5"] for row in values]), device=device,
        ).float()
        hist = torch.as_tensor(
            np.stack([row["hist"] for row in values]), device=device,
        ).float()
        context = policy.ctx_from(hp10, low5, hist)
        x0 = torch.as_tensor(
            np.stack([row["K_x0"] for row in values]), device=device,
        ).float()
        with torch.no_grad():
            controls = BE.integrate_latents(
                policy,
                x0.reshape(-1, policy.d),
                context.repeat_interleave(16, dim=0),
                nfe=8,
            ).reshape(len(values), 16, SP.H, 2)
            target = torch.as_tensor(
                np.stack([row["target_controls"] for row in values]),
                device=device,
            ).float()
            target_x0 = torch.as_tensor(
                np.stack([row["target_x0"] for row in values]),
                device=device,
            ).float()
            phi = policy.phi_s_from_x0(
                target,
                context,
                target_x0,
                s=0.9,
            )
        controls_np = controls.detach().cpu().numpy().astype(np.float32)
        generated.extend(controls_np)
        features.extend(
            BR.l2_normalize(phi).detach().cpu().numpy().astype(np.float32)
        )
        reconstruction.extend([
            float(np.sqrt(np.mean(
                (controls_np[index] - values[index][
                    "original_K_controls"
                ]) ** 2
            )))
            for index in range(len(values))
        ])

    tasks = []
    for anchor_index, (anchor, windows) in enumerate(
        zip(anchors, generated)
    ):
        for candidate_id, controls in enumerate(windows):
            tasks.append((
                anchor_index,
                candidate_id,
                anchor["state"],
                controls,
                anchor["ped_xy"],
                anchor["ped_vel"],
                anchor["gamma"],
            ))
    result_rows = list(executor.map(SM.verify_in_worker, tasks))
    by_anchor = defaultdict(dict)
    for anchor_index, candidate_id, result in result_rows:
        if not result.get("resolved"):
            raise RuntimeError("paired trigger probe verifier error")
        by_anchor[int(anchor_index)][int(candidate_id)] = result

    rows = []
    for anchor_index, (anchor, windows, feature) in enumerate(
        zip(anchors, generated, features)
    ):
        candidates = []
        admissible_count = 0
        for candidate_id, controls in enumerate(windows):
            result = by_anchor[anchor_index][candidate_id]
            margin, hp_old, hp_new = BC.nominal_hp_margin(
                anchor["state"],
                controls[0],
                anchor["ped_xy"],
                anchor["gamma"],
            )
            admissible = bool(
                int(result["y"]) == 1 and margin >= -1.0e-9
            )
            admissible_count += int(admissible)
            candidates.append({
                "candidate_id": candidate_id,
                "controls": controls,
                "result": result,
                "hp_margin": float(margin),
                "hp_old": float(hp_old),
                "hp_new": float(hp_new),
                "admissible": admissible,
            })
        B_orig = [
            candidates[index] for index in anchor["original_B_ids"]
        ]
        selected = BC.select_admissible(
            B_orig,
            selector=selector,
            state=anchor["state"],
            ped_xy=anchor["ped_xy"],
            ped_vel=anchor["ped_vel"],
            gamma=anchor["gamma"],
        )
        target = anchor["target_controls"]
        difference = windows - target[None]
        same_latent = difference[int(anchor["target_latent_id"])]
        target_latent = candidates[int(anchor["target_latent_id"])]
        if selected is None:
            selected_id = None
            one_progress = None
            H_progress = None
            selected_margin = None
            selected_clearance = None
        else:
            selected_id = int(selected["candidate_id"])
            one_progress, H_progress = _goal_progress(
                anchor["state"], selected["controls"],
            )
            selected_margin = float(selected["hp_margin"])
            selected_clearance = _predicted_clearance(
                anchor["state"],
                selected["controls"],
                anchor["ped_xy"],
                anchor["ped_vel"],
            )
        rows.append({
            "anchor_id": int(anchor["anchor_id"]),
            "round": int(anchor["round"]),
            "scenario_id": int(anchor["scenario_id"]),
            "gamma": float(anchor["gamma"]),
            "step": int(anchor["step"]),
            "trigger": str(anchor["trigger"]),
            "population": str(anchor["population"]),
            "execution_source": str(anchor["execution_source"]),
            "phase": str(phase),
            "target_verifier_y": int(anchor["target_verifier_y"]),
            "target_full_rmse": float(
                np.sqrt(np.mean(same_latent ** 2))
            ),
            "target_first_rmse": float(
                np.sqrt(np.mean(same_latent[0] ** 2))
            ),
            "best_K_target_full_rmse": float(
                np.sqrt(np.mean(difference ** 2, axis=(1, 2))).min()
            ),
            "baseline_K_reconstruction_rmse": float(
                reconstruction[anchor_index]
            ),
            "all_K_exact_positive": int(sum(
                int(row["result"]["y"]) for row in candidates
            )),
            "all_K_admissible": int(admissible_count),
            "B_orig_admissible": int(sum(
                row["admissible"] for row in B_orig
            )),
            "B_orig_NVP": selected is None,
            "trace_B_exact_positive": int(
                anchor["trace_B_exact_positive"]
            ),
            "trace_B_admissible": int(anchor["trace_B_admissible"]),
            "baseline_B_admissible_delta_from_trace": int(
                sum(row["admissible"] for row in B_orig)
                - anchor["trace_B_admissible"]
            ),
            "target_latent_exact_y": int(
                target_latent["result"]["y"]
            ),
            "target_latent_admissible": bool(
                target_latent["admissible"]
            ),
            "selected_candidate_id": selected_id,
            "selected_one_step_progress": one_progress,
            "selected_H10_progress": H_progress,
            "selected_hp_margin": selected_margin,
            "selected_predicted_clearance": selected_clearance,
            "_feature": np.asarray(feature, np.float32),
            "_windows": np.asarray(windows, np.float32),
        })
    del policy
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _probe_comparison(before, after):
    before_lookup = {int(row["anchor_id"]): row for row in before}
    after_lookup = {int(row["anchor_id"]): row for row in after}
    if set(before_lookup) != set(after_lookup):
        raise RuntimeError("paired trigger probe anchor set changed")
    rows = []
    for anchor_id in sorted(before_lookup):
        old = before_lookup[anchor_id]
        new = after_lookup[anchor_id]
        old_nvp = bool(old["B_orig_NVP"])
        new_nvp = bool(new["B_orig_NVP"])
        old_target_admissible = bool(old["target_latent_admissible"])
        new_target_admissible = bool(new["target_latent_admissible"])
        progress = new["selected_one_step_progress"]
        actual_repair = bool(
            old["population"] == "D0"
            and old["trigger"] == "finite_B_NVP"
            and old_nvp
            and not new_nvp
            and progress is not None
            and float(progress) > 0.0
        )
        safe_stall = bool(
            old["population"] == "D0"
            and old["trigger"] == "finite_B_NVP"
            and old_nvp
            and not new_nvp
            and (progress is None or float(progress) <= 0.0)
        )
        imitation_only = bool(
            old["population"] == "D0"
            and new["target_full_rmse"] < old["target_full_rmse"]
            and new_nvp
        )
        feature_cosine = float(np.dot(
            old["_feature"], new["_feature"],
        ) / (
            np.linalg.norm(old["_feature"])
            * np.linalg.norm(new["_feature"])
            + 1.0e-12
        ))
        rows.append({
            "anchor_id": anchor_id,
            "gamma": old["gamma"],
            "trigger": old["trigger"],
            "population": old["population"],
            "old_NVP": old_nvp,
            "new_NVP": new_nvp,
            "D0_actual_repair": actual_repair,
            "D0_safe_stall": safe_stall,
            "D0_imitation_only": imitation_only,
            "Dplus_retained": bool(
                old["population"] == "Dplus"
                and old_target_admissible
                and new_target_admissible
            ),
            "Dplus_regressed": bool(
                old["population"] == "Dplus"
                and old_target_admissible
                and not new_target_admissible
            ),
            "Dplus_target_latent_to_safe": bool(
                old["population"] == "Dplus"
                and not old_target_admissible
                and new_target_admissible
            ),
            "delta_all_K_admissible_fraction": (
                new["all_K_admissible"]
                - old["all_K_admissible"]
            ) / 16.0,
            "delta_B_orig_admissible": (
                new["B_orig_admissible"]
                - old["B_orig_admissible"]
            ),
            "delta_target_full_rmse": (
                new["target_full_rmse"]
                - old["target_full_rmse"]
            ),
            "delta_selected_one_step_progress": (
                None
                if old["selected_one_step_progress"] is None
                or new["selected_one_step_progress"] is None
                else (
                    new["selected_one_step_progress"]
                    - old["selected_one_step_progress"]
                )
            ),
            "target_phi_s_cosine": feature_cosine,
            "generated_K_RMS_drift": float(np.sqrt(np.mean(
                (new["_windows"] - old["_windows"]) ** 2
            ))),
        })

    def summarize(values):
        D0 = [row for row in values if row["population"] == "D0"]
        Dplus = [
            row for row in values if row["population"] == "Dplus"
        ]
        return {
            "contexts": len(values),
            "D0_contexts": len(D0),
            "Dplus_contexts": len(Dplus),
            "D0_actual_repairs": sum(
                row["D0_actual_repair"] for row in D0
            ),
            "D0_safe_stalls": sum(
                row["D0_safe_stall"] for row in D0
            ),
            "D0_imitation_only": sum(
                row["D0_imitation_only"] for row in D0
            ),
            "Dplus_retained": sum(
                row["Dplus_retained"] for row in Dplus
            ),
            "Dplus_regressed": sum(
                row["Dplus_regressed"] for row in Dplus
            ),
            "Dplus_target_latent_to_safe": sum(
                row["Dplus_target_latent_to_safe"] for row in Dplus
            ),
            "old_B_orig_NVP_rate": _mean([
                row["old_NVP"] for row in values
            ]),
            "new_B_orig_NVP_rate": _mean([
                row["new_NVP"] for row in values
            ]),
            "mean_delta_all_K_admissible_fraction": _mean([
                row["delta_all_K_admissible_fraction"]
                for row in values
            ]),
            "mean_delta_target_full_rmse": _mean([
                row["delta_target_full_rmse"] for row in values
            ]),
            "mean_target_phi_s_cosine": _mean([
                row["target_phi_s_cosine"] for row in values
            ]),
            "mean_generated_K_RMS_drift": _mean([
                row["generated_K_RMS_drift"] for row in values
            ]),
        }

    return {
        "pooled": summarize(rows),
        "per_gamma": {
            str(gamma): summarize([
                row for row in rows
                if round(row["gamma"], 8) == round(float(gamma), 8)
            ])
            for gamma in SP.GAMMAS
        },
        "per_trigger": {
            trigger: summarize([
                row for row in rows if row["trigger"] == trigger
            ])
            for trigger in sorted({row["trigger"] for row in rows})
        },
        "rows": rows,
    }


def _strip_probe_rows(rows):
    return [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in rows
    ]


def _raw_evaluation(
    arms,
    output_dir,
    *,
    ep0,
    M,
    bank_role,
    noise_seed,
    device,
    workers,
    executor=None,
):
    OE.M_PER_GAMMA = int(M)
    probe, _ = GPS.load_sfm_policy(arms[0]["checkpoint"], device="cpu")
    noise, noise_meta = OE._noise_bank(
        ep0=int(ep0), d=int(probe.d), seed=int(noise_seed),
    )
    del probe
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    records = []
    owns_executor = executor is None
    if owns_executor:
        context = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=int(workers), mp_context=context,
        )
    try:
        for arm in arms:
            cell = OE._evaluate_checkpoint(
                arm["checkpoint"],
                scene_profile="double_density_velocity_ood",
                ep0=int(ep0),
                noise=noise,
                noise_meta=noise_meta,
                device=device,
                cache_dir=cache_dir,
                executor=executor,
            )
            records.append({
                "name": arm["name"],
                "round": arm["round"],
                "phase": arm["phase"],
                "checkpoint": arm["checkpoint"],
                "checkpoint_sha256": FA._sha256_file(
                    arm["checkpoint"]
                ),
                "cell": cell,
            })
    finally:
        if owns_executor:
            executor.shutdown(wait=True)
    report = {
        "status": f"SFM_B1_RAW_M{int(M)}_COMPLETE",
        "scene_profile": "double_density_velocity_ood",
        "bank_role": str(bank_role),
        "bank": {
            "ep0": int(ep0),
            "M_per_gamma": int(M),
            "scenario_ids": list(range(int(ep0), int(ep0) + int(M))),
        },
        "noise_bank": noise_meta,
        "policy_semantics": (
            "raw temperature=1, NFE=8, no acquisition, verifier, repair, "
            "guidance, or fallback"
        ),
        "records": records,
    }
    _write_json(os.path.join(output_dir, "RAW_EVALUATION.json"), report)
    return report


def _save_checkpoint(policy, path, metadata):
    BX._save_checkpoint(policy, path, metadata)
    return {
        "checkpoint": os.path.abspath(path),
        "checkpoint_sha256": FA._sha256_file(path),
    }


def _parse_eval_rounds(value, final_round):
    rounds = tuple(sorted({
        int(value)
        for value in str(value).split(",")
        if str(value).strip()
    }))
    if (
        not rounds
        or rounds[0] != 0
        or any(value < 0 or value > int(final_round) for value in rounds)
        or int(final_round) not in rounds
    ):
        raise ValueError(
            "eval_rounds must include 0 and the final round, with no "
            "out-of-range entries"
        )
    return rounds


def _validate_resume_config(previous_config, current_config):
    compatibility_defaults = {
        "encoder_lr_ratio": 0.0,
        "neutral_replay": True,
    }
    compatibility_normalized = {}
    for key, value in current_config.items():
        if key in {"name", "rounds"}:
            continue
        previous_value = previous_config.get(key)
        if key not in previous_config and key in compatibility_defaults:
            previous_value = compatibility_defaults[key]
            compatibility_normalized[key] = previous_value
        if previous_value != value:
            raise RuntimeError(f"resume config changed: {key}")
    return compatibility_normalized


def _round_record_ref(path, record):
    marker = os.path.abspath(os.fspath(path))
    checkpoint = os.path.abspath(record["checkpoint"])
    round_i = int(record["round"])
    post_positive = os.path.join(
        os.path.dirname(checkpoint), f"round_{round_i:02d}_post_positive.pt",
    )
    if FA._sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("round checkpoint digest mismatch")
    return {
        "path": marker,
        "sha256": FA._sha256_file(marker),
        "round": round_i,
        "post_D0": checkpoint,
        "post_D0_sha256": record["checkpoint_sha256"],
        "post_Dplus": post_positive,
        "post_Dplus_sha256": FA._sha256_file(post_positive),
    }


def _delivery_lineage_refs(delivery, delivery_path, seen=()):
    """Authenticate current and recursively resumed round artifacts."""
    delivery_path = os.path.abspath(os.fspath(delivery_path))
    if delivery_path in seen:
        raise RuntimeError("cyclic resume delivery chain")
    current_paths = [
        os.path.abspath(path) for path in delivery.get("round_records", ())
    ]
    frozen_current = list(delivery.get("round_record_refs", ()))
    computed_current = []
    for path in current_paths:
        with open(path) as stream:
            record = json.load(stream)
        if record.get("status") != ROUND_STATUS:
            raise RuntimeError("invalid resume round marker")
        computed_current.append(_round_record_ref(path, record))
    if frozen_current:
        if frozen_current != computed_current:
            raise RuntimeError("current resume round snapshot changed")
        current_refs = frozen_current
    else:
        current_refs = computed_current

    prior_refs = []
    resume = delivery.get("resume")
    if resume:
        prior_path = os.path.abspath(resume["delivery"])
        payload = open(prior_path, "rb").read()
        if hashlib.sha256(payload).hexdigest() != resume.get("delivery_sha256"):
            raise RuntimeError("nested resume delivery digest mismatch")
        prior_delivery = json.loads(payload)
        expected = _delivery_lineage_refs(
            prior_delivery, prior_path, seen=(*seen, delivery_path),
        )
        snapshot = list(resume.get("round_record_refs", ()))
        if not snapshot:
            raise RuntimeError("nested resume lacks frozen prior-round refs")
        if snapshot != expected:
            raise RuntimeError("nested resume round snapshot changed")
        prior_refs = snapshot
    return [*prior_refs, *current_refs]


def _resume_artifacts(resume_root, cfg, *, source_sha, scenario_ep0):
    root = os.path.abspath(os.fspath(resume_root))
    delivery_path = os.path.join(root, "DELIVERY_COMPLETE.json")
    if not os.path.isfile(delivery_path):
        raise FileNotFoundError(delivery_path)
    with open(delivery_path) as stream:
        delivery = json.load(stream)
    if delivery.get("status") != STATUS:
        raise RuntimeError("resume source is not a completed neutral run")
    if delivery.get("checkpoint_sha256") != source_sha:
        raise RuntimeError("resume source uses another pretrained checkpoint")
    previous_config = delivery.get("config", {})
    current_config = asdict(cfg)
    compatibility_normalized = _validate_resume_config(
        previous_config, current_config,
    )
    current_round_records = list(delivery.get("round_records", ()))
    if not current_round_records:
        raise RuntimeError("resume delivery has no round records")
    round_record_refs = _delivery_lineage_refs(delivery, delivery_path)
    records = []
    for ref in round_record_refs:
        if FA._sha256_file(ref["path"]) != ref["sha256"]:
            raise RuntimeError("resume round marker digest mismatch")
        with open(ref["path"]) as stream:
            record = json.load(stream)
        if int(record["round"]) != int(ref["round"]):
            raise RuntimeError("resume round marker index changed")
        records.append(record)
    rounds = [int(record["round"]) for record in records]
    if rounds != list(range(1, max(rounds) + 1)):
        raise RuntimeError("resume rounds are not contiguous from one")
    resume_round = rounds[-1]
    if resume_round >= int(cfg.rounds):
        raise RuntimeError("resume source already reaches requested final round")
    expected_scenarios = set(range(
        int(scenario_ep0), int(scenario_ep0) + 2 * resume_round
    ))
    observed_scenarios = {
        int(scenario)
        for record in records for scenario in record.get("scenarios", ())
    }
    if observed_scenarios != expected_scenarios:
        raise RuntimeError("resume scenario schedule is incomplete or changed")
    final = records[-1]
    checkpoint = os.path.join(
        root, "checkpoints", f"round_{resume_round:02d}.pt"
    )
    optimizer = os.path.join(
        root, "checkpoints", f"round_{resume_round:02d}_optimizers.pt"
    )
    previous_executed = os.path.join(
        root, "rounds", f"round_{resume_round:02d}",
        "gather", "executed_round.pt",
    )
    encoder_reference = delivery.get("encoder_reference_probe")
    if encoder_reference is not None:
        encoder_reference_path = os.path.abspath(encoder_reference["path"])
        if (
            not os.path.isfile(encoder_reference_path)
            or FA._sha256_file(encoder_reference_path)
            != encoder_reference.get("sha256")
        ):
            raise RuntimeError("resume encoder reference digest mismatch")
    else:
        encoder_reference_path = None
    if FA._sha256_file(checkpoint) != final["checkpoint_sha256"]:
        raise RuntimeError("resume checkpoint hash mismatch")
    if FA._sha256_file(optimizer) != final["optimizer_state"]["sha256"]:
        raise RuntimeError("resume optimizer hash mismatch")
    expected_shard_sha = final["gather"]["executed_shard"]["sha256"]
    if FA._sha256_file(previous_executed) != expected_shard_sha:
        raise RuntimeError("resume GP-support shard hash mismatch")
    executed = OS.ExecutedRoundShard.load(previous_executed)
    positive = OS.positive_records(executed)
    support = {
        float(holder.contexts[int(row["context_id"])]["gamma"])
        for holder, row in positive
    }
    if support != set(map(float, SP.GAMMAS)):
        raise RuntimeError("resume GP support does not cover all gammas")
    return {
        "root": root,
        "delivery": delivery_path,
        "delivery_sha256": FA._sha256_file(delivery_path),
        "round_record_refs": round_record_refs,
        "resume_round": resume_round,
        "checkpoint": checkpoint,
        "checkpoint_sha256": final["checkpoint_sha256"],
        "optimizer": optimizer,
        "optimizer_sha256": final["optimizer_state"]["sha256"],
        "previous_executed": previous_executed,
        "previous_executed_sha256": expected_shard_sha,
        "encoder_reference": encoder_reference_path,
        "visual_encoder_sha256": delivery["visual_encoder_sha256"],
        "next_scenarios": [
            int(scenario_ep0) + 2 * resume_round,
            int(scenario_ep0) + 2 * resume_round + 1,
        ],
        "legacy_config_defaults": compatibility_normalized,
    }


def _restore_optimizer(
    optimizer, path, parameters, *, resume_round, inner_steps,
    parameter_names=None,
    neutral_replay=True,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("round", -1)) != int(resume_round):
        raise RuntimeError("optimizer round mismatch")
    state = payload.get("optimizer", {})
    groups = list(state.get("param_groups", ()))
    target_groups = list(optimizer.param_groups)
    if len(groups) != len(target_groups):
        raise RuntimeError("optimizer hyperparameters changed")
    for source, target in zip(groups, target_groups):
        source_hyper = {key: value for key, value in source.items() if key != "params"}
        target_hyper = {key: value for key, value in target.items() if key != "params"}
        if source_hyper != target_hyper:
            raise RuntimeError("optimizer hyperparameters changed")
    saved_names = payload.get("parameter_names")
    if saved_names is not None and list(saved_names) != list(parameter_names or ()):
        raise RuntimeError("optimizer named parameter order changed")
    ids = [
        parameter_id
        for group in groups for parameter_id in group.get("params", ())
    ]
    if len(ids) != len(parameters) or len(state.get("state", {})) != len(parameters):
        raise RuntimeError("optimizer parameter support is incomplete")
    updates_per_round = 1 + int(bool(neutral_replay))
    expected_step = updates_per_round * int(resume_round) * int(inner_steps)
    for parameter_id, parameter in zip(ids, parameters):
        values = state["state"].get(parameter_id)
        if values is None:
            raise RuntimeError("optimizer parameter state is missing")
        step = int(torch.as_tensor(values["step"]).item())
        if step != expected_step:
            raise RuntimeError("optimizer Adam step does not match global round")
        for key in ("exp_avg", "exp_avg_sq"):
            if tuple(values[key].shape) != tuple(parameter.shape):
                raise RuntimeError("optimizer moment shape mismatch")
    optimizer.load_state_dict(state)
    return {
        "round": int(resume_round),
        "expected_adam_step": expected_step,
        "updates_per_round": updates_per_round,
        "parameters": len(parameters),
        "parameter_names_authenticated": saved_names is not None,
        "legacy_parameter_order_reconstructed": saved_names is None,
        "param_group_hyperparameters": [
            {key: value for key, value in group.items() if key != "params"}
            for group in groups
        ],
    }


def run(args):
    eval_rounds = _parse_eval_rounds(args.eval_rounds, args.rounds)
    locked_eval_values = (
        args.locked_eval_checkpoint,
        args.locked_eval_round,
        args.locked_eval_sha256,
    )
    if any(value is not None for value in locked_eval_values) and not all(
        value is not None for value in locked_eval_values
    ):
        raise ValueError("locked prior-best evaluation requires path, round, and SHA")
    locked_eval = None
    if all(value is not None for value in locked_eval_values):
        locked_round = int(args.locked_eval_round)
        locked_checkpoint = os.path.abspath(args.locked_eval_checkpoint)
        locked_sha = str(args.locked_eval_sha256)
        if locked_round not in eval_rounds or locked_round in (0, int(args.rounds)):
            raise ValueError("locked prior-best round must be an interior eval round")
        if FA._sha256_file(locked_checkpoint) != locked_sha:
            raise RuntimeError("locked prior-best checkpoint SHA changed")
        locked_eval = {
            "round": locked_round,
            "checkpoint": locked_checkpoint,
            "checkpoint_sha256": locked_sha,
        }
    cfg = StudyConfig(
        name=str(args.name),
        rounds=int(args.rounds),
        scenarios_per_round=2,
        lr=float(args.lr),
        inner_steps=int(args.inner_steps),
        batch=128,
        ell=float(args.ell),
        gp_cap=int(args.gp_cap),
        selector=str(args.selector),
        encoder_lr_ratio=float(args.encoder_lr_ratio),
        neutral_replay=bool(args.neutral_replay),
        sample_seed=int(args.sample_seed),
        audit_seed=int(args.audit_seed),
        train_seed=int(args.train_seed),
        probe_seed=int(args.probe_seed),
    ).validate()
    output_root = os.path.abspath(args.output_root)
    if os.path.exists(output_root):
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    source = FA._source()
    if not source["tracked_worktree_clean"]:
        raise RuntimeError("study requires a clean frozen worktree")
    checkpoint = os.path.abspath(args.checkpoint)
    source_sha = FA._sha256_file(checkpoint)
    if source_sha != RA.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("source pretrained checkpoint SHA changed")
    os.makedirs(output_root)
    checkpoints_dir = os.path.join(output_root, "checkpoints")
    os.makedirs(checkpoints_dir)
    rounds_dir = os.path.join(output_root, "rounds")
    os.makedirs(rounds_dir)

    resume = (
        _resume_artifacts(
            args.resume_run_root,
            cfg,
            source_sha=source_sha,
            scenario_ep0=int(args.scenario_ep0),
        )
        if args.resume_run_root else None
    )
    if resume and int(resume["resume_round"]) not in eval_rounds:
        raise ValueError("resumed evaluation must include the anchor round")
    policy_checkpoint = resume["checkpoint"] if resume else checkpoint
    policy, _ = GPS.load_sfm_policy(policy_checkpoint, device=args.device)
    frozen = BS.configure_expansion_trainability(policy)
    if cfg.encoder_lr_ratio > 0.0:
        policy.enc_grid.requires_grad_(True)
    initial_visual_sha = BS.module_sha256(policy.enc_grid)
    if resume and initial_visual_sha != resume["visual_encoder_sha256"]:
        raise RuntimeError("resume visual encoder hash mismatch")
    trainable_names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    effective_frozen_names = [
        name for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    ]
    encoder_parameter_ids = {
        id(parameter) for parameter in policy.enc_grid.parameters()
    }
    main_parameters = [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad
        and id(parameter) not in encoder_parameter_ids
    ]
    encoder_parameters = [
        parameter for parameter in policy.enc_grid.parameters()
        if parameter.requires_grad
    ]
    parameters = [*main_parameters, *encoder_parameters]
    name_by_parameter_id = {
        id(parameter): name for name, parameter in policy.named_parameters()
    }
    optimizer_parameter_names = [
        name_by_parameter_id[id(parameter)] for parameter in parameters
    ]
    optimizer_groups = [{"params": main_parameters, "lr": cfg.lr}]
    if encoder_parameters:
        optimizer_groups.append({
            "params": encoder_parameters,
            "lr": cfg.lr * cfg.encoder_lr_ratio,
        })
    optimizer = torch.optim.Adam(optimizer_groups)
    optimizer_restore = (
        _restore_optimizer(
            optimizer,
            resume["optimizer"],
            parameters,
            resume_round=resume["resume_round"],
            inner_steps=cfg.inner_steps,
            parameter_names=optimizer_parameter_names,
            neutral_replay=cfg.neutral_replay,
        )
        if resume else None
    )
    round0_path = os.path.join(checkpoints_dir, "round_00.pt")
    if not resume:
        _save_checkpoint(policy, round0_path, {
            "study": STATUS,
            "round": 0,
            "phase": "pretrained",
            "source_checkpoint": checkpoint,
            "source_sha256": source_sha,
            "study_config": asdict(cfg),
        })
    current_checkpoint = policy_checkpoint
    encoder_reference_path = (
        resume["encoder_reference"] if resume else None
    )
    encoder_reference = (
        _load_encoder_reference(encoder_reference_path)
        if encoder_reference_path else None
    )
    previous_executed_path = (
        resume["previous_executed"] if resume else None
    )
    history = []
    milestone_arms = [{
        "name": "r0",
        "round": 0,
        "phase": "pretrained",
        "checkpoint": checkpoint,
    }]
    start_round = int(resume["resume_round"]) if resume else 0
    if resume:
        milestone_arms.append({
            "name": f"r{start_round}",
            "round": start_round,
            "phase": "resume_anchor_after_D0",
            "checkpoint": current_checkpoint,
        })
    if locked_eval is not None:
        if any(item["round"] == locked_eval["round"] for item in milestone_arms):
            raise ValueError("locked prior-best duplicates an existing milestone")
        milestone_arms.append({
            "name": f"locked_r{locked_eval['round']}",
            "round": locked_eval["round"],
            "phase": "locked_prior_best",
            "checkpoint": locked_eval["checkpoint"],
        })

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=context,
    ) as probe_executor:
        for round_i in range(start_round + 1, cfg.rounds + 1):
            started = time.perf_counter()
            round_dir = os.path.join(rounds_dir, f"round_{round_i:02d}")
            os.makedirs(round_dir)
            scenarios = tuple(range(
                int(args.scenario_ep0) + (round_i - 1) * 2,
                int(args.scenario_ep0) + round_i * 2,
            ))
            gather_dir = os.path.join(round_dir, "gather")
            current_sha = FA._sha256_file(current_checkpoint)
            RA.collect(
                current_checkpoint,
                scenarios=scenarios,
                gammas=tuple(map(float, SP.GAMMAS)),
                scene_profile=cfg.scene_profile,
                selector=cfg.selector,
                device=args.device,
                verifier_workers=int(args.workers),
                sample_seed=cfg.sample_seed,
                audit_seed=cfg.audit_seed,
                ell=cfg.ell,
                neutral_continuation=True,
                round_i=round_i,
                expected_checkpoint_sha256=current_sha,
                previous_executed_path=previous_executed_path,
                gp_cap=cfg.gp_cap,
                verifier_executor=probe_executor,
                T=cfg.T,
                outdir=gather_dir,
            )
            executed = OS.ExecutedRoundShard.load(
                os.path.join(gather_dir, "executed_round.pt")
            )
            _, neutral_records = _neutral_records(
                os.path.join(gather_dir, "neutral_round.pt")
            )
            positive_records = OS.positive_records(executed)
            if executed.Dminus:
                raise RuntimeError("ordinary executed D unexpectedly has D-")
            if not positive_records or not neutral_records:
                raise RuntimeError("round requires nonempty D+ and D0")

            anchor_path = os.path.join(
                round_dir, "trigger_anchors.pt",
            )
            anchors = _anchor_catalog(
                os.path.join(gather_dir, "repair_trace.pt"),
                per_gamma=int(args.probe_per_gamma),
                seed=cfg.probe_seed + round_i,
                output_path=anchor_path,
            )
            if not anchors:
                raise RuntimeError("round produced no guidance-trigger anchors")

            phase_before = _probe_checkpoint(
                current_checkpoint,
                anchors,
                phase="before_update",
                device=args.device,
                executor=probe_executor,
                batch=cfg.batch,
                selector=cfg.selector,
            )
            gradient = _gradient_diagnostics(
                policy,
                positive_records,
                neutral_records,
                batch=cfg.batch,
                device=args.device,
                seed=cfg.train_seed + round_i * 10_000_019,
            )
            fixed_before = {
                "Dplus": NS._fixed_loss(
                    policy,
                    positive_records,
                    batch=cfg.batch,
                    device=args.device,
                    seed=cfg.train_seed + round_i,
                ),
                "D0": NS._fixed_loss(
                    policy,
                    neutral_records,
                    batch=cfg.batch,
                    device=args.device,
                    seed=cfg.train_seed + round_i,
                ),
            }
            encoder_probe_records = [
                *positive_records, *neutral_records,
            ]
            if encoder_reference is None:
                encoder_reference_path = os.path.join(
                    output_root, "encoder_reference_probe.pt",
                )
                encoder_reference = _create_encoder_reference(
                    encoder_reference_path,
                    policy,
                    encoder_probe_records,
                    device=args.device,
                    anchor_round=start_round,
                    checkpoint_sha256=current_sha,
                )
            encoder_probe_before = _encoder_probe(
                policy, encoder_probe_records, device=args.device,
            )

            before_parameters = R2._module_snapshot(policy)
            positive_update = _population_update(
                policy,
                optimizer,
                positive_records,
                population="Dplus",
                inner_steps=cfg.inner_steps,
                batch=cfg.batch,
                device=args.device,
                seed=cfg.train_seed + round_i * 1_000_003,
            )
            after_positive_parameters = R2._module_snapshot(policy)
            post_positive_path = os.path.join(
                checkpoints_dir,
                f"round_{round_i:02d}_post_positive.pt",
            )
            _save_checkpoint(policy, post_positive_path, {
                "study": STATUS,
                "round": round_i,
                "phase": "post_Dplus",
                "study_config": asdict(cfg),
                "Dplus_records": len(positive_records),
                "D0_used": False,
            })
            phase_positive = _probe_checkpoint(
                post_positive_path,
                anchors,
                phase="after_Dplus",
                device=args.device,
                executor=probe_executor,
                batch=cfg.batch,
                selector=cfg.selector,
            )
            gradient_after_positive = _gradient_diagnostics(
                policy,
                positive_records,
                neutral_records,
                batch=cfg.batch,
                device=args.device,
                seed=cfg.train_seed + round_i * 10_000_019,
            )
            fixed_after_positive = {
                "Dplus": NS._fixed_loss(
                    policy,
                    positive_records,
                    batch=cfg.batch,
                    device=args.device,
                    seed=cfg.train_seed + round_i,
                ),
                "D0": NS._fixed_loss(
                    policy,
                    neutral_records,
                    batch=cfg.batch,
                    device=args.device,
                    seed=cfg.train_seed + round_i,
                ),
            }

            if cfg.neutral_replay:
                neutral_update = _population_update(
                    policy,
                    optimizer,
                    neutral_records,
                    population="D0",
                    inner_steps=cfg.inner_steps,
                    batch=cfg.batch,
                    device=args.device,
                    seed=(
                        cfg.train_seed + round_i * 1_000_003
                        + 500_000_000
                    ),
                )
            else:
                neutral_update = _skipped_population_update(
                    neutral_records, population="D0",
                )
            after_neutral_parameters = R2._module_snapshot(policy)
            round_checkpoint = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}.pt",
            )
            checkpoint_marker = _save_checkpoint(policy, round_checkpoint, {
                "study": STATUS,
                "round": round_i,
                "phase": (
                    "post_Dplus_then_D0"
                    if cfg.neutral_replay else "post_Dplus_D0_audit_only"
                ),
                "study_config": asdict(cfg),
                "Dplus_records": len(positive_records),
                "D0_records": len(neutral_records),
                "D0_original_verifier_y": 0,
                "D0_gp_eligible": False,
                "D0_used": bool(cfg.neutral_replay),
            })
            optimizer_path = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}_optimizers.pt",
            )
            torch.save({
                "round": round_i,
                "optimizer": optimizer.state_dict(),
                "parameter_names": optimizer_parameter_names,
            }, optimizer_path)
            phase_neutral = _probe_checkpoint(
                round_checkpoint,
                anchors,
                phase=(
                    "after_D0" if cfg.neutral_replay
                    else "after_Dplus_D0_audit_only"
                ),
                device=args.device,
                executor=probe_executor,
                batch=cfg.batch,
                selector=cfg.selector,
            )
            fixed_after_neutral = {
                "Dplus": NS._fixed_loss(
                    policy,
                    positive_records,
                    batch=cfg.batch,
                    device=args.device,
                    seed=cfg.train_seed + round_i,
                ),
                "D0": NS._fixed_loss(
                    policy,
                    neutral_records,
                    batch=cfg.batch,
                    device=args.device,
                    seed=cfg.train_seed + round_i,
                ),
            }
            encoder_probe_after = _encoder_probe(
                policy, encoder_probe_records, device=args.device,
            )
            encoder_cumulative = _encoder_reference_comparison(
                policy, encoder_reference, device=args.device,
            )
            if (
                cfg.encoder_lr_ratio == 0.0
                and BS.module_sha256(policy.enc_grid) != initial_visual_sha
            ):
                raise RuntimeError("visual encoder changed")

            paired_positive = _probe_comparison(
                phase_before, phase_positive,
            )
            paired_neutral_increment = _probe_comparison(
                phase_positive, phase_neutral,
            )
            paired_total = _probe_comparison(
                phase_before, phase_neutral,
            )
            probe_payload = {
                "status": PROBE_STATUS,
                "round": round_i,
                "semantics": {
                    "contexts": (
                        "same stored Markov context and pedestrian state"
                    ),
                    "latent_bank": "same original K=16 x0 rows",
                    "B_orig": "same original RBF-selected candidate IDs",
                    "verifier": SM.verifier_manifest(),
                    "selection": (
                        "exact y=1 AND nominal-Hp gate, then "
                        f"{cfg.selector}"
                    ),
                    "claim": (
                        "local generator correction/resubstitution; not "
                        "closed-loop generalization"
                    ),
                },
                "phases": {
                    "before_update": _strip_probe_rows(phase_before),
                    "after_Dplus": _strip_probe_rows(phase_positive),
                    "after_D0": _strip_probe_rows(phase_neutral),
                },
                "comparisons": {
                    "Dplus_increment": paired_positive,
                    "D0_increment": paired_neutral_increment,
                    "total": paired_total,
                },
            }
            _write_json(
                os.path.join(round_dir, "PAIRED_TRIGGER_PROBE.json"),
                probe_payload,
            )

            phase_arms = [
                {
                    "name": f"r{round_i}_before",
                    "round": round_i,
                    "phase": "before_update",
                    "checkpoint": current_checkpoint,
                },
                {
                    "name": f"r{round_i}_post_Dplus",
                    "round": round_i,
                    "phase": "after_Dplus",
                    "checkpoint": post_positive_path,
                },
                {
                    "name": f"r{round_i}_post_D0",
                    "round": round_i,
                    "phase": (
                        "after_D0" if cfg.neutral_replay
                        else "after_Dplus_D0_audit_only"
                    ),
                    "checkpoint": round_checkpoint,
                },
            ]
            raw_m2_dir = os.path.join(round_dir, "same_lineage_raw_M2")
            os.makedirs(raw_m2_dir)
            raw_m2 = _raw_evaluation(
                phase_arms,
                raw_m2_dir,
                ep0=scenarios[0],
                M=2,
                bank_role=(
                    "same two scenarios as this round's gather; "
                    "paired fit diagnostic only"
                ),
                noise_seed=int(args.noise_seed) + round_i,
                device=args.device,
                workers=int(args.workers),
                executor=probe_executor,
            )

            gather_complete = json.load(open(
                os.path.join(gather_dir, "COMPLETE.json")
            ))
            record = {
                "status": ROUND_STATUS,
                "round": round_i,
                "scenarios": list(scenarios),
                "lineages": len(scenarios) * len(SP.GAMMAS),
                "gather": gather_complete,
                "Dplus": len(positive_records),
                "D0": len(neutral_records),
                "gradient_before_update": gradient,
                "gradient_after_Dplus_before_D0": (
                    gradient_after_positive
                ),
                "fixed_loss": {
                    "before": fixed_before,
                    "after_Dplus": fixed_after_positive,
                    "after_D0": fixed_after_neutral,
                },
                "updates": {
                    "Dplus": positive_update,
                    "D0": neutral_update,
                    "phase_order": (
                        ["Dplus", "D0"]
                        if cfg.neutral_replay else ["Dplus"]
                    ),
                    "same_lr_and_inner_steps": bool(cfg.neutral_replay),
                },
                "parameter_relative_drift": {
                    "Dplus_increment": R2._module_relative_drift(
                        before_parameters, after_positive_parameters,
                    ),
                    "D0_increment": R2._module_relative_drift(
                        after_positive_parameters, after_neutral_parameters,
                    ),
                    "total": R2._module_relative_drift(
                        before_parameters, after_neutral_parameters,
                    ),
                },
                "encoder_diagnostics": {
                    **_encoder_probe_comparison(
                        encoder_probe_before, encoder_probe_after,
                    ),
                    "relative_parameter_drift": (
                        R2._module_relative_drift(
                            before_parameters, after_neutral_parameters,
                        )["E_g"]
                    ),
                    "lr": (
                        0.0 if cfg.encoder_lr_ratio == 0.0
                        else cfg.lr * cfg.encoder_lr_ratio
                    ),
                    "trainable": bool(cfg.encoder_lr_ratio > 0.0),
                    "cumulative_from_reference": encoder_cumulative,
                },
                "paired_trigger_probe": {
                    "file": os.path.join(
                        round_dir, "PAIRED_TRIGGER_PROBE.json",
                    ),
                    "Dplus_increment": paired_positive["pooled"],
                    "D0_increment": paired_neutral_increment["pooled"],
                    "total": paired_total["pooled"],
                },
                "same_lineage_raw_M2": {
                    "file": os.path.join(
                        raw_m2_dir, "RAW_EVALUATION.json",
                    ),
                    "records": [
                        {
                            "name": row["name"],
                            "pooled": row["cell"]["summary"]["pooled"],
                        }
                        for row in raw_m2["records"]
                    ],
                },
                **checkpoint_marker,
                "optimizer_state": {
                    "path": optimizer_path,
                    "sha256": FA._sha256_file(optimizer_path),
                    "persistent_across_rounds": True,
                    "single_Adam_across_active_updates": True,
                    "D0_replay_enabled": bool(cfg.neutral_replay),
                },
                "wall_seconds": time.perf_counter() - started,
            }
            _write_json(
                os.path.join(round_dir, "ROUND_COMPLETE.json"), record,
            )
            with open(
                os.path.join(output_root, "metrics.jsonl"), "a",
            ) as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({
                "round": round_i,
                "Dplus": len(positive_records),
                "D0": len(neutral_records),
                "D0_actual_repairs": (
                    paired_neutral_increment["pooled"][
                        "D0_actual_repairs"
                    ]
                ),
                "Dplus_regressed": (
                    paired_positive["pooled"]["Dplus_regressed"]
                ),
                "wall_seconds": record["wall_seconds"],
            }), flush=True)
            history.append(record)
            if round_i in eval_rounds:
                milestone_arms.append({
                    "name": f"r{round_i}",
                    "round": round_i,
                    "phase": (
                        "after_D0" if cfg.neutral_replay
                        else "after_Dplus_D0_audit_only"
                    ),
                    "checkpoint": round_checkpoint,
                })
            current_checkpoint = round_checkpoint
            previous_executed_path = os.path.join(
                gather_dir, "executed_round.pt",
            )

    disjoint_dir = os.path.join(
        output_root, f"disjoint_raw_M{int(args.eval_M)}",
    )
    os.makedirs(disjoint_dir)
    disjoint = _raw_evaluation(
        milestone_arms,
        disjoint_dir,
        ep0=int(args.eval_ep0),
        M=int(args.eval_M),
        bank_role="disjoint fixed CRN metric bank",
        noise_seed=int(args.noise_seed),
        device=args.device,
        workers=int(args.workers),
    )
    complete = {
        "status": STATUS,
        "source": source,
        "checkpoint": checkpoint,
        "checkpoint_sha256": source_sha,
        "config": asdict(cfg),
        "trainability_configure_initially_frozen": frozen,
        "frozen_parameters": effective_frozen_names,
        "initial_visual_encoder_sha256": initial_visual_sha,
        "visual_encoder_sha256": BS.module_sha256(policy.enc_grid),
        "encoder_reference_probe": {
            "path": encoder_reference_path,
            "sha256": FA._sha256_file(encoder_reference_path),
            "anchor_round": int(encoder_reference["anchor_round"]),
            "records": int(encoder_reference["records"]),
        },
        "optimizer_groups": [
            {
                "name": "trunk_head_low_history",
                "lr": cfg.lr,
                "parameters": len(main_parameters),
            },
            *([{
                "name": "enc_grid",
                "lr": cfg.lr * cfg.encoder_lr_ratio,
                "parameters": len(encoder_parameters),
            }] if encoder_parameters else []),
        ],
        "rounds": int(cfg.rounds),
        "rounds_run_this_invocation": len(history),
        "resume": resume,
        "optimizer_restore": optimizer_restore,
        "locked_prior_best": locked_eval,
        "trainable_parameter_names": trainable_names,
        "round_records": [
            os.path.join(
                rounds_dir,
                f"round_{record['round']:02d}",
                "ROUND_COMPLETE.json",
            )
            for record in history
        ],
        "round_record_refs": [
            _round_record_ref(
                os.path.join(
                    rounds_dir,
                    f"round_{record['round']:02d}",
                    "ROUND_COMPLETE.json",
                ),
                record,
            )
            for record in history
        ],
        "disjoint_raw_evaluation": {
            "file": os.path.join(
                disjoint_dir, "RAW_EVALUATION.json",
            ),
            "M_per_gamma": int(args.eval_M),
            "ep0": int(args.eval_ep0),
            "rounds": list(eval_rounds),
            "records": [
                {
                    "name": row["name"],
                    "round": row["round"],
                    "checkpoint": row["cell"]["checkpoint"],
                    "checkpoint_sha256": row["cell"]["checkpoint_sha256"],
                    "pooled": row["cell"]["summary"]["pooled"],
                }
                for row in disjoint["records"]
            ],
        },
        "scientific_scope": {
            "same_lineage_M2": "fit/behavior diagnostic only",
            "fixed_trigger_probe": (
                "causal local generator audit at remembered contexts"
            ),
            f"disjoint_M{int(args.eval_M)}": (
                "small screening metric, not final confirmation"
            ),
            "D0": (
                "teacher action with immutable exact y=0; never a "
                "certificate-positive claim"
            ),
            "ordinary_replay_window": (
                "W=1 current-round executed D+ only; this follows the "
                "full-trajectory one-pass study, not the older query-W2 B1"
            ),
        },
    }
    finished_source = FA._source()
    if (
        not finished_source["tracked_worktree_clean"]
        or finished_source["commit"] != source["commit"]
    ):
        raise RuntimeError("source worktree changed during neutral training")
    _write_json(
        os.path.join(output_root, "DELIVERY_COMPLETE.json"), complete,
    )
    return complete


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--resume-run-root",
        help=(
            "completed neutral run to continue with model, Adam, global "
            "round/seed schedule, and previous-round GP support intact"
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--name", default="lr3em5_s04")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--scenario-ep0", type=int, default=DEFAULT_SCENARIO_EP0)
    parser.add_argument("--eval-ep0", type=int, default=DEFAULT_EVAL_EP0)
    parser.add_argument("--eval-M", type=int, default=20)
    parser.add_argument(
        "--eval-rounds",
        default="0,1,2",
        help=(
            "comma-separated disjoint-M20 checkpoints; must include 0 "
            "and the final round"
        ),
    )
    parser.add_argument("--locked-eval-checkpoint")
    parser.add_argument("--locked-eval-round", type=int)
    parser.add_argument("--locked-eval-sha256")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    parser.add_argument(
        "--selector",
        choices=("margin", "progress_gated_margin"),
        default="margin",
    )
    parser.add_argument(
        "--encoder-lr-ratio", type=float, default=0.0,
        help="0 keeps E_g frozen; the only creative sanity value is 0.1",
    )
    parser.add_argument(
        "--no-neutral-replay", dest="neutral_replay",
        action="store_false",
        help="collect and audit D0 but skip its optimizer update",
    )
    parser.set_defaults(neutral_replay=True)
    parser.add_argument("--ell", type=float, default=RA.DEFAULT_ELL)
    parser.add_argument("--gp-cap", type=int, default=DEFAULT_GP_CAP)
    parser.add_argument("--probe-per-gamma", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=700_000)
    parser.add_argument("--audit-seed", type=int, default=2_026_073_0)
    parser.add_argument("--train-seed", type=int, default=2_026_073_1)
    parser.add_argument("--probe-seed", type=int, default=2_026_073_2)
    parser.add_argument("--noise-seed", type=int, default=2_026_073_3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if int(args.eval_M) not in (10, 20):
        raise ValueError("screening metric bank must be M=10 or M=20/gamma")
    run(args)
    print(os.path.join(
        args.output_root, "DELIVERY_COMPLETE.json",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
