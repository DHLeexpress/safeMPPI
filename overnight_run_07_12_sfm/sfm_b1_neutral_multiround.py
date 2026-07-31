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
        if self.selector != "margin":
            raise ValueError("this study is pinned to max-step-margin")
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
        optimizer.step()
        losses.append(float(loss))
        exposure_hashes.append(_sha256_jsonable(identities))
    policy.eval()
    encoder_after = BS.module_sha256(policy.enc_grid)
    if encoder_after != encoder_before:
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
        "mass": _compact_mass(accounting),
        "encoder_sha_before": encoder_before,
        "encoder_sha_after": encoder_after,
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
            selector="margin",
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


def run(args):
    eval_rounds = _parse_eval_rounds(args.eval_rounds, args.rounds)
    cfg = StudyConfig(
        name=str(args.name),
        rounds=int(args.rounds),
        scenarios_per_round=2,
        lr=float(args.lr),
        inner_steps=int(args.inner_steps),
        batch=128,
        ell=float(args.ell),
        gp_cap=int(args.gp_cap),
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

    policy, _ = GPS.load_sfm_policy(checkpoint, device=args.device)
    frozen = BS.configure_expansion_trainability(policy)
    visual_sha = BS.module_sha256(policy.enc_grid)
    parameters = [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
    round0_path = os.path.join(checkpoints_dir, "round_00.pt")
    _save_checkpoint(policy, round0_path, {
        "study": STATUS,
        "round": 0,
        "phase": "pretrained",
        "source_checkpoint": checkpoint,
        "source_sha256": source_sha,
        "study_config": asdict(cfg),
    })
    current_checkpoint = checkpoint
    previous_executed_path = None
    history = []
    milestone_arms = [{
        "name": "r0",
        "round": 0,
        "phase": "pretrained",
        "checkpoint": checkpoint,
    }]

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=context,
    ) as probe_executor:
        for round_i in range(1, cfg.rounds + 1):
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
                selector="margin",
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

            neutral_update = _population_update(
                policy,
                optimizer,
                neutral_records,
                population="D0",
                inner_steps=cfg.inner_steps,
                batch=cfg.batch,
                device=args.device,
                seed=cfg.train_seed + round_i * 1_000_003 + 500_000_000,
            )
            after_neutral_parameters = R2._module_snapshot(policy)
            round_checkpoint = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}.pt",
            )
            checkpoint_marker = _save_checkpoint(policy, round_checkpoint, {
                "study": STATUS,
                "round": round_i,
                "phase": "post_Dplus_then_D0",
                "study_config": asdict(cfg),
                "Dplus_records": len(positive_records),
                "D0_records": len(neutral_records),
                "D0_original_verifier_y": 0,
                "D0_gp_eligible": False,
            })
            optimizer_path = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}_optimizers.pt",
            )
            torch.save({
                "round": round_i,
                "optimizer": optimizer.state_dict(),
            }, optimizer_path)
            phase_neutral = _probe_checkpoint(
                round_checkpoint,
                anchors,
                phase="after_D0",
                device=args.device,
                executor=probe_executor,
                batch=cfg.batch,
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
            if BS.module_sha256(policy.enc_grid) != visual_sha:
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
                        "exact y=1 AND nominal-Hp gate, then max margin"
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
                    "phase": "after_D0",
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
                    "phase_order": ["Dplus", "D0"],
                    "same_lr_and_inner_steps": True,
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
                    "single_Adam_across_Dplus_and_D0": True,
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
                    "phase": "after_D0",
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
        "frozen_parameters": frozen,
        "visual_encoder_sha256": visual_sha,
        "rounds": len(history),
        "round_records": [
            os.path.join(
                rounds_dir,
                f"round_{index:02d}",
                "ROUND_COMPLETE.json",
            )
            for index in range(1, len(history) + 1)
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
            "disjoint_M20": "small screening metric, not final confirmation",
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
    _write_json(
        os.path.join(output_root, "DELIVERY_COMPLETE.json"), complete,
    )
    return complete


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
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
    if int(args.eval_M) != 20:
        raise ValueError("sanity metric bank is pinned to M=20/gamma")
    run(args)
    print(os.path.join(
        args.output_root, "DELIVERY_COMPLETE.json",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
