"""Round-1 sanity study for isolated neutral-repair CFM replay.

The study has one immutable gather shared by every arm:

1. collect ordinary exact-positive executed windows in ``D+``;
2. collect guided, exact-negative executed repairs separately in ``D0``;
3. apply the usual alpha=0 positive-only B1 update once;
4. clone that checkpoint and apply a dedicated whole-D0 CFM objective.

``D0`` is never relabeled, inserted into D+/D-, or exposed to the RBF GP.
This is an in-sample diagnostic on the same twenty scenario seeds used for
gathering, not a generalization or safety claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import multiprocessing as mp
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_offline_eval as OE
import sfm_b1_offline_store as OS
import sfm_b1_r2_alpha_replay as R2
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


STATUS = "SFM_B1_NEUTRAL_TEACHER_SANITY_COMPLETE"
TRAINING_STATUS = "SFM_B1_NEUTRAL_TEACHER_TRAINING_COMPLETE"
PROBE_STATUS = "SFM_B1_NEUTRAL_TEACHER_PROBE_COMPLETE"
DEFAULT_EP0 = 250_000
DEFAULT_M = 20
DEFAULT_ORDINARY_LR = 1.0e-5
DEFAULT_NEUTRAL_LRS = (3.0e-5, 1.0e-4)
DEFAULT_NEUTRAL_STEPS = (1, 4, 16)
DEFAULT_NOISE_SEED = 2_026_073_0
DEFAULT_TRAIN_SEED = 2_026_073_1
DEFAULT_PROBE_PER_GAMMA = 5


class _SingleRecent:
    """Minimal one-round view required by the original B1 replay."""

    def __init__(self, shard):
        self._shard = shard
        self.window = 1

    @property
    def rounds(self):
        return [self._shard]

    def positive_records(self):
        return OS.positive_records(self._shard)

    def negative_records(self):
        return OS.negative_records(self._shard)


class _NeutralHolder:
    """Context holder compatible with the production hierarchy helpers."""

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


def _neutral_records(payload):
    if payload.get("status") != "SFM_B1_NEUTRAL_ROUND_COMPLETE":
        raise ValueError("not an authenticated neutral-round payload")
    holder = _NeutralHolder(payload["round"])
    records = []
    for expected, source in enumerate(payload["records"]):
        verifier = source.get("verifier_result", {})
        if (
            int(source["neutral_id"]) != expected
            or source["population"] != "D0"
            or source["semantic_label"] != "neutral"
            or int(source["verifier_y"]) != 0
            or not verifier.get("resolved")
            or int(verifier.get("y", -1)) != 0
            or not verifier.get("full_h")
            or int(verifier.get("terminal_step", -1)) != SP.H
            or source["train_eligible"]
            or source["replay_default"]
            or source["gp_eligible"]
        ):
            raise RuntimeError("D0 semantics changed before teacher replay")
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
            "original_verifier_y": 0,
        }
        if (
            tuple(row["controls"].shape) != (SP.H, 2)
            or tuple(row["x0"].shape) != (2 * SP.H,)
        ):
            raise RuntimeError("invalid D0 controls/x0 shape")
        holder.windows.append(row)
        records.append((holder, row))
    if len(records) != int(payload["summary"]["D0"]):
        raise RuntimeError("D0 payload count mismatch")
    return holder, records


def _objective(
    policy, records, mass, *, batch, device, seed, backward,
):
    ordered = BS.hierarchical_order(records, int(seed))
    visited = []
    total_loss = 0.0
    for start in range(0, len(ordered), int(batch)):
        values = ordered[start:start + int(batch)]
        grid, low, hist, controls = BS._tensor_batch(values, device)
        context = policy.ctx_from(grid, low, hist)
        weights = torch.as_tensor(
            [
                len(values) * mass[
                    (id(holder), int(row["query_id"]))
                ]
                for holder, row in values
            ],
            dtype=controls.dtype,
            device=device,
        )
        torch.manual_seed(int(seed) + start)
        loss = policy.cfm_loss(controls, context, weights=weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite whole-buffer CFM loss")
        if backward:
            loss.backward()
        total_loss += float(loss.detach())
        visited.extend(
            (int(holder.round_i), int(row["query_id"]))
            for holder, row in values
        )
    expected = {
        (int(holder.round_i), int(row["query_id"]))
        for holder, row in records
    }
    if len(visited) != len(expected) or set(visited) != expected:
        raise RuntimeError("whole-buffer objective duplicated or omitted support")
    return total_loss, visited


def _fixed_loss(policy, records, *, batch, device, seed):
    mass, _ = BS.hierarchy_mass(records)
    with torch.no_grad():
        loss, _ = _objective(
            policy,
            records,
            mass,
            batch=batch,
            device=device,
            seed=seed,
            backward=False,
        )
    return float(loss)


def _neutral_update(
    policy,
    optimizer,
    records,
    *,
    steps,
    batch,
    device,
    seed,
):
    if not records or int(steps) < 1:
        raise ValueError("neutral replay requires nonempty D0 and positive steps")
    mass, accounting = BS.hierarchy_mass(records)
    encoder_before = BS.module_sha256(policy.enc_grid)
    losses = []
    policy.train()
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss, visited = _objective(
            policy,
            records,
            mass,
            batch=batch,
            device=device,
            seed=int(seed) + step * 1_000_003,
            backward=True,
        )
        optimizer.step()
        if len(visited) != len(records):
            raise RuntimeError("neutral exposure count mismatch")
        losses.append(float(loss))
    policy.eval()
    encoder_after = BS.module_sha256(policy.enc_grid)
    if encoder_after != encoder_before:
        raise RuntimeError("visual encoder changed during neutral replay")
    return {
        "semantic_label": "neutral_teacher_despite_exact_y0",
        "records": len(records),
        "inner_steps": int(steps),
        "optimizer_steps": int(steps),
        "sample_exposures": len(records) * int(steps),
        "exact_once_per_inner_step": True,
        "fresh_cfm_base_each_exposure": True,
        "stored_x0_used_for_training": False,
        "losses": losses,
        "mass": _compact_mass(accounting),
        "encoder_sha_before": encoder_before,
        "encoder_sha_after": encoder_after,
    }


def _gradient(policy, records, *, batch, device, seed):
    mass, _ = BS.hierarchy_mass(records)
    policy.zero_grad(set_to_none=True)
    _objective(
        policy,
        records,
        mass,
        batch=batch,
        device=device,
        seed=seed,
        backward=True,
    )
    value = BS._gradient_snapshot(policy)
    policy.zero_grad(set_to_none=True)
    return value


def _cosine(first, second):
    common = sorted(set(first) & set(second))
    numerator = sum(
        float((first[key].double() * second[key].double()).sum())
        for key in common
        if first[key] is not None and second[key] is not None
    )
    first_norm = sum(
        float(first[key].double().square().sum())
        for key in common if first[key] is not None
    ) ** 0.5
    second_norm = sum(
        float(second[key].double().square().sum())
        for key in common if second[key] is not None
    ) ** 0.5
    return (
        numerator / (first_norm * second_norm)
        if first_norm and second_norm else None
    )


def _arm_name(lr, steps):
    lr_text = f"{float(lr):.0e}".replace("-", "m")
    return f"neutral_lr{lr_text}_s{int(steps):02d}"


def _summarize_ordinary(value):
    if value["path"] != "positive_only":
        raise RuntimeError("alpha=0 ordinary phase left positive-only path")
    return {
        "path": value["path"],
        "eligible": int(value["eligible"]),
        "visited": len(value["visited"]),
        "loss": float(value["loss"]),
        "optimizer_steps": int(value["optimizer_steps"]),
        "mass": _compact_mass(value["mass"]),
    }


def _train(
    checkpoint,
    gather_dir,
    output_dir,
    *,
    neutral_lrs,
    neutral_steps,
    ordinary_lr,
    batch,
    device,
    seed,
):
    executed = OS.ExecutedRoundShard.load(
        os.path.join(gather_dir, "executed_round.pt")
    )
    neutral_payload = torch.load(
        os.path.join(gather_dir, "neutral_round.pt"),
        map_location="cpu",
        weights_only=False,
    )
    neutral_holder, neutral_records = _neutral_records(neutral_payload)
    if executed.Dminus:
        raise RuntimeError("neutral collector unexpectedly populated ordinary D-")
    if not executed.Dplus or not neutral_records:
        raise RuntimeError("sanity study requires nonempty D+ and D0")

    os.makedirs(output_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    frozen = BS.configure_expansion_trainability(policy)
    encoder_sha = BS.module_sha256(policy.enc_grid)
    positive_records = OS.positive_records(executed)
    ordinary_before = R2._module_snapshot(policy)
    ordinary_fixed_before = {
        "Dplus": _fixed_loss(
            policy,
            positive_records,
            batch=batch,
            device=device,
            seed=int(seed) + 20_000_000,
        ),
        "D0": _fixed_loss(
            policy,
            neutral_records,
            batch=batch,
            device=device,
            seed=int(seed) + 20_000_001,
        ),
    }
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters()
         if parameter.requires_grad],
        lr=float(ordinary_lr),
    )
    ordinary = BS.signed_update(
        policy,
        optimizer,
        _SingleRecent(executed),
        alpha=0.0,
        batch=int(batch),
        device=device,
        seed=int(seed),
    )
    ordinary = _summarize_ordinary(ordinary)
    if ordinary["optimizer_steps"] != 1:
        raise RuntimeError("ordinary alpha=0 phase must use one Adam step")
    policy.eval()
    ordinary_fixed_after = {
        "Dplus": _fixed_loss(
            policy,
            positive_records,
            batch=batch,
            device=device,
            seed=int(seed) + 20_000_000,
        ),
        "D0": _fixed_loss(
            policy,
            neutral_records,
            batch=batch,
            device=device,
            seed=int(seed) + 20_000_001,
        ),
    }
    ordinary_drift = R2._module_relative_drift(
        ordinary_before, R2._module_snapshot(policy)
    )
    if BS.module_sha256(policy.enc_grid) != encoder_sha:
        raise RuntimeError("visual encoder changed during ordinary replay")
    ordinary_checkpoint = os.path.join(
        checkpoint_dir, "post_positive.pt"
    )
    BX._save_checkpoint(policy, ordinary_checkpoint, {
        "study": STATUS,
        "phase": "ordinary_Dplus",
        "alpha": 0.0,
        "lr": float(ordinary_lr),
        "optimizer_steps": 1,
        "neutral_used": False,
    })

    positive_gradient = _gradient(
        policy,
        positive_records,
        batch=batch,
        device=device,
        seed=int(seed) + 10_000_000,
    )
    neutral_gradient = _gradient(
        policy,
        neutral_records,
        batch=batch,
        device=device,
        seed=int(seed) + 10_000_000,
    )
    gradient_cosine = _cosine(positive_gradient, neutral_gradient)
    policy.zero_grad(set_to_none=True)
    del policy, optimizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    arms = [
        {
            "name": "r0",
            "phase": "pretrained",
            "checkpoint": os.path.abspath(checkpoint),
            "checkpoint_sha256": FA._sha256_file(checkpoint),
            "neutral_lr": None,
            "neutral_steps": 0,
        },
        {
            "name": "post_positive",
            "phase": "ordinary_Dplus",
            "checkpoint": ordinary_checkpoint,
            "checkpoint_sha256": FA._sha256_file(ordinary_checkpoint),
            "neutral_lr": None,
            "neutral_steps": 0,
        },
    ]
    arm_logs = {}
    for lr in neutral_lrs:
        for steps in neutral_steps:
            name = _arm_name(lr, steps)
            policy, _ = GPS.load_sfm_policy(
                ordinary_checkpoint, device=device
            )
            BS.configure_expansion_trainability(policy)
            before = R2._module_snapshot(policy)
            fixed_before = {
                "Dplus": _fixed_loss(
                    policy,
                    positive_records,
                    batch=batch,
                    device=device,
                    seed=int(seed) + 20_000_000,
                ),
                "D0": _fixed_loss(
                    policy,
                    neutral_records,
                    batch=batch,
                    device=device,
                    seed=int(seed) + 20_000_001,
                ),
            }
            optimizer = torch.optim.Adam(
                [parameter for parameter in policy.parameters()
                 if parameter.requires_grad],
                lr=float(lr),
            )
            update = _neutral_update(
                policy,
                optimizer,
                neutral_records,
                steps=int(steps),
                batch=int(batch),
                device=device,
                seed=int(seed) + 30_000_000,
            )
            fixed_after = {
                "Dplus": _fixed_loss(
                    policy,
                    positive_records,
                    batch=batch,
                    device=device,
                    seed=int(seed) + 20_000_000,
                ),
                "D0": _fixed_loss(
                    policy,
                    neutral_records,
                    batch=batch,
                    device=device,
                    seed=int(seed) + 20_000_001,
                ),
            }
            drift = R2._module_relative_drift(
                before, R2._module_snapshot(policy)
            )
            arm_checkpoint = os.path.join(checkpoint_dir, f"{name}.pt")
            BX._save_checkpoint(policy, arm_checkpoint, {
                "study": STATUS,
                "phase": "neutral_D0_teacher",
                "ordinary_checkpoint": ordinary_checkpoint,
                "neutral_semantic_label": "neutral",
                "original_verifier_y": 0,
                "neutral_lr": float(lr),
                "neutral_inner_steps": int(steps),
                "neutral_gp_eligible": False,
            })
            arm_logs[name] = {
                "update": update,
                "fixed_loss": {
                    "before": fixed_before,
                    "after": fixed_after,
                },
                "module_relative_parameter_drift": drift,
            }
            arms.append({
                "name": name,
                "phase": "neutral_D0_teacher",
                "checkpoint": arm_checkpoint,
                "checkpoint_sha256": FA._sha256_file(arm_checkpoint),
                "neutral_lr": float(lr),
                "neutral_steps": int(steps),
            })
            del policy, optimizer
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    report = {
        "status": TRAINING_STATUS,
        "source": FA._source(),
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": FA._sha256_file(checkpoint),
        "gather_dir": os.path.abspath(gather_dir),
        "executed_shard": {
            "D": len(executed.D),
            "Dplus": len(executed.Dplus),
            "Dminus": len(executed.Dminus),
        },
        "neutral_shard": {
            "D0": len(neutral_records),
            "all_original_verifier_y": 0,
            "gp_eligible": 0,
            "ordinary_replay_eligible": 0,
        },
        "ordinary": {
            "alpha": 0.0,
            "lr": float(ordinary_lr),
            **ordinary,
            "fixed_loss": {
                "before": ordinary_fixed_before,
                "after": ordinary_fixed_after,
            },
            "module_relative_parameter_drift": ordinary_drift,
        },
        "Dplus_D0_gradient_cosine_at_post_positive": gradient_cosine,
        "frozen_parameters": frozen,
        "visual_encoder_sha256": encoder_sha,
        "arms": arms,
        "arm_logs": arm_logs,
        "neutral_holder_round": neutral_holder.round_i,
    }
    _write_json(os.path.join(output_dir, "TRAINING_COMPLETE.json"), report)
    return report, neutral_holder, neutral_records


def _probe_subset(holder, records, per_gamma, seed):
    grouped = defaultdict(list)
    for pair in records:
        context = holder.contexts[int(pair[1]["context_id"])]
        grouped[round(float(context["gamma"]), 8)].append(pair)
    generator = np.random.default_rng(int(seed))
    selected = []
    for gamma in map(float, SP.GAMMAS):
        values = grouped[round(gamma, 8)]
        order = generator.permutation(len(values))
        selected.extend(values[index] for index in order[:int(per_gamma)])
    return selected


def _probe_one_arm(
    arm,
    probe_records,
    *,
    K,
    batch,
    device,
    seed,
    executor,
):
    policy, _ = GPS.load_sfm_policy(arm["checkpoint"], device=device)
    policy.eval()
    tasks = []
    measurements = []
    for start in range(0, len(probe_records), int(batch)):
        values = probe_records[start:start + int(batch)]
        grid, low, hist, teacher = BS._tensor_batch(values, device)
        context = policy.ctx_from(grid, low, hist)
        latent_rows = []
        for holder, row in values:
            stored = np.asarray(row["x0"], np.float32)
            generator = np.random.default_rng(FA._keyed_seed(
                int(seed),
                int(holder.contexts[int(row["context_id"])]["scenario_id"]),
                f"{float(holder.contexts[int(row['context_id'])]['gamma']):.8f}",
                int(holder.contexts[int(row["context_id"])]["step"]),
                int(row["query_id"]),
            ))
            extra = generator.standard_normal(
                (int(K) - 1, int(policy.d)), dtype=np.float32
            )
            latent_rows.append(np.concatenate([stored[None], extra], axis=0))
        latents = torch.as_tensor(
            np.stack(latent_rows), device=device, dtype=context.dtype
        )
        with torch.no_grad():
            windows = BE.integrate_latents(
                policy,
                latents.reshape(-1, policy.d),
                context.repeat_interleave(int(K), dim=0),
                nfe=8,
            ).reshape(len(values), int(K), SP.H, 2)
        windows_np = windows.detach().cpu().numpy().astype(np.float32)
        teacher_np = teacher.detach().cpu().numpy().astype(np.float32)
        for local, (holder, row) in enumerate(values):
            context_row = holder.contexts[int(row["context_id"])]
            difference = windows_np[local] - teacher_np[local][None]
            full_rmse = np.sqrt(np.mean(difference ** 2, axis=(1, 2)))
            first_rmse = np.sqrt(np.mean(
                difference[:, 0] ** 2, axis=1
            ))
            candidate_rows = []
            for candidate_id, controls in enumerate(windows_np[local]):
                task_id = len(tasks)
                tasks.append((
                    task_id,
                    candidate_id,
                    context_row["state"],
                    controls,
                    context_row["ped_xy"],
                    context_row["ped_vel"],
                    context_row["gamma"],
                ))
                candidate_rows.append({
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "controls": controls,
                })
            measurements.append({
                "neutral_id": int(row["query_id"]),
                "scenario_id": int(context_row["scenario_id"]),
                "gamma": float(context_row["gamma"]),
                "step": int(context_row["step"]),
                "same_latent_full_rmse": float(full_rmse[0]),
                "same_latent_first_rmse": float(first_rmse[0]),
                "min_K_full_rmse": float(full_rmse.min()),
                "min_K_first_rmse": float(first_rmse.min()),
                "candidate_rows": candidate_rows,
                "state": context_row["state"],
                "ped_xy": context_row["ped_xy"],
            })
    results = list(executor.map(SM.verify_in_worker, tasks))
    lookup = {
        int(context_index): result
        for context_index, _, result in results
    }
    for measurement in measurements:
        positives = []
        progresses = []
        margins = []
        for candidate in measurement.pop("candidate_rows"):
            result = lookup[int(candidate["task_id"])]
            if not result.get("resolved"):
                raise RuntimeError("hard-context probe verifier error")
            positives.append(int(result["y"]))
            state = np.asarray(measurement["state"], np.float32)
            next_state = BE._step(state, candidate["controls"][0])
            progresses.append(float(
                np.linalg.norm(state[:2] - SS.GOAL)
                - np.linalg.norm(next_state[:2] - SS.GOAL)
            ))
            margin, _, _ = BC.nominal_hp_margin(
                state,
                candidate["controls"][0],
                measurement["ped_xy"],
                measurement["gamma"],
            )
            margins.append(float(margin))
        measurement.update(
            all_K_positive_fraction=float(np.mean(positives)),
            any_positive_K=bool(any(positives)),
            fixed_B4_any_positive=bool(any(positives[:4])),
            same_latent_positive=bool(positives[0]),
            mean_one_step_progress=float(np.mean(progresses)),
            same_latent_one_step_progress=float(progresses[0]),
            mean_nominal_hp_margin=float(np.mean(margins)),
            same_latent_nominal_hp_margin=float(margins[0]),
        )
        measurement.pop("state")
        measurement.pop("ped_xy")

    def summarize(values):
        if not values:
            return {
                "contexts": 0,
                "all_K_positive_fraction": None,
                "any_positive_K": None,
                "fixed_B4_any_positive": None,
                "fixed_B4_NVP": None,
                "same_latent_positive": None,
                "same_latent_full_rmse": None,
                "same_latent_first_rmse": None,
                "min_K_full_rmse": None,
                "mean_one_step_progress": None,
                "mean_nominal_hp_margin": None,
            }
        return {
            "contexts": len(values),
            "all_K_positive_fraction": float(np.mean([
                value["all_K_positive_fraction"] for value in values
            ])),
            "any_positive_K": float(np.mean([
                value["any_positive_K"] for value in values
            ])),
            "fixed_B4_any_positive": float(np.mean([
                value["fixed_B4_any_positive"] for value in values
            ])),
            "fixed_B4_NVP": float(1.0 - np.mean([
                value["fixed_B4_any_positive"] for value in values
            ])),
            "same_latent_positive": float(np.mean([
                value["same_latent_positive"] for value in values
            ])),
            "same_latent_full_rmse": float(np.mean([
                value["same_latent_full_rmse"] for value in values
            ])),
            "same_latent_first_rmse": float(np.mean([
                value["same_latent_first_rmse"] for value in values
            ])),
            "min_K_full_rmse": float(np.mean([
                value["min_K_full_rmse"] for value in values
            ])),
            "mean_one_step_progress": float(np.mean([
                value["mean_one_step_progress"] for value in values
            ])),
            "mean_nominal_hp_margin": float(np.mean([
                value["mean_nominal_hp_margin"] for value in values
            ])),
        }

    summary = {
        "pooled": summarize(measurements),
        "per_gamma": {
            str(gamma): summarize([
                value for value in measurements
                if float(value["gamma"]) == float(gamma)
            ])
            for gamma in SP.GAMMAS
        },
    }
    del policy
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return summary, measurements


def _run_probe(
    arms,
    holder,
    records,
    output_dir,
    *,
    per_gamma,
    K,
    batch,
    device,
    workers,
    seed,
):
    probe_records = _probe_subset(holder, records, per_gamma, seed)
    actual_per_gamma = {
        str(gamma): sum(
            float(holder.contexts[int(row["context_id"])]["gamma"])
            == float(gamma)
            for _, row in probe_records
        )
        for gamma in SP.GAMMAS
    }
    context = mp.get_context("spawn")
    arm_results = []
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=context
    ) as executor:
        for arm in arms:
            summary, rows = _probe_one_arm(
                arm,
                probe_records,
                K=K,
                batch=batch,
                device=device,
                seed=seed,
                executor=executor,
            )
            arm_results.append({
                "arm": arm["name"],
                "checkpoint": arm["checkpoint"],
                "checkpoint_sha256": arm["checkpoint_sha256"],
                "summary": summary,
                "rows": rows,
            })
    report = {
        "status": PROBE_STATUS,
        "semantics": {
            "population": "fixed gamma-balanced subset of exact-negative D0",
            "stored_x0": "candidate 0 reuses the original flow base",
            "other_latents": "15 deterministic Gaussian controls",
            "fixed_B4": (
                "candidate 0 plus three deterministic controls; diagnostic "
                "only, not RBF acquisition"
            ),
            "teacher_label": "exact verifier y=0 throughout",
        },
        "K": int(K),
        "requested_contexts_per_gamma": int(per_gamma),
        "actual_contexts_per_gamma": actual_per_gamma,
        "arms": arm_results,
    }
    _write_json(os.path.join(output_dir, "HARD_CONTEXT_PROBE.json"), report)
    return report


def _run_raw_evaluation(
    arms,
    output_dir,
    *,
    ep0,
    M,
    noise_seed,
    device,
    workers,
):
    OE.M_PER_GAMMA = int(M)
    probe, _ = GPS.load_sfm_policy(arms[0]["checkpoint"], device="cpu")
    noise, noise_meta = OE._noise_bank(
        ep0=int(ep0), d=int(probe.d), seed=int(noise_seed)
    )
    del probe
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir)
    records = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=context
    ) as executor:
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
                "arm": arm["name"],
                "phase": arm["phase"],
                "neutral_lr": arm["neutral_lr"],
                "neutral_steps": arm["neutral_steps"],
                "checkpoint": arm["checkpoint"],
                "cell": cell,
            })
    report = {
        "status": f"SFM_B1_NEUTRAL_RAW_M{int(M)}_COMPLETE",
        "scene_profile": "double_density_velocity_ood",
        "bank": {
            "ep0": int(ep0),
            "M_per_gamma": int(M),
            "scenario_ids": list(range(int(ep0), int(ep0) + int(M))),
            "same_as_gathering_bank": True,
            "interpretation": "paired in-sample/resubstitution sanity only",
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


def _metric(cell, name):
    if name in ("CR", "SR", "timeout"):
        return float(cell[name])
    key = {
        "Validity": "Validity",
        "clearance": "successful_clearance",
        "time": "successful_time_to_goal",
    }[name]
    value = cell[key]["mean"]
    return None if value is None else float(value)


def _render(raw, probe, output_dir):
    metrics = (
        ("CR", "Collision rate"),
        ("Validity", "Validity"),
        ("clearance", "Successful min. clearance [m]"),
        ("time", "Successful time-to-goal [s]"),
    )
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, len(raw["records"]))
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for color, record in zip(colors, raw["records"]):
            values = [
                _metric(
                    record["cell"]["summary"]["per_gamma"][str(gamma)],
                    metric,
                )
                for gamma in SP.GAMMAS
            ]
            axis.plot(
                SP.GAMMAS,
                [np.nan if value is None else value for value in values],
                marker="o",
                ms=3.5,
                lw=1.3,
                color=color,
                label=record["arm"],
            )
        axis.set_title(title)
        axis.set_xlabel(r"$\gamma$")
        axis.grid(alpha=0.25)
    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    png = os.path.join(output_dir, "raw_gamma_trends.png")
    pdf = os.path.join(output_dir, "raw_gamma_trends.pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    probe_lookup = {
        row["arm"]: row["summary"]["pooled"]
        for row in probe["arms"]
    }
    csv_path = os.path.join(output_dir, "pooled_summary.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "arm", "neutral_lr", "neutral_steps", "SR", "CR", "timeout",
            "Validity", "clearance", "time", "D0_same_latent_RMSE",
            "D0_any_positive_K16", "D0_fixed_B4_NVP",
        ))
        writer.writeheader()
        for record in raw["records"]:
            pooled = record["cell"]["summary"]["pooled"]
            local = probe_lookup[record["arm"]]
            writer.writerow({
                "arm": record["arm"],
                "neutral_lr": record["neutral_lr"],
                "neutral_steps": record["neutral_steps"],
                "SR": _metric(pooled, "SR"),
                "CR": _metric(pooled, "CR"),
                "timeout": _metric(pooled, "timeout"),
                "Validity": _metric(pooled, "Validity"),
                "clearance": _metric(pooled, "clearance"),
                "time": _metric(pooled, "time"),
                "D0_same_latent_RMSE": local["same_latent_full_rmse"],
                "D0_any_positive_K16": local["any_positive_K"],
                "D0_fixed_B4_NVP": local["fixed_B4_NVP"],
            })
    return [png, pdf, csv_path]


def run(args):
    output_root = os.path.abspath(args.output_root)
    if os.path.exists(output_root):
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    source = FA._source()
    if not source["tracked_worktree_clean"]:
        raise RuntimeError("sanity study requires a clean frozen worktree")
    os.makedirs(output_root)
    scenarios = tuple(range(int(args.ep0), int(args.ep0) + int(args.M)))
    gather_dir = os.path.join(output_root, "gather")
    RA.collect(
        args.checkpoint,
        scenarios=scenarios,
        gammas=tuple(map(float, SP.GAMMAS)),
        scene_profile="double_density_velocity_ood",
        selector="margin",
        device=args.device,
        verifier_workers=int(args.workers),
        sample_seed=int(args.sample_seed),
        audit_seed=int(args.audit_seed),
        ell=float(args.ell),
        neutral_continuation=True,
        T=SP.T,
        outdir=gather_dir,
    )
    training_dir = os.path.join(output_root, "training")
    training, holder, neutral_records = _train(
        args.checkpoint,
        gather_dir,
        training_dir,
        neutral_lrs=tuple(map(float, args.neutral_lrs)),
        neutral_steps=tuple(map(int, args.neutral_steps)),
        ordinary_lr=float(args.ordinary_lr),
        batch=int(args.batch),
        device=args.device,
        seed=int(args.train_seed),
    )
    probe_dir = os.path.join(output_root, "probe")
    os.makedirs(probe_dir)
    probe = _run_probe(
        training["arms"],
        holder,
        neutral_records,
        probe_dir,
        per_gamma=int(args.probe_per_gamma),
        K=16,
        batch=int(args.batch),
        device=args.device,
        workers=int(args.workers),
        seed=int(args.probe_seed),
    )
    evaluation_dir = os.path.join(output_root, "evaluation")
    os.makedirs(evaluation_dir)
    raw = _run_raw_evaluation(
        training["arms"],
        evaluation_dir,
        ep0=int(args.ep0),
        M=int(args.M),
        noise_seed=int(args.noise_seed),
        device=args.device,
        workers=int(args.workers),
    )
    outputs = _render(raw, probe, output_root)
    complete = {
        "status": STATUS,
        "source": source,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": FA._sha256_file(args.checkpoint),
        "scene_profile": "double_density_velocity_ood",
        "gather_bank": {
            "ep0": int(args.ep0),
            "M": int(args.M),
            "scenarios": list(scenarios),
            "gammas": list(map(float, SP.GAMMAS)),
            "lineages": len(scenarios) * len(SP.GAMMAS),
        },
        "ordinary": {
            "alpha": 0.0,
            "lr": float(args.ordinary_lr),
            "whole_Dplus_adam_steps": 1,
        },
        "neutral_sweep": {
            "lrs": list(map(float, args.neutral_lrs)),
            "whole_D0_inner_steps": list(map(int, args.neutral_steps)),
            "GP_eligible": False,
            "original_verifier_y": 0,
        },
        "evaluation": {
            "same_scenarios_as_gather": True,
            "M_per_gamma": int(args.M),
            "raw_temperature": 1.0,
            "claim_scope": "paired in-sample sanity; not promotion evidence",
        },
        "training_complete": os.path.join(
            training_dir, "TRAINING_COMPLETE.json"
        ),
        "probe_complete": os.path.join(
            probe_dir, "HARD_CONTEXT_PROBE.json"
        ),
        "raw_evaluation": os.path.join(
            evaluation_dir, "RAW_EVALUATION.json"
        ),
        "outputs": outputs,
    }
    _write_json(os.path.join(output_root, "DELIVERY_COMPLETE.json"), complete)
    return complete


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ep0", type=int, default=DEFAULT_EP0)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--ordinary-lr", type=float, default=DEFAULT_ORDINARY_LR)
    parser.add_argument(
        "--neutral-lrs",
        type=float,
        nargs="+",
        default=DEFAULT_NEUTRAL_LRS,
    )
    parser.add_argument(
        "--neutral-steps",
        type=int,
        nargs="+",
        default=DEFAULT_NEUTRAL_STEPS,
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ell", type=float, default=RA.DEFAULT_ELL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=700_000)
    parser.add_argument("--audit-seed", type=int, default=2_026_073_0)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--probe-seed", type=int, default=2_026_073_2)
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument(
        "--probe-per-gamma",
        type=int,
        default=DEFAULT_PROBE_PER_GAMMA,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if int(args.M) != 20:
        raise ValueError("canonical sanity is pinned to M=20 scenarios")
    if tuple(map(float, args.neutral_lrs)) != DEFAULT_NEUTRAL_LRS:
        raise ValueError(
            f"neutral LR grid must remain {DEFAULT_NEUTRAL_LRS}"
        )
    if tuple(map(int, args.neutral_steps)) != DEFAULT_NEUTRAL_STEPS:
        raise ValueError(
            f"neutral inner-step grid must remain {DEFAULT_NEUTRAL_STEPS}"
        )
    value = run(args)
    print(os.path.join(args.output_root, "DELIVERY_COMPLETE.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
