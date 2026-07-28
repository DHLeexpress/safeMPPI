"""Privileged Codex MPC teacher data kept separate from certified Safe Expansion.

This module is an explicit ablation.  It does *not* change ordinary
``D/D+/D-`` gathering, the RBF-GP, acquisition, verifier labels, or validity
accounting.  At declared hard contexts it asks the historical privileged SFM
MPC controller for one bounded H=10 recovery plan and stores that plan in a
separate ``D_MPC`` buffer even when the compact SOCP would reject it.

The intended round ordering is::

    ordinary executed D+/D- replay
        -> save theta_(n+1/2)
        -> dedicated D_MPC CFM block
        -> save theta_(n+1)

The compact SOCP may be run after selection as an audit only.  Its result never
gates storage and never turns a teacher record into a safety-positive sample.
The privileged controller is never used during raw evaluation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_mpc_pool as MP
import sfm_b1_offline_store as OS
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_scene as SS


BUFFER_STATUS = "SFM_UNVERIFIED_MPC_TEACHER_BUFFER_COMPLETE"
BUFFER_VERSION = 1
TEACHER_SOURCE = "codex_privileged_sfm_mpc"
R1_CHECKPOINT_SHA256 = (
    "141e4ae6592bf73f500ae4c5382258c8463de8519c2c7cf26f5c68ea2563b06a"
)
H = 10
CONTROL_TOL = 1.0e-6


def _controller_config_hash():
    payload = json.dumps(
        asdict(MP.privileged_sfm_config()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _context_snapshot(context):
    return {
        "scenario_id": int(context["scenario_id"]),
        "gamma": float(context["gamma"]),
        "step": int(context["step"]),
        "state": np.asarray(context["state"], np.float32).copy(),
        "ped_xy": np.asarray(context["ped_xy"], np.float32).copy(),
        "ped_vel": np.asarray(context["ped_vel"], np.float32).copy(),
    }


def validate_controls(controls, *, u_max=SS.U_MAX, tol=CONTROL_TOL):
    """Validate a controller command without silently repairing a bad plan."""
    raw = np.asarray(controls, np.float32)
    if raw.shape != (H, 2) or not np.isfinite(raw).all():
        raise ValueError("D_MPC requires finite controls with shape [10,2]")
    maximum = float(np.max(np.abs(raw))) if raw.size else 0.0
    clipped = np.clip(raw, -float(u_max), float(u_max))
    delta = float(np.max(np.abs(clipped - raw))) if raw.size else 0.0
    if delta > float(tol):
        raise ValueError(
            f"privileged controller exceeded U_MAX by {delta:.6g}; "
            "refusing to silently reinterpret the teacher"
        )
    return clipped.astype(np.float32, copy=True), {
        "u_max": float(u_max),
        "max_abs_before": maximum,
        "max_clip_delta": delta,
    }


def _balanced_context_ids(shard, hard_windows, max_contexts):
    """Deterministic gamma-balanced subset of declared hard contexts."""
    grouped = defaultdict(list)
    for window in hard_windows:
        context_id = int(window["context_id"])
        context = shard.contexts[context_id]
        grouped[round(float(context["gamma"]), 8)].append(context_id)
    grouped = {
        gamma: sorted(set(values))
        for gamma, values in grouped.items()
        if values
    }
    if not grouped:
        return []
    if max_contexts is None:
        return sorted(value for values in grouped.values() for value in values)
    limit = int(max_contexts)
    if limit <= 0:
        raise ValueError("max_contexts must be positive or None")
    selected = []
    active = {gamma: list(values) for gamma, values in sorted(grouped.items())}
    while active and len(selected) < limit:
        next_active = {}
        for gamma, values in active.items():
            if len(selected) >= limit:
                break
            # Evenly consume each lineage instead of taking only early steps.
            index = (len(values) - 1) // 2
            selected.append(values.pop(index))
            if values:
                next_active[gamma] = values
        active = next_active
    return sorted(set(selected))


def _selected_family(pool, selected_plan):
    sources = list(pool.get("candidate_sources") or ())
    for plan, source in zip(pool["plans"], sources):
        if np.allclose(
            np.asarray(plan, np.float32),
            np.asarray(selected_plan, np.float32),
            atol=1.0e-6,
        ):
            return dict(source)
    return {"family": "controller_generated_or_escalated", "source_index": -1}


def select_teacher_plan(policy, context, humans, *, device, seed_step):
    """Run the same local privileged SFM MPC selector used by the Codex wrapper."""
    pool = MP.build_codex_pool(
        policy,
        context,
        humans,
        device=device,
        seed_step=seed_step,
        track_sources=True,
    )
    gamma = float(context["gamma"])
    cfg = KZ._gamma_controller_config(
        MP.privileged_sfm_config(), gamma,
    ).validate()
    state = np.asarray(context["state"], np.float32)
    ped_xy = np.asarray(context["ped_xy"], np.float32)
    current_clearance = float(
        np.min(np.linalg.norm(ped_xy - state[:2], axis=1) - SS.R_PED)
    )
    margin = KZ._adaptive_step_filter_margin(cfg, gamma)
    clearance_target = KZ._adaptive_step_filter_clearance_target(
        cfg, gamma, margin,
    )
    action, diagnostics, selected_plan = KZ.exact_sfm_horizon_filter_action(
        humans,
        state,
        pool["nominal_plan"],
        pool["refined_pool"],
        margin=margin,
        horizon=int(cfg.step_filter_horizon),
        n_goal_plans=int(cfg.step_filter_goal_plans),
        n_avoid_plans=int(cfg.step_filter_avoid_plans),
        always_select=bool(cfg.step_filter_always_select),
        min_progress=float(cfg.step_filter_min_progress),
        goal_score_weight=float(cfg.step_filter_goal_score_weight),
        clearance_weight=float(cfg.step_filter_clearance_weight),
        fallback_clearance=bool(current_clearance < float(margin)),
        fallback_lookahead=int(cfg.step_filter_fallback_lookahead),
        viability_lookahead=int(cfg.step_filter_viability_lookahead),
        viability_band=float(cfg.step_filter_viability_band),
        viability_goal_weight=float(cfg.step_filter_viability_goal_weight),
        viability_escalate=bool(cfg.step_filter_viability_escalate),
        viability_escalation_band=float(
            cfg.step_filter_viability_escalation_band
        ),
        viability_escalation_min_progress=float(
            cfg.step_filter_viability_escalation_entry_progress
        ),
        clearance_target=clearance_target,
        clearance_target_weight=float(
            cfg.step_filter_clearance_target_weight
        ),
    )
    selected_plan, clip = validate_controls(selected_plan)
    if not bool(diagnostics.get("filter_feasible")):
        return None, {
            "reason": "privileged_filter_infeasible",
            "selector_diagnostics": diagnostics,
            "pool_manifest": pool["pool_manifest"],
        }
    return selected_plan, {
        "first_action": np.asarray(action, np.float32).tolist(),
        "selector_diagnostics": diagnostics,
        "candidate_source": _selected_family(pool, selected_plan),
        "pool_manifest": dict(pool["pool_manifest"]),
        "clip": clip,
    }


def harvest_round(
    policy,
    shard,
    hard_windows,
    *,
    device,
    environment,
    max_contexts=None,
    executor=None,
    audit_socp=False,
):
    """Harvest one control-bounded privileged teacher per hard context.

    ``audit_socp`` only annotates the selected teacher.  A resolved negative,
    an error, and a positive are all handled identically for storage.
    """
    if audit_socp and executor is None:
        raise ValueError("audit_socp requires an executor")
    context_ids = _balanced_context_ids(shard, hard_windows, max_contexts)
    records = []
    counts = defaultdict(int)
    controller_hash = _controller_config_hash()
    for context_id in context_ids:
        context = shard.contexts[int(context_id)]
        counts["hard_contexts"] += 1
        humans, replay_state = MP.replay_prefix_humans(
            shard,
            int(context["scenario_id"]),
            float(context["gamma"]),
            int(context["step"]),
            environment,
        )
        ped_xy_live, _ = SS.collect_humans(humans)
        if not (
            np.allclose(
                replay_state,
                np.asarray(context["state"], np.float32),
                atol=1.0e-4,
            )
            and np.allclose(
                ped_xy_live,
                np.asarray(context["ped_xy"], np.float32),
                atol=1.0e-4,
            )
        ):
            counts["replay_mismatch"] += 1
            continue
        plan, selection = select_teacher_plan(
            policy,
            context,
            humans,
            device=device,
            seed_step=int(context["step"]),
        )
        if plan is None:
            counts["privileged_infeasible"] += 1
            continue
        audit = None
        if audit_socp:
            task = (
                0,
                0,
                context["state"],
                plan,
                context["ped_xy"],
                context["ped_vel"],
                context["gamma"],
            )
            _, _, result = next(iter(executor.map(SM.verify_in_worker, [task])))
            audit = {
                "resolved": bool(result.get("resolved")),
                "verifier_label": (
                    int(result["y"]) if result.get("resolved") else None
                ),
                "full_h": bool(result.get("full_h", False)),
                "error": result.get("error"),
                "diagnostics": dict(result.get("diagnostics") or {}),
            }
            counts[
                "socp_audit_positive"
                if audit["verifier_label"] == 1
                else "socp_audit_nonpositive"
            ] += 1
        record = {
            "teacher_id": len(records),
            "round": int(shard.round_i),
            "context_id": int(context_id),
            "scenario_id": int(context["scenario_id"]),
            "gamma": float(context["gamma"]),
            "episode_id": int(context["scenario_id"]),
            "step": int(context["step"]),
            "controls": plan,
            "source": TEACHER_SOURCE,
            "candidate_source": selection["candidate_source"],
            "controller_config_hash": controller_hash,
            "selector_diagnostics": selection["selector_diagnostics"],
            "pool_manifest": selection["pool_manifest"],
            "clip": selection["clip"],
            "context_snapshot": _context_snapshot(context),
            "socp_audit": audit,
        }
        forbidden = {"y", "query_id", "train_eligible", "x0"} & set(record)
        if forbidden:
            raise AssertionError(f"D_MPC leaked ordinary-D fields: {forbidden}")
        records.append(record)
        counts["kept"] += 1
    per_gamma = defaultdict(int)
    per_family = defaultdict(int)
    for record in records:
        per_gamma[f"{float(record['gamma']):g}"] += 1
        per_family[str(record["candidate_source"]["family"])] += 1
    return records, {
        "counts": dict(counts),
        "per_gamma": dict(per_gamma),
        "per_family": dict(per_family),
        "audit_socp": bool(audit_socp),
        "storage_gate": (
            "privileged exact-SFM filter_feasible and bounded controls only; "
            "compact SOCP is audit-only"
        ),
    }


def save_buffer(path, round_shard_path, shard, records, audit):
    """Atomically save a teacher buffer authenticated to one round shard."""
    path = os.path.abspath(os.fspath(path))
    round_shard_path = os.path.abspath(os.fspath(round_shard_path))
    if int(shard.round_i) <= 0:
        raise ValueError("teacher buffer requires a positive expansion round")
    forbidden = {"y", "query_id", "train_eligible", "x0"}
    for record in records:
        leaked = forbidden & set(record)
        if leaked:
            raise ValueError(f"D_MPC record leaked ordinary-D fields: {leaked}")
        context_id = int(record["context_id"])
        if not 0 <= context_id < len(shard.contexts):
            raise ValueError("D_MPC record references a missing context")
        validate_controls(record["controls"])
    payload = {
        "status": BUFFER_STATUS,
        "version": BUFFER_VERSION,
        "round": int(shard.round_i),
        "round_shard_path": round_shard_path,
        "round_shard_sha256": OS.sha256_file(round_shard_path),
        "records": list(records),
        "audit": dict(audit),
        "semantics": {
            "ordinary_D_unchanged": True,
            "gp_acquisition_unchanged": True,
            "teacher_is_safety_label": False,
            "teacher_used_at_raw_evaluation": False,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


def _teacher_mass(shard, records):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        context = shard.contexts[int(record["context_id"])]
        gamma = round(float(context["gamma"]), 8)
        episode = int(context["scenario_id"])
        context_id = int(context["context_id"])
        grouped[gamma][episode][context_id].append(record)
    mass = {}
    if not grouped:
        return mass, {"total": 0.0, "gamma": {}}
    gamma_mass = defaultdict(float)
    for gamma, episodes in grouped.items():
        for _, contexts in episodes.items():
            for _, values in contexts.items():
                value = (
                    1.0
                    / len(grouped)
                    / len(episodes)
                    / len(contexts)
                    / len(values)
                )
                for record in values:
                    mass[int(record["teacher_id"])] = value
                    gamma_mass[f"{gamma:g}"] += value
    total = float(sum(mass.values()))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"teacher hierarchy mass sums to {total}, not one")
    return mass, {"total": total, "gamma": dict(gamma_mass)}


def _tensor_batch(shard, records, device):
    contexts = [shard.contexts[int(record["context_id"])] for record in records]
    hp10 = torch.as_tensor(
        np.stack([context["hp10"] for context in contexts]), device=device,
    ).float()
    low = torch.as_tensor(
        np.stack([context["low5"] for context in contexts]), device=device,
    ).float()
    hist = torch.as_tensor(
        np.stack([context["hist"] for context in contexts]), device=device,
    ).float()
    controls = torch.as_tensor(
        np.stack([record["controls"] for record in records]), device=device,
    ).float()
    return hp10, low, hist, controls


def distill_block(
    policy,
    optimizer,
    shard,
    records,
    *,
    epochs,
    batch,
    seed,
):
    """Dedicated teacher-only CFM update with fresh bases every exposure."""
    if int(epochs) < 0 or int(batch) <= 0:
        raise ValueError("epochs must be non-negative and batch positive")
    records = list(records)
    if not records or int(epochs) == 0:
        return {
            "steps": 0,
            "optimizer_steps": 0,
            "sample_exposures": 0,
            "records": len(records),
            "epochs": int(epochs),
            "loss_first": None,
            "loss_last": None,
            "mass": {"total": 0.0, "gamma": {}},
        }
    mass, accounting = _teacher_mass(shard, records)
    rng = np.random.default_rng(int(seed))
    device = next(policy.parameters()).device
    losses = []
    base_rng_states = []
    policy.train()
    for epoch in range(int(epochs)):
        order = list(rng.permutation(len(records)))
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for start in range(0, len(order), int(batch)):
            chunk = [records[index] for index in order[start:start + int(batch)]]
            hp10, low, hist, controls = _tensor_batch(shard, chunk, device)
            context = policy.ctx_from(hp10, low, hist)
            chunk_mass = float(sum(
                mass[int(record["teacher_id"])] for record in chunk
            ))
            weights = torch.as_tensor(
                [
                    mass[int(record["teacher_id"])]
                    for record in chunk
                ],
                dtype=controls.dtype,
                device=device,
            )
            exposure_seed = int(seed) + epoch * 1_000_003 + start
            torch.manual_seed(exposure_seed)
            base_rng_states.append(exposure_seed)
            loss = policy.cfm_loss(controls, context, weights=weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite D_MPC distillation loss")
            # policy.cfm_loss returns a normalized weighted mean.  Multiplying
            # by this chunk's global hierarchy mass and accumulating every
            # chunk before one Adam step yields the declared whole-buffer
            # gamma->episode->context->teacher objective exactly.
            scaled = loss * chunk_mass
            scaled.backward()
            epoch_loss += float(scaled.detach())
        optimizer.step()
        losses.append(epoch_loss)
    policy.eval()
    return {
        "steps": len(losses),
        "optimizer_steps": len(losses),
        "sample_exposures": len(records) * int(epochs),
        "records": len(records),
        "epochs": int(epochs),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
        "mass": accounting,
        "fresh_base_seed_count": len(set(base_rng_states)),
    }
