#!/usr/bin/env python3
"""Deterministic trust-step audit for corrected P2 fine tuning.

This is an analysis-only script.  It compares checkpoints on identical fixed
late-goal contexts/noise, reads existing faithful evaluations, and writes the
compact JSON evidence used by ``trust_step_probe.md``.  It never trains or
modifies a checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent.parent
sys.path[:0] = [str(WORK), str(ROOT.parent), str(ROOT)]

import eval_ae  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
import grid_scene as GS  # noqa: E402


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
CHECKPOINTS = {
    "incumbent": ROOT / "results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt",
    "lr1e4_unfreeze": ROOT / "analysis/runs/corrected_equiv_full_s31/ckpt_102.pt",
    "lr2e5_unfreeze": ROOT / "analysis/runs/corrected_lr2e5_unfreeze_s41/ckpt_102.pt",
    "lr2e5_freeze": ROOT / "analysis/runs/corrected_lr2e5_freeze_s41/ckpt_102.pt",
}
EVAL_DIRS = {
    "lr1e4_unfreeze": ROOT / "analysis/runs/eval_corrected_it102_m25",
    "lr2e5_unfreeze": ROOT / "analysis/runs/eval_corrected_unfreeze_lr2e5_it102_m25",
    "lr2e5_freeze": ROOT / "analysis/runs/eval_corrected_freeze_lr2e5_it102_m25",
}


def fixed_sample(policy, grid, low5, hist, x0, draws=128, nfe=8):
    with torch.no_grad():
        ctx = policy.ctx_from(grid, low5, hist).repeat_interleave(draws, 0)
        x = x0.clone()
        for i in range(nfe):
            tau = torch.full((len(x),), i / nfe)
            x = x + policy(x, tau, ctx) / nfe
    return x.reshape(len(grid), draws, 10, 2).clamp(-1, 1).numpy()


def late_context_audit():
    db_path = ROOT / "results/p2/finalunit_q50_k14_s15_from_it18/viz_db/it19.pt"
    db = torch.load(db_path, map_location="cpu", weights_only=False)
    low = db["low5"].numpy()
    pos = np.array([5.0, 5.0]) - 5.0 * low[:, :2]
    idx = np.where(np.linalg.norm(pos - np.array([5.0, 5.0]), axis=1) < 1.0)[0]
    idx = idx[np.linspace(0, len(idx) - 1, min(64, len(idx))).astype(int)]
    grid, low5 = db["grid"][idx], db["low5"][idx]
    hist = torch.zeros(len(idx), 16, 2)
    draws = 128
    x0 = torch.randn(len(idx) * draws, 20, generator=torch.Generator().manual_seed(120))
    samples = {}
    for name, path in CHECKPOINTS.items():
        policy, _ = HP.load_hp(str(path), device="cpu")
        samples[name] = fixed_sample(policy.eval(), grid, low5, hist, x0, draws=draws)

    base = samples["incumbent"]
    out = {}
    for name, u in samples.items():
        a0 = u[:, :, 0]
        full = u.sum(2)
        ad = a0[:, :, 0] - a0[:, :, 1]
        fd = full[:, :, 0] - full[:, :, 1]
        step_bias = (u[..., 0] - u[..., 1]).mean((0, 1))
        delta = np.linalg.norm(u - base, axis=-1)
        denom = np.linalg.norm(base, axis=-1).mean((0, 1))
        rel = delta.mean((0, 1)) / np.maximum(denom, 1e-12)
        out[name] = {
            "late_a0_x_minus_y": float(ad.mean()),
            "late_full_x_minus_y": float(fd.mean()),
            "late_y_dominant_fraction": float((ad < 0).mean()),
            "first_full_correlation": float(np.corrcoef(ad.ravel(), fd.ravel())[0, 1]),
            "mean_a0_x": float(a0[..., 0].mean()),
            "mean_a0_y": float(a0[..., 1].mean()),
            "per_action_x_minus_y": [float(x) for x in step_bias],
            "per_action_relative_l2_vs_incumbent": [float(x) for x in rel],
            "first_action_relative_l2_vs_incumbent": float(rel[0]),
            "early_half_relative_l2_vs_incumbent": float(rel[:5].mean()),
            "late_half_relative_l2_vs_incumbent": float(rel[5:].mean()),
            "late_to_early_drift_ratio": float(rel[5:].mean() / max(rel[:5].mean(), 1e-12)),
        }
    return {
        "source_contexts": str(db_path.relative_to(ROOT)),
        "context_count": int(len(idx)),
        "noise_draws_per_context": draws,
        "nfe": 8,
        "checkpoints": out,
    }


def incumbent_m25():
    env = GS.make_grid()
    out = {}
    source = ROOT / "results/p2/eval_finalunit_s15_it100"
    for gamma in GAMMAS:
        paths = eval_ae.load_paths(source / f"paths_g{gamma}.npz")[:25]
        out[str(gamma)] = eval_ae.summarize_paths(paths, env, gamma, "incumbent-M25")
    return out


def read_eval_rows(directory):
    out = {}
    for gamma in GAMMAS:
        path = directory / f"row_g{gamma}.json"
        if not path.exists():
            return None
        out[str(gamma)] = json.loads(path.read_text())
    return out


def eval_summary(rows, incumbent_rows):
    sr = np.array([rows[str(g)]["SR"] for g in GAMMAS], dtype=float)
    cr = np.array([rows[str(g)]["CR"] for g in GAMMAS], dtype=float)
    inc = np.array([incumbent_rows[str(g)]["SR"] for g in GAMMAS], dtype=float)
    return {
        "per_gamma": rows,
        "mean_SR": float(sr.mean()),
        "minimum_SR": float(sr.min()),
        "maximum_CR": float(cr.max()),
        "maximum_paired_SR_drop_vs_incumbent": float(np.max(inc - sr)),
        "mean_paired_SR_change_vs_incumbent": float(np.mean(sr - inc)),
    }


def main():
    vector = json.loads((ROOT / "analysis/trust_step_vector_field.json").read_text())
    inc_eval = incumbent_m25()
    comparisons = {}
    for name, ck in CHECKPOINTS.items():
        field = vector["checkpoints"][name]
        pair = None if name == "incumbent" else vector[
            "pairwise_origin_field_relative_l2"
        ][f"incumbent__{name}"]
        rows = inc_eval if name == "incumbent" else read_eval_rows(EVAL_DIRS[name])
        comparisons[name] = {
            "checkpoint": str(ck.relative_to(ROOT)),
            "parameter_relative_l2": {
                k: field["parameters"][k] for k in ("trunk", "head", "encoder")
            },
            "origin_field_relative_l2_vs_incumbent": None if pair is None else pair["mean"],
            "origin_field_relative_l2_per_gamma": None if pair is None else pair["per_gamma"],
            "balanced_demo_fixed_cfm": field["balanced_demo_fixed_cfm"]["all"],
            "origin_gamma_0.5": field["origin"]["0.5"],
            "faithful_M25": None if rows is None else eval_summary(rows, inc_eval),
        }

    result = {
        "scope": "P2 incumbent trust-step audit; no Mizuta/Kazuki modification",
        "comparison": comparisons,
        "late_context_fixed_noise": late_context_audit(),
        "recommendation": {
            "learning_rate": 2e-5,
            "freeze_encoder": True,
            "inner_steps_per_gather": 1,
            "hard_per_update_acceptance": {
                "origin_field_relative_l2_max": 0.005,
                "trunk_parameter_relative_l2_max": 0.0002,
                "head_parameter_relative_l2_max": 0.00013,
                "late_a0_x_minus_y_min": 0.10,
                "late_a0_x_minus_y_drop_max": 0.03,
                "late_y_dominant_fraction_max": 0.45,
                "late_context_first_action_relative_l2_max": 0.03,
                "late_context_early_half_relative_l2_max": 0.03,
                "balanced_demo_cfm_mse_relative_increase_max": 0.005,
                "all_gamma_M25_CR_max": 0.0,
                "paired_any_gamma_M25_SR_drop_max": 0.08,
                "cumulative_origin_field_relative_l2_from_M100_anchor_max": 0.01,
            },
            "stop_and_measure": {
                "origin_field_relative_l2": 0.004,
                "late_a0_x_minus_y_drop": 0.02,
                "late_context_early_half_relative_l2": 0.02,
                "paired_any_gamma_M25_SR_drop": 0.04,
                "cumulative_origin_field_relative_l2_from_M100_anchor": 0.008,
            },
            "rollback_if": "any hard bound is exceeded; restore both model and optimizer/train state",
        },
    }
    out = ROOT / "analysis/trust_step_probe.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
