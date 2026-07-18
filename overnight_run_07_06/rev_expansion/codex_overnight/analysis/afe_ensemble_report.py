"""Eight-panel report for one completed AFE deep-ensemble run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
COLORS = {
    gamma: plt.get_cmap("plasma")(0.08 + 0.77 * index / 6)
    for index, gamma in enumerate(GAMMAS)
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed_run(root):
    paths = {
        name: os.path.join(root, name)
        for name in ("recipe.json", "probe.jsonl", "COMPLETE.json")
    }
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    with open(paths["recipe.json"]) as stream:
        recipe = json.load(stream)
    with open(paths["COMPLETE.json"]) as stream:
        complete = json.load(stream)
    records = [json.loads(line) for line in open(paths["probe.jsonl"]) if line.strip()]
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("trainer completion marker is not COMPLETE")
    if recipe.get("algorithm") not in {
        "afe_deep_ensemble_parallel_v1",
        "afe_deep_ensemble_adaptive_ess_parallel_v2",
        "afe_low7_deep_ensemble_adaptive_ess_parallel_v1",
    }:
        raise RuntimeError("report accepts only the declared AFE ensemble algorithm")
    for relative in ("recipe.json", "probe.jsonl"):
        expected = complete.get("artifact_sha256", {}).get(relative)
        if not expected or sha256_file(os.path.join(root, relative)) != expected:
            raise RuntimeError(f"trainer artifact hash mismatch: {relative}")
    expected_rounds = list(range(int(recipe["rounds"]) + 1))
    if [int(record["round"]) for record in records] != expected_rounds:
        raise RuntimeError("probe does not contain exactly round 0..R")
    return recipe, complete, records


def _values(records, key, default=np.nan):
    return [record.get(key, default) for record in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    recipe, _, records = load_completed_run(args.run)
    rounds = [int(record["round"]) for record in records]
    adapted = records[1:]
    adapted_rounds = rounds[1:]
    figure, axes = plt.subplots(3, 3, figsize=(18, 15.5))

    ax = axes.flat[0]
    for key, label, color in (("SR", "SR", "#118833"), ("NVP", "NVP", "#cc3311"),
                              ("CR", "CR", "#332288")):
        ax.plot(rounds, [record["ctrl_pooled"][key] for record in records], "-o",
                label=label, color=color)
    ax.set(title="A. Expert-free verified controller", xlabel="round", ylim=(-0.03, 1.03))
    ax.legend()

    ax = axes.flat[1]
    for gamma in GAMMAS:
        ax.plot(rounds, [record["ctrl"][gamma]["SR"] for record in records], "-o",
                color=COLORS[gamma], label=f"γ={gamma}")
    ax.set(title="B. Controller SR by γ", xlabel="round", ylim=(-0.03, 1.03))
    ax.legend(fontsize=7, ncol=2)

    ax = axes.flat[2]
    for gamma in GAMMAS:
        ax.plot(rounds, [record["V_safe_gamma"][gamma] for record in records], "-o",
                color=COLORS[gamma], lw=1.6)
        ax.plot(rounds, [record["V_full_gamma"][gamma] for record in records], "--",
                color=COLORS[gamma], lw=1.0, alpha=0.8)
    ax.set(title="C. Untilted validity (safe solid / +progress dashed)", xlabel="round",
           ylim=(-0.03, 1.03))

    ax = axes.flat[3]
    ax.plot(adapted_rounds, _values(adapted, "cfm"), "-o", color="#0077bb",
            label="CFM loss")
    ax.plot(adapted_rounds, _values(adapted, "cfm_last"), "--o", color="#33bbee",
            label="last-step CFM")
    ax2 = ax.twinx()
    ax2.plot(rounds, _values(records, "rep_cos"), "-s", color="#ee7733", label="φ cosine")
    ax.set(title="D. Update and representation drift", xlabel="round", ylabel="loss")
    ax2.set_ylabel("fixed-probe cosine")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes.flat[4]
    ax.plot(adapted_rounds, _values(adapted, "sig_all_med"), "-o",
            label="all-K ensemble std")
    ax.plot(adapted_rounds, _values(adapted, "sig_sel_med"), "-o",
            label="selected-B ensemble std")
    correlations = [
        record.get("ensemble_round_start", {}).get(
            "member_prediction_correlation", np.nan
        )
        for record in adapted
    ]
    ax2 = ax.twinx()
    ax2.plot(adapted_rounds, correlations, "-s", color="#cc3311",
             label="acquisition-model member corr.")
    ax.set(title="E. Neural epistemic uncertainty", xlabel="round", ylabel="ensemble std")
    ax2.set_ylabel("member prediction corr.")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes.flat[5]
    ax.plot(adapted_rounds, _values(adapted, "ess_first_med"), "-o",
            label="first-step ESS/K")
    ax.plot(adapted_rounds, _values(adapted, "ess_med"), "-o",
            label="median ESS/M remaining")
    ax.plot(adapted_rounds, _values(adapted, "ent_med"), "-o", label="entropy")
    ess_target = recipe.get("adaptive_ess_target") or 0.375
    ax.axhline(ess_target, color="0.5", ls=":", label=f"ESS target {ess_target:g}")
    ax2 = ax.twinx()
    ax2.plot(adapted_rounds, _values(adapted, "uplift_med"), "-s", color="#cc3311",
             label="σ uplift")
    ax.set(title="F. Acquisition selectivity (r1 uniform)", xlabel="round",
           ylim=(-0.03, 1.03))
    ax2.set_ylabel("selected − pool σ")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes.flat[6]
    totals = np.asarray(_values(records, "n_D", 0), dtype=float)
    positives = np.asarray(_values(records, "n_Dpos", 0), dtype=float)
    estimator_sizes = [record.get("ensemble", {}).get("n", 0) for record in records]
    ax.plot(rounds, totals, "-o", label="uncertainty D (all labels)")
    ax.plot(rounds, positives, "-o", label="flow D+ (positive)")
    ax.plot(rounds, totals - positives, "-o", label="negative labels")
    ax.plot(rounds, estimator_sizes, "--s", label="ensemble fit rows")
    ax.set(title="G. Uncertainty vs learning memory", xlabel="round", ylabel="samples")
    ax.legend(fontsize=8)

    ax = axes.flat[7]
    gathering = np.asarray(_values(adapted, "t_gather", 0.0), dtype=float)
    sampling = np.asarray([
        record.get("gather_timing", {}).get("sampling", 0.0) for record in adapted
    ])
    verifier = np.asarray([
        record.get("gather_timing", {}).get("verifier_wall", 0.0) for record in adapted
    ])
    cfm = np.asarray(_values(adapted, "t_update", 0.0), dtype=float)
    ensemble = (
        np.asarray(_values(adapted, "t_ensemble", 0.0), dtype=float)
        + np.asarray(_values(adapted, "t_beta_calibration", 0.0), dtype=float)
    )
    audit = np.asarray(_values(adapted, "t_audit", 0.0), dtype=float)
    controller_eval = np.asarray(
        _values(adapted, "t_controller_eval", 0.0), dtype=float
    )
    checkpoint = np.asarray(_values(adapted, "t_checkpoint", 0.0), dtype=float)
    total = np.asarray(_values(adapted, "t_round_total", 0.0), dtype=float)
    gather_other = np.maximum(gathering - sampling - verifier, 0.0)
    ax.bar(adapted_rounds, sampling, label="GPU proposal/φ/σ")
    ax.bar(adapted_rounds, verifier, bottom=sampling, label="verifier wall")
    ax.bar(adapted_rounds, gather_other, bottom=sampling + verifier,
           label="gather bookkeeping")
    bottom = sampling + verifier + gather_other
    for values, label in (
        (cfm, "CFM update"),
        (ensemble, "reembed + ensemble (+ one-time β)"),
        (audit, "untilted audit"),
        (controller_eval, "verified-controller eval"),
        (checkpoint, "artifacts"),
    ):
        ax.bar(adapted_rounds, values, bottom=bottom, label=label)
        bottom = bottom + values
    ax.plot(adapted_rounds, total, "-ko", label="total round wall")
    ax.set(title="H. Measured round wall time", xlabel="round", ylabel="seconds")
    ax.legend(fontsize=7)

    ax = axes.flat[8]
    beta_used = [record.get("beta_used", record.get("beta")) for record in records]
    beta_next = [record.get("beta_next", record.get("beta")) for record in records]
    ax.plot(rounds, beta_used, "-o", label=r"$\beta_n$ used")
    ax.plot(rounds, beta_next, "--s", label=r"$\beta_{n+1}$ calibrated")
    positive_beta = [value for value in beta_next if value is not None]
    if positive_beta and all(float(value) > 0.0 for value in positive_beta):
        ax.set_yscale("log")
    target = recipe.get("adaptive_ess_target")
    achieved = [
        ((record.get("beta_calibration") or {}).get("solution") or {})
        .get("achieved", {}).get("ess_med", np.nan)
        for record in records
    ]
    ax2 = ax.twinx()
    ax2.plot(rounds, achieved, ":^", color="#cc3311", label="calibration ESS")
    if target is not None:
        ax2.axhline(float(target), color="0.4", ls="--", label=f"ESS target {target:g}")
    ax.set(title="I. Acquisition temperature schedule", xlabel="round", ylabel=r"$\beta$")
    ax2.set_ylabel("normalized ESS")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    final = records[-1]
    scene = recipe["scene"]["profile"]["name"]
    figure.suptitle(
        f"Single-arm AFE deep ensemble — {scene}; {recipe['rollout_replicas']} "
        f"rollouts/γ/round; β={recipe['beta']:.4g}; 5×(100,100) MLP\n"
        f"final Vsafe={final['V_safe']:.3f}, controller SR={final['ctrl_pooled']['SR']:.3f}, "
        f"NVP={final['ctrl_pooled']['NVP']:.3f}; r1 uniform bootstrap",
        fontsize=13,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    figure.savefig(args.out, dpi=140)
    plt.close(figure)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
