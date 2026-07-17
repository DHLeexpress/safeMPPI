"""Eight-panel report for one completed AFE-RBF expansion run."""
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
    if recipe.get("algorithm") != "afe_rbf_previous_round_parallel_v1":
        raise RuntimeError("report accepts only the declared AFE-RBF algorithm")
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
    recipe, complete, records = load_completed_run(args.run)
    rounds = [int(record["round"]) for record in records]
    adapted = records[1:]
    adapted_rounds = rounds[1:]
    figure, axes = plt.subplots(2, 4, figsize=(22, 10.5))

    ax = axes[0, 0]
    for key, label, color in (("SR", "SR", "#118833"), ("NVP", "NVP", "#cc3311"),
                              ("CR", "CR", "#332288")):
        ax.plot(rounds, [record["ctrl_pooled"][key] for record in records], "-o",
                label=label, color=color)
    ax.set(title="A. Expert-free verified controller", xlabel="round", ylim=(-0.03, 1.03))
    ax.legend()

    ax = axes[0, 1]
    for gamma in GAMMAS:
        ax.plot(rounds, [record["ctrl"][gamma]["SR"] for record in records], "-o",
                color=COLORS[gamma], label=f"γ={gamma}")
    ax.set(title="B. Controller SR by γ", xlabel="round", ylim=(-0.03, 1.03))
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 2]
    for gamma in GAMMAS:
        ax.plot(rounds, [record["V_safe_gamma"][gamma] for record in records], "-o",
                color=COLORS[gamma], lw=1.6)
        ax.plot(rounds, [record["V_full_gamma"][gamma] for record in records], "--",
                color=COLORS[gamma], lw=1.0, alpha=0.8)
    ax.set(title="C. Untilted validity (safe solid / +progress dashed)", xlabel="round",
           ylim=(-0.03, 1.03))

    ax = axes[0, 3]
    ax.plot(adapted_rounds, _values(adapted, "cfm"), "-o", color="#0077bb", label="CFM loss")
    ax.plot(adapted_rounds, _values(adapted, "cfm_last"), "--o", color="#33bbee",
            label="last-step CFM")
    ax2 = ax.twinx()
    ax2.plot(rounds, _values(records, "rep_cos"), "-s", color="#ee7733", label="φ cosine")
    ax.set(title="D. Update and representation drift", xlabel="round", ylabel="loss")
    ax2.set_ylabel("fixed-probe cosine")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes[1, 0]
    ax.plot(adapted_rounds, _values(adapted, "sig_all_med"), "-o", label="all-K median")
    ax.plot(adapted_rounds, _values(adapted, "sig_sel_med"), "-o", label="selected-B median")
    ax.plot(adapted_rounds, _values(adapted, "sig_iqr_med"), "--o", label="all-K IQR")
    ax.set(title="E. RBF posterior uncertainty", xlabel="round", ylabel="σ")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(adapted_rounds, _values(adapted, "ess_med"), "-o", label="ESS/K")
    ax.plot(adapted_rounds, _values(adapted, "ent_med"), "-o", label="entropy")
    ax.axhline(0.375, color="0.5", ls=":", label="calibration target")
    ax2 = ax.twinx()
    ax2.plot(adapted_rounds, _values(adapted, "uplift_med"), "-s", color="#cc3311",
             label="σ uplift")
    ax.set(title="F. Acquisition selectivity", xlabel="round", ylim=(-0.03, 1.03))
    ax2.set_ylabel("selected − pool σ")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes[1, 2]
    ax.plot(rounds, _values(records, "n_D", 0), "-o", label="all verified queries D")
    ax.plot(rounds, _values(records, "n_Dpos", 0), "-o", label="cumulative D+")
    gp_sizes = [record.get("gp_buffer", {}).get("n", 0) for record in records]
    ax.plot(rounds, gp_sizes, "-s", label="round-local GP buffer")
    ax.set(title="G. Learning vs acquisition memory", xlabel="round", ylabel="samples")
    ax.legend(fontsize=8)

    ax = axes[1, 3]
    gathering = np.asarray(_values(adapted, "t_gather", 0.0), dtype=float)
    sampling = np.asarray([
        record.get("gather_timing", {}).get("sampling", 0.0) for record in adapted
    ])
    verifier = np.asarray([
        record.get("gather_timing", {}).get("verifier_wall", 0.0) for record in adapted
    ])
    update = np.asarray(_values(adapted, "t_update", 0.0), dtype=float)
    ax.bar(adapted_rounds, sampling, label="GPU proposal/φ/σ")
    ax.bar(adapted_rounds, verifier, bottom=sampling, label="verifier wall")
    ax.bar(adapted_rounds, np.maximum(gathering - sampling - verifier, 0.0),
           bottom=sampling + verifier, label="bookkeeping/other")
    ax.plot(adapted_rounds, update, "-ko", label="CFM update")
    ax.set(title="H. Measured round time", xlabel="round", ylabel="seconds")
    ax.legend(fontsize=7)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    final = records[-1]
    scene = recipe["scene"]["profile"]["name"]
    figure.suptitle(
        f"Single-arm AFE-RBF — {scene}; {recipe['rollout_replicas']} rollouts/γ/round; "
        f"GP cap {recipe['gp_cap']}; β={recipe['beta']:.4g}; ell={recipe['lengthscale']:.4g}\n"
        f"final Vsafe={final['V_safe']:.3f}, controller SR={final['ctrl_pooled']['SR']:.3f}, "
        f"NVP={final['ctrl_pooled']['NVP']:.3f}",
        fontsize=13,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    figure.savefig(args.out, dpi=140)
    plt.close(figure)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
