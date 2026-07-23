#!/usr/bin/env python3
"""Aggregate the completed two-round SFM alpha/replay factorial.

The per-arm evaluator remains the source of scientific metrics.  This module
only validates their shared contracts, renders pooled comparisons, and
computes paired scenario-bootstrap changes against the common r0 checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALPHAS = (0.0, 0.01, 0.1)
REPLAY_EPOCHS = (1, 10, 100)
ROUNDS = (0, 1, 2)


def arm_name(alpha: float, epochs: int) -> str:
    return f"margin_alpha{str(float(alpha)).replace('.', 'p')}_epochs{int(epochs):03d}"


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _metric(cell: dict, key: str) -> float | None:
    if key in ("SR", "CR", "timeout", "V_safe"):
        return float(cell[key])
    entry = (
        cell["successful_clearance"]
        if key == "clearance"
        else cell["successful_time_to_goal"]
    )
    return None if entry["mean"] is None else float(entry["mean"])


def _post_expansion_key(row: dict) -> tuple:
    """Safety-first deterministic selection among measured r1/r2 cells."""
    clearance = -math.inf if row["clearance"] is None else float(row["clearance"])
    time = math.inf if row["time"] is None else float(row["time"])
    return (
        float(row["CR"]),
        -float(row["SR"]),
        -clearance,
        time,
        int(row["round"]),
        float(row["alpha"]),
        int(row["replay_epochs"]),
    )


def paired_cluster_delta(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    key: str,
    *,
    seed: int = 20260723,
    draws: int = 20_000,
) -> dict:
    baseline = {
        (int(row["episode"]), float(row["gamma"])): float(bool(row[key]))
        for row in baseline_rows
    }
    candidate = {
        (int(row["episode"]), float(row["gamma"])): float(bool(row[key]))
        for row in candidate_rows
    }
    if set(baseline) != set(candidate):
        raise ValueError("paired evaluator rows do not share the same scenario/gamma keys")
    episode_ids = sorted({episode for episode, _ in baseline})
    per_episode = np.asarray([
        np.mean([
            candidate[(episode, gamma)] - baseline[(episode, gamma)]
            for current_episode, gamma in baseline if current_episode == episode
        ])
        for episode in episode_ids
    ], dtype=float)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(per_episode), size=(int(draws), len(per_episode))
    )
    samples = per_episode[indices].mean(axis=1)
    return {
        "estimate": float(per_episode.mean()),
        "scenario_cluster_bootstrap95": list(
            map(float, np.quantile(samples, (0.025, 0.975)))
        ),
        "paired_scenarios": len(episode_ids),
        "paired_gamma_cells": len(baseline),
    }


def _load(run_root: Path) -> tuple[list[dict], dict, str]:
    rows = []
    reference_payload = None
    noise_sha = None
    baseline_cell_key = None
    for alpha in ALPHAS:
        for epochs in REPLAY_EPOCHS:
            arm = arm_name(alpha, epochs)
            training_path = run_root / "arms" / arm / "COMPLETE.json"
            evaluation_path = (
                run_root / "evaluation" / arm / "raw_m50_r0_r2_metrics.json"
            )
            with training_path.open() as stream:
                training = json.load(stream)
            with evaluation_path.open() as stream:
                evaluation = json.load(stream)
            if training.get("status") != "R2_ALPHA_REPLAY_COMPLETE":
                raise RuntimeError(f"incomplete training arm: {training_path}")
            if evaluation.get("status") != "SFM_B1_R2_RAW_M50_COMPLETE":
                raise RuntimeError(f"incomplete evaluation arm: {evaluation_path}")
            current_noise = evaluation["noise_bank"]["sha256"]
            if noise_sha is None:
                noise_sha = current_noise
                reference_payload = evaluation["archived_M100_reference"]
            elif current_noise != noise_sha:
                raise RuntimeError("arm evaluations do not share one CRN bank")
            records = evaluation["records"]
            if [int(record["round"]) for record in records] != list(ROUNDS):
                raise RuntimeError(f"{arm} does not contain r0/r1/r2")
            if baseline_cell_key is None:
                baseline_cell_key = records[0]["cell"]["cell_key"]
            elif records[0]["cell"]["cell_key"] != baseline_cell_key:
                raise RuntimeError("arm evaluations do not reuse the same r0 cell")
            history = {int(item["round"]): item for item in training["history"]}
            for record in records:
                round_i = int(record["round"])
                cell = record["cell"]["summary"]["pooled"]
                item = {
                    "arm": arm,
                    "alpha": float(alpha),
                    "replay_epochs": int(epochs),
                    "round": round_i,
                    "SR": _metric(cell, "SR"),
                    "CR": _metric(cell, "CR"),
                    "timeout": _metric(cell, "timeout"),
                    "V_safe": _metric(cell, "V_safe"),
                    "clearance": _metric(cell, "clearance"),
                    "time": _metric(cell, "time"),
                    "cell_key": record["cell"]["cell_key"],
                    "checkpoint_sha256": record["cell"]["checkpoint_sha256"],
                    "evaluation_rows": record["cell"]["rows"],
                    "training": None if round_i == 0 else history[round_i],
                }
                rows.append(item)
    assert reference_payload is not None and noise_sha is not None
    return rows, reference_payload, noise_sha


def _render(rows: list[dict], outdir: Path, best: dict) -> list[str]:
    specs = (
        ("CR", "Collision rate"),
        ("V_safe", r"$V_{\mathrm{safe}}$"),
        ("clearance", "Successful min. clearance [m]"),
        ("time", "Successful time-to-goal [s]"),
    )
    colors = {1: "#0072B2", 10: "#E69F00", 100: "#CC79A7"}
    linestyles = {0.0: "-", 0.01: "--", 0.1: ":"}
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
    })
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9))
    for axis, (metric, title) in zip(axes.flat, specs):
        for alpha in ALPHAS:
            for epochs in REPLAY_EPOCHS:
                values = sorted(
                    [
                        row for row in rows
                        if row["alpha"] == alpha
                        and row["replay_epochs"] == epochs
                    ],
                    key=lambda row: row["round"],
                )
                axis.plot(
                    [row["round"] for row in values],
                    [
                        np.nan if row[metric] is None else row[metric]
                        for row in values
                    ],
                    color=colors[epochs],
                    linestyle=linestyles[alpha],
                    marker="o",
                    lw=2,
                    alpha=0.85,
                )
        axis.scatter(
            [best["round"]],
            [best[metric]],
            marker="*",
            s=210,
            c="#009E73",
            edgecolors="black",
            zorder=8,
        )
        axis.set(
            title=title,
            xlabel="expansion round",
            xticks=ROUNDS,
        )
        axis.grid(alpha=0.25)
        if metric in ("CR", "V_safe"):
            axis.set_ylim(-0.03, 1.03)
    handles = [
        plt.Line2D([0], [0], color=colors[value], lw=2.5, label=f"{value} epochs")
        for value in REPLAY_EPOCHS
    ]
    handles.extend([
        plt.Line2D(
            [0], [0], color="black", linestyle=linestyles[value], lw=2,
            label=rf"$\alpha={value:g}$",
        )
        for value in ALPHAS
    ])
    handles.append(plt.Line2D(
        [0], [0], marker="*", color="none", markerfacecolor="#009E73",
        markeredgecolor="black", markersize=14, label="best post-expansion cell",
    ))
    figure.legend(
        handles=handles, loc="upper center", ncol=7, frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"factorial_pooled_curves.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def run(run_root: str) -> dict:
    root = Path(run_root).resolve()
    rows, archived, noise_sha = _load(root)
    baseline = next(
        row for row in rows
        if row["round"] == 0
        and row["alpha"] == 0.0
        and row["replay_epochs"] == 1
    )
    candidates = [row for row in rows if row["round"] > 0]
    best = min(candidates, key=_post_expansion_key)
    output_dir = root / "evaluation" / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "factorial_pooled_metrics.csv"
    fields = (
        "arm", "alpha", "replay_epochs", "round",
        "SR", "CR", "timeout", "V_safe", "clearance", "time",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows, key=lambda item: (
                item["alpha"], item["replay_epochs"], item["round"]
            )
        ):
            writer.writerow({key: row[key] for key in fields})
    outputs = _render(rows, output_dir, best)
    summary = {
        "status": "SFM_B1_R2_FACTORIAL_AGGREGATE_COMPLETE",
        "run_root": str(root),
        "training_source_commit": "58ec896f87a5859149a39f5f7796560cd53da518",
        "noise_bank_sha256": noise_sha,
        "common_r0_cell_key": baseline["cell_key"],
        "baseline": {key: baseline[key] for key in fields},
        "best_post_expansion": {key: best[key] for key in fields},
        "paired_changes_best_minus_r0": {
            key: paired_cluster_delta(
                baseline["evaluation_rows"],
                best["evaluation_rows"],
                key,
                seed=20260723 + index,
            )
            for index, key in enumerate(("success", "collision", "timeout", "v_safe"))
        },
        "archived_M100_reference": archived,
        "selection_rule": (
            "post-expansion cells only; lower raw CR, then higher raw SR, then "
            "higher successful-only clearance, then lower successful-only time"
        ),
        "artifacts": {
            "csv": str(csv_path),
            "csv_sha256": _sha256_file(csv_path),
            "figures": [
                {"path": path, "sha256": _sha256_file(path)}
                for path in outputs
            ],
            "best_per_gamma_png": str(
                root / "evaluation" / best["arm"] / "raw_m50_r0_r2_curves.png"
            ),
        },
    }
    summary_path = output_dir / "factorial_summary.json"
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    result = run(parser.parse_args().run_root)
    print(json.dumps({
        "status": result["status"],
        "baseline": result["baseline"],
        "best_post_expansion": result["best_post_expansion"],
        "paired_changes": result["paired_changes_best_minus_r0"],
        "artifacts": result["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
