"""Overlay actual executed D+ and D0 acquisition windows for ESS comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_scene as SS


BLUE = "#0066ff"
MAGENTA = "#d000b5"
GAMMAS = (0.1, 0.3, 0.5, 1.0)


def _load(path):
    return torch.load(os.path.abspath(path), map_location="cpu", weights_only=False)


def _rollout(state, controls):
    state = np.asarray(state, np.float32).copy()
    values = [state[:2].copy()]
    for action in np.asarray(controls, np.float32):
        state[:2] += SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
        state[2:4] += SS.DT * action
        values.append(state[:2].copy())
    return np.asarray(values)


def _records(gather_dir):
    executed = _load(os.path.join(gather_dir, "executed_round.pt"))
    neutral = _load(os.path.join(gather_dir, "neutral_round.pt"))
    contexts = {int(row["context_id"]): row for row in executed["contexts"]}
    rows = []
    for window in executed["windows"]:
        if int(window["y"]) != 1 or not bool(window["full_h"]):
            continue
        context = contexts[int(window["context_id"])]
        rows.append({
            "population": "D+",
            "scenario_id": int(context["scenario_id"]),
            "gamma": float(context["gamma"]),
            "step": int(context["step"]),
            "state": np.asarray(context["state"], np.float32),
            "controls": np.asarray(window["controls"], np.float32),
            "sigma": float(window["sigma"]),
        })
    for record in neutral["records"]:
        if record["population"] != "D0" or int(record["verifier_y"]) != 0:
            raise RuntimeError("neutral population semantics changed")
        rows.append({
            "population": "D0",
            "scenario_id": int(record["scenario_id"]),
            "gamma": float(record["gamma"]),
            "step": int(record["step"]),
            "state": np.asarray(record["state"], np.float32),
            "controls": np.asarray(record["controls"], np.float32),
            "sigma": float(record["sigma"]),
        })
    return rows


def _diversity(rows):
    if not rows:
        return {"samples": 0, "occupied_025m_bins": 0, "heading_entropy": 0.0,
                "spatial_rms": 0.0, "median_sigma": None}
    xy = np.stack([row["state"][:2] for row in rows])
    bins = np.floor((xy - np.array([SS.TASK_LO, SS.TASK_LO])) / 0.25).astype(int)
    occupied = len({tuple(value) for value in bins})
    actions = np.stack([row["controls"][0] for row in rows])
    angle = np.arctan2(actions[:, 1], actions[:, 0])
    hist, _ = np.histogram(angle, bins=12, range=(-math.pi, math.pi))
    probability = hist[hist > 0] / max(hist.sum(), 1)
    entropy = float(-(probability * np.log(probability)).sum() / np.log(12.0))
    center = xy.mean(0)
    return {
        "samples": len(rows),
        "occupied_025m_bins": occupied,
        "heading_entropy": entropy,
        "spatial_rms": float(np.sqrt(np.square(xy - center).sum(1).mean())),
        "median_sigma": float(np.median([row["sigma"] for row in rows])),
    }


def render(rows_by_name, labels, output_stem):
    figure, axes = plt.subplots(3, 4, figsize=(13.0, 9.7), sharex=True, sharey=True)
    manifest = {"gammas": list(GAMMAS), "rows": {}}
    for row_index, (name, rows) in enumerate(rows_by_name.items()):
        manifest["rows"][name] = {}
        for column, gamma in enumerate(GAMMAS):
            axis = axes[row_index, column]
            selected = [value for value in rows if abs(value["gamma"] - gamma) < 1e-6]
            grouped = defaultdict(list)
            for value in selected:
                grouped[value["scenario_id"]].append(value)
            for values in grouped.values():
                values.sort(key=lambda value: value["step"])
                trajectory = np.stack([value["state"][:2] for value in values])
                axis.plot(trajectory[:, 0], trajectory[:, 1], color="black", lw=0.55, alpha=0.5)
            for value in selected:
                branch = _rollout(value["state"], value["controls"])
                color = BLUE if value["population"] == "D+" else MAGENTA
                axis.plot(branch[:, 0], branch[:, 1], color=color, lw=0.55, alpha=0.13)
            for population, color, marker in (("D+", BLUE, "o"), ("D0", MAGENTA, "s")):
                values = [value for value in selected if value["population"] == population]
                if values:
                    xy = np.stack([value["state"][:2] for value in values])
                    axis.scatter(xy[:, 0], xy[:, 1], s=8, marker=marker, color=color,
                                 alpha=0.72, linewidths=0, zorder=3)
            axis.scatter([0.0], [0.0], s=28, marker="o", facecolor="white",
                         edgecolor="black", linewidth=0.8, zorder=5)
            axis.scatter([SS.GOAL[0]], [SS.GOAL[1]], s=75, marker="*", color="#f0b000",
                         edgecolor="black", linewidth=0.5, zorder=5)
            axis.set_aspect("equal")
            axis.set_xlim(SS.TASK_LO, SS.TASK_HI)
            axis.set_ylim(SS.TASK_LO, SS.TASK_HI)
            axis.grid(alpha=0.16, linewidth=0.5)
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$")
            if column == 0:
                axis.set_ylabel(labels[name])
            if row_index == 2:
                axis.set_xlabel("x [m]")
            positive = sum(value["population"] == "D+" for value in selected)
            neutral = sum(value["population"] == "D0" for value in selected)
            metrics = _diversity(selected)
            metrics.update(Dplus=positive, D0=neutral)
            manifest["rows"][name][str(gamma)] = metrics
            axis.text(0.025, 0.975,
                      f"D+ {positive}  D0 {neutral}\nbins {metrics['occupied_025m_bins']}  "
                      f"Hθ {metrics['heading_entropy']:.2f}",
                      transform=axis.transAxes, va="top", ha="left", fontsize=7.5,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5))
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=BLUE, markersize=5,
                   label=r"executed exact-positive $D^+$"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=MAGENTA, markersize=5,
                   label=r"guided exact-negative neutral $D_0$"),
        plt.Line2D([0], [0], color="black", lw=0.8, label="executed context path"),
    ]
    figure.legend(handles=handles, ncol=3, loc="upper center", frameon=False)
    figure.suptitle("Actual acquisition support after one expansion round", y=0.965)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(output_stem)), exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output_stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    with open(f"{output_stem}.json", "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-gather", required=True)
    parser.add_argument("--ess05-gather", required=True)
    parser.add_argument("--ess01-gather", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args(argv)
    rows = {
        "pretrained": _records(args.pretrained_gather),
        "ess05": _records(args.ess05_gather),
        "ess01": _records(args.ess01_gather),
    }
    labels = {
        "pretrained": "r0 gather",
        "ess05": "post-r1 · ESS 0.5",
        "ess01": "post-r1 · ESS 0.1",
    }
    render(rows, labels, os.path.abspath(args.output_stem))


if __name__ == "__main__":
    main()
