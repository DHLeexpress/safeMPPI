"""Spatial D+ / D0 / G+ acquisition-support figure for guided-positive gathers.

This is the ESS acquisition panel (``sfm_b1_ess_acquisition_viz.py``) extended
with the *tested teacher* population.  Every panel shows, for one policy
variant (row) and one gamma (column), the actual spatial support of the three
populations a guided-positive gather produces on the two shared OOD scenarios:

* ``D+``  blue dots -- the exact full-H10 executed positive windows
  (``executed_round.pt``: ``y == 1`` and ``full_h``).
* ``D0``  magenta squares -- the guided-repair exact-negative neutral windows
  that were executed at the steps where the finite-B acquisition produced no
  verified candidate (``neutral_round.pt``).
* ``G+``  green triangles -- the certified teacher windows harvested by the
  proactive locked-Kazuki generator below ``guided_collect_until_step``
  (``guided_positive_round.pt``).  These are *tested* (exact verifier ``y=1``,
  ``full_h``) but never executed and never in the GP, so they mark support the
  policy could have reached but did not.

Each G+ marker carries the 10-step planned path the teacher proposed,
integrated from the stored context state with the stored ``controls[10, 2]``
through the canonical double integrator (``sfm_metrics2.rollout_positions``).
The thin black line is the actual executed context path (D+ union D0, the two
populations that were executed, ordered by step); G+ is deliberately excluded
from it because those windows were never executed.

The per-panel box reports the three counts plus the guided yield, i.e. how many
of the contexts where the teacher was actually invoked produced at least one
certified window.  The denominator comes from ``repair_trace.pt`` (one trace per
live step, with ``guided_proactive_rows``); when that file is absent the
denominator falls back to the executed-plus-neutral step keys below the guided
horizon and the panel is marked ``est``.

Usage::

    python claude_gplus_acquisition_viz.py \
        --gather .../round_01/gather --label "r0 pretrained" \
        --output-stem .../trackE_gplus_support_r0
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import claude_guided_positive_pilot as GPP
import sfm_b1_kazuki_repair_audit as RA
import sfm_metrics2 as M2
import sfm_scene as SS


BLUE = "#0066ff"
MAGENTA = "#d000b5"
GREEN = "#0a9b46"
GAMMAS = (0.1, 0.3, 0.5, 1.0)
POPULATIONS = ("D+", "D0", "G+")
# G+ is harvested at the *same* live contexts the policy executed, so its
# markers sit on top of the D+/D0 ones by construction.  They are therefore
# drawn last as hollow triangles: a green ring around a blue dot means the
# teacher also certified there, a bare green triangle means the teacher
# certified at a step where the policy itself produced no executed positive.
STYLE = {
    "D+": dict(color=BLUE, marker="o", size=8.0, filled=True, zorder=3.0),
    "D0": dict(color=MAGENTA, marker="s", size=8.0, filled=True, zorder=3.1),
    "G+": dict(color=GREEN, marker="^", size=19.0, filled=False, zorder=3.4),
}
DEFAULT_TITLE = (
    "Actual D+/D0/G+ acquisition support - two shared OOD scenarios - "
    "teacher = full locked Kazuki (t<{until})"
)


def _load(path):
    return torch.load(os.path.abspath(path), map_location="cpu", weights_only=False)


def _rollout(state, controls):
    """Planned 10-step path of one window under the canonical dynamics."""
    return np.asarray(
        M2.rollout_positions(np.asarray(state, np.float32)[:4], controls),
        np.float32,
    )


def _records(gather_dir):
    """Authenticated D+ / D0 / G+ rows for one gather directory."""
    gather_dir = os.path.abspath(gather_dir)
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
    guided_path = os.path.join(gather_dir, RA.GUIDED_POSITIVE_STORE)
    guided_meta = None
    if os.path.exists(guided_path):
        payload = _load(guided_path)
        guided_meta = {
            "guided_collect_until_step": int(
                payload["guided_collect_until_step"]
            ),
            "guided_generator": str(payload["guided_generator"]),
            "Gplus": int(payload["summary"]["Gplus"]),
            "contexts": int(payload["summary"]["contexts"]),
        }
        # The pilot loader is the authentication authority for this store.
        holder, records = GPP._guided_positive_records(guided_path)
        for _, row in records:
            context = holder.contexts[int(row["context_id"])]
            rows.append({
                "population": "G+",
                "scenario_id": int(row["scenario_id"]),
                "gamma": float(row["gamma"]),
                "step": int(row["step"]),
                "state": np.asarray(context["state"], np.float32),
                "controls": np.asarray(row["controls"], np.float32),
                # kazuki_full windows never come from the policy proposal
                # pool, so their stored sigma is 0 by construction.
                "sigma": None,
                "reused_from_repair": bool(row["reused_from_repair"]),
            })
    return rows, guided_meta


def _guided_budget(gather_dir, rows, guided_meta):
    """Per-gamma teacher invocation budget and certified yield.

    Preferred source is ``repair_trace.pt`` (one trace per live step, carrying
    the proactive rows the verifier was actually asked about).  The fallback
    counts the executed-plus-neutral step keys below the guided horizon, which
    misses live steps that produced neither population, so it is reported as an
    estimate.
    """
    if guided_meta is None:
        return {}, "none"
    until = int(guided_meta["guided_collect_until_step"])
    trace_path = os.path.join(os.path.abspath(gather_dir), "repair_trace.pt")
    budget = defaultdict(lambda: dict(
        attempted_contexts=0, guided_windows=0,
        certified_windows=0, certified_contexts=set(),
    ))
    if os.path.exists(trace_path):
        payload = _load(trace_path)
        for trace in payload["traces"]:
            if int(trace["step"]) >= until:
                continue
            cell = budget[round(float(trace["gamma"]), 8)]
            cell["attempted_contexts"] += 1
            for row in trace.get("guided_proactive_rows") or []:
                cell["guided_windows"] += 1
                result = row.get("result") or {}
                if int(result.get("y", -1)) == 1 and result.get("full_h"):
                    cell["certified_windows"] += 1
                    cell["certified_contexts"].add(
                        (int(trace["scenario_id"]), int(trace["step"]))
                    )
        source = "repair_trace"
    else:
        seen = set()
        for row in rows:
            if row["population"] == "G+" or int(row["step"]) >= until:
                continue
            key = (round(float(row["gamma"]), 8), int(row["scenario_id"]),
                   int(row["step"]))
            if key in seen:
                continue
            seen.add(key)
            budget[key[0]]["attempted_contexts"] += 1
        for row in rows:
            if row["population"] != "G+":
                continue
            cell = budget[round(float(row["gamma"]), 8)]
            cell["certified_windows"] += 1
            cell["guided_windows"] += 1
            cell["certified_contexts"].add(
                (int(row["scenario_id"]), int(row["step"]))
            )
        source = "estimated_from_shards"
    resolved = {}
    for gamma, cell in budget.items():
        resolved[gamma] = dict(
            attempted_contexts=int(cell["attempted_contexts"]),
            guided_windows=int(cell["guided_windows"]),
            certified_windows=int(cell["certified_windows"]),
            certified_contexts=len(cell["certified_contexts"]),
        )
    return resolved, source


def _protocol(gather_dir):
    """Optional GP/ESS protocol header, mirroring the ESS panel."""
    path = os.path.join(os.path.abspath(gather_dir), "repair_trace.pt")
    if not os.path.exists(path):
        return None
    protocol = _load(path)["protocol"]
    return {
        "beta": float(protocol["beta"]),
        "calibrated_ess_over_K": float(protocol["calibrated_ess_over_K"]),
        "realized_ess_over_K": float(protocol["realized_ess_over_K"]),
        "uplift": float(protocol["acquisition"]["uplift"]),
        "gp_effective_cap": int(protocol["gp_selection"]["effective_cap"]),
    }


def _diversity(rows):
    """Reference ESS-panel diversity summary for one population subset."""
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
    sigmas = [row["sigma"] for row in rows if row.get("sigma") is not None]
    return {
        "samples": len(rows),
        "occupied_025m_bins": occupied,
        "heading_entropy": entropy,
        "spatial_rms": float(np.sqrt(np.square(xy - center).sum(1).mean())),
        "median_sigma": (float(np.median(sigmas)) if sigmas else None),
    }


def _panel_metrics(selected, budget_cell):
    metrics = {
        "counts": {
            population: sum(
                row["population"] == population for row in selected
            )
            for population in POPULATIONS
        },
        "diversity": {
            population: _diversity(
                [row for row in selected if row["population"] == population]
            )
            for population in POPULATIONS
        },
    }
    metrics["diversity"]["all"] = _diversity(selected)
    metrics["guided_budget"] = dict(budget_cell or {})
    return metrics


def _count_text(metrics, budget_source):
    counts = metrics["counts"]
    lines = [
        f"D+ {counts['D+']}  D0 {counts['D0']}  G+ {counts['G+']}",
    ]
    budget = metrics["guided_budget"]
    if budget and int(budget.get("attempted_contexts", 0)) > 0:
        attempted = int(budget["attempted_contexts"])
        certified = int(budget["certified_contexts"])
        suffix = "" if budget_source == "repair_trace" else " est"
        lines.append(
            f"yield {certified}/{attempted} ctx "
            f"({100.0 * certified / attempted:.0f}%){suffix}"
        )
    else:
        lines.append("yield n/a")
    lines.append(
        f"bins {metrics['diversity']['all']['occupied_025m_bins']}  "
        f"Hθ {metrics['diversity']['all']['heading_entropy']:.2f}"
    )
    return "\n".join(lines)


def render(gathers, output_stem, *, gammas=GAMMAS, title=None,
           show_planned_paths=True):
    """One row per gather, one column per gamma."""
    gammas = tuple(float(value) for value in gammas)
    rows_by_name = {}
    order = []
    manifest = {
        "gammas": list(gammas),
        "populations": list(POPULATIONS),
        "rows": {},
    }
    until_steps = set()
    for label, gather_dir in gathers:
        rows, guided_meta = _records(gather_dir)
        budget, budget_source = _guided_budget(gather_dir, rows, guided_meta)
        rows_by_name[label] = dict(
            rows=rows, budget=budget, budget_source=budget_source,
            guided=guided_meta, gather=os.path.abspath(gather_dir),
            protocol=_protocol(gather_dir),
        )
        order.append(label)
        if guided_meta is not None:
            until_steps.add(int(guided_meta["guided_collect_until_step"]))
    until = min(until_steps) if until_steps else 0

    n_rows = len(order)
    # Fixed 0.95in header (title + one legend row) regardless of row count.
    header = 0.95
    fig_height = 3.05 * n_rows + header
    figure, axes = plt.subplots(
        n_rows, len(gammas), figsize=(13.0, fig_height),
        sharex=True, sharey=True, squeeze=False,
    )
    for row_index, label in enumerate(order):
        bundle = rows_by_name[label]
        manifest["rows"][label] = {
            "gather": bundle["gather"],
            "guided": bundle["guided"],
            "guided_budget_source": bundle["budget_source"],
            "protocol": bundle["protocol"],
            "panels": {},
        }
        for column, gamma in enumerate(gammas):
            axis = axes[row_index, column]
            selected = [
                row for row in bundle["rows"]
                if abs(row["gamma"] - gamma) < 1e-6
            ]
            executed = [
                row for row in selected if row["population"] in ("D+", "D0")
            ]
            grouped = defaultdict(list)
            for row in executed:
                grouped[row["scenario_id"]].append(row)
            for values in grouped.values():
                values.sort(key=lambda value: value["step"])
                trajectory = np.stack([value["state"][:2] for value in values])
                axis.plot(trajectory[:, 0], trajectory[:, 1], color="black",
                          lw=0.55, alpha=0.5)
            for row in executed:
                branch = _rollout(row["state"], row["controls"])
                color = BLUE if row["population"] == "D+" else MAGENTA
                axis.plot(branch[:, 0], branch[:, 1], color=color, lw=0.55,
                          alpha=0.13)
            if show_planned_paths:
                for row in selected:
                    if row["population"] != "G+":
                        continue
                    branch = _rollout(row["state"], row["controls"])
                    axis.plot(branch[:, 0], branch[:, 1], color=GREEN, lw=0.5,
                              alpha=0.16, zorder=2)
            for population in POPULATIONS:
                values = [
                    row for row in selected if row["population"] == population
                ]
                if not values:
                    continue
                style = STYLE[population]
                xy = np.stack([row["state"][:2] for row in values])
                if style["filled"]:
                    axis.scatter(
                        xy[:, 0], xy[:, 1], s=style["size"],
                        marker=style["marker"], color=style["color"],
                        alpha=0.72, linewidths=0, zorder=style["zorder"],
                    )
                else:
                    axis.scatter(
                        xy[:, 0], xy[:, 1], s=style["size"],
                        marker=style["marker"], facecolors="none",
                        edgecolors=style["color"], alpha=0.85, linewidths=0.55,
                        zorder=style["zorder"],
                    )
            axis.scatter([0.0], [0.0], s=28, marker="o", facecolor="white",
                         edgecolor="black", linewidth=0.8, zorder=5)
            axis.scatter([SS.GOAL[0]], [SS.GOAL[1]], s=75, marker="*",
                         color="#f0b000", edgecolor="black", linewidth=0.5,
                         zorder=5)
            axis.set_aspect("equal")
            axis.set_xlim(SS.TASK_LO, SS.TASK_HI)
            axis.set_ylim(SS.TASK_LO, SS.TASK_HI)
            axis.grid(alpha=0.16, linewidth=0.5)
            if row_index == 0:
                axis.set_title(rf"$\gamma={gamma:g}$")
            if column == 0:
                axis.set_ylabel(label, fontsize=9.0)
            if row_index == n_rows - 1:
                axis.set_xlabel("x [m]")
            metrics = _panel_metrics(
                selected, bundle["budget"].get(round(gamma, 8)),
            )
            manifest["rows"][label]["panels"][f"{gamma:g}"] = metrics
            axis.text(0.025, 0.975, _count_text(metrics, bundle["budget_source"]),
                      transform=axis.transAxes, va="top", ha="left",
                      fontsize=7.0,
                      bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.72, pad=1.5))
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=BLUE,
                   markersize=5, label=r"executed exact-positive $D^+$"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=MAGENTA,
                   markersize=5,
                   label=r"guided exact-negative neutral $D_0$"),
        plt.Line2D([0], [0], marker="^", linestyle="", markersize=6.0,
                   markerfacecolor="none", markeredgecolor=GREEN,
                   markeredgewidth=0.9, color="none",
                   label=r"certified teacher window $G^+$ (never executed)"),
        plt.Line2D([0], [0], color=GREEN, lw=0.9, alpha=0.5,
                   label="teacher 10-step planned path"),
        plt.Line2D([0], [0], color="black", lw=0.8,
                   label="executed context path"),
    ]
    figure.legend(handles=handles, ncol=5, frameon=False, fontsize=8.5,
                  loc="upper center",
                  bbox_to_anchor=(0.5, 1.0 - 0.42 / fig_height))
    figure.suptitle(
        title if title else DEFAULT_TITLE.format(until=until),
        y=1.0 - 0.16 / fig_height, fontsize=11.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 1.0 - header / fig_height))
    output_stem = os.path.abspath(output_stem)
    os.makedirs(os.path.dirname(output_stem), exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = f"{output_stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    with open(f"{output_stem}.json", "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
    written.append(f"{output_stem}.json")
    return manifest, written


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gather", action="append", required=True,
        help="gather directory (executed/neutral/[guided] round shards); "
             "repeat once per figure row, in row order",
    )
    parser.add_argument(
        "--label", action="append", default=None,
        help="row label; repeat to match --gather (\\n allowed)",
    )
    parser.add_argument("--output-stem", required=True)
    parser.add_argument(
        "--copy-to", action="append", default=None,
        help="directory to copy the rendered png/pdf/json into",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--no-planned-paths", action="store_true",
        help="draw only the G+ markers, not their 10-step planned paths",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    gathers = [os.path.abspath(path) for path in args.gather]
    labels = args.label
    if labels is None:
        labels = [os.path.basename(os.path.dirname(path)) for path in gathers]
    if len(labels) != len(gathers):
        raise ValueError("--label must be given once per --gather")
    labels = [label.replace("\\n", "\n") for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("row labels must be distinct")
    manifest, written = render(
        list(zip(labels, gathers)), args.output_stem,
        gammas=tuple(args.gammas), title=args.title,
        show_planned_paths=not args.no_planned_paths,
    )
    for directory in (args.copy_to or []):
        os.makedirs(os.path.abspath(directory), exist_ok=True)
        for path in written:
            shutil.copy2(path, os.path.abspath(directory))
    for path in written:
        print(path)
    for label, row in manifest["rows"].items():
        for gamma, panel in sorted(row["panels"].items()):
            counts = panel["counts"]
            budget = panel["guided_budget"]
            print(
                f"[{label.replace(chr(10), ' ')}] gamma {gamma}: "
                f"D+ {counts['D+']} D0 {counts['D0']} G+ {counts['G+']} "
                f"yield {budget.get('certified_contexts', 0)}/"
                f"{budget.get('attempted_contexts', 0)} ctx"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
