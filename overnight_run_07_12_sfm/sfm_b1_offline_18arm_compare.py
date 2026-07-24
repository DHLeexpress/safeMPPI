"""Paired four-metric comparison of margin and SafeMPPI-cost 9-arm sweeps."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_sfm_b1_offline_9arm as RUN


STATUS = "SFM_B1_OFFLINE_SELECTOR_COMPARISON_COMPLETE"
SELECTOR_ORDER = ("margin", "safemppi_cost", "balanced_rank")


def _read_json(path):
    with open(path) as stream:
        return json.load(stream)


def _load(root, selector):
    root = Path(root).resolve()
    delivery = _read_json(root / "DELIVERY_COMPLETE.json")
    if delivery.get("status") != "SFM_B1_OFFLINE_9ARM_DELIVERY_COMPLETE":
        raise ValueError(f"incomplete 9-arm delivery: {root}")
    contract = dict(delivery["contract"])
    observed = contract.get("execution_selector", "margin")
    if observed != selector:
        raise ValueError(
            f"expected {selector} sweep, observed {observed} at {root}"
        )
    aggregate = _read_json(
        root / "evaluation" / "aggregate" / "AGGREGATE_COMPLETE.json"
    )
    rows = list(aggregate["rows"])
    if len(rows) != 9 * (RUN.ROUNDS + 1):
        raise ValueError(f"expected 99 aggregate rows at {root}, got {len(rows)}")
    for row in rows:
        row["selector"] = selector
    return root, delivery, contract, aggregate, rows


def _paired_r0(rows):
    fields = ("SR", "CR", "timeout", "Validity", "clearance", "time_to_goal")
    values = [
        tuple(row[field] for field in fields)
        for row in rows if int(row["round"]) == 0
    ]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError("all 18 arms must share an identical raw-M50 r0 cell")
    return dict(zip(fields, values[0]))


def _plot(rows, output):
    selectors = tuple(
        selector for selector in SELECTOR_ORDER
        if any(row["selector"] == selector for row in rows)
    )
    combinations = [
        (float(alpha), int(exposure))
        for alpha in RUN.ALPHAS for exposure in RUN.EXPOSURE_EPOCHS
    ]
    colors = plt.get_cmap("tab10")
    color_for = {
        combination: colors(index)
        for index, combination in enumerate(combinations)
    }
    linestyles = {
        "margin": "-",
        "safemppi_cost": "--",
        "balanced_rank": ":",
    }
    specs = (
        ("CR", "Collision rate", (-.03, 1.03)),
        ("Validity", "Validity", (-.03, 1.03)),
        ("clearance", "Min. clearance [m]", None),
        ("time_to_goal", "Time-to-goal [s]", None),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    for axis, (key, title, ylim) in zip(axes.flat, specs):
        for selector in selectors:
            for alpha, exposure in combinations:
                values = [
                    row for row in rows
                    if row["selector"] == selector
                    and float(row["alpha"]) == alpha
                    and int(row["exposure_epochs"]) == exposure
                ]
                values.sort(key=lambda row: int(row["round"]))
                axis.plot(
                    [row["round"] for row in values],
                    [
                        float("nan") if row[key] is None else float(row[key])
                        for row in values
                    ],
                    color=color_for[(alpha, exposure)],
                    linestyle=linestyles[selector],
                    linewidth=1.55, alpha=.9,
                )
        axis.set_title(title)
        axis.set_xlabel("Expansion round")
        axis.set_xticks(range(RUN.ROUNDS + 1))
        axis.grid(alpha=.24)
        if ylim is not None:
            axis.set_ylim(*ylim)
    handles = [
        plt.Line2D(
            [0], [0], color=color_for[(alpha, exposure)], lw=2.6,
            label=rf"$\alpha={alpha:g}$, exposure={exposure}",
        )
        for alpha, exposure in combinations
    ]
    selector_labels = {
        "margin": "max one-step margin",
        "safemppi_cost": "native SafeMPPI cost",
        "balanced_rank": "balanced safety + performance rank",
    }
    handles.extend(
        plt.Line2D(
            [0], [0], color="black", lw=2.4,
            linestyle=linestyles[selector], label=selector_labels[selector],
        )
        for selector in selectors
    )
    figure.legend(
        handles=handles, ncol=4, loc="upper center",
        frameon=False, fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 1, .89))
    artifacts = []
    for suffix in ("png", "pdf"):
        path = output / f"paired_{9 * len(selectors)}arm_raw_m50.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        artifacts.append(str(path.resolve()))
    plt.close(figure)
    return artifacts


def compare(margin_root, cost_root, output_dir, *, balanced_root=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    loaded = [
        _load(margin_root, "margin"),
        _load(cost_root, "safemppi_cost"),
    ]
    if balanced_root is not None:
        loaded.append(_load(balanced_root, "balanced_rank"))
    rows = [row for item in loaded for row in item[-1]]
    r0 = _paired_r0(rows)
    csv_path = output / f"paired_{9 * len(loaded)}arm_raw_m50.csv"
    fields = (
        "selector", "arm", "alpha", "exposure_epochs", "round",
        "SR", "CR", "timeout", "Validity", "clearance", "time_to_goal",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    figures = _plot(rows, output)
    best_by_selector = {}
    selectors = tuple(item[-1][0]["selector"] for item in loaded)
    for selector in selectors:
        candidates = [
            row for row in rows
            if row["selector"] == selector and int(row["round"]) > 0
        ]
        best_by_selector[selector] = min(candidates, key=RUN._screening_key)
    report = {
        "status": STATUS,
        "comparison_role": (
            "paired common-bank raw-M50 screening; selector is the only "
            "factor added to the existing alpha x exposure grid"
        ),
        "margin_root": str(loaded[0][0]),
        "safemppi_cost_root": str(loaded[1][0]),
        "balanced_rank_root": (
            None if len(loaded) == 2 else str(loaded[2][0])
        ),
        "paired_r0": r0,
        "best_screening_cell_by_selector": best_by_selector,
        "rows": len(rows),
        "csv": str(csv_path.resolve()),
        "figures": figures,
    }
    marker = output / "COMPARISON_COMPLETE.json"
    temporary = marker.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.replace(temporary, marker)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-root", required=True)
    parser.add_argument("--safemppi-cost-root", required=True)
    parser.add_argument("--balanced-rank-root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    compare(
        args.margin_root, args.safemppi_cost_root, args.output_dir,
        balanced_root=args.balanced_rank_root,
    )


if __name__ == "__main__":
    main()
