"""Paired-difference confirmation analysis for the Stage-E M100 comparison.

Inputs: the raw-evaluator metrics JSON (r0 + selected checkpoint on one CRN
bank) and the locked-Kazuki metrics JSON on the same episode bank.  All three
methods share the same scenario ids per gamma (CRN pedestrian banks); r0 and
the selected checkpoint additionally share the same latent bank.

For each pair (selected - r0, selected - Kazuki, Kazuki - r0) and each of the
four study metrics we report the mean difference with a 95% scenario-cluster
bootstrap interval: episodes are grouped by scenario id (keeping all seven
paired gamma rows together) and clusters are resampled with replacement.
Collision and Validity use all episodes; clearance and time-to-goal are
success-conditioned, so each bootstrap draw recomputes the per-method mean
over its successful episodes inside the resampled clusters (a paired
difference of success-conditioned means, not a per-episode paired delta).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


METRICS = ("CR", "validity", "successful_clearance", "time_to_goal")


def _rows(path, record_index=None, expect_label=None):
    with open(path) as stream:
        payload = json.load(stream)
    if "records" in payload:
        record = payload["records"][record_index]
        if expect_label is not None and record["label"] != expect_label:
            raise ValueError(
                f"{path}: expected label {expect_label}, got {record['label']}"
            )
        return payload, record["cell"]["rows"]
    return payload, payload["rows"]


def _by_scenario(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    return grouped


def _metric_values(rows, metric):
    if metric == "CR":
        return [float(bool(row["collision"])) for row in rows]
    if metric == "validity":
        return [float(row["validity"]) for row in rows]
    key = (
        "successful_clearance" if metric == "successful_clearance"
        else "time_to_goal"
    )
    return [
        float(row[key]) for row in rows if row[key] is not None
    ]


def _cluster_mean(grouped, scenarios, metric):
    values = []
    for scenario in scenarios:
        values.extend(_metric_values(grouped[scenario], metric))
    return float(np.mean(values)) if values else float("nan")


def paired_difference(rows_a, rows_b, *, seed, draws=10_000):
    grouped_a, grouped_b = _by_scenario(rows_a), _by_scenario(rows_b)
    scenarios = sorted(set(grouped_a) & set(grouped_b))
    if set(grouped_a) != set(grouped_b):
        raise ValueError("methods do not share the scenario bank")
    generator = np.random.default_rng(seed)
    out = {}
    for metric in METRICS:
        point = (
            _cluster_mean(grouped_a, scenarios, metric)
            - _cluster_mean(grouped_b, scenarios, metric)
        )
        samples = []
        for _ in range(draws):
            resample = generator.choice(scenarios, size=len(scenarios))
            samples.append(
                _cluster_mean(grouped_a, resample, metric)
                - _cluster_mean(grouped_b, resample, metric)
            )
        finite = [s for s in samples if np.isfinite(s)]
        low, high = np.quantile(finite, [0.025, 0.975])
        out[metric] = dict(
            difference=point, ci95=[float(low), float(high)],
            draws=len(finite),
        )
    return out


def summarize_method(rows):
    n = len(rows)
    values = {m: _metric_values(rows, m) for m in METRICS}
    return dict(
        n=n,
        SR=float(np.mean([bool(r["success"]) for r in rows])),
        CR=float(np.mean(values["CR"])),
        timeout=float(np.mean([bool(r["timeout"]) for r in rows])),
        Validity=float(np.mean(values["validity"])),
        successful_clearance=float(np.mean(values["successful_clearance"])),
        successful_time_to_goal=float(np.mean(values["time_to_goal"])),
        successes=int(sum(bool(r["success"]) for r in rows)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-metrics", required=True,
                        help="raw M100 metrics json containing r0 + selected")
    parser.add_argument("--kazuki-metrics", required=True)
    parser.add_argument("--selected-label", required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    raw_payload, r0_rows = _rows(args.raw_metrics, 0, "r0")
    _, selected_rows = _rows(args.raw_metrics, 1, args.selected_label)
    kazuki_payload, kazuki_rows = _rows(args.kazuki_metrics)

    result = dict(
        status="CLAUDE_M100_CONFIRMATION_ANALYSIS",
        bank=raw_payload.get("bank"),
        kazuki_config=kazuki_payload.get("kazuki_config"),
        methods=dict(
            r0=summarize_method(r0_rows),
            selected=summarize_method(selected_rows),
            kazuki=summarize_method(kazuki_rows),
        ),
        paired_differences=dict(
            selected_minus_r0=paired_difference(
                selected_rows, r0_rows, seed=args.seed,
            ),
            selected_minus_kazuki=paired_difference(
                selected_rows, kazuki_rows, seed=args.seed + 1,
            ),
            kazuki_minus_r0=paired_difference(
                kazuki_rows, r0_rows, seed=args.seed + 2,
            ),
        ),
        semantics=(
            "scenario-cluster bootstrap (10k draws) keeping the seven paired "
            "gamma rows per scenario together; clearance/time are "
            "success-conditioned means recomputed inside each draw"
        ),
    )
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["methods"], indent=1))
    print(json.dumps(result["paired_differences"]["selected_minus_r0"],
                     indent=1))


if __name__ == "__main__":
    main()
