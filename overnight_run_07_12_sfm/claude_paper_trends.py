"""Convert raw offline evaluator metrics into the paper trends contract.

Reads one or more ``raw_m{M}_offline_metrics.json`` files produced by
``sfm_b1_offline_eval.py`` (each containing per-round records with the full
per-episode rows) and writes the per-(round,gamma) JSONL consumed by
``safe_flow_expansion@87063d3:scripts/paper_b1_margin50_trends.py``:

    {"round": r, "gamma": g, "m": M, "temp": 1.0,
     "CR": {"mean": .., "se": ..}, "v_safe": {"mean": .., "se": ..},
     "clearance": {"mean": .., "se": ..}, "time": {"mean": .., "se": ..}}

Standard errors are computed numerically from the stored rows (binomial SE
for CR; sample SE of per-trajectory validity fractions; sample SE over
successful episodes for clearance and time) and stored alongside the means,
as the study contract requires.  The band semantics of the paper script are
unchanged: it applies Wilson intervals to CR/v_safe from (mean, m) and
mean +/- 1.96*se to clearance/time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys


def _se_binomial(p, n):
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n else float("nan")


def _mean_se(values):
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return mean, math.sqrt(var / len(finite))


def convert(metrics_paths, jsonl_path):
    rows_out = []
    for path in metrics_paths:
        with open(path) as stream:
            payload = json.load(stream)
        for record in payload["records"]:
            cell = record["cell"]
            for gamma_key in cell["summary"]["per_gamma"]:
                gamma = float(gamma_key)
                rows = [
                    row for row in cell["rows"]
                    if float(row["gamma"]) == gamma
                ]
                m = len(rows)
                cr = sum(bool(r["collision"]) for r in rows) / m
                v_mean, v_se = _mean_se([r["validity"] for r in rows])
                c_mean, c_se = _mean_se(
                    [r["successful_clearance"] for r in rows],
                )
                t_mean, t_se = _mean_se([r["time_to_goal"] for r in rows])
                rows_out.append(dict(
                    round=int(record["round"]), gamma=gamma, m=m, temp=1.0,
                    CR=dict(mean=cr, se=_se_binomial(cr, m)),
                    v_safe=dict(mean=v_mean, se=v_se),
                    clearance=dict(mean=c_mean, se=c_se),
                    time=dict(mean=t_mean, se=t_se),
                ))
    rows_out.sort(key=lambda r: (r["round"], r["gamma"]))
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    with open(jsonl_path, "w") as stream:
        for row in rows_out:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return rows_out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="raw_m*_offline_metrics.json files (rounds merge)")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--stem", default="b1_margin50_metric_trends")
    parser.add_argument(
        "--paper-script",
        default=("/home/dohyun/projects/safe_flow_expansion-claude-plot-"
                 "87063d3/scripts/paper_b1_margin50_trends.py"),
    )
    args = parser.parse_args(argv)
    outdir = os.path.abspath(args.outdir)
    jsonl_path = os.path.join(outdir, f"{args.label}_trends_rows.jsonl")
    rows = convert(args.metrics, jsonl_path)
    print(f"{len(rows)} rows -> {jsonl_path}")
    command = [
        sys.executable, args.paper_script,
        "--arm", f"{args.label}={jsonl_path}",
        "--outdir", outdir, "--stem", args.stem,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
