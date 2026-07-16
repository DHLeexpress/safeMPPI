#!/usr/bin/env python3
"""Requirement-by-requirement P2 audit against the authoritative expert rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GAMMAS = (.1, .2, .3, .4, .5, .7, 1.0)


def load(folder, gamma):
    return json.loads((folder / f"row_g{float(gamma)}.json").read_text())


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-dir", type=Path, default=Path("results/expert_gt"))
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--coverage-close", type=int, default=14,
                    help="operational 'close to 16' threshold; still separately requires > expert")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    expert = [load(args.expert_dir, g) for g in GAMMAS]
    cand = [load(args.candidate_dir, g) for g in GAMMAS]
    rows = []
    for g, e, c in zip(GAMMAS, expert, cand):
        checks = {
            "SR_100": c["SR"] == 1.0,
            "CR_0": c["CR"] == 0.0,
            "safer_than_expert": c["clearance_mean"] > e["clearance_mean"],
            "faster_than_expert": c["time_mean_s"] < e["time_mean_s"],
            "coverage_beats_expert": c["coverage"] > e["coverage"],
            "coverage_close_to_16": c["coverage"] >= args.coverage_close,
            "M_at_least_100": c["M"] >= 100,
        }
        rows.append({"gamma": g, "expert": e, "candidate": c, "checks": checks,
                     "pass": all(checks.values())})
    eclear = np.array([x["clearance_mean"] for x in expert])
    cclear = np.array([x["clearance_mean"] for x in cand])
    etime = np.array([x["time_mean_s"] for x in expert])
    ctime = np.array([x["time_mean_s"] for x in cand])
    trends = {
        "clearance_pearson": corr(eclear, cclear),
        "time_pearson": corr(etime, ctime),
        "low_gamma_has_max_clearance": int(np.argmax(cclear)) == 0,
        "fastest_gamma_is_medium_or_high": int(np.argmin(ctime)) >= 2,
    }
    trends["pass"] = (trends["low_gamma_has_max_clearance"] and
                      trends["fastest_gamma_is_medium_or_high"] and
                      trends["clearance_pearson"] > 0 and trends["time_pearson"] > 0)
    result = {"candidate_dir": str(args.candidate_dir), "coverage_close_threshold": args.coverage_close,
              "rows": rows, "trends": trends,
              "all_goals_pass": all(r["pass"] for r in rows) and trends["pass"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"all_goals_pass": result["all_goals_pass"], "trends": trends,
                      "row_pass": {str(r["gamma"]): r["pass"] for r in rows}}, indent=2))


if __name__ == "__main__":
    main()
