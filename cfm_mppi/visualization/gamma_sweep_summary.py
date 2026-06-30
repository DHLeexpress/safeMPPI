from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {"n": 0, "success_rate": 0.0, "collision_rate": 0.0, "mean_min_clearance": 0.0, "mean_final_goal_distance": 0.0, "mean_planning_time_ms": 0.0}
    def mean(key: str) -> float:
        return float(np.mean([float(r.get(key, 0.0) or 0.0) for r in records]))
    def rate(key: str) -> float:
        return float(np.mean([1.0 if r.get(key) else 0.0 for r in records]))
    return dict(
        n=len(records),
        success_rate=rate("success"),
        collision_rate=rate("collision"),
        goal_reached_rate=rate("goal_reached"),
        mean_min_clearance=mean("min_clearance"),
        mean_final_goal_distance=mean("final_goal_distance"),
        mean_control_effort=mean("control_effort"),
        mean_control_smoothness=mean("control_smoothness"),
        mean_planning_time_ms=1000.0 * mean("planning_wall_time_mean"),
        p95_planning_time_ms=1000.0 * mean("planning_wall_time_p95"),
    )


def summarize_records(records: Sequence[Dict[str, Any]], gammas: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mizuta_cfm_mppi": aggregate([r for r in records if r["method"] == "mizuta_cfm_mppi"]), "safemppi_gamma": {}}
    for gamma in gammas:
        key = f"{gamma:.10g}"
        rows = [r for r in records if r["method"] == "safemppi_gamma" and abs(float(r["gamma"]) - gamma) < 1e-9]
        out["safemppi_gamma"][key] = {"gamma": float(gamma), **aggregate(rows)}
    return out


def write_summary(root: Path, summary: Dict[str, Any], gammas: Sequence[float]) -> None:
    rows = [{"method": "mizuta_cfm_mppi", "gamma": "", **summary["mizuta_cfm_mppi"]}]
    rows += [{"method": "safemppi_gamma", **summary["safemppi_gamma"][f"{g:.10g}"]} for g in gammas]
    with (root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    keys = sorted({k for r in rows for k in r})
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Mizuta CFM-MPPI vs online safeMPPI gamma sweep\n\n")
        f.write("| method | gamma | success | collision | clearance | final dist | plan ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.get('method','')} | {r.get('gamma','')} | {float(r.get('success_rate',0)):.3f} | "
                f"{float(r.get('collision_rate',0)):.3f} | {float(r.get('mean_min_clearance',0)):.3f} | "
                f"{float(r.get('mean_final_goal_distance',0)):.3f} | {float(r.get('mean_planning_time_ms',0)):.2f} |\n"
            )
