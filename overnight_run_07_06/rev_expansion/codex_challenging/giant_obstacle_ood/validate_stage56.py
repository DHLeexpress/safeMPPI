#!/usr/bin/env python3
"""Independent mechanics/artifact audit for giant-obstacle Stages 5--6."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
STAGE5 = HERE / "stage_results/05_window_expand"
RUNS = STAGE5 / "runs/temp0.5_stable"
OUT = HERE / "stage_results/06_exact_reports"
ARMS = ("full", "no_socp", "no_progress", "no_curriculum")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: dict[str, object] = {}
    budgets = {}
    probes = {}
    for arm in ARMS:
        run = RUNS / arm
        recipe = json.loads((run / "recipe.json").read_text())
        probes[arm] = rows(run / "probe.jsonl")
        budgets[arm] = {int(key): int(value) for key, value in json.loads(
            (run / "accepted_window_budget.json").read_text()).items()}
        require(len(probes[arm]) == 20, f"{arm}: expected 20 probes")
        require(set(budgets[arm]) == set(range(1, 21)), f"{arm}: incomplete budget")
        require(recipe["aggregation_semantics"] == "window_native_v1", f"{arm}: wrong aggregation")
        require(recipe["trajectory_status"] == "diagnostic_only", f"{arm}: trajectory gate active")
        require(recipe["goal_reach_gate"] is False, f"{arm}: reach is a gather gate")
        require(recipe["gather_temperature"] == .5 == recipe["evaluation_temperature"],
                f"{arm}: unmatched temperature")
        require(recipe["demo_frac_schedule"] == [[0, .5], [11, .25]],
                f"{arm}: wrong demo schedule")
        require(recipe["lr"] == 5e-6, f"{arm}: unstable calibration LR promoted")
        require(all(row["gather_audit"]["ready"] for row in probes[arm]), f"{arm}: gather starved")
        require(all(row["functional_step"] > 0 for row in probes[arm]), f"{arm}: missing update")
        require(not any(row["rollback"] for row in probes[arm]), f"{arm}: rollback detected")
        require([row["demo_req"] for row in probes[arm]][:10] == [8] * 10,
                f"{arm}: early demo mass is not 50%")
        require([row["demo_req"] for row in probes[arm]][10:] == [4] * 10,
                f"{arm}: late demo mass is not 25%")
        checks[f"{arm}_accepted_total"] = sum(budgets[arm].values())
    require(budgets["full"] == budgets["no_curriculum"], "-Curriculum is not exact-count matched")
    checks["no_curriculum_exact_window_count_match"] = True

    def total(arm: str, key: str) -> int:
        return sum(int(row["gather_audit"].get(key, 0)) for row in probes[arm])

    require(total("full", "socp_evaluated") > 0 and total("full", "progress_evaluated") > 0,
            "Full did not execute both validity predicates")
    require(total("no_socp", "socp_evaluated") == 0, "-SOCP called SOCP")
    require(total("no_socp", "safe_space_evaluated") > 0, "-SOCP did not use geometric safety")
    require(total("no_progress", "progress_evaluated") == 0, "-Progress called progress")
    require(total("no_progress", "socp_evaluated") > 0, "-Progress also removed SOCP")
    require(all(row["batch_f"] == 0 for row in probes["no_curriculum"]),
            "-Curriculum still used a frontier batch")
    checks["predicate_call_totals"] = {
        arm: {key: total(arm, key) for key in
              ("progress_evaluated", "socp_evaluated", "safe_space_evaluated")}
        for arm in ARMS
    }

    selection = json.loads((STAGE5 / "temperature_probe/selection.json").read_text())
    require(selection["selected_temperature"] == .5, "wrong promoted temperature")
    require(selection["gather"]["0.5"]["accepted_windows"] >= 100, "temp .5 gather starved")
    checks["temperature_selection"] = selection

    evaluation = json.loads((STAGE5 / "evaluation/evaluation_manifest.json").read_text())
    require(evaluation["temperature_sweep"] == [.1, .5, 1.0], "incomplete final temperature sweep")
    for arm in ARMS:
        require(any((STAGE5 / f"evaluation/{arm}/temp0.5").glob("rollouts_m*.npz")),
                f"{arm}: missing matched evaluation")
    checks["evaluation"] = evaluation

    images = {}
    for name in ("rollouts_v4.png", "internals_v4.png", "scatter_v4.png"):
        path = OUT / "viz" / name
        require(path.exists() and path.stat().st_size > 10_000, f"missing/small {name}")
        with Image.open(path) as image:
            images[name] = {"size": list(image.size), "bytes": path.stat().st_size}
            require(min(image.size) >= 650, f"{name}: resolution too small")
    checks["images"] = images

    video = OUT / "viz/curriculum_it20.mp4"
    require(video.exists() and video.stat().st_size > 100_000, "missing/small curriculum video")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,nb_frames",
        "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    video_info = json.loads(probe.stdout)
    duration = float(video_info["format"]["duration"])
    require(duration >= 20.0, "curriculum video is not slow/full-length")
    checks["video"] = {"bytes": video.stat().st_size, "ffprobe": video_info}

    route = json.loads((STAGE5 / "evaluation/route_mode_audit.json").read_text())
    checks["scientific_outcome_status"] = route["status"]
    checks["scientific_outcome_alerts"] = route["alerts"]
    payload = {"status": "PASS", "scope": "mechanics and artifacts", "checks": checks}
    log_dir = OUT / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "independent_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    (log_dir / "independent_audit.log").write_text(
        "PASS mechanics/artifacts\n" +
        f"scientific outcome: {route['status']} {route['alerts']}\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
