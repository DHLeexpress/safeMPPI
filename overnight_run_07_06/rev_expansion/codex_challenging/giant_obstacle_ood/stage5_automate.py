#!/usr/bin/env python3
"""Resumable Stage-5 orchestration: selected temperature plus four matched arms."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE = HERE / "stage_results/05_window_expand"
DRIVER = HERE / "stage5_window_expand.py"
RUNS = STAGE / "runs/temp0.5_stable"
LOGS = STAGE / "logs"
TEMPS = (0.1, 0.5, 1.0)


def env_gpu2() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    prefix = "/home/dohyun/miniforge3/lib:/usr/local/cuda/compat"
    env["LD_LIBRARY_PATH"] = prefix + (":" + env["LD_LIBRARY_PATH"]
                                          if env.get("LD_LIBRARY_PATH") else "")
    return env


def complete(path: Path, iters: int) -> bool:
    if not (path / "final.pt").exists() or not (path / "run_manifest.json").exists():
        return False
    try:
        payload = json.loads((path / "run_manifest.json").read_text())
        return payload.get("status") == "PASS" and int(payload.get("iters", -1)) == int(iters)
    except Exception:
        return False


def command(arm: str, outdir: Path, iters: int, extra: tuple[str, ...] = ()) -> list[str]:
    return [
        sys.executable, str(DRIVER), "run", "--arm", arm,
        "--outdir", str(outdir), "--temperature", "0.5", "--iters", str(iters),
        *extra,
    ]


def launch(name: str, cmd: list[str], env: dict[str, str]):
    log_path = LOGS / f"stable_{name}.log"
    handle = log_path.open("w")
    process = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
    )
    print(f"[{name}] pid={process.pid} log={log_path.relative_to(ROOT)}", flush=True)
    return process, handle, log_path


def wait_jobs(jobs: dict[str, tuple], poll_seconds: float = 10.0) -> None:
    pending = set(jobs)
    while pending:
        time.sleep(poll_seconds)
        status = []
        for name in sorted(pending):
            process, _handle, _path = jobs[name]
            code = process.poll()
            status.append(f"{name}:{'running' if code is None else 'exit='+str(code)}")
            if code is not None:
                pending.remove(name)
        print("[stage5] " + " | ".join(status), flush=True)
    failures = []
    for name, (process, handle, path) in jobs.items():
        handle.close()
        if process.returncode:
            tail = "\n".join(path.read_text().splitlines()[-40:])
            failures.append(f"{name} exit={process.returncode}\n{tail}")
    if failures:
        raise RuntimeError("\n\n".join(failures))


def write_temperature_selection() -> dict:
    rollout = json.loads((STAGE / "temperature_probe/rollout_summary.json").read_text())
    gather = {}
    for temperature in TEMPS:
        candidates = (
            STAGE / f"temperature_probe/gather_t{temperature:g}",
            STAGE / f"temperature_probe/gather_t{temperature}",
        )
        root = next((path for path in candidates if path.exists()), candidates[0])
        budget = json.loads((root / "accepted_window_budget.json").read_text())
        probes = [json.loads(line) for line in (root / "probe.jsonl").read_text().splitlines() if line]
        gather[str(temperature)] = {
            "accepted_windows": int(sum(int(value) for value in budget.values())),
            "classes_ready": bool(probes[-1]["gather_audit"]["classes_ready"]),
            "gamma_ready": bool(probes[-1]["gather_audit"]["gamma_ready"]),
        }
    action_delta = {
        key: value["overall"]["mean_control_delta"]
        for key, value in rollout["summaries"].items()
    }
    selected = 0.5
    if not (gather["0.5"]["classes_ready"] and gather["0.5"]["gamma_ready"]
            and gather["0.5"]["accepted_windows"] >= 100):
        selected = 1.0
    payload = {
        "status": "PASS",
        "selected_temperature": selected,
        "selection_rule": "prefer 0.5 over 1.0 when it supplies >=100 all-gamma two-class valid2 windows",
        "temperature_0.1_status": "excluded: user-observed and measured over-smoothing",
        "mean_control_delta": action_delta,
        "gather": gather,
    }
    (STAGE / "temperature_probe/selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    if selected != 0.5:
        raise RuntimeError(f"probe selected {selected}; this orchestrator path is explicitly temp0.5")
    return payload


def health_audit(iters: int) -> dict:
    arms = ("full", "no_socp", "no_progress", "no_curriculum")
    audit = {"status": "PASS", "arms": {}}
    budgets = {}
    for arm in arms:
        root = RUNS / arm
        probes = [json.loads(line) for line in (root / "probe.jsonl").read_text().splitlines() if line]
        budget = {int(k): int(v) for k, v in json.loads(
            (root / "accepted_window_budget.json").read_text()).items()}
        budgets[arm] = budget
        if len(probes) != iters or set(budget) != set(range(1, iters + 1)):
            raise RuntimeError(f"{arm}: incomplete probe/budget history")
        if not all(row.get("functional_step", 0.0) > 0.0 for row in probes):
            raise RuntimeError(f"{arm}: one or more iterations did not update")
        if any(bool(row.get("rollback", False)) for row in probes):
            raise RuntimeError(f"{arm}: unstable update rollback detected")
        if not all(row["gather_audit"].get("ready", False) for row in probes):
            raise RuntimeError(f"{arm}: faithful window gather starved")
        audit["arms"][arm] = {
            "accepted_total": sum(budget.values()),
            "functional_step_max": max(row["functional_step"] for row in probes),
            "rollbacks": sum(bool(row.get("rollback", False)) for row in probes),
            "demo_requested": [row["demo_req"] for row in probes],
            "socp_evaluated": sum(row["gather_audit"].get("socp_evaluated", 0) for row in probes),
            "progress_evaluated": sum(row["gather_audit"].get("progress_evaluated", 0) for row in probes),
        }
    if budgets["full"] != budgets["no_curriculum"]:
        raise RuntimeError("-Curriculum accepted-window counts do not exactly match Full")
    if audit["arms"]["no_socp"]["socp_evaluated"] != 0:
        raise RuntimeError("-SOCP called SOCP")
    if audit["arms"]["no_progress"]["progress_evaluated"] != 0:
        raise RuntimeError("-Progress called progress")
    audit["no_curriculum_exact_window_count_match"] = True
    (STAGE / "logs/health_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.iters < 1:
        raise ValueError("--iters must be positive")
    LOGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    selection = write_temperature_selection()
    print(f"[temperature] selected {selection['selected_temperature']}", flush=True)
    env = env_gpu2()
    started = time.perf_counter()

    first = {}
    for arm in ("full", "no_socp", "no_progress"):
        path = RUNS / arm
        if complete(path, args.iters):
            print(f"[{arm}] already complete; skipping", flush=True)
            continue
        extra = (("--overwrite",) if path.exists() and any(path.iterdir()) else ())
        first[arm] = launch(arm, command(arm, path, args.iters, extra), env)
    if first:
        wait_jobs(first)

    full_budget = RUNS / "full/accepted_window_budget.json"
    no_curr = RUNS / "no_curriculum"
    if not complete(no_curr, args.iters):
        extra = ["--budget-path", str(full_budget)]
        if no_curr.exists() and any(no_curr.iterdir()):
            extra.append("--overwrite")
        wait_jobs({
            "no_curriculum": launch(
                "no_curriculum", command("no_curriculum", no_curr, args.iters, tuple(extra)), env
            )
        })
    else:
        print("[no_curriculum] already complete; skipping", flush=True)

    audit = health_audit(args.iters)
    manifest = {
        "status": "PASS",
        "physical_gpu": 2,
        "temperature": 0.5,
        "iters": args.iters,
        "arms": {arm: str((RUNS / arm).resolve()) for arm in
                 ("full", "no_socp", "no_progress", "no_curriculum")},
        "health_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (STAGE / "automation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS -> {STAGE / 'automation_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
