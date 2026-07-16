"""Launch the matched Full / three-No window-native expansion sanity run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "reference" / "window_expand_hardtail.py"
DEFAULT_CKPT = HERE / "stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt"
DEFAULT_OUT = HERE / "stage_results/05_window_native/runs"


def command(checkpoint: Path, outdir: Path, iters: int, tag: str, extra=()):
    return [
        sys.executable, str(TRAINER),
        "--ckpt", str(checkpoint), "--outdir", str(outdir),
        "--iters", str(iters), "--m-measure", "2", "--measure-every", str(iters),
        "--seed", "6010", "--no-freeze", "--enc-lr-mult", ".3",
        "--beta", ".2", "--lr", "5e-6",
        "--rollouts-per-iter", "14", "--gather-attempt-cap", "28", "--batch", "16",
        "--gp-buf", "200", "--qbuf-cap", "200", "--valid-prog-floor", ".15",
        "--mix-start", ".4", ".6", "--mix-end", ".4", ".6",
        "--quantile-schedule", "0:0.30",
        "--early-inner", "2", "--inner-steps", "2", "--cooldown-inner", "2",
        "--targeted-frac", "0", "--min-modes-per-gamma", "0",
        "--recovery-frac", "0", "--hard-quota", "0", "--guard-quota", "0",
        "--strip-probe-every", "0", "--wall-plugs", "8", "--start-eps", ".3",
        "--goal-xy", "4.7", "4.7", "--reach", ".15",
        "--legacy-prime-iters", "0", "--viz-db-every", "1",
        "--ckpt-every", str(iters), "--log-comp-every", "1", "--tag", tag,
        *map(str, extra),
    ]


def launch(name, cmd, logs, env):
    log_path = logs / f"{name}.log"
    handle = log_path.open("w")
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(f"[{name}] pid={proc.pid} log={log_path.relative_to(HERE)}", flush=True)
    return proc, handle, log_path


def wait_all(jobs):
    failures = []
    for name, (proc, handle, log_path) in jobs.items():
        code = proc.wait()
        handle.close()
        print(f"[{name}] exit={code}", flush=True)
        if code:
            failures.append((name, code, log_path))
    if failures:
        summary = ", ".join(f"{n}={c} ({p})" for n, c, p in failures)
        raise RuntimeError(f"window-native arm failure(s): {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iters", type=int, default=6)
    args = parser.parse_args()
    if args.iters < 1:
        raise ValueError("--iters must be positive")

    out_root = args.out_root.resolve()
    logs = out_root.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    arm_dirs = {name: out_root / f"sanity_v1_{name}" for name in
                ("full", "no_socp", "no_progress", "no_curriculum")}
    occupied = [p for p in arm_dirs.values() if p.exists()]
    if occupied:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, occupied)))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    lib = "/home/dohyun/miniforge3/lib:/usr/local/cuda/compat"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"]
                                          if env.get("LD_LIBRARY_PATH") else "")

    started = time.time()
    first = {
        "full": command(args.checkpoint, arm_dirs["full"], args.iters, "window_native_full"),
        "no_socp": command(args.checkpoint, arm_dirs["no_socp"], args.iters,
                           "window_native_no_socp", ("--ablate-socp",)),
        "no_progress": command(args.checkpoint, arm_dirs["no_progress"], args.iters,
                               "window_native_no_progress", ("--ablate-progress",)),
    }
    jobs = {name: launch(name, cmd, logs, env) for name, cmd in first.items()}
    wait_all(jobs)

    budget_path = arm_dirs["full"] / "accepted_window_budget.json"
    budget = {int(k): int(v) for k, v in json.loads(budget_path.read_text()).items()}
    expected = set(range(1, args.iters + 1))
    if set(budget) != expected:
        raise RuntimeError(f"Full window budget iterations {sorted(budget)} != {sorted(expected)}")
    no_curr_cmd = command(
        args.checkpoint, arm_dirs["no_curriculum"], args.iters,
        "window_native_no_curriculum",
        ("--ablate-curriculum", "--accepted-window-budget-file", budget_path),
    )
    wait_all({"no_curriculum": launch("no_curriculum", no_curr_cmd, logs, env)})

    matched = {int(k): int(v) for k, v in json.loads(
        (arm_dirs["no_curriculum"] / "accepted_window_budget.json").read_text()).items()}
    if matched != budget:
        raise RuntimeError(f"-Curriculum window budget mismatch: full={budget}, no_curr={matched}")

    manifest = {
        "status": "PASS",
        "trainer": str(TRAINER),
        "checkpoint": str(args.checkpoint.resolve()),
        "physical_gpu": 2,
        "iters": args.iters,
        "arms": {k: str(v) for k, v in arm_dirs.items()},
        "full_window_budget": budget,
        "no_curriculum_exact_count_match": True,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = out_root.parent / "sanity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS -> {manifest_path.relative_to(HERE)}", flush=True)


if __name__ == "__main__":
    main()
