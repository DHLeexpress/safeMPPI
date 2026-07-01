"""Orchestrator for the Safe-Flow-Expansion smoke test.

Chains the pipeline stages (each stage persists artifacts to results/ so stages are independent):
    env  ->  data  ->  pretrain  ->  expand  ->  video

Usage:
    python run_smoke.py --stage all --smoke            # fast end-to-end (CPU, minutes)
    python run_smoke.py --stage all                    # full run
    python run_smoke.py --stage expand --gammas 0.3 0.5 0.7
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["env", "data", "pretrain", "expand", "video"]


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *[str(a) for a in args]]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--n-paths", type=int, default=None)
    ap.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    ap.add_argument("--wandb-project", default="cfm-mppi-safeflow")
    args = ap.parse_args()

    smoke = ["--smoke"] if args.smoke else []
    wb = ["--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project]
    todo = STAGES if args.stage == "all" else [args.stage]

    if "env" in todo:
        run("env.py")
    if "data" in todo:
        run("build_dataset.py", *smoke)
    if "pretrain" in todo:
        run("pretrain.py", *smoke, "--device", args.device, *wb)
    if "expand" in todo:
        run("expand.py", *smoke, "--device", args.device, "--gammas", *args.gammas, *wb)
    if "video" in todo:
        npaths = args.n_paths if args.n_paths is not None else (120 if args.smoke else 300)
        run("video.py", "--device", args.device, "--n-paths", npaths)

    print("\n=== pipeline complete — see figures/ and results/ ===", flush=True)


if __name__ == "__main__":
    main()
