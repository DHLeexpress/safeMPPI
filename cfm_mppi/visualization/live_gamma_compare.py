from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

from cfm_mppi.evaluation.eval_benchmark import DEFAULTS, _set_seed
from cfm_mppi.visualization.gamma_sweep_data import build_gamma_grid, rollout_mizuta, rollout_safemppi
from cfm_mppi.visualization.gamma_sweep_render import render_animation
from cfm_mppi.visualization.gamma_sweep_summary import summarize_records, write_summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _set_seed(args.seed)
    args.u_min = tuple(float(x) for x in args.u_min)
    args.u_max = tuple(float(x) for x in args.u_max)
    gammas = build_gamma_grid(args.gamma_values, args.gamma_count)
    if args.smoke:
        args.horizon = min(args.horizon, 20)
        args.num_episodes = min(args.num_episodes, 2)
        args.safemppi_num_samples = min(args.safemppi_num_samples, 128)
        args.no_video = True
    root = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S") / args.dataset / args.dynamics
    root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    jsonl = root / "gamma_sweep_records.jsonl"
    with jsonl.open("w", encoding="utf-8", buffering=1) as f:
        for ep in range(args.num_episodes):
            rec = rollout_mizuta(args, ep)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            print(f"mizuta ep={ep+1}/{args.num_episodes} success={int(rec['success'])} collision={int(rec['collision'])} min_clearance={rec['min_clearance']:.3f}", flush=True)
        for gamma in gammas:
            for ep in range(args.num_episodes):
                rec = rollout_safemppi(args, ep, gamma)
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
                print(f"safeMPPI gamma={gamma:.3f} ep={ep+1}/{args.num_episodes} success={int(rec['success'])} collision={int(rec['collision'])} min_clearance={rec['min_clearance']:.3f} plan_ms={1000*rec['planning_wall_time_mean']:.2f}", flush=True)
    summary = summarize_records(records, gammas)
    write_summary(root, summary, gammas)
    artifacts = {
        "output_dir": str(root),
        "records_jsonl": str(jsonl),
        "summary_json": str(root / "summary.json"),
        "summary_csv": str(root / "summary.csv"),
        "summary_md": str(root / "summary.md"),
    }
    artifacts.update(render_animation(root, records, summary, gammas, args))
    with (root / "artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)
    print(json.dumps(artifacts, indent=2), flush=True)
    return artifacts


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Three-panel live video comparing Mizuta CFM-MPPI and safeMPPI with gamma in [0,1].")
    p.add_argument("--dataset", default="sfm", choices=["sfm", "ucy", "sdd"])
    p.add_argument("--dynamics", default="doubleintegrator", choices=["doubleintegrator", "unicycle"])
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--video-episode", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/visualization/gamma_sweep")
    p.add_argument("--gamma-count", type=int, default=21)
    p.add_argument("--gamma-values", nargs="*", type=float, default=None)
    p.add_argument("--horizon", type=int, default=DEFAULTS["horizon"])
    p.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    p.add_argument("--safety-margin", type=float, default=DEFAULTS["safety_margin"])
    p.add_argument("--success-threshold", type=float, default=DEFAULTS["success_threshold"])
    p.add_argument("--u-min", nargs=2, type=float, default=list(DEFAULTS["u_min"]))
    p.add_argument("--u-max", nargs=2, type=float, default=list(DEFAULTS["u_max"]))
    p.add_argument("--safemppi-num-samples", type=int, default=1024)
    p.add_argument("--safemppi-horizon", type=int, default=20)
    p.add_argument("--safemppi-noise-sigma", type=float, default=0.6)
    p.add_argument("--safemppi-temperature", type=float, default=1.0)
    p.add_argument("--check-first-control-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mizuta-checkpoint", default="output_dir/cfm_transformer/checkpoint.pth")
    p.add_argument("--safe-cfm-checkpoint", default="output_dir/safe_contextual_cfm/checkpoint_best.pth")
    p.add_argument("--drifting-checkpoint", default="output_dir/drifting_generator/checkpoint_best.pth")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--show-live", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
