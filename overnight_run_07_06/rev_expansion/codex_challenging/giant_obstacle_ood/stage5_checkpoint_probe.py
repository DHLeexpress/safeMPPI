#!/usr/bin/env python3
"""Persist matched temperature/NFE probes for one Stage-5 checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import grid_hp_expt as HP  # noqa: E402
from giant_obstacle_ood.stage4_frozen_ood import (  # noqa: E402
    rollout_policy,
    save_records,
    summarize_method,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=92500)
    parser.add_argument("--persistent-route-bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-latent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent-correlation", type=float, default=0.0)
    parser.add_argument("--ensemble-size", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.nfe <= 0 or args.steps <= 0:
        parser.error("repetitions, nfe, and steps must be positive")
    if not 0.0 <= args.latent_correlation <= 1.0:
        parser.error("--latent-correlation must be in [0,1]")
    if args.ensemble_size <= 0:
        parser.error("--ensemble-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, payload = HP.load_hp(args.checkpoint.resolve(), device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for temperature in args.temperatures:
        records = rollout_policy(
            policy,
            repetitions=args.repetitions,
            temperature=float(temperature),
            nfe=args.nfe,
            T=args.steps,
            seed0=args.seed,
            device=device,
            method=f"checkpoint it{int(payload.get('iter', 0))} T={temperature:g}",
            persistent_route_bit=bool(args.persistent_route_bit),
            persistent_latent=bool(args.persistent_latent),
            latent_correlation=float(args.latent_correlation),
            ensemble_size=int(args.ensemble_size),
        )
        key = f"{float(temperature):g}"
        summaries[key] = summarize_method(records)
        save_records(
            records,
            args.output / f"rollouts_temp{key}_m{args.repetitions}_nfe{args.nfe}_T{args.steps}.npz",
            checkpoint=np.asarray(str(args.checkpoint.resolve())),
            iteration=np.asarray(int(payload.get("iter", 0))),
            temperature=np.asarray(float(temperature)),
            matched_seed0=np.asarray(args.seed),
            nfe=np.asarray(args.nfe),
            max_steps=np.asarray(args.steps),
            persistent_route_bit=np.asarray(bool(args.persistent_route_bit)),
            persistent_latent=np.asarray(bool(args.persistent_latent)),
            latent_correlation=np.asarray(float(args.latent_correlation)),
            ensemble_size=np.asarray(int(args.ensemble_size)),
        )
    audit = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": int(payload.get("iter", 0)),
        "temperatures": [float(value) for value in args.temperatures],
        "M_per_gamma": args.repetitions,
        "nfe": args.nfe,
        "max_steps": args.steps,
        "matched_seed0": args.seed,
        "persistent_route_bit": bool(args.persistent_route_bit),
        "persistent_latent": bool(args.persistent_latent),
        "latent_correlation": float(args.latent_correlation),
        "ensemble_size": int(args.ensemble_size),
        "summaries": summaries,
    }
    (args.output / "metrics.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
