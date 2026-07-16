#!/usr/bin/env python3
"""One declared heuristic-gamma tuning sweep on training-only seeds 100+."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

import adaptive_gamma_eval as AGE
import eval_ae as EVAL
import grid_scene as GS
import grid_expand_hardtail as HT


PAIRS = ((.2, .8), (.2, 1.0), (.3, .8), (.3, 1.0), (.4, 1.0), (.4, 1.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--M", type=int, default=25)
    ap.add_argument("--seed0", type=int, default=100,
                    help="must stay outside fixed evaluation seeds 0--99")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.seed0 < 100:
        raise ValueError("heuristic tuning must use training-only seeds >=100")
    policy, _ = HT.HP.load_hp(args.ckpt, device=args.device)
    env = GS.make_grid(); HT._apply_wall_plugs(env, 4)
    results = []
    for d_lo, d_hi in PAIRS:
        paths = []
        for i in range(args.M):
            out = AGE.deploy(policy, env, "heuristic", args.seed0 + i, d_lo=d_lo, d_hi=d_hi)
            paths.append(out["path"])
        row = EVAL.summarize_paths(paths, env, float("nan"), f"heuristic d=({d_lo},{d_hi})")
        row.update(d_lo=d_lo, d_hi=d_hi)
        results.append(row)
        print(f"d=({d_lo:.1f},{d_hi:.1f}) SR={row['SR']:.3f} CR={row['CR']:.3f} "
              f"clearance={row['clearance_mean']:.3f} time={row['time_mean_s']:.2f}", flush=True)
    # Reliability is lexicographically primary; among equal CR/SR, prefer safety,
    # then speed. This objective is declared before looking at evaluation seeds.
    def rank(row):
        clearance = row["clearance_mean"] if np.isfinite(row["clearance_mean"]) else -np.inf
        time = row["time_mean_s"] if np.isfinite(row["time_mean_s"]) else np.inf
        return (row["CR"], -row["SR"], -clearance, time)
    selected = min(results, key=rank)
    artifact = dict(protocol=dict(seed0=args.seed0, M=args.M, seeds="training-only",
                                  pairs=[list(x) for x in PAIRS],
                                  selection="min CR, max SR, max clearance, min time (lexicographic)"),
                    checkpoint=str(Path(args.ckpt).resolve()), results=results,
                    selected=dict(d_lo=selected["d_lo"], d_hi=selected["d_hi"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, allow_nan=True) + "\n")
    print("selected", artifact["selected"], "->", args.out)


if __name__ == "__main__":
    main()
