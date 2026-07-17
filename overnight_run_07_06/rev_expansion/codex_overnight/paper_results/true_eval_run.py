"""TRUE evaluation runner (user 2026-07-16c): bare-policy random rollouts, M=100 per gamma, for
EVERY AFE2 checkpoint (round 0 = purely pretrained .. round 10). No verifier, no NVP, no fallback,
no curation: each rollout terminates only on actual collision/OOB, goal reach, or the T cap.
Saves raw paths per (round, gamma); all metrics are computed later from the raw paths by
true_eval_fig.py with one uniform code path (SR, CR, min obstacle clearance, time-to-success,
trajectory validity2 = taskspace AND approach AND SOCP at that gamma).

Usage (split across processes):
  python paper_results/true_eval_run.py --ckpt-dir results/afe2/afe_s910 \
      --rounds 0 1 2 3 --outdir results/true_eval/afe
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch

import _paths  # noqa: F401
import grid_scene as GS
import grid_rollout as GR
import grid_hp_expt as HP
import grid_expand_hardtail as HT
import grid_metrics2 as GM2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="results/afe2/afe_s910")
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--outdir", default="results/true_eval/afe")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = HT._apply_wall_plugs(GS.make_grid(), 8)
    env.x0 = torch.tensor([0.3, 0.3, 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor([4.7, 4.7], dtype=env.goal.dtype)
    GM2.GOAL_XY = np.array([4.7, 4.7], dtype=float)
    os.makedirs(args.outdir, exist_ok=True)
    for n in args.rounds:
        ck = os.path.join(args.ckpt_dir, f"ckpt_{n}.pt")
        if not os.path.exists(ck):
            ck = os.path.join(args.ckpt_dir, "final.pt")
        pol, _ = HP.load_hp(ck, device=dev)
        for gi, g in enumerate(args.gammas):
            fout = os.path.join(args.outdir, f"paths_r{n}_g{g}.npz")
            if os.path.exists(fout):
                print(f"skip r{n} g{g} (exists)", flush=True)
                continue
            paths, reached = [], []
            for m in range(args.M):
                torch.manual_seed(7_000_000 + n * 10_000 + gi * 1_000 + m)   # true random, reproducible
                out = GR.fm_deploy(pol, env, float(g), T=args.T, temp=1.0, nfe=8,
                                   reach=args.reach, device=dev)
                paths.append(np.asarray(out["path"], np.float32))
                reached.append(bool(out["reached"]))
            pa = np.empty(len(paths), dtype=object)
            for i, p in enumerate(paths):
                pa[i] = p
            np.savez_compressed(fout, paths=pa, gamma=float(g), round=int(n),
                                reached=np.asarray(reached, bool))
            print(f"r{n} g{g}: SR(raw reach flag) {np.mean(reached):.2f} -> {fout}", flush=True)
    with open(os.path.join(args.outdir, f"DONE_{'_'.join(map(str, args.rounds))}"), "w") as f:
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()
