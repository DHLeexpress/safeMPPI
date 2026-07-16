"""Fast PAIRED a-d eval for the greedy hill-climb: load a checkpoint once, deploy the SAME fixed-seed
episodes on gammas {0.1,0.5,1.0}, return pooled (SR, CR, clearance, time, coverage) as one JSON line.

Fixed seed0 => every checkpoint is scored on IDENTICAL initial conditions, so "strictly improves" is a
paired comparison (low variance), not an independent-sample race. Faithful deploy (temp=1, NFE 8),
reach 0.1 — the same metric code (summarize_paths) used by eval_ae and every paper row.

  python analysis/greedy_eval.py --ckpt <ckpt> --M 8            # -> {"SR":..,"CR":..,"clr":..,"time":..,"cov":..}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]

import grid_hp_expt as HP          # noqa: E402
import grid_scene as GS            # noqa: E402
import sr_cr_eval as SR            # noqa: E402
from eval_ae import summarize_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--reach", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pol, _ = HP.load_hp(args.ckpt, device=args.device)
    pol.eval()
    env = GS.make_grid()
    rows = []
    for g in args.gammas:
        _, _, paths_by_g = SR.eval_policy(pol, env, gammas=[float(g)], M=args.M, T_max=args.T,
                                          reach=args.reach, temp=1.0, device=args.device,
                                          seed0=args.seed0, keep_paths=args.M, log=lambda *a, **k: None)
        paths = paths_by_g[float(g)]
        rows.append(summarize_paths(paths, env, g, "greedy", args.reach))

    # pool across gammas: SR/CR/time averaged; clearance averaged over gammas that HAD successes;
    # coverage summed (distinct modes discovered across gammas). NaNs (no success) drop from clearance.
    sr = float(np.mean([r["SR"] for r in rows]))
    cr = float(np.mean([r["CR"] for r in rows]))
    clr_vals = [r["clearance_mean"] for r in rows if r.get("clearance_mean") is not None
                and np.isfinite(r["clearance_mean"])]
    tim_vals = [r["time_mean_s"] for r in rows if r.get("time_mean_s") is not None
                and np.isfinite(r["time_mean_s"])]
    clr = float(np.mean(clr_vals)) if clr_vals else float("nan")
    tim = float(np.mean(tim_vals)) if tim_vals else float("nan")
    cov = int(sum(r["coverage"] for r in rows))
    out = dict(SR=sr, CR=cr, clr=clr, time=tim, cov=cov,
               per_gamma={str(g): dict(SR=r["SR"], CR=r["CR"], clr=r["clearance_mean"],
                                       time=r["time_mean_s"], cov=r["coverage"])
                          for g, r in zip(args.gammas, rows)})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
