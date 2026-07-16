"""Fixed-seed gate: per-seed diff of a candidate M25 evaluation against the t104 baseline.

Gate (codex handoff): the 11 fixed failures (seed 12 origin OOB at all 7 gammas; near-goal overshoots
g.1/s22 g.4/s8 g.5/s3 g.7/s5) must flip to success AND no currently-passing seed may regress.
Reads two eval directories in the eval_ae worker format (paths_g*.npz per gamma) and reuses the
origin_window_failure_probe taxonomy so 'origin OOB' / 'near-goal OOB' mean exactly what they meant.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

ORP = importlib.import_module("origin_window_failure_probe")

GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
FIXED = [("0.1", 12), ("0.2", 12), ("0.3", 12), ("0.4", 12), ("0.5", 12), ("0.7", 12), ("1.0", 12),
         ("0.1", 22), ("0.4", 8), ("0.5", 3), ("0.7", 5)]


def load_kinds(evdir):
    kinds = {}
    for g in GAMMAS:
        f = Path(evdir) / f"paths_g{g}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        for seed, path in zip(z["seeds"], z["paths"]):
            kinds[(g, int(seed))] = ORP.path_kind(np.asarray(path, float))
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="candidate M25 eval dir (eval_ae worker format)")
    ap.add_argument("--baseline-dir", default="results/p2/eval_corrected_mode2_it104_m25")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cand = load_kinds(args.eval_dir)
    base = load_kinds(args.baseline_dir)
    missing = [g for g in GAMMAS if not any(k[0] == g for k in cand)]
    if missing:
        print(f"[WARN] candidate missing gammas: {missing}")

    fixed_status = {}
    for g, s in FIXED:
        fixed_status[f"g{g}_s{s}"] = dict(baseline=base.get((g, s)), candidate=cand.get((g, s)),
                                          flipped=cand.get((g, s)) == "success")
    n_flip = sum(v["flipped"] for v in fixed_status.values())

    regressions, improvements = [], []
    for key, bk in base.items():
        ck = cand.get(key)
        if ck is None:
            continue
        if bk == "success" and ck != "success":
            regressions.append(dict(gamma=key[0], seed=key[1], was="success", now=ck))
        if bk != "success" and ck == "success":
            improvements.append(dict(gamma=key[0], seed=key[1], was=bk, now="success"))

    per_gamma = {}
    for g in GAMMAS:
        ks = [v for k, v in cand.items() if k[0] == g]
        if ks:
            per_gamma[g] = dict(SR=float(np.mean([k == "success" for k in ks])), M=len(ks),
                                fails={f"s{k[1]}": v for k, v in cand.items()
                                       if k[0] == g and v != "success"})

    verdict = dict(fixed_flipped=f"{n_flip}/{len(FIXED)}", all_fixed_flipped=n_flip == len(FIXED),
                   regressions=regressions, n_regressions=len(regressions),
                   improvements=improvements, per_gamma_SR=per_gamma,
                   gate_pass=(n_flip == len(FIXED) and not regressions),
                   candidate=str(args.eval_dir), baseline=str(args.baseline_dir),
                   fixed_status=fixed_status)
    print(json.dumps({k: v for k, v in verdict.items() if k != "fixed_status"}, indent=1))
    for k, v in fixed_status.items():
        print(f"  {k}: {v['baseline']} -> {v['candidate']}  {'FLIP' if v['flipped'] else 'still failing'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
