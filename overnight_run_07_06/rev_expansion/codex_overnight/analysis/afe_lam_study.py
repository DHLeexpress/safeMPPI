"""Pick lambda for a LIVE-sigma arm, from measured data (no assumptions).

The round-k viz db stores A_inv of the cumulative A = I + (1/lam0) * S (lam0 = 1e-2).  Recover
S = lam0 * (A - I), rebuild A' = I + (1/lam') S for candidate lambdas, and report the sigma profile
of FRESH policy queries (drawn-query features of a later round) under each lambda' -- i.e. what the
acquisition signal WOULD have been.  Choose the lambda whose sigma still discriminates at the run's
mid/late query volume.
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/afe/A_s910")
    ap.add_argument("--lams", nargs="+", type=float, default=[0.01, 1.0, 10.0, 100.0, 1000.0])
    args = ap.parse_args()
    dbs = sorted(glob.glob(os.path.join(args.run, "viz_db", "round*.pt")),
                 key=lambda p: int(re.findall(r"round(\d+)\.pt", p)[0]))
    lam0 = 0.01
    for p in [dbs[0], dbs[len(dbs) // 2], dbs[-1]]:
        db = torch.load(p, map_location="cpu", weights_only=False)
        A_inv = db["A_inv"].to(torch.float64)
        A = torch.linalg.inv(A_inv)
        S = lam0 * (A - torch.eye(A.shape[0], dtype=torch.float64))
        n = int(db["blr_n"])
        # eigen-spectrum of S tells the query-mass distribution over feature directions
        ev = torch.linalg.eigvalsh(S).clamp_min(0)
        print(f"\n{os.path.basename(p)}: n={n} queries | S eigmass p50={ev.median():.1f} "
              f"max={ev.max():.1f} min={ev.min():.3f}")
        # sigma of a hypothetical fresh unit feature along each eigendirection, per lambda
        for lam in args.lams:
            sig_dir = (lam / (lam + ev)).sqrt()      # sigma along each eigendirection
            print(f"  lam={lam:7.2f}: sigma p10={sig_dir.quantile(0.1):.3f} "
                  f"p50={sig_dir.median():.3f} p90={sig_dir.quantile(0.9):.3f} "
                  f"(spread p90-p10 = {(sig_dir.quantile(0.9) - sig_dir.quantile(0.1)):.3f})")


if __name__ == "__main__":
    main()
