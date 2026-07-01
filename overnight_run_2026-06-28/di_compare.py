"""Compare the BALANCED DI config at horizon H=10 vs H=15 (u_max*H*dt = 2.0 vs 3.0 = sensing; the DTCBF level sets
only bind if the H-step rollout can REACH the sensing-radius faces). 100 eps/dataset x 2 ds x gamma{0.1,0.5,1.0}.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/di_compare.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import param_oat_di as OAT

EPS100 = list(range(0, 300, 3))   # 100 eps/dataset
BALANCED = dict(centroid_gain=0.2, sigma_volume_gain=0.0, sigma_aniso=2.5, sensing=3.0, num_samples=512,
                temperature=0.1, noise=0.3, predict_gain=0.6, centroid_smooth=0.5, centroid_eps=0.15, random_backup_frac=0.2)
HORIZONS = {"H10": 10, "H15": 15}


def cfg_for(H):
    c = OAT.build_cfg(BALANCED); c["horizon"] = H; return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfgs = {nm: cfg_for(H) for nm, H in HORIZONS.items()}
    TUP = [(nm, ds, g, ep) for nm in HORIZONS for ds in OAT.DATASETS for g in OAT.GAMMAS for ep in EPS100]
    mine = TUP[args.shard::args.nshard]; acc = {nm: [0, 0, 0, 0.0] for nm in HORIZONS}
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} episode-rollouts", flush=True)
    for i, (nm, ds, g, ep) in enumerate(mine):
        su, co, a = OAT.one(ds, ep, g, cfgs[nm], args.device)
        acc[nm][0] += su; acc[nm][1] += co; acc[nm][2] += 1; acc[nm][3] += a
        if (i + 1) % 100 == 0:
            json.dump(acc, open(args.out, "w")); print(f"[shard {args.shard}] {i+1}/{len(mine)}", flush=True)
    json.dump(acc, open(args.out, "w")); print(f"[shard {args.shard}] done", flush=True)


if __name__ == "__main__":
    main()
