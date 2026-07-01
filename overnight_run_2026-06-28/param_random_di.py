"""DOUBLE-INTEGRATOR LOCAL random search AROUND the current-best config (explores parameter INTERACTIONS that the
one-at-a-time sweep can't see). Each config = the best config with 1-5 randomly changed params. Full eval
(50 eps/dataset x gamma{0.1,0.5,1.0}, gamma=0.1 extra steps). Incremental save (deadline-safe). Sharded over 4 GPUs.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/param_random_di.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import param_oat_di as OAT   # reuse one(), build_cfg, EPS, GAMMAS, DATASETS

# current-best config (OAT combined-best: 88% succ / 12% col / 60% acc)
BASE = dict(centroid_gain=0.3, sigma_volume_gain=1.0, sigma_aniso=2.5, sensing=3.0, num_samples=512,
            temperature=0.1, noise=0.5, predict_gain=0.6, centroid_smooth=0.5, centroid_eps=0.15, random_backup_frac=0.0)
CANDS = {
    "centroid_gain": [0.05, 0.1, 0.2, 0.3], "sigma_volume_gain": [0.0, 0.5, 1.0, 1.5],
    "sigma_aniso": [1.0, 2.0, 2.5, 3.0], "sensing": [2.0, 2.5, 3.0, 3.5], "num_samples": [256, 512],
    "temperature": [0.1, 0.3, 0.5, 1.0], "noise": [0.3, 0.5, 0.7], "predict_gain": [0.0, 0.2, 0.4, 0.6],
    "centroid_smooth": [0.0, 0.25, 0.5, 0.75], "centroid_eps": [0.05, 0.15, 0.3, 0.5],
    "random_backup_frac": [0.0, 0.05, 0.1, 0.2],
}


def gen_configs(n, seed=0):
    rng = random.Random(seed); keys = list(CANDS); cfgs = [dict(BASE)]; seen = {frozenset(BASE.items())}
    while len(cfgs) < n:
        c = dict(BASE)
        for k in rng.sample(keys, rng.randint(1, 5)):   # change 1-5 params -> "around the best"
            c[k] = rng.choice(CANDS[k])
        fs = frozenset(c.items())
        if fs not in seen:
            seen.add(fs); cfgs.append(c)
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--nconfigs", type=int, default=160); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=17000.0)
    args = ap.parse_args()
    allc = list(enumerate(gen_configs(args.nconfigs, args.seed)))
    mine = allc[args.shard::args.nshard]; res = []; t0 = time.time()
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs", flush=True)
    for idx, c in mine:
        if time.time() - t0 > args.deadline:
            print(f"[shard {args.shard}] deadline; {len(res)}/{len(mine)}", flush=True); break
        S = Co = n = 0; A = []
        for ds in OAT.DATASETS:
            for g in OAT.GAMMAS:
                for ep in OAT.EPS:
                    su, co, a = OAT.one(ds, ep, g, OAT.build_cfg(c), args.device); S += su; Co += co; A.append(a); n += 1
        r = dict(idx=idx, succ=round(100 * S / n), col=round(100 * Co / n), acc=round(100 * np.mean(A)), **c)
        res.append(r); json.dump(res, open(args.out, "w"), indent=1)   # incremental save (deadline-safe)
        print(f"  cfg{idx}: succ={r['succ']}% col={r['col']}% acc={r['acc']}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"[shard {args.shard}] done {len(res)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
