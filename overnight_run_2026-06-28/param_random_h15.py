"""Random search at FIXED (sensing=3, H=15) — find the best config tuned FOR the long horizon (the earlier H=15 test
reused the H=10-tuned BALANCED, which was unfair). ~50 random configs (BALANCED ± 1-5 changed params, sensing fixed=3,
H=15), compared against the recent best evaluated at (sensing=3,H=10) AND (sensing=2,H=10). 50 eps/dataset.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/param_random_h15.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import param_oat_di as OAT, param_random_di as RD

BALANCED = dict(RD.BASE)                                   # sensing=3 BALANCED config (best @ H=10)
CANDS = {k: v for k, v in RD.CANDS.items() if k != "sensing"}   # vary everything EXCEPT sensing (fixed=3)


def gen_random(n, seed=0):
    rng = random.Random(seed); keys = list(CANDS); base = dict(BALANCED, sensing=3.0)
    cfgs = []; seen = set()
    while len(cfgs) < n:
        c = dict(base)
        for k in rng.sample(keys, rng.randint(1, 5)):
            c[k] = rng.choice(CANDS[k])
        fs = frozenset(c.items())
        if fs not in seen:
            seen.add(fs); cfgs.append(c)
    return cfgs


def build_list(nconfigs, seed):
    L = [("CUR_s3_H10", dict(BALANCED, sensing=3.0), 3.0, 10),     # recent best (the real winner)
         ("BAL_s2_H10", dict(BALANCED, sensing=2.0), 2.0, 10)]      # literal "(2,10)"
    for i, c in enumerate(gen_random(nconfigs, seed)):
        L.append((f"r{i}_s3_H15", c, 3.0, 15))
    return L


def cfg_of(pd, sensing, H):
    c = dict(pd); c["sensing"] = sensing; cfg = OAT.build_cfg(c); cfg["horizon"] = H; return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--nconfigs", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=18000.0)
    args = ap.parse_args()
    allc = list(enumerate(build_list(args.nconfigs, args.seed)))
    mine = allc[args.shard::args.nshard]; res = []; t0 = time.time()
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs (sensing=3,H=15 + 2 baselines)", flush=True)
    for idx, (nm, pd, sensing, H) in mine:
        if time.time() - t0 > args.deadline:
            print(f"[shard {args.shard}] deadline; {len(res)}/{len(mine)}", flush=True); break
        cfg = cfg_of(pd, sensing, H); S = Co = n = 0; A = []
        for ds in OAT.DATASETS:
            for g in OAT.GAMMAS:
                for ep in OAT.EPS:
                    su, co, a = OAT.one(ds, ep, g, cfg, args.device); S += su; Co += co; A.append(a); n += 1
        r = dict(idx=idx, name=nm, H=H, succ=round(100 * S / n), col=round(100 * Co / n), acc=round(100 * np.mean(A)), **pd)  # pd has 'sensing'
        res.append(r); json.dump(res, open(args.out, "w"), indent=1)
        print(f"  {nm}: succ={r['succ']}% col={r['col']}% acc={r['acc']}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"[shard {args.shard}] done {len(res)}", flush=True)


if __name__ == "__main__":
    main()
