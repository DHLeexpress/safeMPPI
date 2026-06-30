"""Fine-tune the mean/variance-control parameters on BOTH UCY+SDD (20-ep spread each) for gamma {0.1,0.5,1.0}.
Grid: centroid_gain x sigma_volume_gain x control_weight x centroid_horizon(K). Fixed: nominal=0 + warm-start,
margin=0, polytope barrier, noise=0.5, predict_gain=0.4, sensing=3.0, temp=0.3, H=10. Sharded across GPUs, budgeted.
Metric per config = mean over (UCY+SDD) x (gamma) x (episodes): success / near / collision / acceptance.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/param_finetune.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from polytope_explainer import rollout
from cfm_mppi.mppi.sweep import _load, DT

GRID = list(itertools.product([0.05, 0.1, 0.15], [0.5, 1.0, 1.5], [0.03, 0.15, 0.4], [2, 3]))  # cg, sv, cw, K
EPISODES = list(range(0, 300, 15))   # 20-episode spread per dataset
GAMMAS = [0.1, 0.5, 1.0]; DATASETS = ["ucy", "sdd"]
BASE = dict(horizon=10, dt=DT, num_samples=128, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
            safety_margin=0.0, temperature=0.3, dynamics_type="singleintegrator", barrier_activation_radius=3.0,
            use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, predict_gain=0.4, polytope_nbase=16)


def one(ds, ep, g, cg, sv, cw, K, dev):
    cfg = dict(BASE, centroid_gain=cg, sigma_volume_gain=sv, control_weight=cw, centroid_horizon=K)
    rec, path, goal = rollout(ds, ep, g, cfg, dev=dev); _, _, obs, _ = _load(ds, ep, 80)
    acc = float(np.mean([st["n_acc"] / (st["n_acc"] + st["n_rej"] + 1e-9) for st in rec if st.get("poly") is not None]))
    mc = np.inf
    for t in range(min(len(path), 80)):
        ob = obs[min(t, obs.shape[0] - 1)]; ok = ~np.isnan(ob[:, :2]).any(1); o = ob[ok]
        if o.shape[0]:
            mc = min(mc, float(np.min(np.linalg.norm(o[:, :2] - path[t], axis=1) - o[:, 2] - 0.2)))
    fd = float(np.linalg.norm(path[-1] - goal))
    return int(fd < 0.6 and mc >= 0), int(fd < 1.5 and mc >= 0), int(mc < 0), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--deadline", type=float, default=1620.0)
    args = ap.parse_args(); mine = GRID[args.shard::args.nshard]; res = []; t0 = time.time()
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs", flush=True)
    for cg, sv, cw, K in mine:
        if time.time() - t0 > args.deadline:
            print(f"[shard {args.shard}] deadline; did {len(res)}/{len(mine)} configs", flush=True); break
        S = N = NE = C = 0; accs = []; cells = {}
        for ds in DATASETS:
            for g in GAMMAS:
                s = ne = c = 0; ac = []
                for ep in EPISODES:
                    su, nr, co, a = one(ds, ep, g, cg, sv, cw, K, args.device)
                    s += su; ne += nr; c += co; ac.append(a); S += su; NE += nr; C += co; accs.append(a); N += 1
                cells[f"{ds}_g{g}"] = dict(succ=round(100 * s / len(EPISODES)), col=round(100 * c / len(EPISODES)), acc=round(100 * np.mean(ac)))
        res.append(dict(cg=cg, sv=sv, cw=cw, K=K, succ=round(100 * S / N), near=round(100 * NE / N), col=round(100 * C / N), acc=round(100 * np.mean(accs)), cells=cells))
        r = res[-1]; print(f"  cg={cg} sv={sv} cw={cw} K={K}: succ={r['succ']}% near={r['near']}% col={r['col']}% acc={r['acc']}%", flush=True)
    json.dump(res, open(args.out, "w"), indent=2); print(f"[shard {args.shard}] saved {len(res)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
