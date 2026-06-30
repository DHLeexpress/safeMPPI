"""Fine-tune the mean/cov-shifting params at 50 episodes/config (sharded across GPUs).
Grid: centroid_gain x centroid_horizon(K) x sigma_volume_gain x noise. Fixed: nominal=0 + warm-start, margin=0,
polytope barrier, predict_gain=0.4, sensing=3.0, temperature=0.3, decay=0.7, eps=0.15, H=10, γ=0.5.
Metrics: acc% (mean acceptance), succ% (final dist<0.6 & no collision), near% (final dist<1.5 & no collision), col%.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/meancov_finetune.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from polytope_explainer import rollout
from cfm_mppi.mppi.sweep import _load, DT

GRID = list(itertools.product([0.05, 0.1, 0.15], [1, 2, 3], [0.5, 1.0, 1.5], [0.5, 0.7]))  # cg, K, sv, noise


def ev(cg, K, sv, noise, g=0.5, neps=50, dev="cuda"):
    accs = []; succ = near = coll = n = 0
    for ep in range(neps):
        try:
            s0, goal, obs, _ = _load("ucy", ep, 80)
        except Exception:
            continue
        cfg = dict(horizon=10, dt=DT, num_samples=128, noise_sigma=(noise, noise), u_min=(-2., -2.), u_max=(2., 2.),
                   safety_margin=0.0, temperature=0.3, dynamics_type="singleintegrator", barrier_activation_radius=3.0,
                   use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, centroid_gain=cg,
                   centroid_horizon=K, sigma_volume_gain=sv, predict_gain=0.4, polytope_nbase=16)
        rec, path, goal = rollout("ucy", ep, g, cfg, dev=dev); n += 1
        accs.append(np.mean([st["n_acc"] / (st["n_acc"] + st["n_rej"] + 1e-9) for st in rec if st.get("poly") is not None]))
        mc = np.inf
        for t in range(min(len(path), 80)):
            ob = obs[min(t, obs.shape[0] - 1)]; ok = ~np.isnan(ob[:, :2]).any(1); o = ob[ok]
            if o.shape[0]:
                mc = min(mc, float(np.min(np.linalg.norm(o[:, :2] - path[t], axis=1) - o[:, 2] - 0.2)))
        fd = float(np.linalg.norm(path[-1] - goal))
        succ += int(fd < 0.6 and mc >= 0); near += int(fd < 1.5 and mc >= 0); coll += int(mc < 0)
    n = max(n, 1)
    return round(np.mean(accs) * 100), round(succ / n * 100), round(near / n * 100), round(coll / n * 100), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--neps", type=int, default=50); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    args = ap.parse_args(); mine = GRID[args.shard::args.nshard]; res = []
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs", flush=True)
    for cg, K, sv, noise in mine:
        a, s, nr, c, n = ev(cg, K, sv, noise, neps=args.neps, dev=args.device)
        res.append(dict(cg=cg, K=K, sv=sv, noise=noise, acc=a, succ=s, near=nr, col=c, n=n))
        print(f"  cg={cg} K={K} sv={sv} noise={noise}: acc={a}% succ={s}% near={nr}% col={c}% (n={n})", flush=True)
    json.dump(res, open(args.out, "w"), indent=2); print(f"[shard {args.shard}] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
