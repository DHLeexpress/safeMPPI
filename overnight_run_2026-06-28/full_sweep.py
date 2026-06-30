"""Full UCY+SDD sweep of the FINE-TUNED polytope SafeMPPI config, over gamma, sharded across GPUs, time-budgeted.
Config (50-eps fine-tuned): nominal=0 + warm-start, margin=0, polytope barrier, cg=0.1/K=3/sv=1.0/noise=0.5,
predict_gain=0.4, sensing=3.0, temp=0.3, H=10. Eval set = 300 distinct episodes/dataset.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/full_sweep.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from polytope_explainer import rollout
from cfm_mppi.mppi.sweep import _load, DT

GAMMAS = [0.1, 0.3, 0.5, 0.7, 1.0]; NEPS = 300; DATASETS = ["ucy", "sdd"]
CFG = dict(horizon=10, dt=DT, num_samples=512, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
           safety_margin=0.0, temperature=0.3, dynamics_type="singleintegrator", barrier_activation_radius=2.0,
           use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, centroid_gain=0.1,
           centroid_smooth=0.5, sigma_volume_gain=0.5, sigma_aniso=2.0, control_weight=0.03,
           predict_gain=0.4, polytope_nbase=16)


def metrics(ds, ep, g, dev):
    rec, path, goal = rollout(ds, ep, g, dict(CFG), dev=dev); s0, _, obs, _ = _load(ds, ep, 80)
    acc = float(np.mean([st["n_acc"] / (st["n_acc"] + st["n_rej"] + 1e-9) for st in rec if st.get("poly") is not None]))
    mc = np.inf
    for t in range(min(len(path), 80)):
        ob = obs[min(t, obs.shape[0] - 1)]; ok = ~np.isnan(ob[:, :2]).any(1); o = ob[ok]
        if o.shape[0]:
            mc = min(mc, float(np.min(np.linalg.norm(o[:, :2] - path[t], axis=1) - o[:, 2] - 0.2)))
    fd = float(np.linalg.norm(path[-1] - goal))
    return dict(ds=ds, g=g, ep=ep, acc=acc, succ=int(fd < 0.6 and mc >= 0), near=int(fd < 1.5 and mc >= 0), col=int(mc < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--deadline", type=float, default=1620.0)   # 27 min wall budget
    args = ap.parse_args()
    TUP = [(ds, g, ep) for ds in DATASETS for g in GAMMAS for ep in range(NEPS)]
    mine = TUP[args.shard::args.nshard]; res = []; t0 = time.time()
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} rollouts", flush=True)
    for i, (ds, g, ep) in enumerate(mine):
        if time.time() - t0 > args.deadline:
            print(f"[shard {args.shard}] deadline hit at {i}/{len(mine)}", flush=True); break
        try:
            res.append(metrics(ds, ep, g, args.device))
        except Exception:
            pass
        if (i + 1) % 150 == 0:
            print(f"[shard {args.shard}] {i+1}/{len(mine)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(args.out, "w")); print(f"[shard {args.shard}] saved {len(res)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
