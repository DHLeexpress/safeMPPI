"""Wide Stage-2 sweep over the escape-bias family, sharded across GPUs.
Axes: escape_gain x sensing(barrier_activation_radius) x horizon x noise_sigma x gamma.
Each config evaluated on UCY+SDD (50 eps). Target: >=90% success & 0 collision on BOTH.

  # launch one shard per GPU:
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/wide_sweep.py --shard 0 --nshard 4 --out .../grid_s0.json
"""
from __future__ import annotations
import argparse, itertools, json, os
from copy import deepcopy
from cfm_mppi.mppi.sweep import evaluate

BASE = dict(dt=0.1, num_samples=128, temperature=1.0, u_min=(-2., -2.), u_max=(2., 2.), safety_margin=0.5,
            dynamics_type="singleintegrator", use_ho_barrier=False, eta=0.0, use_guidance=False,
            use_aniso_cov=False, barrier_topk=0, predict_gain=0.4, control_weight=0.10)
AX = dict(escape_gain=[0.03, 0.05, 0.08], sensing=[2.0, 2.5], horizon=[10, 15], noise=[0.4, 0.6])
GAMMAS = [0.3, 0.5, 0.7]


def grid():
    out = []
    for e, s, H, nz in itertools.product(AX["escape_gain"], AX["sensing"], AX["horizon"], AX["noise"]):
        for g in GAMMAS:
            cfg = dict(BASE, escape_gain=e, barrier_activation_radius=s, horizon=H, noise_sigma=(nz, nz))
            out.append((cfg, g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    G = grid(); mine = G[args.shard::args.nshard]
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs", flush=True)
    res = []
    for cfg, g in mine:
        r = {ds: evaluate(cfg, ds, args.episodes, g, args.device) for ds in ["ucy", "sdd"]}
        rec = {"escape": cfg["escape_gain"], "sensing": cfg["barrier_activation_radius"], "horizon": cfg["horizon"],
               "noise": cfg["noise_sigma"][0], "gamma": g, "ucy": r["ucy"], "sdd": r["sdd"]}
        res.append(rec)
        print(f"  esc={rec['escape']} sens={rec['sensing']} H={rec['horizon']} nz={rec['noise']} γ={g}: "
              f"UCY {r['ucy']['success_rate']*100:.0f}/{r['ucy']['collide_rate']*100:.0f}  "
              f"SDD {r['sdd']['success_rate']*100:.0f}/{r['sdd']['collide_rate']*100:.0f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[shard {args.shard}] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
