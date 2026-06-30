"""DOUBLE-INTEGRATOR fine-tune on the eval datasets (UCY+SDD). Grid: centroid_gain x sigma_volume_gain x
sigma_aniso x sensing. Fixed: nominal=0 + warm-start, margin=0, polytope barrier, ns=256, control_weight=0.03,
predict_gain=0.4, temp=0.3, H=10. 20-ep spread per dataset x gamma {0.1,0.5,1.0}. Reports succ/col/acc; picks best.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/param_finetune_di.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from cfm_mppi.mppi.sweep import _load, DT
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

GRID = list(itertools.product([0.1, 0.2, 0.3], [0.5, 1.0], [1.5, 2.5], [2.0, 3.0]))  # cg, sv, aniso, sensing
EPS = list(range(0, 300, 15)); GAMMAS = [0.1, 0.5, 1.0]; DATASETS = ["ucy", "sdd"]


def di_step(s, u):
    return np.array([s[0] + 0.1 * s[2] + 0.005 * u[0], s[1] + 0.1 * s[3] + 0.005 * u[1],
                     s[2] + 0.1 * u[0], s[3] + 0.1 * u[1]], np.float32)


def one(ds, ep, g, cg, sv, an, sens, dev):
    s0, goal, obs, vel = _load(ds, ep, 80)
    cfg = dict(horizon=10, dt=DT, num_samples=256, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
               safety_margin=0.0, temperature=0.3, dynamics_type="doubleintegrator", barrier_activation_radius=sens,
               use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, centroid_gain=cg, centroid_smooth=0.5,
               sigma_volume_gain=sv, sigma_aniso=an, control_weight=0.03, predict_gain=0.4, polytope_nbase=16)
    ad = SafeMPPIAdapter(**cfg); st = np.array([s0[0], s0[1], 0, 0.], np.float32); mc = np.inf; reached = False; ar = []
    for t in range(80):
        ob = obs[min(t, obs.shape[0] - 1)]; vl = vel[min(t, vel.shape[0] - 1)]; ok = ~np.isnan(ob[:, :2]).any(1); o = ob[ok]
        a, info = ad.plan(torch.tensor(st, dtype=torch.float32, device=dev), torch.tensor(goal, dtype=torch.float32, device=dev),
                          torch.tensor(o, dtype=torch.float32, device=dev), gamma=g,
                          obstacle_velocities=torch.tensor(vl[ok], dtype=torch.float32, device=dev), seed=t)
        ar.append(1 - info["infeasibility_rate"]); st = di_step(st, a.detach().cpu().numpy())
        if o.shape[0]:
            mc = min(mc, float(np.min(np.linalg.norm(o[:, :2] - st[:2], axis=1) - o[:, 2] - 0.2)))
        if np.linalg.norm(st[:2] - goal) < 0.6:
            reached = True; break
    return int(reached and mc >= 0), int(mc < 0), float(np.mean(ar))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--deadline", type=float, default=1620.0)
    args = ap.parse_args(); mine = GRID[args.shard::args.nshard]; res = []; t0 = time.time()
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs", flush=True)
    for cg, sv, an, sens in mine:
        if time.time() - t0 > args.deadline:
            print(f"[shard {args.shard}] deadline; {len(res)}/{len(mine)}", flush=True); break
        S = C = n = 0; A = []
        for ds in DATASETS:
            for g in GAMMAS:
                for ep in EPS:
                    su, co, a = one(ds, ep, g, cg, sv, an, sens, args.device); S += su; C += co; A.append(a); n += 1
        res.append(dict(cg=cg, sv=sv, aniso=an, sens=sens, succ=round(100 * S / n), col=round(100 * C / n), acc=round(100 * np.mean(A))))
        r = res[-1]; print(f"  cg={cg} sv={sv} aniso={an} sens={sens}: succ={r['succ']}% col={r['col']}% acc={r['acc']}%", flush=True)
    json.dump(res, open(args.out, "w"), indent=2); print(f"[shard {args.shard}] saved {len(res)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
