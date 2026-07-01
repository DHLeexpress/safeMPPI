"""DOUBLE-INTEGRATOR one-at-a-time (OAT) parameter sweep on UCY+SDD with the 3-mode mixture (Mode C always-on).
Vary each parameter over its candidates while the rest sit at the di_grid center; report acc/succ/col.
50 eps/dataset x gamma {0.1,0.5,1.0}; gamma=0.1 gets more rollout steps. random_backup_frac fixed 0.03.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/param_oat_di.py --shard 0 --nshard 4 --out .../s0.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from cfm_mppi.mppi.sweep import _load, DT
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

CENTER = dict(centroid_gain=0.1, sigma_volume_gain=1.0, sigma_aniso=2.5, sensing=2.0, num_samples=256,
              temperature=0.3, noise=0.5, predict_gain=0.4, centroid_smooth=0.5, centroid_eps=0.15,
              random_backup_frac=0.1)
SWEEP = {
    "centroid_gain": [0.05, 0.1, 0.2, 0.3], "sigma_volume_gain": [0.0, 0.5, 1.0, 1.5],
    "sigma_aniso": [1.0, 2.0, 2.5, 3.0], "sensing": [2.0, 2.5, 3.0, 3.5], "num_samples": [256, 512],
    "temperature": [0.1, 0.3, 0.5, 1.0], "noise": [0.3, 0.5, 0.7], "predict_gain": [0.0, 0.2, 0.4, 0.6],
    "centroid_smooth": [0.0, 0.25, 0.5, 0.75], "centroid_eps": [0.05, 0.15, 0.3, 0.5],
    "random_backup_frac": [0.0, 0.05, 0.1, 0.2],   # 11th: Mode-C always-on backup fraction p_c
}
CONFIGS = [(p, v, {**CENTER, p: v}) for p, vals in SWEEP.items() for v in vals]   # 37 OAT configs
EPS = list(range(0, 300, 6)); GAMMAS = [0.1, 0.5, 1.0]; DATASETS = ["ucy", "sdd"]   # 50 eps/dataset


def di_step(s, u):
    return np.array([s[0] + 0.1 * s[2] + 0.005 * u[0], s[1] + 0.1 * s[3] + 0.005 * u[1],
                     s[2] + 0.1 * u[0], s[3] + 0.1 * u[1]], np.float32)


def build_cfg(c):
    return dict(horizon=10, dt=DT, num_samples=int(c["num_samples"]), noise_sigma=(c["noise"], c["noise"]),
                u_min=(-2., -2.), u_max=(2., 2.), safety_margin=0.0, temperature=c["temperature"],
                dynamics_type="doubleintegrator", barrier_activation_radius=c["sensing"], use_polytope_barrier=True,
                use_goal_nominal=False, warm_start=True, centroid_gain=c["centroid_gain"], centroid_smooth=c["centroid_smooth"],
                centroid_eps=c["centroid_eps"], sigma_volume_gain=c["sigma_volume_gain"], sigma_aniso=c["sigma_aniso"],
                random_backup_frac=c["random_backup_frac"], control_weight=0.03, predict_gain=c["predict_gain"], polytope_nbase=16)


def one(ds, ep, g, cfg, dev):
    steps = 120 if g <= 0.1 else 80                                    # gamma=0.1 gets more completion time
    s0, goal, obs, vel = _load(ds, ep, 120)
    ad = SafeMPPIAdapter(**cfg); st = np.array([s0[0], s0[1], 0, 0.], np.float32); mc = np.inf; reached = False; ar = []
    for t in range(steps):
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
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--deadline", type=float, default=7200.0)
    args = ap.parse_args(); mine = CONFIGS[args.shard::args.nshard]; res = []; t0 = time.time()
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} OAT configs", flush=True)
    for pname, v, c in mine:
        if time.time() - t0 > args.deadline:
            print(f"[shard {args.shard}] deadline; {len(res)}/{len(mine)}", flush=True); break
        S = Co = n = 0; A = []
        for ds in DATASETS:
            for g in GAMMAS:
                for ep in EPS:
                    su, co, a = one(ds, ep, g, build_cfg(c), args.device); S += su; Co += co; A.append(a); n += 1
        res.append(dict(param=pname, value=v, succ=round(100 * S / n), col=round(100 * Co / n), acc=round(100 * np.mean(A))))
        r = res[-1]; print(f"  {pname}={v}: succ={r['succ']}% col={r['col']}% acc={r['acc']}% ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(args.out, "w"), indent=2); print(f"[shard {args.shard}] saved {len(res)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
