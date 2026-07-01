"""Importance-sampling (Mode-B = polytope-AREA rays) sweep on the di_grid_current_best.gif episodes (UCY 16,47,90,150)
x gamma{0.1,0.5,1.0}. Fine grid over predict_gain x centroid_smooth x centroid_eps x centroid_gain x urgency-mode{1,4}.
PRIORITY metric: min accepted-per-step across ALL (ep,gamma,step) >= 1 (never hit the fallback). Then success/collision.
No random_backup_frac / sigma_aniso (polytope sampling supplies Mode-B diversity); base noise kept for goal-seeking.
  CUDA_VISIBLE_DEVICES=0 python overnight_run_2026-06-28/area_sweep.py --shard 0 --nshard 4 --steps 80 --out .../s0.json
"""
from __future__ import annotations
import argparse, itertools, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from cfm_mppi.mppi.sweep import _load, DT
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

EPISODES = [16, 47, 90, 150]; GAMMAS = [0.1, 0.5, 1.0]
GRID = list(itertools.product([0.0, 0.2, 0.4], [0.25, 0.5, 0.75], [0.15, 0.3], [0.1, 0.2, 0.3], [False, True]))  # predict,smooth,eps,cg,mode4


def di_step(s, u):
    return np.array([s[0] + 0.1 * s[2] + 0.005 * u[0], s[1] + 0.1 * s[3] + 0.005 * u[1],
                     s[2] + 0.1 * u[0], s[3] + 0.1 * u[1]], np.float32)


def cfg_for(predict, smooth, eps, cg, mode4):
    return dict(horizon=10, dt=DT, num_samples=512, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
                safety_margin=0.0, temperature=0.1, dynamics_type="doubleintegrator", barrier_activation_radius=3.0,
                use_polytope_barrier=True, use_goal_nominal=False, warm_start=True, centroid_gain=cg, centroid_smooth=smooth,
                centroid_eps=eps, sigma_volume_gain=0.0, sigma_aniso=2.5, random_backup_frac=0.0, predict_gain=predict,
                polytope_nbase=16, polytope_area_sampling=True, urgency_size_diff=mode4)


def one(ep, g, cfg, steps, dev):
    s0, goal, obs, vel = _load("ucy", ep, max(steps, 120))
    ad = SafeMPPIAdapter(**cfg); st = np.array([s0[0], s0[1], 0, 0.], np.float32); mc = np.inf; reached = False; minacc = 10 ** 9
    for t in range(steps):
        ob = obs[min(t, obs.shape[0] - 1)]; vl = vel[min(t, vel.shape[0] - 1)]; ok = ~np.isnan(ob[:, :2]).any(1); o = ob[ok]
        a, info = ad.plan(torch.tensor(st, dtype=torch.float32, device=dev), torch.tensor(goal, dtype=torch.float32, device=dev),
                          torch.tensor(o, dtype=torch.float32, device=dev), gamma=g,
                          obstacle_velocities=torch.tensor(vl[ok], dtype=torch.float32, device=dev), seed=t)
        nrej = int(info["num_barrier_violations"]); rate = float(info["infeasibility_rate"]); ntot = int(round(nrej / rate)) if rate > 1e-9 else int(cfg["num_samples"])
        minacc = min(minacc, ntot - nrej)
        st = di_step(st, a.detach().cpu().numpy())
        if o.shape[0]:
            mc = min(mc, float(np.min(np.linalg.norm(o[:, :2] - st[:2], axis=1) - o[:, 2] - 0.2)))
        if np.linalg.norm(st[:2] - goal) < 0.6:
            reached = True; break
    return minacc, int(reached and mc >= 0), int(mc < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=80)
    args = ap.parse_args(); mine = GRID[args.shard::args.nshard]; res = []
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} configs", flush=True)
    for predict, smooth, eps, cg, mode4 in mine:
        cfg = cfg_for(predict, smooth, eps, cg, mode4); worst = 10 ** 9; S = Co = n = 0
        for ep in EPISODES:
            for g in GAMMAS:
                ma, su, co = one(ep, g, cfg, args.steps, args.device); worst = min(worst, ma); S += su; Co += co; n += 1
        r = dict(predict=predict, smooth=smooth, eps=eps, cg=cg, mode=(4 if mode4 else 1), worst_minacc=worst,
                 succ=round(100 * S / n), col=round(100 * Co / n))
        res.append(r); json.dump(res, open(args.out, "w"), indent=1)
        print(f"  predict={predict} smooth={smooth} eps={eps} cg={cg} mode={'4' if mode4 else '1'}: worst_minacc={worst} succ={r['succ']}% col={r['col']}%", flush=True)
    print(f"[shard {args.shard}] done {len(res)}", flush=True)


if __name__ == "__main__":
    main()
