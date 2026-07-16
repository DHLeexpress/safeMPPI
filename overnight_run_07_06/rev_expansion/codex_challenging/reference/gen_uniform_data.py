"""Generate SafeMPPI rollout data from a UNIFORM grid of starts (rev_expansion, user 2026-07-08).
Replaces dr05's random off-diagonal sampling with a uniform 32x32 grid (|y-x|>=1, obstacle-free, +-0.02 jitter)
so the state distribution is dense + balanced (anti-collapse) and the on-diagonal/origin region is in-distribution.
Saves dataset/druni_windows_g{gamma}.pt in the same schema as dr05. One process per gamma (CPU). No git/wandb.
"""
import sys, os
sys.path.insert(0, '/home/dohyun/projects/cfm_mppi/overnight_run_07_06')
import _paths  # noqa
import argparse, time
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '24')))   # avoid 7-proc oversubscription (256 cores)
import grid_scene as GS
from di_grid_viz import di_step
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from stage2_grid_data import windows_from

DATASET = '/home/dohyun/projects/cfm_mppi/overnight_run_07_06/dataset'


def uniform_starts():
    env = GS.make_grid(); obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    LO, HI, N = 0.1, 4.9, 32; cell = (HI - LO) / N
    cen = LO + cell * (np.arange(N) + 0.5); X, Y = np.meshgrid(cen, cen)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    pts = pts + np.random.default_rng(0).uniform(-0.02, 0.02, pts.shape)   # fixed jitter (same starts every gamma)
    band = np.abs(pts[:, 1] - pts[:, 0]) >= 1.0
    clr = np.linalg.norm(pts[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr
    free = clr.min(1) > 0.05
    return pts[band & free]


def rollout_from(env, cfg, gamma, start_xy, seed, reach=0.4):
    ad = SafeMPPIAdapter(**cfg)
    st = np.array([start_xy[0], start_xy[1], 0.0, 0.0], dtype=np.float32)
    goal_t = env.goal.detach().cpu().float(); obs_plan = GS.planner_obstacles(env)
    goal = env.goal.detach().cpu().numpy()
    states, controls = [st.copy()], []
    for t in range(env.T):
        a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_plan, gamma=float(gamma), seed=seed * 1000 + t)
        a = a.detach().cpu().numpy().astype(np.float32); st = di_step(st, a, dt=env.dt)
        states.append(st.copy()); controls.append(a)
        if np.linalg.norm(st[:2] - goal) < reach:
            break
    return np.array(states, np.float32), np.array(controls, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma', type=float, required=True)
    ap.add_argument('--out-prefix', default='druni_')
    args = ap.parse_args()
    env = GS.make_grid(); cfg = GS.mode1_config()
    starts = uniform_starts()
    G, L, Hh, U, st_ok = [], [], [], [], []
    n_ok = 0; t0 = time.time()
    for i, s in enumerate(starts):
        states, controls = rollout_from(env, cfg, args.gamma, s, seed=i)
        ok, _ = GS.is_success(states[:, :2], env)
        if not ok or len(controls) < 2:
            continue
        n_ok += 1
        g, l, h, u = windows_from(states, controls, env, args.gamma)
        G += g; L += l; Hh += h; U += u; st_ok.append([float(s[0]), float(s[1]), 0.0, 0.0])
        if (i + 1) % 100 == 0:
            print(f'g{args.gamma}: {i+1}/{len(starts)} starts, {n_ok} ok, {len(G)} windows, '
                  f'{(time.time()-t0)/(i+1):.2f}s/start', flush=True)
    out = os.path.join(DATASET, f'{args.out_prefix}windows_g{args.gamma}.pt')
    torch.save(dict(grid=torch.tensor(np.array(G)), low5=torch.tensor(np.array(L)),
                    hist=torch.tensor(np.array(Hh)), U=torch.tensor(np.array(U)),
                    starts=torch.tensor(np.array(st_ok)), gamma=float(args.gamma),
                    n_traj=n_ok, n_seeds=len(starts)), out)
    print(f'SAVED {out}: {n_ok}/{len(starts)} trajs, {len(G)} windows ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
