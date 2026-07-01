"""SafeMPPI data engine (Pillar 2 -> Pillar 4 data): the scarce conservative dataset.

Deploys the FROZEN ``best_area_mode4`` double-integrator planner closed-loop in each static
cluttered scene of the bank, across a gamma grid and a few seeds, and records the EXECUTED
control sequence (the reward-weighted first action per receding-horizon step) as the FM training
target U, conditioned on (start, goal, obstacle-set, gamma).  Episodes are run a fixed length
``T_exec`` (no early stop -> no padding ambiguity); only goal-reaching, collision-free episodes
are kept (the conservative safe demos we distil into the seed policy).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import _paths
import env as E
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter


def load_config():
    with open(_paths.BEST_CONFIG) as f:
        return json.load(f)["config"]


def _min_clear_np(S: np.ndarray, env) -> float:
    obs = env.obstacles.detach().cpu().numpy()
    if obs.shape[0] == 0:
        return float("inf")
    p = S[:, :2]
    d = np.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - float(env.r_robot)
    return float(d.min())


def run_episode(env, gamma: float, seed: int, cfg: dict):
    """Fresh adapter per episode; fixed-length receding-horizon rollout. Returns (U, S, reached, minclr)."""
    ad = SafeMPPIAdapter(**cfg)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    goal_t = env.goal.detach().cpu().float()
    obs_t = env.obstacles.detach().cpu().float()
    Us, states = [], [st.copy()]
    for t in range(env.T):
        a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_t,
                       gamma=gamma, seed=seed * 1000 + t, return_rollouts=False)  # static: no obs_vel
        u = a.detach().cpu().numpy().astype(np.float32)
        Us.append(u)
        st = E.di_step(st, u, dt=env.dt)
        states.append(st.copy())
    U = np.stack(Us).astype(np.float32)               # [T,2]
    S = np.stack(states).astype(np.float32)           # [T+1,4]
    d_goal = np.linalg.norm(S[:, :2] - env.goal.detach().cpu().numpy(), axis=1)
    reached = bool(d_goal.min() < 0.4 and d_goal[-1] < 0.6)
    return U, S, reached, _min_clear_np(S, env)


def screen_navigable_bank(n_scenes, base_seed, scene_kwargs, cfg,
                          gammas_probe=(0.3, 0.5, 0.7), max_scan=80, log=print):
    """Keep only scenes the frozen planner can solve (challenging-but-solvable clutter)."""
    envs, seeds, s = [], [], base_seed
    while len(envs) < n_scenes and s < base_seed + max_scan:
        e = E.make_clutter_env(s, **scene_kwargs)
        ok = False
        for g in gammas_probe:
            _, _, reached, mc = run_episode(e, g, 0, cfg)
            if reached and mc >= 0.0:
                ok = True
                break
        if ok:
            envs.append(e)
            seeds.append(s)
        s += 1
    if len(envs) < n_scenes:
        raise RuntimeError(f"Only {len(envs)}/{n_scenes} navigable scenes in {max_scan} seeds.")
    log(f"screened {len(envs)} navigable scenes from seeds {seeds}")
    return envs, seeds


def build_dataset(bank, gammas, seeds, cfg, log=print):
    Us, starts, goals, gams, sids = [], [], [], [], []
    n_try = n_reach = n_safe = 0
    t0 = time.time()
    for si, env in enumerate(bank):
        for g in gammas:
            for sd in seeds:
                n_try += 1
                U, S, reached, minclr = run_episode(env, g, sd, cfg)
                if reached:
                    n_reach += 1
                if reached and minclr >= 0.0:
                    n_safe += 1
                    Us.append(U)
                    starts.append(env.x0[:2].detach().cpu().numpy())
                    goals.append(env.goal.detach().cpu().numpy())
                    gams.append(g)
                    sids.append(si)
        log(f"  scene {si}: kept {n_safe}/{n_try} so far ({time.time()-t0:.0f}s)")
    if not Us:
        raise RuntimeError("No safe reaching episodes collected — check config/scene scale.")
    data = {
        "U": torch.tensor(np.stack(Us), dtype=torch.float32),
        "start": torch.tensor(np.stack(starts), dtype=torch.float32),
        "goal": torch.tensor(np.stack(goals), dtype=torch.float32),
        "gamma": torch.tensor(gams, dtype=torch.float32),
        "scene_id": torch.tensor(sids, dtype=torch.long),
        "scenes": [{"obstacles": e.obstacles.detach().cpu(),
                    "start": e.x0[:2].detach().cpu(),
                    "goal": e.goal.detach().cpu()} for e in bank],
        "T": bank[0].T, "dt": bank[0].dt, "u_max": bank[0].u_max, "r_robot": bank[0].r_robot,
        "gammas": list(gammas), "reached_rate": n_reach / max(n_try, 1),
        "safe_rate": n_safe / max(n_try, 1),
    }
    log(f"dataset: {len(Us)} demos kept / {n_try} tried "
        f"(reach {data['reached_rate']:.2f}, safe {data['safe_rate']:.2f}) in {time.time()-t0:.0f}s")
    return data


def bank_params(smoke: bool):
    """(n_scenes, base_seed, scene kwargs), gamma grid, seed grid. One place so pretrain reuses it."""
    if smoke:
        scene = dict(n_obs=8, box=3.2, r_min=0.2, r_max=0.4, clearance=0.55, T_exec=48)
        return dict(n_scenes=4, base_seed=100, scene=scene), [0.3, 0.5, 0.7], [0, 1]
    scene = dict(n_obs=9, box=3.0, r_min=0.22, r_max=0.42, clearance=0.5, T_exec=80)
    return dict(n_scenes=8, base_seed=100, scene=scene), [0.1, 0.3, 0.5, 0.7], [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(_paths.HERE, "results", "dataset.pt"))
    args = ap.parse_args()

    cfg = load_config()
    bp, gammas, seeds = bank_params(args.smoke)
    print(f"=== build_dataset smoke={args.smoke} | target {bp['n_scenes']} scenes x {len(gammas)} gamma x "
          f"{len(seeds)} seeds, T={bp['scene']['T_exec']} ===", flush=True)
    bank, scene_seeds = screen_navigable_bank(bp["n_scenes"], bp["base_seed"], bp["scene"], cfg)
    data = build_dataset(bank, gammas, seeds, cfg)
    data["bank_params"] = bp
    data["scene_seeds"] = scene_seeds
    data["config"] = cfg
    torch.save(data, args.out)
    print("saved", args.out, "| demos", data["U"].shape, flush=True)


if __name__ == "__main__":
    main()
