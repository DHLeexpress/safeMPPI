"""Freeze the r1 self-anchor buffer for the stable-continuation study.

Runs the immutable r1 checkpoint RAW (temperature 1, NFE 8, CRN latents) on
the private anchor bank, keeps only SUCCESSFUL episodes, exact-verifies every
executed sliding window of full length H_t=10, and stores the y=1 windows
with their exact gathering-format contexts (hp10/low5/hist/state/ped_xy/
ped_vel).  This is self-replay of the policy's own certified successful
behavior — not expert data.  The buffer is frozen once, before round 1, with
a deterministic gamma-balanced cap (order by (episode, step), first 200 per
gamma).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_b1_offline_store as OS
import sfm_hp_history as HH
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS

CAP_PER_GAMMA = 200
H = 10


@torch.no_grad()
def _rollout_with_contexts(policy, episode, gamma, noise, device, environment):
    humans = SS.make_humans(
        int(episode), 0, environment["n_ped"],
        tuple(environment["ped_speed_range"]),
    )
    state = np.zeros(4, np.float32)
    history = HH.HpHistory()
    controls_list, contexts = [], []
    status = None
    minimum_clearance = float("inf")
    for step in range(int(SP.T)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        clearance = float(
            np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED
        ) if len(ped_xy) else float("inf")
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            status = "collision"
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < 0.5:
            status = "success"
            break
        obstacles = np.concatenate([
            ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32),
        ], axis=1)
        raw_grid = torch.as_tensor(GF.axis_grid(
            state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE,
        ))
        hp10 = history.append(raw_grid)
        low = torch.as_tensor(GF.low5(state, SS.GOAL, gamma))
        hist = torch.as_tensor(GF.hist_pad(
            np.asarray(controls_list[-16:])
            if controls_list else np.zeros((0, 2)), 16,
        ))
        ctx = policy.ctx_from(
            hp10[None].float().to(device), low[None].float().to(device),
            hist[None].float().to(device),
        )
        window = BE.integrate_latents(
            policy,
            torch.as_tensor(noise[step][None], device=device),
            ctx, nfe=8,
        ).reshape(H, 2).cpu().numpy().astype(np.float32)
        contexts.append(dict(
            step=int(step), state=state.copy(),
            hp10=hp10.numpy().astype(np.float32),
            low5=low.numpy().astype(np.float32),
            hist=hist.numpy().astype(np.float32),
            ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
        ))
        action = window[0]
        controls_list.append(action)
        state[:2] = state[:2] + SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
        state[2:4] = state[2:4] + SS.DT * action
        SS.advance_humans(humans, state)
    if status is None:
        status = "timeout"
    return dict(
        episode=int(episode), gamma=float(gamma), status=status,
        steps=len(controls_list),
        controls=np.asarray(controls_list, np.float32),
        contexts=contexts, min_clearance=float(minimum_clearance),
    )


def collect(checkpoint, outpath, *, ep0, m_per_gamma, noise_seed, device,
            workers):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    environment = SS.scene_profile("double_density_velocity_ood")
    generator = np.random.default_rng(int(noise_seed))
    noise = generator.standard_normal(
        (len(SP.GAMMAS), int(m_per_gamma), int(SP.T), int(policy.d)),
        dtype=np.float32,
    )
    rollouts = []
    for gamma_index, gamma in enumerate(SP.GAMMAS):
        for rollout_index in range(int(m_per_gamma)):
            rollouts.append(_rollout_with_contexts(
                policy, int(ep0) + rollout_index, float(gamma),
                noise[gamma_index, rollout_index], device, environment,
            ))
    successes = [r for r in rollouts if r["status"] == "success"]
    tasks, meta = [], []
    for run_index, run in enumerate(successes):
        n = run["steps"]
        for start in range(0, n - H + 1):
            tasks.append((
                len(meta), 0, run["contexts"][start]["state"],
                run["controls"][start:start + H],
                run["contexts"][start]["ped_xy"],
                run["contexts"][start]["ped_vel"], run["gamma"],
            ))
            meta.append((run_index, start))
    context_mp = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=context_mp,
    ) as executor:
        results = {i: r for i, _, r in executor.map(SM.verify_in_worker, tasks)}
    per_gamma = {}
    for task_index, (run_index, start) in enumerate(meta):
        result = results[task_index]
        if not result.get("resolved") or int(result.get("y", 0)) != 1:
            continue
        run = successes[run_index]
        per_gamma.setdefault(round(run["gamma"], 8), []).append(dict(
            episode=run["episode"], gamma=run["gamma"], step=start,
            context=run["contexts"][start],
            controls=run["controls"][start:start + H].copy(),
        ))
    records = []
    audit_counts = {}
    for gamma in sorted(per_gamma):
        rows = sorted(per_gamma[gamma], key=lambda r: (r["episode"], r["step"]))
        kept = rows[:CAP_PER_GAMMA]
        audit_counts[str(gamma)] = dict(
            certified_available=len(rows), kept=len(kept),
        )
        records.extend(kept)
    payload = dict(
        status="R1_ANCHOR_BUFFER_FROZEN",
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=OS.sha256_file(checkpoint),
        bank=dict(ep0=int(ep0), m_per_gamma=int(m_per_gamma),
                  noise_seed=int(noise_seed)),
        cap_per_gamma=CAP_PER_GAMMA,
        rollout_outcomes={
            status: sum(r["status"] == status for r in rollouts)
            for status in ("success", "collision", "timeout")
        },
        windows_verified=len(tasks),
        per_gamma=audit_counts,
        n_records=len(records),
        records=records,
    )
    torch.save(payload, outpath)
    summary = {k: payload[k] for k in (
        "status", "checkpoint_sha256", "rollout_outcomes",
        "windows_verified", "per_gamma", "n_records",
    )}
    with open(outpath + ".summary.json", "w") as stream:
        json.dump(summary, stream, indent=1)
    print(json.dumps(summary, indent=1))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ep0", type=int, default=380_000)
    parser.add_argument("--m-per-gamma", type=int, default=12)
    parser.add_argument("--noise-seed", type=int, default=20_260_739)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    collect(
        args.checkpoint, args.out, ep0=args.ep0,
        m_per_gamma=args.m_per_gamma, noise_seed=args.noise_seed,
        device=args.device, workers=args.workers,
    )


if __name__ == "__main__":
    main()
