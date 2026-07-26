"""Locked Kazuki comparator on a fixed M/gamma bank with executed-window Validity.

Additive evaluation-only script: it changes nothing in the B1 pipeline.  The
comparator is the existing generate-guide-refine ``kazuki_sfm_deploy`` with the
locked configuration (safe_coefs=(0.3,), goal_coef=0.5, zero gamma spans,
sample_seed=700000) on the same Hp10 pretrained prior.  Episodes are seeded with
``SS.make_humans(episode, 0, n_ped, speed_range)`` exactly as the raw offline
evaluator, so a shared ep0 gives the same pedestrian bank.  Validity is the
identical terminal-truncated executed-window metric from
``sfm_b1_offline_eval._verify_executed_episode``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os

import numpy as np

import _paths  # noqa: F401


LOCKED = dict(safe_coef=0.3, goal_coef=0.5, sample_seed=700_000)


def _rollout_gamma(payload):
    """Worker: run every episode of one gamma cell and attach validity."""
    (checkpoint, scene_profile, ep0, m_per_gamma, gamma, device) = payload
    import torch  # noqa: F401  (worker-local import keeps spawn cheap to reason about)
    import grid_policy_sfm as GPS
    import sfm_b1_offline_eval as OE
    import sfm_kazuki as KZ
    import sfm_protocol as SP
    import sfm_scene as SS

    environment = SS.scene_profile(scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    config = KZ.KazukiConfig(
        safe_coefs=(float(LOCKED["safe_coef"]),),
        goal_coef=float(LOCKED["goal_coef"]),
    ).validate()
    rows = []
    for episode in range(int(ep0), int(ep0) + int(m_per_gamma)):
        rollout = KZ.kazuki_sfm_deploy(
            policy, episode, float(gamma), cfg=config,
            n_ped=environment["n_ped"], T=SP.T, device=device,
            ped_speed_range=tuple(environment["ped_speed_range"]),
            sample_seed=int(LOCKED["sample_seed"]), collect_diagnostics=False,
        )
        success = bool(rollout["success"])
        collision = bool(rollout["collision"])
        steps = int(rollout["steps"])
        row = {
            "episode": int(episode),
            "gamma": float(gamma),
            "status": (
                "success" if success
                else "collision" if collision else "timeout"
            ),
            "success": success,
            "collision": collision,
            "timeout": bool(not success and not collision),
            "steps": steps,
            "time_to_goal": steps * SS.DT if success else None,
            "min_clearance": float(rollout["min_clear"]),
            "successful_clearance": (
                float(rollout["min_clear"]) if success else None
            ),
            "states": np.asarray(rollout["states"], np.float32),
            "controls": np.asarray(rollout["controls"], np.float32),
            "ped_xy": np.asarray(rollout["peds"], np.float32),
            "ped_vel": np.asarray(rollout["ped_vels"], np.float32),
        }
        validity = OE._verify_executed_episode(row)
        for key in ("states", "controls", "ped_xy", "ped_vel"):
            row.pop(key)
        row.update(validity)
        rows.append(row)
    return rows


def run(args) -> dict:
    import sfm_b1_offline_eval as OE
    import sfm_kazuki as KZ
    import sfm_protocol as SP
    import sfm_scene as SS

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = os.path.abspath(args.checkpoint)
    checkpoint_sha = OE._sha256_file(checkpoint)
    payloads = [
        (
            checkpoint, args.scene_profile, int(args.ep0),
            int(args.m_per_gamma), float(gamma), args.device,
        )
        for gamma in SP.GAMMAS
    ]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(len(payloads), int(args.workers)),
        mp_context=context,
    ) as executor:
        cell_rows = list(executor.map(_rollout_gamma, payloads))
    rows = [row for cell in cell_rows for row in cell]
    summary = OE.summarize(
        rows, seed=int(args.ep0) + int(checkpoint_sha[:8], 16) % 100_000,
    )
    OE._assert_zero_verifier_errors(summary)
    result = {
        "status": "CLAUDE_KAZUKI_FIXED_BANK_COMPLETE",
        "method": "default Kazuki generate-guide-refine (locked)",
        "kazuki_config": dict(LOCKED),
        "kazuki_config_full": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(KZ.KazukiConfig(
                safe_coefs=(float(LOCKED["safe_coef"]),),
                goal_coef=float(LOCKED["goal_coef"]),
            ).validate()).items()
            if isinstance(value, (int, float, str, bool, tuple, type(None)))
        },
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "scene_profile": args.scene_profile,
        "environment": SS.scene_profile(args.scene_profile),
        "bank": {
            "ep0": int(args.ep0),
            "M_per_gamma": int(args.m_per_gamma),
            "same_scenario_ids_for_every_gamma": True,
            "pedestrian_seeding": "SS.make_humans(episode, 0, n_ped, speed_range)",
        },
        "summary": summary,
        "rows": rows,
        "metric_semantics": {
            "Validity": (
                "identical executed sliding-window metric as "
                "sfm_b1_offline_eval (H_t=min(10,N_tau-t), exact GREEN verifier)"
            ),
            "comparator_semantics": (
                "learned prior plus reward guidance and MPPI refinement; "
                "no retuning; no external shield or fallback"
            ),
        },
    }
    path = os.path.join(
        output_dir, f"kazuki_m{int(args.m_per_gamma)}_metrics.json"
    )
    OE._write_json(path, result)
    print(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--scene-profile", default="double_density_velocity_ood",
    )
    parser.add_argument("--ep0", type=int, required=True)
    parser.add_argument("--m-per-gamma", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
