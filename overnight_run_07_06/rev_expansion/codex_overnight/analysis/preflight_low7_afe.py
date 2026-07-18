"""Authenticate the low7 checkpoint, context schema, and canonical giant scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import afe_context as CX
from afe2_scene_profiles import build_scene, get_scene_profile
from codex_challenging.afe_restart.evaluate_low7_pretrained import (
    validate_scene_contract,
)
import grid_expand_afe2 as AFE2
import grid_hp_expt as HP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--scene-profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    checkpoint_sha = AFE2._sha256_file(args.checkpoint)
    if checkpoint_sha != args.expected_sha256:
        raise RuntimeError("low7 checkpoint file SHA-256 mismatch")
    policy, payload = HP.load_hp(args.checkpoint, device="cpu")
    model_sha, checkpoint_contract, contract_sha = AFE2.validate_checkpoint_contract(
        args.scene_profile, policy, payload, checkpoint_sha
    )
    policy_contract = CX.policy_contract(policy)
    profile = get_scene_profile(args.scene_profile)
    env = build_scene(profile)
    scene = validate_scene_contract(profile.name, env)
    probes = []
    for position in ((1.25, 2.5), (3.75, 2.5), (2.5, 1.25), (2.5, 3.75)):
        state = np.asarray((*position, 0.0, 0.0), dtype=np.float32)
        context = CX.build_context(
            state, env.goal.numpy(), 0.5, [], env, CX.LOW7_SCHEMA
        )
        probes.append({
            "position": list(position),
            "closest_boundary_vector_scaled": [
                float(value) for value in context.low5[4:6]
            ],
        })
    if not (
        probes[0]["closest_boundary_vector_scaled"][0] > 0.0
        and probes[1]["closest_boundary_vector_scaled"][0] < 0.0
        and probes[2]["closest_boundary_vector_scaled"][1] > 0.0
        and probes[3]["closest_boundary_vector_scaled"][1] < 0.0
    ):
        raise RuntimeError("low7 boundary-vector probe orientation is incorrect")
    source = AFE2._git_state()
    if (
        source["commit"] is None
        or source["tracked_dirty"]
        or source["untracked_runtime_sources"]
    ):
        raise RuntimeError(f"preflight requires clean committed source: {source}")
    result = {
        "status": "LOW7_AFE_PREFLIGHT_COMPLETE",
        "source": source,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "file_sha256": checkpoint_sha,
            "model_state_sha256": model_sha,
            "contract": checkpoint_contract,
            "contract_sha256": contract_sha,
        },
        "policy": {
            "conditioning_schema": policy_contract.schema,
            "raw_condition_dim": policy_contract.raw_condition_dim,
            "ctx_dim": policy_contract.ctx_dim,
            "trunk_input_dim": policy_contract.trunk_input_dim,
            "parameter_count": sum(p.numel() for p in policy.parameters()),
        },
        "scene": scene,
        "boundary_vector_probes": probes,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
