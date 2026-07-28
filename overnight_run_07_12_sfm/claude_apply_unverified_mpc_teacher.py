"""Apply one isolated privileged-MPC teacher block after an ordinary SFE update.

The input checkpoint is treated as ``theta_(n+1/2)``: ordinary executed
``D+/D-`` replay has already happened.  This command harvests a separate
control-bounded ``D_MPC`` from the matching round shard, applies a dedicated
CFM block, and writes ``theta_(n+1)``.  It never edits the round shard and never
routes teacher rows into the verifier, GP, acquisition, or validity metrics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_offline_aug as AUG
import claude_unverified_mpc_teacher as T
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_exec as OE
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_scene as SS


STATUS = "SFM_UNVERIFIED_MPC_TEACHER_BLOCK_COMPLETE"


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _hard_windows(shard):
    _, population_b, stats = AUG.tag_populations(shard)
    rows = {int(row["window_id"]): row for row in population_b}
    for row in shard.windows:
        if bool(row.get("nvp_context")):
            rows[int(row["window_id"])] = row
    return [rows[key] for key in sorted(rows)], stats


def _probe_objective(policy, shard, records, *, teacher, device, seed):
    if not records:
        return None, {}
    if teacher:
        values = list(records)[:128]
        hp10, low, hist, controls = T._tensor_batch(shard, values, device)
    else:
        values = list(records)[:128]
        contexts = [shard.contexts[int(row["context_id"])] for row in values]
        hp10 = torch.as_tensor(
            np.stack([row["hp10"] for row in contexts]), device=device,
        ).float()
        low = torch.as_tensor(
            np.stack([row["low5"] for row in contexts]), device=device,
        ).float()
        hist = torch.as_tensor(
            np.stack([row["hist"] for row in contexts]), device=device,
        ).float()
        controls = torch.as_tensor(
            np.stack([row["controls"] for row in values]), device=device,
        ).float()
    torch.manual_seed(int(seed))
    context = policy.ctx_from(hp10, low, hist)
    loss = policy.cfm_loss(controls, context)
    parameters = [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=False,
    )
    snapshot = {
        name: gradient.detach().cpu()
        for (name, parameter), gradient in zip(
            (
                (name, parameter)
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            ),
            gradients,
        )
        if gradient is not None
    }
    return float(loss.detach()), snapshot


def _gradient_cosine(left, right):
    common = sorted(set(left) & set(right))
    if not common:
        return None
    numerator = sum(
        float((left[name].double() * right[name].double()).sum())
        for name in common
    )
    left_norm = sum(
        float(left[name].double().square().sum()) for name in common
    ) ** 0.5
    right_norm = sum(
        float(right[name].double().square().sum()) for name in common
    ) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(numerator / (left_norm * right_norm))


def _conflict_probe(policy, shard, teachers, *, device, seed):
    was_training = policy.training
    policy.eval()
    positive_loss, positive_gradient = _probe_objective(
        policy, shard, shard.Dplus, teacher=False,
        device=device, seed=seed,
    )
    negative_loss, negative_gradient = _probe_objective(
        policy, shard, shard.Dminus, teacher=False,
        device=device, seed=seed,
    )
    teacher_loss, teacher_gradient = _probe_objective(
        policy, shard, teachers, teacher=True,
        device=device, seed=seed,
    )
    policy.train(was_training)
    return {
        "ordinary_positive_loss": positive_loss,
        "ordinary_negative_loss": negative_loss,
        "teacher_loss": teacher_loss,
        "cos_teacher_positive": _gradient_cosine(
            teacher_gradient, positive_gradient,
        ),
        "cos_teacher_negative": _gradient_cosine(
            teacher_gradient, negative_gradient,
        ),
        "probe_seed": int(seed),
        "ordinary_positive_records": min(len(shard.Dplus), 128),
        "ordinary_negative_records": min(len(shard.Dminus), 128),
        "teacher_records": min(len(teachers), 128),
    }


def run(args):
    checkpoint = os.path.abspath(args.checkpoint)
    round_shard_path = os.path.abspath(args.round_shard)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError(output_dir)
    checkpoint_sha = OS.sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA mismatch: "
            f"expected {args.expected_checkpoint_sha256}, got {checkpoint_sha}"
        )
    if args.expected_checkpoint_sha256 == T.R1_CHECKPOINT_SHA256:
        parent_contract = "immutable accepted r1"
    else:
        parent_contract = "caller-pinned post-ordinary checkpoint"
    shard = OS.ExecutedRoundShard.load(round_shard_path)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=args.device)
    BS.configure_expansion_trainability(policy)
    encoder_sha_before = BS.module_sha256(policy.enc_grid)
    hard, population_stats = _hard_windows(shard)
    environment = SS.scene_profile(args.scene_profile)
    os.makedirs(output_dir)
    with ProcessPoolExecutor(max_workers=args.verifier_workers) as executor:
        records, harvest = T.harvest_round(
            policy,
            shard,
            hard,
            device=args.device,
            environment=environment,
            max_contexts=args.max_contexts,
            executor=executor if args.audit_socp else None,
            audit_socp=args.audit_socp,
        )
    buffer_path = os.path.join(output_dir, "D_MPC.pt")
    T.save_buffer(
        buffer_path,
        round_shard_path,
        shard,
        records,
        harvest,
    )
    probe_before = _conflict_probe(
        policy,
        shard,
        records,
        device=args.device,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters()
         if parameter.requires_grad],
        lr=args.teacher_lr,
    )
    update = T.distill_block(
        policy,
        optimizer,
        shard,
        records,
        epochs=args.teacher_epochs,
        batch=args.batch,
        seed=args.seed,
    )
    if BS.module_sha256(policy.enc_grid) != encoder_sha_before:
        raise RuntimeError("visual encoder changed during teacher block")
    probe_after = _conflict_probe(
        policy,
        shard,
        records,
        device=args.device,
        seed=args.seed,
    )
    output_checkpoint = os.path.join(output_dir, "post_teacher.pt")
    BX._save_checkpoint(
        policy,
        output_checkpoint,
        {
            "phase": "post_ordinary_then_unverified_privileged_mpc_teacher",
            "source_checkpoint": checkpoint,
            "source_checkpoint_sha256": checkpoint_sha,
            "round_shard": round_shard_path,
            "round_shard_sha256": OS.sha256_file(round_shard_path),
            "teacher_buffer": buffer_path,
            "teacher_lr": float(args.teacher_lr),
            "teacher_epochs": int(args.teacher_epochs),
            "teacher_seed": int(args.seed),
        },
    )
    report = {
        "status": STATUS,
        "parent_contract": parent_contract,
        "source_checkpoint": checkpoint,
        "source_checkpoint_sha256": checkpoint_sha,
        "round": int(shard.round_i),
        "round_shard": round_shard_path,
        "round_shard_sha256": OS.sha256_file(round_shard_path),
        "teacher_buffer": buffer_path,
        "teacher_buffer_sha256": OS.sha256_file(buffer_path),
        "post_teacher_checkpoint": output_checkpoint,
        "post_teacher_checkpoint_sha256": OS.sha256_file(output_checkpoint),
        "ordinary_replay_precedes_this_command": True,
        "teacher_is_not_a_safety_label": True,
        "teacher_used_at_evaluation": False,
        "population_stats": population_stats,
        "harvest": harvest,
        "update": update,
        "conflict_probe_before": probe_before,
        "conflict_probe_after": probe_after,
    }
    _write_json(os.path.join(output_dir, "COMPLETE.json"), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=T.R1_CHECKPOINT_SHA256,
    )
    parser.add_argument("--round-shard", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-profile", default=OE.SCENE_PROFILE)
    parser.add_argument("--teacher-lr", type=float, required=True)
    parser.add_argument("--teacher-epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-contexts", type=int, default=400)
    parser.add_argument("--audit-socp", action="store_true")
    parser.add_argument("--verifier-workers", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not 0.0 < float(args.teacher_lr) <= 1.0e-3:
        parser.error("--teacher-lr must be in (0,1e-3]")
    if not 0 <= int(args.teacher_epochs) <= 32:
        parser.error("--teacher-epochs must be in [0,32]")
    run(args)


if __name__ == "__main__":
    main()
