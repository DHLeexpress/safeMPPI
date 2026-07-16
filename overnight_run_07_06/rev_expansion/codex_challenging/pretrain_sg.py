#!/usr/bin/env python3
"""Fine-tune the original low5+H_P policy after removing raw endpoints.

This loader consumes all seven Stage 2 ``w8sg`` datasets.  Validation is
grouped by pair index, so every window of a held-out episode pair stays out of
the training set at every gamma.  The complete tensor pool is placed on GPU
for fast, deterministic training without per-batch host transfers.  By
default, compatible weights are migrated from the rejected 41-D checkpoint by
deleting only its four raw-endpoint columns from the first trunk layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import grid_hp_expt as HP
from viz_style import GAMMAS


HERE = Path(__file__).resolve().parent
STAGE = HERE / "stage_results" / "03_pretrain"
DEFAULT_DATA = HERE / "stage_results" / "02_demos" / "data"
if Path(HP.__file__).resolve().parent != HERE:
    raise ImportError(f"expected local grid_hp_expt.py, imported {HP.__file__}")


def gamma_tag(gamma: float) -> str:
    return str(float(gamma))


def jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def load_pool(data_dir: Path) -> dict[str, torch.Tensor]:
    pieces: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("grid", "low5", "hist", "U", "pair", "gamma_id")
    }
    expected_hw = (32, 32)
    for gamma_id, gamma in enumerate(GAMMAS):
        path = data_dir / f"w8sg_windows_g{gamma_tag(gamma)}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        count = int(payload["grid"].shape[0])
        if tuple(payload["grid"].shape[1:]) != (3, *expected_hw):
            raise ValueError(f"{path}: unexpected grid shape {tuple(payload['grid'].shape)}")
        for key in ("low5", "hist", "U", "window_starts", "window_goals", "window_pair_indices"):
            if len(payload[key]) != count:
                raise ValueError(f"{path}: {key} length {len(payload[key])} != {count}")
        if not torch.allclose(
            payload["low5"][:, 4],
            torch.full_like(payload["low5"][:, 4], float(gamma)),
        ):
            raise ValueError(f"{path}: gamma channel mismatch")
        pieces["grid"].append(payload["grid"].float())
        pieces["low5"].append(payload["low5"].float())
        pieces["hist"].append(payload["hist"].float())
        pieces["U"].append(payload["U"].float())
        pieces["pair"].append(payload["window_pair_indices"].long())
        pieces["gamma_id"].append(torch.full((count,), gamma_id, dtype=torch.long))
        print(f"[load] gamma={gamma:g} windows={count:,} <- {path}", flush=True)
        del payload
    pool = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    count = len(pool["grid"])
    if any(len(value) != count for value in pool.values()):
        raise RuntimeError("concatenated pool lengths differ")
    return pool


def split_pairs(pair_indices: torch.Tensor, n_val_pairs: int, seed: int):
    unique = np.unique(pair_indices.numpy())
    if not (0 < n_val_pairs < len(unique)):
        raise ValueError(f"val pairs must be between 1 and {len(unique)-1}")
    rng = np.random.default_rng(seed)
    val_pairs = np.sort(rng.choice(unique, size=n_val_pairs, replace=False)).astype(np.int64)
    val_mask = torch.from_numpy(np.isin(pair_indices.numpy(), val_pairs))
    train_index = torch.nonzero(~val_mask, as_tuple=False).flatten()
    val_index = torch.nonzero(val_mask, as_tuple=False).flatten()
    return train_index, val_index, val_pairs


def migrate_raw_endpoint_checkpoint(policy: HP.GridHPFlowPolicy, path: Path) -> dict[str, Any]:
    """Load every compatible weight and delete only the four rejected columns.

    Flow-trunk input layout before migration:
      U[20] + low5[5] + raw endpoints[4] + E(H_P)[32] + time[32]
    After migration:
      U[20] + low5[5] + E(H_P)[32] + time[32]
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not config.get("raw_start_goal", False) or config.get("ctx_dim") != 41:
        raise ValueError(f"{path} is not the expected rejected 41-D raw-endpoint checkpoint")
    old_state = checkpoint["state_dict"]
    new_state = policy.state_dict()
    first_key = "trunk.0.weight"
    old_weight = old_state[first_key]
    new_weight = new_state[first_key]
    if old_weight.shape[0] != new_weight.shape[0] or old_weight.shape[1] != new_weight.shape[1] + 4:
        raise ValueError(f"cannot migrate first trunk layer {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}")
    endpoint_start = policy.d + policy.low_raw_dim
    endpoint_stop = endpoint_start + 4
    keep = torch.cat(
        (
            torch.arange(endpoint_start),
            torch.arange(endpoint_stop, old_weight.shape[1]),
        )
    )
    migrated = {}
    for key, target in new_state.items():
        source = old_state[key]
        migrated[key] = source[:, keep].clone() if key == first_key else source.clone()
        if migrated[key].shape != target.shape:
            raise ValueError(f"migrated {key} shape {tuple(migrated[key].shape)} != {tuple(target.shape)}")
    policy.load_state_dict(migrated, strict=True)
    removed = old_weight[:, endpoint_start:endpoint_stop]
    return {
        "source": str(path.resolve()),
        "source_schema": config.get("schema_version"),
        "source_best_epoch": checkpoint.get("best_epoch"),
        "source_best_val": checkpoint.get("best_val"),
        "old_first_layer_shape": list(old_weight.shape),
        "new_first_layer_shape": list(new_weight.shape),
        "removed_column_range": [endpoint_start, endpoint_stop],
        "removed_weight_l2": float(removed.norm()),
        "all_other_tensors_copied_exactly": True,
    }


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_loss(
    policy: HP.GridHPFlowPolicy,
    pool: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch: int,
    *,
    seed: int,
    amp: bool,
    low5_override: torch.Tensor | None = None,
) -> float:
    policy.eval()
    device = pool["grid"].device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    total = 0.0
    samples = 0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        for offset in range(0, len(indices), batch):
            index = indices[offset:offset + batch]
            low5 = pool["low5"][index] if low5_override is None else low5_override[offset:offset + len(index)]
            with autocast_context(device, amp):
                context = policy.ctx_from(
                    pool["grid"][index],
                    low5,
                    pool["hist"][index],
                )
                loss = policy.cfm_loss(pool["U"][index], context)
            total += float(loss) * len(index)
            samples += len(index)
    return total / max(samples, 1)


@torch.no_grad()
def representation_diagnostics(
    policy: HP.GridHPFlowPolicy,
    pool: dict[str, torch.Tensor],
    indices: torch.Tensor,
    max_samples: int = 4096,
) -> dict[str, Any]:
    policy.eval()
    chosen = indices[: min(max_samples, len(indices))]
    tokens = policy.hp_token(pool["grid"][chosen]).float()
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    std = centered.std(dim=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
    return {
        "samples": len(chosen),
        "token_norm_mean": float(tokens.norm(dim=1).mean()),
        "feature_std_mean": float(std.mean()),
        "feature_std_min": float(std.min()),
        "feature_std_max": float(std.max()),
        "active_dimensions_std_gt_1e-3": int((std > 1e-3).sum()),
        "covariance_effective_rank": float(effective_rank),
    }


def per_gamma_validation(
    policy: HP.GridHPFlowPolicy,
    pool: dict[str, torch.Tensor],
    val_index: torch.Tensor,
    batch: int,
    amp: bool,
) -> dict[str, float]:
    result = {}
    val_gamma = pool["gamma_id"][val_index]
    for gamma_id, gamma in enumerate(GAMMAS):
        indices = val_index[val_gamma == gamma_id]
        result[gamma_tag(gamma)] = evaluate_loss(
            policy,
            pool,
            indices,
            batch,
            seed=9000 + gamma_id,
            amp=amp,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=HERE / "pretrained_sg_walls8.pt")
    parser.add_argument(
        "--stage-copy",
        type=Path,
        default=STAGE / "data" / "pretrained_sg_walls8.pt",
    )
    parser.add_argument("--history", type=Path, default=STAGE / "logs" / "pretrain_history.csv")
    parser.add_argument("--summary", type=Path, default=STAGE / "logs" / "pretrain_summary.json")
    parser.add_argument("--split", type=Path, default=STAGE / "data" / "pair_split.npz")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=STAGE / "rejected_raw_endpoints" / "data" / "pretrained_raw_endpoints.pt",
        help="rejected 41-D checkpoint whose compatible weights will be migrated",
    )
    parser.add_argument("--from-scratch", action="store_true", help="skip checkpoint migration")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--val-batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--repr", type=int, default=20)
    parser.add_argument("--trunk-hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--enc-depth", type=int, default=2)
    parser.add_argument("--val-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--split-seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resident", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    for path in (args.out, args.stage_copy, args.history, args.summary, args.split):
        path.parent.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.data_dir)
    total_windows = len(pool["grid"])
    train_index, val_index, val_pairs = split_pairs(pool["pair"], args.val_pairs, args.split_seed)
    train_pairs = np.setdiff1d(np.unique(pool["pair"].numpy()), val_pairs)
    np.savez_compressed(
        args.split,
        train_pairs=train_pairs.astype(np.int32),
        val_pairs=val_pairs.astype(np.int32),
        split_seed=np.int64(args.split_seed),
    )
    print(
        f"[split] windows={total_windows:,} train={len(train_index):,} val={len(val_index):,} "
        f"pairs={len(train_pairs)}/{len(val_pairs)}",
        flush=True,
    )

    if args.resident:
        move_started = time.perf_counter()
        pool = {key: value.to(device) for key, value in pool.items()}
        train_index = train_index.to(device)
        val_index = val_index.to(device)
        print(f"[data] resident on {device} in {time.perf_counter()-move_started:.1f}s", flush=True)
    elif device.type == "cuda":
        raise ValueError("non-resident CUDA training is not implemented; use --resident")

    policy = HP.GridHPFlowPolicy(
        repr_dim=args.repr,
        grid_hw=(32, 32),
        trunk_hidden=tuple(args.trunk_hidden),
        enc_depth=args.enc_depth,
    )
    migration = None
    if not args.from_scratch:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(args.init_checkpoint)
        migration = migrate_raw_endpoint_checkpoint(policy, args.init_checkpoint)
        print(
            f"[migration] {migration['old_first_layer_shape']} -> {migration['new_first_layer_shape']} "
            f"removed columns {migration['removed_column_range']}; copied every other tensor",
            flush=True,
        )
    policy = policy.to(device)
    parameters = sum(parameter.numel() for parameter in policy.parameters())
    encoder_parameters = sum(parameter.numel() for parameter in policy.enc_grid.parameters())
    print(
        f"[model] params={parameters:,} encoder={encoder_parameters:,} ctx={policy.ctx_dim} "
        f"unfrozen_encoder={all(parameter.requires_grad for parameter in policy.enc_grid.parameters())}",
        flush=True,
    )

    initial_val_loss = evaluate_loss(
        policy,
        pool,
        val_index,
        args.val_batch,
        seed=81000,
        amp=args.amp,
    )
    print(f"[initial] grouped-pair val={initial_val_loss:.5f}", flush=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(epoch: int) -> float:
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    history_fields = (
        "epoch",
        "train_cfm",
        "val_cfm",
        "lr",
        "epoch_seconds",
        "encoder_grad_norm",
        "gpu_memory_allocated_mib",
    )
    with args.history.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()

    best_loss = initial_val_loss
    best_epoch = -1
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
    }
    epoch_rows = []
    train_generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        policy.train()
        permutation = train_index[torch.randperm(len(train_index), generator=train_generator, device=device)]
        train_total = 0.0
        train_samples = 0
        encoder_grad_total = 0.0
        batches = 0
        for offset in range(0, len(permutation), args.batch):
            index = permutation[offset:offset + args.batch]
            with autocast_context(device, args.amp):
                context = policy.ctx_from(
                    pool["grid"][index],
                    pool["low5"][index],
                    pool["hist"][index],
                )
                loss = policy.cfm_loss(pool["U"][index], context)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            encoder_grad = torch.sqrt(
                sum(
                    (parameter.grad.detach().float() ** 2).sum()
                    for parameter in policy.enc_grid.parameters()
                    if parameter.grad is not None
                )
            )
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            train_total += float(loss.detach()) * len(index)
            train_samples += len(index)
            encoder_grad_total += float(encoder_grad)
            batches += 1
        scheduler.step()

        train_loss = train_total / train_samples
        val_loss = evaluate_loss(
            policy,
            pool,
            val_index,
            args.val_batch,
            seed=81000,
            amp=args.amp,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in policy.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_cfm": train_loss,
            "val_cfm": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "encoder_grad_norm": encoder_grad_total / max(batches, 1),
            "gpu_memory_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            ),
        }
        epoch_rows.append(row)
        with args.history.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=history_fields).writerow(row)
        if epoch % 5 == 0 or epoch == args.epochs - 1 or epoch == best_epoch:
            print(
                f"[epoch {epoch:03d}] train={train_loss:.5f} val={val_loss:.5f} "
                f"lr={row['lr']:.2e} enc_grad={row['encoder_grad_norm']:.3e} "
                f"time={row['epoch_seconds']:.1f}s best={best_loss:.5f}@{best_epoch}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    policy.load_state_dict(best_state)
    policy.eval()

    representation = representation_diagnostics(policy, pool, val_index)
    per_gamma = per_gamma_validation(policy, pool, val_index, args.val_batch, args.amp)
    goal_sample = val_index[: min(4096, len(val_index))]
    low5_values = pool["low5"][goal_sample]
    shuffle_generator = torch.Generator(device=device).manual_seed(args.seed + 99)
    permutation = torch.randperm(len(low5_values), generator=shuffle_generator, device=device)
    goal_shuffled_low5 = low5_values.clone()
    goal_shuffled_low5[:, :2] = low5_values[permutation, :2]
    goal_loss_correct = evaluate_loss(
        policy,
        pool,
        goal_sample,
        args.val_batch,
        seed=82000,
        amp=args.amp,
    )
    goal_loss_shuffled = evaluate_loss(
        policy,
        pool,
        goal_sample,
        args.val_batch,
        seed=82000,
        amp=args.amp,
        low5_override=goal_shuffled_low5,
    )

    finished = datetime.now(timezone.utc)
    summary = {
        "status": "PASS",
        "started_at_utc": started_wall.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "amp_bfloat16": bool(args.amp and device.type == "cuda"),
        "dataset": {
            "directory": str(args.data_dir.resolve()),
            "total_windows": total_windows,
            "train_windows": len(train_index),
            "val_windows": len(val_index),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "val_pair_indices": val_pairs.tolist(),
            "gammas": list(GAMMAS),
        },
        "model": {
            "parameters": parameters,
            "encoder_parameters": encoder_parameters,
            "context_dimension": policy.ctx_dim,
            "context_layout": "low5(relative_goal, velocity, gamma) + E(H_P)[32]",
            "raw_start_goal": False,
            "encoder_trainable": all(parameter.requires_grad for parameter in policy.enc_grid.parameters()),
            "config": policy.config(),
        },
        "training": {
            "mode": "from_scratch" if args.from_scratch else "remove_raw_endpoint_columns_then_finetune",
            "migration": migration,
            "epochs": args.epochs,
            "batch": args.batch,
            "learning_rate": args.lr,
            "initial_val_cfm": initial_val_loss,
            "best_epoch": best_epoch,
            "best_val_cfm": best_loss,
            "final_train_cfm": epoch_rows[-1]["train_cfm"],
            "final_val_cfm": epoch_rows[-1]["val_cfm"],
            "per_gamma_val_cfm": per_gamma,
        },
        "diagnostics": {
            "visual_encoder": representation,
            "correct_relative_goal_val_cfm": goal_loss_correct,
            "shuffled_relative_goal_val_cfm": goal_loss_shuffled,
            "shuffled_over_correct_ratio": goal_loss_shuffled / goal_loss_correct,
        },
        "artifacts": {
            "checkpoint": str(args.out.resolve()),
            "stage_checkpoint": str(args.stage_copy.resolve()),
            "history": str(args.history.resolve()),
            "pair_split": str(args.split.resolve()),
        },
        "args": jsonable(vars(args)),
    }

    # Save the portable best checkpoint with CPU tensors, then mirror it into
    # the stage-scoped artifact tree required by the experiment protocol.
    policy = policy.cpu()
    HP.save_hp(
        policy,
        args.out,
        extra={
            "best_val": best_loss,
            "best_epoch": best_epoch,
            "pretrain_summary": summary,
        },
    )
    shutil.copy2(args.out, args.stage_copy)
    args.summary.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(
        f"[saved] {args.out} and {args.stage_copy} | best val={best_loss:.5f}@{best_epoch} "
        f"wall={summary['wall_seconds']:.1f}s",
        flush=True,
    )
    print(
        f"[diagnostic] encoder std={representation['feature_std_mean']:.4g} "
        f"active={representation['active_dimensions_std_gt_1e-3']}/32 "
        f"effective_rank={representation['covariance_effective_rank']:.2f} "
        f"relative_goal_shuffle_ratio={summary['diagnostics']['shuffled_over_correct_ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
