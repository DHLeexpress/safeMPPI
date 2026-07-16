#!/usr/bin/env python3
"""Fresh endpoint-free pretraining on the balanced fixed-pair ID dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT.parents[1], ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import grid_hp_expt as HP  # noqa: E402
from viz_style import GAMMAS  # noqa: E402
from giant_obstacle_ood.stage2b_balanced_id_data import ALL_SIGNATURES  # noqa: E402


STAGE = HERE / "stage_results/03_pretrain"
DEFAULT_DATA = HERE / "stage_results/02b_balanced_id/data"
H = 10


def gamma_tag(gamma: float) -> str:
    return str(float(gamma))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def load_pool(data_dir: Path) -> tuple[dict[str, torch.Tensor], list[dict]]:
    keys = (
        "grid", "low5", "hist", "U", "window_trajectory_ids",
        "window_signature_ids", "window_seeds",
    )
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    pieces["gamma_id"] = []
    sources = []
    for gamma_id, gamma in enumerate(GAMMAS):
        path = data_dir / f"balanced_id_windows_g{gamma_tag(gamma)}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if payload.get("schema_version") != "giant_ood_id_balanced_v2_full_horizon":
            raise ValueError(f"{path}: not the final full-horizon Stage-2B schema")
        count = len(payload["grid"])
        if count != 1536 or int(payload["n_traj"]) != 24:
            raise ValueError(f"{path}: expected 1,536 windows from 24 paths")
        if bool(payload["padded_mask"].any()) or not bool(payload["physical_collision_free_mask"].all()):
            raise ValueError(f"{path}: padding or physically unsafe window reached Stage 3")
        if not torch.allclose(payload["low5"][:, 4], torch.full_like(payload["low5"][:, 4], gamma)):
            raise ValueError(f"{path}: gamma feature mismatch")
        for key in keys:
            pieces[key].append(payload[key].clone() if key.startswith("window_") else payload[key].float())
        pieces["gamma_id"].append(torch.full((count,), gamma_id, dtype=torch.long))
        sources.append({"gamma": gamma, "path": str(path.resolve()), "sha256": sha256(path), "windows": count})
        print(f"[load] gamma={gamma:g} windows={count:,} <- {path}", flush=True)
    pool = {key: torch.cat(values, dim=0) for key, values in pieces.items()}
    if len(pool["grid"]) != 10_752:
        raise RuntimeError(f"expected 10,752 windows, found {len(pool['grid'])}")
    return pool, sources


def stratified_split(pool: dict[str, torch.Tensor], split_seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Hold out one of four trajectories in every gamma/signature stratum."""
    trajectory = pool["window_trajectory_ids"]
    signature = pool["window_signature_ids"]
    gamma_id = pool["gamma_id"]
    val_trajectories: list[int] = []
    strata = []
    for gid, gamma in enumerate(GAMMAS):
        gamma_rows = gamma_id == gid
        for sid in sorted(int(value) for value in torch.unique(signature[gamma_rows])):
            rows = gamma_rows & (signature == sid)
            tids = sorted(int(value) for value in torch.unique(trajectory[rows]))
            if len(tids) != 4:
                raise RuntimeError(f"gamma={gamma:g}, signature={sid}: expected four trajectories, got {tids}")
            rng = np.random.default_rng(split_seed + gid * 10_003 + sid * 101)
            heldout = tids[int(rng.integers(0, len(tids)))]
            val_trajectories.append(heldout)
            strata.append({
                "gamma": float(gamma), "gamma_id": gid, "signature_id": sid,
                "all_trajectory_ids": tids, "validation_trajectory_id": heldout,
            })
    val_mask = torch.from_numpy(np.isin(trajectory.numpy(), np.asarray(val_trajectories, dtype=np.int64)))
    train_index = torch.where(~val_mask)[0]
    val_index = torch.where(val_mask)[0]
    if len(train_index) != 8064 or len(val_index) != 2688:
        raise RuntimeError(f"unexpected split sizes {len(train_index)}/{len(val_index)}")

    audit: dict[str, Any] = {"split_seed": split_seed, "strata": strata, "per_gamma": {}}
    for gid, gamma in enumerate(GAMMAS):
        entry = {}
        for name, indices, expected in (("train", train_index, 1152), ("validation", val_index, 384)):
            selected = indices[gamma_id[indices] == gid]
            words = signature[selected]
            counts = {int(sid): int((words == sid).sum()) for sid in torch.unique(words)}
            r_first = int(sum(count for sid, count in counts.items() if _signature_from_id(pool, sid).startswith("R")))
            u_first = int(sum(count for sid, count in counts.items() if _signature_from_id(pool, sid).startswith("U")))
            if len(selected) != expected or r_first != expected // 2 or u_first != expected // 2:
                raise RuntimeError(f"{name} balance failed at gamma={gamma:g}: {len(selected)}, {r_first}/{u_first}")
            if len(set(counts.values())) != 1:
                raise RuntimeError(f"{name} signature mass failed at gamma={gamma:g}: {counts}")
            entry[name] = {"windows": len(selected), "r_first": r_first, "u_first": u_first,
                           "signature_window_counts": counts}
        audit["per_gamma"][gamma_tag(gamma)] = entry
    return train_index, val_index, audit


def _signature_from_id(pool: dict[str, torch.Tensor], signature_id: int) -> str:
    # All C(8,4) words in the same lexicographic combinations order as Stage 2B.
    import itertools
    words = tuple(
        "".join("R" if index in right else "U" for index in range(8))
        for right in itertools.combinations(range(8), 4)
    )
    return words[signature_id]


def reflect_batch(grid: torch.Tensor, low5: torch.Tensor, hist: torch.Tensor,
                  controls: torch.Tensor | None = None):
    """Exact x<->y reflection for the axis-aligned polar representation."""
    n_theta = grid.shape[-2]
    if n_theta % 4:
        raise ValueError("polar theta resolution must be divisible by four")
    source = (n_theta // 4 - 1 - torch.arange(n_theta, device=grid.device)) % n_theta
    reflected_grid = grid.index_select(-2, source)
    reflected_low = low5.clone()
    reflected_low[:, 0:2] = low5[:, [1, 0]]
    reflected_low[:, 2:4] = low5[:, [3, 2]]
    reflected_hist = hist.index_select(-1, torch.tensor((1, 0), device=hist.device))
    reflected_controls = None if controls is None else controls.index_select(
        -1, torch.tensor((1, 0), device=controls.device)
    )
    return reflected_grid, reflected_low, reflected_hist, reflected_controls


def reflect_flat_controls(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(len(values), H, 2).index_select(
        -1, torch.tensor((1, 0), device=values.device)
    ).reshape(len(values), -1)


def symmetric_cfm_loss(policy: HP.GridHPFlowPolicy, grid: torch.Tensor, low5: torch.Tensor,
                       hist: torch.Tensor, controls: torch.Tensor, equivariance_weight: float,
                       ot_coupling: bool = False,
                       mode_labels: torch.Tensor | None = None,
                       mode_noise_coupling: bool = False):
    """Pair every real row with its exact scene reflection and paired flow noise."""
    rgrid, rlow, rhist, rcontrols = reflect_batch(grid, low5, hist, controls)
    context = policy.ctx_from(grid, low5, hist)
    reflected_context = policy.ctx_from(rgrid, rlow, rhist)
    batch = len(controls)
    x1 = (controls / policy.u_max).reshape(batch, policy.d)
    reflected_x1 = (rcontrols / policy.u_max).reshape(batch, policy.d)
    x0 = torch.randn_like(x1)
    if mode_noise_coupling:
        if mode_labels is None:
            raise ValueError("mode-noise coupling requires R/U labels")
        # Encode the reflection-antisymmetric route bit in otherwise iid
        # Gaussian noise. Swapping x/y of the first action flips both the bit
        # and the route under exact scene reflection. With balanced R/U mass,
        # the aggregate source remains N(0,I).
        wrong_side = (x0[:, 0] - x0[:, 1]) * mode_labels.to(x0) < 0
        first_pair = x0[wrong_side, :2].clone()
        x0[wrong_side, 0] = first_pair[:, 1]
        x0[wrong_side, 1] = first_pair[:, 0]
    if ot_coupling and batch > 1:
        # Minibatch 2-Wasserstein coupling shortens Gaussian-to-control paths.
        # Keep each target attached to its own context; only permute iid source
        # noises, which preserves N(0,I) while reducing unsafe mode averaging.
        cost = torch.cdist(x0.float(), x1.float()).square().detach().cpu().numpy()
        noise_rows, target_columns = linear_sum_assignment(cost)
        assigned = torch.empty_like(x0)
        assigned[torch.as_tensor(target_columns, device=x0.device)] = x0[
            torch.as_tensor(noise_rows, device=x0.device)
        ]
        x0 = assigned
    reflected_x0 = reflect_flat_controls(x0)
    tau = torch.rand(batch, device=x1.device).clamp(1e-4, 1.0)
    x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
    reflected_x_tau = (1 - tau)[:, None] * reflected_x0 + tau[:, None] * reflected_x1
    prediction = policy(x_tau, tau, context)
    reflected_prediction = policy(reflected_x_tau, tau, reflected_context)
    target = x1 - x0
    reflected_target = reflected_x1 - reflected_x0
    cfm = 0.5 * (((prediction - target) ** 2).mean() +
                 ((reflected_prediction - reflected_target) ** 2).mean())
    equivariance = ((reflected_prediction - reflect_flat_controls(prediction)) ** 2).mean()
    return cfm + equivariance_weight * equivariance, cfm.detach(), equivariance.detach()


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(policy, pool, indices, batch: int, seed: int, amp: bool,
             symmetry_augment: bool, equivariance_weight: float,
             ot_coupling: bool = False,
             mode_noise_coupling: bool = False) -> dict[str, float]:
    policy.eval()
    device = pool["grid"].device
    devices = [device.index or 0] if device.type == "cuda" else []
    totals = np.zeros(3, dtype=float)
    samples = 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        for offset in range(0, len(indices), batch):
            idx = indices[offset:offset + batch]
            with amp_context(device, amp):
                if symmetry_augment:
                    loss, cfm, equiv = symmetric_cfm_loss(
                        policy, pool["grid"][idx], pool["low5"][idx], pool["hist"][idx],
                        pool["U"][idx], equivariance_weight, ot_coupling,
                        pool["mode_label"][idx], mode_noise_coupling,
                    )
                else:
                    context = policy.ctx_from(pool["grid"][idx], pool["low5"][idx], pool["hist"][idx])
                    loss = policy.cfm_loss(pool["U"][idx], context)
                    cfm, equiv = loss.detach(), torch.zeros((), device=device)
            totals += np.asarray((float(loss), float(cfm), float(equiv))) * len(idx)
            samples += len(idx)
    return {"loss": totals[0] / samples, "cfm": totals[1] / samples, "equivariance": totals[2] / samples}


@torch.no_grad()
def representation_diagnostics(policy, pool, indices) -> dict:
    chosen = indices[: min(2688, len(indices))]
    token = policy.hp_token(pool["grid"][chosen]).float()
    centered = token - token.mean(0, keepdim=True)
    std = centered.std(0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probability = eig / eig.sum().clamp_min(1e-12)
    rank = torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
    return {
        "samples": len(chosen), "feature_std_mean": float(std.mean()),
        "feature_std_min": float(std.min()), "feature_std_max": float(std.max()),
        "active_dimensions_std_gt_1e-3": int((std > 1e-3).sum()),
        "covariance_effective_rank": float(rank), "token_norm_mean": float(token.norm(dim=1).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch", type=int, default=512, help="real rows per step; reflection is paired internally")
    parser.add_argument("--val-batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--repr", type=int, default=20)
    parser.add_argument("--trunk-hidden", type=int, nargs="+", default=(128, 64))
    parser.add_argument("--enc-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--split-seed", type=int, default=31711)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symmetry-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--equivariance-weight", type=float, default=0.0)
    parser.add_argument("--ot-coupling", action=argparse.BooleanOptionalAction, default=False,
                        help="minibatch Hungarian coupling between Gaussian noise and controls")
    parser.add_argument("--mode-noise-coupling", action=argparse.BooleanOptionalAction, default=False,
                        help="couple the antisymmetric Gaussian component to balanced R/U route mode")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="final refit on all approved rows after using the stratified split for recipe selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.stage / "runs" / args.tag
    checkpoint_path = run / "checkpoint_best.pt"
    history_path = run / "history.csv"
    summary_path = run / "summary.json"
    split_path = run / "split_audit.json"
    run.mkdir(parents=True, exist_ok=True)
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pool, sources = load_pool(args.data_dir)
    mode_lookup = torch.tensor(
        [1.0 if word.startswith("R") else -1.0 for word in ALL_SIGNATURES], dtype=torch.float32
    )
    pool["mode_label"] = mode_lookup[pool["window_signature_ids"].long()]
    train_index, val_index, split_audit = stratified_split(pool, args.split_seed)
    split_path.write_text(json.dumps(jsonable(split_audit), indent=2) + "\n")
    heldout_train_windows = len(train_index)
    if args.train_all:
        train_index = torch.arange(len(pool["grid"]), dtype=torch.long)
    print(
        f"[split] optimization={len(train_index):,}, monitoring={len(val_index):,}; "
        f"exact R/U in every gamma; train_all={args.train_all}", flush=True,
    )
    pool = {key: value.to(device) for key, value in pool.items()}
    train_index = train_index.to(device)
    val_index = val_index.to(device)

    policy = HP.GridHPFlowPolicy(
        repr_dim=args.repr, grid_hw=(32, 32), trunk_hidden=tuple(args.trunk_hidden), enc_depth=args.enc_depth,
    ).to(device)
    if policy.ctx_dim != 37 or policy.config()["raw_start_goal"]:
        raise RuntimeError("Stage 3 must use the original endpoint-free 37-D policy")
    parameters = sum(parameter.numel() for parameter in policy.parameters())
    encoder_parameters = sum(parameter.numel() for parameter in policy.enc_grid.parameters())
    print(
        f"[model] fresh parameters={parameters:,}; encoder={encoder_parameters:,}; ctx=37; "
        f"symmetry_augment={args.symmetry_augment}; equiv_w={args.equivariance_weight:g}; "
        f"ot_coupling={args.ot_coupling}; mode_noise={args.mode_noise_coupling}", flush=True,
    )

    initial = evaluate(policy, pool, val_index, args.val_batch, 81000, args.amp,
                       args.symmetry_augment, args.equivariance_weight, args.ot_coupling,
                       args.mode_noise_coupling)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(epoch: int) -> float:
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    fields = ("epoch", "train_loss", "train_cfm", "train_equivariance", "val_loss", "val_cfm",
              "val_equivariance", "lr", "encoder_grad_norm", "epoch_seconds", "gpu_memory_mib")
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

    best_loss = float(initial["loss"])
    best_epoch = -1
    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    rows = []
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        policy.train()
        permutation = train_index[torch.randperm(len(train_index), generator=generator, device=device)]
        totals = np.zeros(3, dtype=float)
        samples = 0
        encoder_grad_sum = 0.0
        batches = 0
        for offset in range(0, len(permutation), args.batch):
            idx = permutation[offset:offset + args.batch]
            with amp_context(device, args.amp):
                if args.symmetry_augment:
                    loss, cfm, equiv = symmetric_cfm_loss(
                        policy, pool["grid"][idx], pool["low5"][idx], pool["hist"][idx],
                        pool["U"][idx], args.equivariance_weight, args.ot_coupling,
                        pool["mode_label"][idx], args.mode_noise_coupling,
                    )
                else:
                    context = policy.ctx_from(pool["grid"][idx], pool["low5"][idx], pool["hist"][idx])
                    loss = policy.cfm_loss(pool["U"][idx], context)
                    cfm, equiv = loss.detach(), torch.zeros((), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            encoder_grad = torch.sqrt(sum(
                (parameter.grad.detach().float() ** 2).sum()
                for parameter in policy.enc_grid.parameters() if parameter.grad is not None
            ))
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            totals += np.asarray((float(loss.detach()), float(cfm), float(equiv))) * len(idx)
            samples += len(idx)
            encoder_grad_sum += float(encoder_grad)
            batches += 1
        scheduler.step()
        validation = evaluate(policy, pool, val_index, args.val_batch, 81000, args.amp,
                              args.symmetry_augment, args.equivariance_weight, args.ot_coupling,
                              args.mode_noise_coupling)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        row = {
            "epoch": epoch, "train_loss": totals[0] / samples, "train_cfm": totals[1] / samples,
            "train_equivariance": totals[2] / samples, "val_loss": validation["loss"],
            "val_cfm": validation["cfm"], "val_equivariance": validation["equivariance"],
            "lr": optimizer.param_groups[0]["lr"], "encoder_grad_norm": encoder_grad_sum / batches,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
        }
        rows.append(row)
        with history_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        if epoch % 10 == 0 or epoch == args.epochs - 1 or epoch == best_epoch:
            print(
                f"[epoch {epoch:03d}] train={row['train_cfm']:.5f} val={row['val_cfm']:.5f} "
                f"eq={row['val_equivariance']:.5f} lr={row['lr']:.2e} "
                f"enc_grad={row['encoder_grad_norm']:.2e} best={best_loss:.5f}@{best_epoch}", flush=True,
            )

    policy.load_state_dict(best_state)
    policy.eval()
    per_gamma = {}
    for gid, gamma in enumerate(GAMMAS):
        selected = val_index[pool["gamma_id"][val_index] == gid]
        per_gamma[gamma_tag(gamma)] = evaluate(
            policy, pool, selected, args.val_batch, 82000 + gid, args.amp,
            args.symmetry_augment, args.equivariance_weight, args.ot_coupling,
            args.mode_noise_coupling,
        )
    representation = representation_diagnostics(policy, pool, val_index)
    summary = {
        "status": "TRAINED_AWAITING_ROLLOUT_GATE",
        "started_at_utc": started_wall.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset": {"sources": sources, "total_windows": len(pool["grid"]),
                    "optimization_windows": len(train_index), "monitoring_windows": len(val_index),
                    "recipe_selection_train_windows": heldout_train_windows,
                    "monitoring_rows_seen_by_optimizer": bool(args.train_all), "split": split_audit},
        "model": {"fresh_from_scratch": True, "raw_start_goal": False, "context_dimension": 37,
                  "parameters": parameters, "encoder_parameters": encoder_parameters,
                  "encoder_trainable": True, "config": policy.config()},
        "training": {"epochs": args.epochs, "batch_real_rows": args.batch,
                     "symmetry_augment": args.symmetry_augment,
                     "final_refit_on_all_approved_rows": bool(args.train_all),
                     "reflection_semantics": "exact x<->y symmetry of ID scene; paired noise" if args.symmetry_augment else None,
                     "equivariance_weight": args.equivariance_weight,
                     "ot_coupling": args.ot_coupling,
                     "mode_noise_coupling": args.mode_noise_coupling,
                     "learning_rate": args.lr,
                     "initial_validation": initial, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                     "best_per_gamma_validation": per_gamma,
                     "final_epoch": rows[-1]},
        "diagnostics": {"visual_encoder": representation},
        "artifacts": {"checkpoint": str(checkpoint_path.resolve()), "history": str(history_path.resolve()),
                      "summary": str(summary_path.resolve()), "split_audit": str(split_path.resolve())},
        "args": jsonable(vars(args)),
    }
    policy = policy.cpu()
    HP.save_hp(policy, checkpoint_path, extra={
        "best_val": best_loss, "best_epoch": best_epoch, "stage3_pretrain_summary": summary,
    })
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(
        f"[saved] {checkpoint_path} best={best_loss:.5f}@{best_epoch} "
        f"wall={summary['wall_seconds']:.1f}s encoder_rank={representation['covariance_effective_rank']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
