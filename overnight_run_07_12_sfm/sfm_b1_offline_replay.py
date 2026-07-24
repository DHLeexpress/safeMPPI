"""Minibatch replay for one offline executed-window round.

One exposure epoch visits every resolved executed positive and negative exactly
once.  Adam steps after each mixed minibatch, rather than after accumulating a
single full-dataset gradient.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch

import sfm_b1_r2_alpha_replay as R2
import sfm_b1_store as BS
import sfm_b1_offline_store as OS


def _set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _identity(record):
    shard, window = record
    return int(shard.round_i), int(window["window_id"])


def _proportional_interleave(positive, negative):
    """Spread both deterministic sign orders over the complete epoch."""
    positive = list(positive)
    negative = list(negative)
    p_index = n_index = 0
    merged = []
    while p_index < len(positive) or n_index < len(negative):
        p_progress = (
            (p_index + 1) / len(positive)
            if p_index < len(positive) else float("inf")
        )
        n_progress = (
            (n_index + 1) / len(negative)
            if n_index < len(negative) else float("inf")
        )
        if p_progress <= n_progress:
            merged.append(positive[p_index])
            p_index += 1
        else:
            merged.append(negative[n_index])
            n_index += 1
    return merged


def stratified_batches(shard, *, batch, seed):
    positives = BS.hierarchical_order(OS.positive_records(shard), int(seed))
    negatives = BS.hierarchical_order(OS.negative_records(shard), int(seed) + 1)
    merged = _proportional_interleave(positives, negatives)
    batches = [
        merged[start:start + int(batch)]
        for start in range(0, len(merged), int(batch))
    ]
    identities = [_identity(record) for values in batches for record in values]
    expected = [_identity(record) for record in positives + negatives]
    if len(identities) != len(set(identities)) or set(identities) != set(expected):
        raise RuntimeError("offline minibatch replay duplicated or omitted support")
    return batches, positives, negatives


def _weighted_loss(policy, records, mass, population, device):
    if not records:
        return None
    grid, low, hist, controls = BS._tensor_batch(records, device)
    context = policy.ctx_from(grid, low, hist)
    weights = torch.as_tensor([
        int(population) * mass[(id(shard), int(window["query_id"]))]
        for shard, window in records
    ], dtype=controls.dtype, device=controls.device)
    return policy.cfm_loss(controls, context, weights=weights)


def _finite_trainable(policy):
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in policy.parameters()
        if parameter.requires_grad
    )


def _one_batch(
    policy, optimizer, values, *, positive_mass, negative_mass,
    positive_population, negative_population, alpha, device, seed,
):
    positive = [record for record in values if int(record[1]["y"]) == 1]
    negative = [record for record in values if int(record[1]["y"]) == 0]
    _set_seed(seed)
    optimizer.zero_grad(set_to_none=True)
    positive_loss = _weighted_loss(
        policy, positive, positive_mass, positive_population, device,
    )
    if positive_loss is None:
        return dict(
            stepped=False, positive_loss=None, negative_loss=None, rho=0.0,
            positive_norm=0.0, negative_norm=0.0, gradient_cosine=None,
            positive=len(positive), negative=len(negative),
        )
    if not bool(torch.isfinite(positive_loss)):
        raise FloatingPointError("non-finite positive CFM loss")
    positive_loss.backward()
    positive_gradient = BS._gradient_snapshot(policy)
    positive_norm = BS._gradient_norm(positive_gradient)

    negative_loss = None
    negative_gradient = {}
    negative_norm = 0.0
    rho = 0.0
    cosine = None
    if float(alpha) > 0.0 and negative:
        optimizer.zero_grad(set_to_none=True)
        negative_loss = _weighted_loss(
            policy, negative, negative_mass, negative_population, device,
        )
        if not bool(torch.isfinite(negative_loss)):
            raise FloatingPointError("non-finite negative CFM loss")
        negative_loss.backward()
        negative_gradient = BS._gradient_snapshot(policy)
        negative_norm = BS._gradient_norm(negative_gradient)
        rho = float(alpha) * positive_norm / (negative_norm + 1.0e-12)
        cosine = R2._gradient_cosine(positive_gradient, negative_gradient)

    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        pos = positive_gradient.get(name)
        neg = negative_gradient.get(name)
        if pos is None and neg is None:
            parameter.grad = None
        elif pos is None:
            parameter.grad = -rho * neg
        elif neg is None:
            parameter.grad = pos
        else:
            parameter.grad = pos - rho * neg
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite gradient in {name}")
    optimizer.step()
    if not _finite_trainable(policy):
        raise FloatingPointError("optimizer produced non-finite parameters")
    return dict(
        stepped=True,
        positive_loss=float(positive_loss.detach()),
        negative_loss=(
            None if negative_loss is None else float(negative_loss.detach())
        ),
        rho=float(rho),
        positive_norm=float(positive_norm),
        negative_norm=float(negative_norm),
        gradient_cosine=cosine,
        positive=len(positive),
        negative=len(negative),
    )


def _summary(values):
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return dict(
        mean=float(np.mean(finite)),
        first=finite[0],
        last=finite[-1],
        minimum=min(finite),
        maximum=max(finite),
    )


def replay(
    policy, optimizer, shard, *, alpha, exposure_epochs, batch, device, seed,
):
    if float(alpha) not in (0.0, 0.01, 0.1):
        raise ValueError("alpha must be one of {0,0.01,0.1}")
    if int(exposure_epochs) not in (1, 10, 100):
        raise ValueError("exposure_epochs must be one of {1,10,100}")
    policy.train()
    positives = OS.positive_records(shard)
    negatives = OS.negative_records(shard)
    positive_mass, positive_mass_accounting = BS.hierarchy_mass(positives)
    negative_mass, negative_mass_accounting = BS.hierarchy_mass(negatives)
    probe_seed = int(seed) + 9_000_001
    fixed_probe_before = dict(
        positive=R2._fixed_probe_loss(
            policy, positives, batch=batch, device=device, seed=probe_seed,
        ),
        negative=R2._fixed_probe_loss(
            policy, negatives, batch=batch, device=device, seed=probe_seed + 1,
        ),
    )
    module_before = R2._module_snapshot(policy)
    encoder_before = BS.module_sha256(policy.enc_grid)
    epoch_rows = []
    total_steps = 0
    for epoch_i in range(int(exposure_epochs)):
        epoch_seed = int(seed) + epoch_i * 100_003
        batches, positive_order, negative_order = stratified_batches(
            shard, batch=batch, seed=epoch_seed,
        )
        batch_rows = []
        for batch_i, values in enumerate(batches):
            row = _one_batch(
                policy, optimizer, values,
                positive_mass=positive_mass,
                negative_mass=negative_mass,
                positive_population=len(positives),
                negative_population=len(negatives),
                alpha=alpha,
                device=device,
                seed=epoch_seed + batch_i,
            )
            batch_rows.append(row)
            total_steps += int(row["stepped"])
        positive_visits = sum(row["positive"] for row in batch_rows)
        negative_visits = sum(row["negative"] for row in batch_rows)
        if positive_visits != len(positive_order):
            raise RuntimeError("positive exposure count mismatch")
        if negative_visits != len(negative_order):
            raise RuntimeError("negative exposure count mismatch")
        epoch_rows.append(dict(
            epoch=epoch_i + 1,
            seed=epoch_seed,
            batches=len(batches),
            optimizer_steps=sum(int(row["stepped"]) for row in batch_rows),
            positive_visits=positive_visits,
            negative_visits=negative_visits,
            positive_loss=_summary(row["positive_loss"] for row in batch_rows),
            negative_loss=_summary(row["negative_loss"] for row in batch_rows),
            rho=_summary(row["rho"] for row in batch_rows),
            positive_norm=_summary(row["positive_norm"] for row in batch_rows),
            negative_norm=_summary(row["negative_norm"] for row in batch_rows),
            gradient_cosine=_summary(
                row["gradient_cosine"] for row in batch_rows
            ),
        ))

    encoder_after = BS.module_sha256(policy.enc_grid)
    if encoder_after != encoder_before:
        raise RuntimeError("visual encoder changed during offline replay")
    fixed_probe_after = dict(
        positive=R2._fixed_probe_loss(
            policy, positives, batch=batch, device=device, seed=probe_seed,
        ),
        negative=R2._fixed_probe_loss(
            policy, negatives, batch=batch, device=device, seed=probe_seed + 1,
        ),
    )
    expected_batches = (
        math.ceil((len(positives) + len(negatives)) / int(batch))
        if positives or negatives else 0
    )
    if any(row["batches"] != expected_batches for row in epoch_rows):
        raise RuntimeError("unexpected minibatch count")
    expected_steps = expected_batches * int(exposure_epochs) if positives else 0
    if total_steps != expected_steps:
        raise RuntimeError(
            f"expected {expected_steps} Adam steps, observed {total_steps}"
        )
    policy.eval()
    return dict(
        alpha=float(alpha),
        exposure_epochs=int(exposure_epochs),
        positive_eligible=len(positives),
        negative_eligible=len(negatives),
        total_eligible=len(positives) + len(negatives),
        batches_per_epoch=expected_batches,
        optimizer_steps=total_steps,
        positive_total_visits=len(positives) * int(exposure_epochs),
        negative_total_visits=len(negatives) * int(exposure_epochs),
        negative_used_for_training=bool(float(alpha) > 0.0 and negatives),
        alpha_zero_semantics=(
            "D- remains stored and occupies its deterministic mixed-minibatch "
            "slots, but contributes exactly zero gradient when alpha=0"
        ),
        fixed_probe=dict(before=fixed_probe_before, after=fixed_probe_after),
        module_relative_parameter_drift=R2._module_relative_drift(
            module_before, R2._module_snapshot(policy),
        ),
        visual_encoder_sha_before=encoder_before,
        visual_encoder_sha_after=encoder_after,
        positive_mass=R2._compact_mass(positive_mass_accounting),
        negative_mass=R2._compact_mass(negative_mass_accounting),
        exact_once_per_exposure_epoch=True,
        epochs=epoch_rows,
    )
