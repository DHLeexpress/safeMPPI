"""Corrected ordinary-B1 replay and corrected privileged-teacher objective.

Fixes the two implementation errors named in the study directive:

A. Ordinary B1 uses the ORIGINAL online semantics via ``BS.signed_update``:
   one full-dataset accumulated gradient and exactly one Adam step per replay
   epoch, over a fail-closed W=2 window holding BOTH declared recent shards.

B. The teacher optimizes exactly  L_T = sum_i m_i L_CFM(i)  for normalized
   hierarchy masses (sum m_i = 1): each chunk of size b passes
   ``weights = b * m_i`` to the production ``FlowPolicy.cfm_loss`` (which
   multiplies then means) and the returned chunk losses are SUMMED — never
   multiplied by a chunk mass again.  One accumulated Adam step per epoch;
   deterministic per-chunk CFM noise via ``torch.manual_seed``.

The module also provides the label-based metrics reader (the previous
``records[0]`` reader silently returned r0 as the r1 baseline) and the
fixed-context output-RMSE probe.
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_b1_eval as BE
import sfm_b1_store as BS


class CorrectedRecent:
    """Fail-closed W=2 window over exactly two distinct executed shards."""

    def __init__(self, shards):
        shards = list(shards)
        if len(shards) != 2:
            raise RuntimeError(
                f"W=2 requires BOTH declared recent shards; got {len(shards)} "
                "— aborting instead of silently using W=1"
            )
        rounds = [int(shard.round_i) for shard in shards]
        if len(set(rounds)) != 2:
            raise RuntimeError(f"W=2 shards must be distinct rounds: {rounds}")
        for shard in shards:
            if not shard.Dplus and not shard.Dminus:
                raise RuntimeError(
                    f"declared recent shard round {shard.round_i} is empty"
                )
        self._rounds = sorted(shards, key=lambda s: int(s.round_i))
        self.window = 2

    @property
    def rounds(self):
        return list(self._rounds)

    def positive_records(self):
        return [(s, row) for s in self._rounds for row in s.Dplus]

    def negative_records(self):
        return [(s, row) for s in self._rounds for row in s.Dminus]


def corrected_ordinary_epochs(policy, optimizer, recent, *, alpha, epochs,
                              batch, device, seed):
    """One ``BS.signed_update`` call (= one Adam step) per epoch."""
    results = []
    for epoch in range(int(epochs)):
        result = BS.signed_update(
            policy, optimizer, recent, alpha=float(alpha), batch=int(batch),
            device=device, seed=int(seed) + epoch,
        )
        if int(result["optimizer_steps"]) != 1:
            raise RuntimeError(
                "corrected ordinary epoch must take exactly one Adam step, "
                f"observed {result['optimizer_steps']}"
            )
        results.append({
            key: result[key] for key in (
                "path", "rho", "positive_norm", "negative_norm",
                "positive_loss", "negative_loss", "positive_eligible",
                "negative_eligible", "optimizer_steps",
            )
        })
    return dict(epochs=int(epochs), adam_steps=len(results), per_epoch=results)


def normalized_teacher_mass(records):
    """Hierarchy masses over (holder, row) records, verified to sum to 1."""
    mass, accounting = BS.hierarchy_mass(records)
    total = sum(mass.values())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise RuntimeError(f"teacher hierarchy mass sums to {total}, not 1")
    return mass, accounting


def _chunks(sequence, size):
    for start in range(0, len(sequence), int(size)):
        yield start, sequence[start:start + int(size)]


def teacher_epoch_loss(policy, records, mass, *, batch, device, seed,
                       backward):
    """SUM of chunk losses with weights = b * m_i (exactly sum_i m_i L_i)."""
    total = 0.0
    for start, values in _chunks(records, batch):
        grid, low, hist, controls = BS._tensor_batch(values, device)
        context = policy.ctx_from(grid, low, hist)
        weights = torch.as_tensor([
            len(values) * mass[(id(holder), int(row["query_id"]))]
            for holder, row in values
        ], dtype=controls.dtype, device=device)
        torch.manual_seed(int(seed) + start)
        chunk_loss = policy.cfm_loss(controls, context, weights=weights)
        if not bool(torch.isfinite(chunk_loss)):
            raise FloatingPointError("non-finite teacher chunk loss")
        if backward:
            chunk_loss.backward()
        total += float(chunk_loss.detach())
    return total


def corrected_teacher_epochs(policy, optimizer, records, *, epochs, batch,
                             device, seed):
    """One accumulated Adam step per epoch on the exact L_T objective."""
    mass, accounting = normalized_teacher_mass(records)
    policy.train()
    losses = []
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = teacher_epoch_loss(
            policy, records, mass, batch=batch, device=device,
            seed=int(seed) + epoch * 1_000_003, backward=True,
        )
        optimizer.step()
        losses.append(loss)
    policy.eval()
    return dict(
        epochs=int(epochs), adam_steps=int(epochs), losses=losses,
        records=len(records), mass_accounting_gamma=accounting["gamma"],
    )


def replayed_chunk_noise(controls_shape, device, seed):
    """Replay cfm_loss's internal draws (x0 then tau) for one chunk (CPU)."""
    b, d = controls_shape
    torch.manual_seed(int(seed))
    x0 = torch.randn(b, d, device=device)
    tau = torch.rand(b, device=device).clamp(1e-4, 1.0)
    return x0, tau


def direct_teacher_loss(policy, records, mass, *, batch, device, seed):
    """Independent  sum_i m_i L_i  using the replayed per-chunk noise."""
    total = None
    for start, values in _chunks(records, batch):
        grid, low, hist, controls = BS._tensor_batch(values, device)
        context = policy.ctx_from(grid, low, hist)
        b = len(values)
        x1 = (controls / policy.u_max).reshape(b, policy.d)
        x0, tau = replayed_chunk_noise((b, policy.d), device,
                                       int(seed) + start)
        x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
        prediction = policy.forward(
            x_tau, tau, policy._expand_ctx(context, b),
        )
        per = ((prediction - (x1 - x0)) ** 2).mean(dim=1)
        masses = torch.as_tensor([
            mass[(id(holder), int(row["query_id"]))] for holder, row in values
        ], dtype=per.dtype, device=device)
        contribution = (per * masses).sum()
        total = contribution if total is None else total + contribution
    return total


@torch.no_grad()
def fixed_context_rmse(policy_a, policy_b, probes, *, device, seed):
    """First-action and full-H10 RMSE between two policies on fixed probes."""
    first, full = [], []
    for index, probe in enumerate(probes):
        hp10 = torch.as_tensor(probe["hp10"], device=device)[None].float()
        low = torch.as_tensor(probe["low5"], device=device)[None].float()
        hist = torch.as_tensor(probe["hist"], device=device)[None].float()
        generator = np.random.default_rng(int(seed) + index)
        latents = torch.as_tensor(generator.standard_normal(
            (8, int(policy_a.d)), dtype=np.float32,
        ), device=device)
        windows = []
        for policy in (policy_a, policy_b):
            ctx = policy.ctx_from(hp10, low, hist)
            windows.append(BE.integrate_latents(
                policy, latents, ctx.repeat_interleave(8, dim=0), nfe=8,
            ).reshape(8, 10, 2))
        delta = (windows[0] - windows[1])
        first.append(float(delta[:, 0].square().mean().sqrt()))
        full.append(float(delta.square().mean().sqrt()))
    return dict(
        probes=len(probes),
        first_action_rmse=float(np.mean(first)),
        h10_window_rmse=float(np.mean(full)),
    )


def pooled_by_label(path, label):
    """Label-based reader (fixes the records[0]-as-r1 bug)."""
    import json

    with open(path) as stream:
        payload = json.load(stream)
    matches = [r for r in payload["records"] if r["label"] == str(label)]
    if len(matches) != 1:
        raise KeyError(
            f"{path}: expected exactly one record labeled {label!r}, "
            f"found {len(matches)}"
        )
    cell = matches[0]["cell"]["summary"]
    p = cell["pooled"]

    def _gamma(g):
        c = cell["per_gamma"][g]
        return dict(
            clearance=c["successful_clearance"]["mean"],
            time=c["successful_time_to_goal"]["mean"],
        )

    return dict(
        SR=float(p["SR"]), CR=float(p["CR"]), timeout=float(p["timeout"]),
        Validity=float(p["Validity"]["mean"]),
        clearance=p["successful_clearance"]["mean"],
        time=p["successful_time_to_goal"]["mean"],
        g01=_gamma("0.1"), g10=_gamma("1.0"),
    )
