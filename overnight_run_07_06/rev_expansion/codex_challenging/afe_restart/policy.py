"""Flow-policy sampling and frozen-feature adapters.

Sampling accepts an explicit :class:`torch.Generator`; unlike the legacy
``sample_window`` helper it never mutates or relies on global RNG state.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .schemas import QueryContext


def _as_batch_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    # QueryContext arrays are intentionally read-only.  Copy before exposing
    # their storage to torch so no tensor kernel can mutate ledger identity.
    value = torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=device)
    return value.unsqueeze(0)


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def context_tensors(context: QueryContext, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _as_batch_tensor(context.grid, device),
        _as_batch_tensor(context.low5, device),
        _as_batch_tensor(context.hist, device),
    )


@torch.inference_mode()
def sample_plans(
    model: torch.nn.Module,
    context: QueryContext,
    count: int,
    *,
    temperature: float,
    nfe: int,
    generator: torch.Generator,
) -> np.ndarray:
    """Draw complete ``[count,10,2]`` plans using seeded Euler integration."""
    if count <= 0:
        raise ValueError("count must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    device = next(model.parameters()).device
    grid, low5, hist = context_tensors(context, device)
    encoded = model.ctx_from(grid, low5, hist)
    encoded = model._expand_ctx(encoded[0], count)
    x = temperature * torch.randn(
        count, int(model.d), generator=generator, device=device, dtype=encoded.dtype,
    )
    for step in range(nfe):
        tau = torch.full((count,), step / nfe, device=device, dtype=encoded.dtype)
        x = x + model(x, tau, encoded) / nfe
    plans = (x.reshape(count, int(model.T), 2) * float(model.u_max)).clamp(
        -float(model.u_max), float(model.u_max),
    )
    return plans.detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass
class FrozenFeatureModel:
    """Immutable hash-checked copy of the pretrained representation."""

    model: torch.nn.Module
    s: float = 0.9
    expected_dim: int = 32

    @classmethod
    def from_pretrained(
        cls, model: torch.nn.Module, *, s: float = 0.9, expected_dim: int = 32,
    ) -> "FrozenFeatureModel":
        frozen = copy.deepcopy(model).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        instance = cls(frozen, float(s), int(expected_dim))
        instance._initial_hash = model_state_hash(frozen)
        return instance

    def __post_init__(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._initial_hash = getattr(self, "_initial_hash", model_state_hash(self.model))

    @property
    def state_hash(self) -> str:
        current = model_state_hash(self.model)
        if current != self._initial_hash:
            raise RuntimeError("frozen feature model changed during expansion")
        return current

    @torch.inference_mode()
    def encode(self, context: QueryContext, plans: np.ndarray | torch.Tensor) -> np.ndarray:
        self.state_hash
        device = next(self.model.parameters()).device
        grid, low5, hist = context_tensors(context, device)
        controls = torch.as_tensor(plans, dtype=torch.float32, device=device)
        if controls.ndim == 2:
            controls = controls.unsqueeze(0)
        features = self.model.phi_s_at(controls, grid, low5, hist, s=self.s)
        if features.ndim != 2 or features.shape[1] != self.expected_dim:
            raise RuntimeError(
                f"expected frozen feature shape [B,{self.expected_dim}], got {tuple(features.shape)}"
            )
        norms = torch.linalg.vector_norm(features.double(), dim=1, keepdim=True)
        if bool((norms <= 1e-12).any()):
            raise RuntimeError("cannot normalize a zero frozen feature")
        normalized = features.double() / norms
        return normalized.cpu().numpy()


def batch_context_arrays(records: Sequence[object], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Stack ledger contexts for the proximal CFM loss adapter."""
    grids = torch.as_tensor(
        np.stack([np.asarray(record.context.grid) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    low5 = torch.as_tensor(
        np.stack([np.asarray(record.context.low5) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    hist = torch.as_tensor(
        np.stack([np.asarray(record.context.hist) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    plans = torch.as_tensor(
        np.stack([np.asarray(record.plan) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    return grids, low5, hist, plans


def ledger_cfm_loss(
    model: torch.nn.Module,
    records: Sequence[object],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """CFM loss with explicit RNG, suitable for the proximal solver."""
    device = next(model.parameters()).device
    grid, low5, hist, plans = batch_context_arrays(records, device)
    context = model.ctx_from(grid, low5, hist)
    batch = plans.shape[0]
    x1 = (plans / float(model.u_max)).reshape(batch, int(model.d))
    x0 = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
    tau = torch.rand(batch, generator=generator, device=device, dtype=x1.dtype).clamp(1e-4, 1.0)
    x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
    target = x1 - x0
    prediction = model(x_tau, tau, model._expand_ctx(context, batch))
    return ((prediction - target) ** 2).mean()
