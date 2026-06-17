from __future__ import annotations

import torch


def barrier_clearance(pos: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    if obstacles.numel() == 0:
        return torch.full(pos.shape[:-1], float("inf"), device=pos.device, dtype=pos.dtype)
    centers = obstacles[..., :2]
    radii = obstacles[..., 2]
    d = torch.linalg.norm(pos.unsqueeze(-2) - centers, dim=-1) - radii
    return torch.min(d, dim=-1).values


def affine_barrier_h(x0: torch.Tensor, x: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    """
    Port of safeGPC DoubleIntegrator2D.hnew_torch/huniversal_proj_torch.

    It selects the nearest circle by current position and projects the current
    position onto the initial state's nearest-boundary normal.
    """
    if obstacles.numel() == 0:
        return torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
    obs = obstacles.to(device=x.device, dtype=x.dtype)
    if obs.ndim == 2:
        obs = obs.unsqueeze(0).expand(x.shape[0], -1, -1)
    centers = obs[:, :, :2]
    radii = obs[:, :, 2]
    pos0 = x0[:, :2]
    pos = x[:, :2]
    d_current = torch.linalg.norm(pos[:, None, :] - centers, dim=2) - radii
    idx = torch.argmin(d_current, dim=1)
    batch = torch.arange(x.shape[0], device=x.device)
    c_sel = centers[batch, idx]
    r_sel = radii[batch, idx]
    diff0 = pos0 - c_sel
    dist0 = torch.linalg.norm(diff0, dim=1).clamp_min(1e-12)
    nearest0 = c_sel + diff0 / dist0.unsqueeze(1) * r_sel.unsqueeze(1)
    d0b = (dist0 - r_sel).clamp_min(1e-12)
    normal = nearest0 - pos0
    normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
    raw_proj = torch.sum((nearest0 - pos) * normal, dim=1)
    return raw_proj / d0b
