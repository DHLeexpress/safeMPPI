from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .barrier import affine_barrier_h, barrier_clearance


@dataclass
class SafeMPPIConfig:
    horizon: int = 20
    dt: float = 0.1
    num_samples: int = 128
    gamma: float = 0.5
    temperature: float = 1.0
    noise_sigma: float = 0.6
    u_min: Tuple[float, float] = (-2.0, -2.0)
    u_max: Tuple[float, float] = (2.0, 2.0)
    check_first_control_only: bool = False
    dynamics_type: str = "doubleintegrator"


class SafeMPPIAdapter:
    """
    Minimal PyTorch port of local safeGPC MPPI sample rejection.

    Source parity:
    safeGPC `utils/alg_base.py` rejects samples when
    `h_new < (1 - gamma) * h_old`; `tasks/doubleIntegrator.py` supplies
    `huniversal_proj_torch`, which reduces to the affine circle projection
    implemented in `affine_barrier_h`.
    """

    def __init__(self, **kwargs):
        self.config = SafeMPPIConfig(**kwargs)
        self.u_min = torch.tensor(self.config.u_min, dtype=torch.float32)
        self.u_max = torch.tensor(self.config.u_max, dtype=torch.float32)

    def _step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        dt = self.config.dt
        if self.config.dynamics_type == "doubleintegrator":
            new_state = state.clone()
            new_state[:, 0] = state[:, 0] + dt * state[:, 2] + 0.5 * dt * dt * control[:, 0]
            new_state[:, 1] = state[:, 1] + dt * state[:, 3] + 0.5 * dt * dt * control[:, 1]
            new_state[:, 2] = state[:, 2] + dt * control[:, 0]
            new_state[:, 3] = state[:, 3] + dt * control[:, 1]
            return new_state
        if self.config.dynamics_type == "unicycle":
            new_state = state.clone()
            new_state[:, 0] = state[:, 0] + dt * control[:, 0] * torch.cos(state[:, 2])
            new_state[:, 1] = state[:, 1] + dt * control[:, 0] * torch.sin(state[:, 2])
            new_state[:, 2] = torch.atan2(
                torch.sin(state[:, 2] + dt * control[:, 1]),
                torch.cos(state[:, 2] + dt * control[:, 1]),
            )
            return new_state
        new_state = state.clone()
        new_state[:, :2] = state[:, :2] + dt * control
        return new_state

    def plan(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        *,
        gamma: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        t0 = time.perf_counter()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim > 1:
            goal = goal[0]
        device = state.device
        dtype = state.dtype
        u_min = self.u_min.to(device=device, dtype=dtype)
        u_max = self.u_max.to(device=device, dtype=dtype)
        obstacles = obstacles.to(device=device, dtype=dtype)
        if obstacles.ndim == 2:
            obstacles_batch = obstacles.unsqueeze(0).expand(self.config.num_samples, -1, -1)
        else:
            obstacles_batch = obstacles
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(int(seed))
        controls = torch.randn(
            self.config.num_samples,
            self.config.horizon,
            2,
            generator=gen,
            device=device,
            dtype=dtype,
        ) * self.config.noise_sigma
        to_goal = goal[:2].to(device=device, dtype=dtype) - state[0, :2]
        nominal = torch.clamp(to_goal / max(self.config.horizon * self.config.dt, 1e-6), u_min, u_max)
        controls = torch.clamp(controls + nominal.view(1, 1, 2), u_min, u_max)
        x0 = state[0].unsqueeze(0).expand(self.config.num_samples, -1)
        x = x0.clone()
        costs = torch.zeros(self.config.num_samples, device=device, dtype=dtype)
        infeasible = torch.zeros(self.config.num_samples, device=device, dtype=torch.bool)
        min_h = torch.full((self.config.num_samples,), float("inf"), device=device, dtype=dtype)
        gamma_value = float(self.config.gamma if gamma is None else gamma)
        for t in range(self.config.horizon):
            h_old = affine_barrier_h(x0, x, obstacles_batch)
            x_next = self._step(x, controls[:, t])
            h_new = affine_barrier_h(x0, x_next, obstacles_batch)
            min_h = torch.minimum(min_h, h_new)
            violation = h_new < (1.0 - gamma_value) * h_old
            if self.config.check_first_control_only:
                if t == 0:
                    infeasible |= violation
            else:
                infeasible |= violation
            goal_cost = torch.linalg.norm(x_next[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1) ** 2
            effort = 0.01 * torch.sum(controls[:, t] ** 2, dim=1)
            costs += goal_cost + effort
            x = x_next
        costs = costs + 10.0 * torch.linalg.norm(x[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1) ** 2
        costs = torch.where(infeasible, torch.full_like(costs, float("inf")), costs)
        if torch.isinf(costs).all():
            costs = torch.nan_to_num(costs, posinf=1e9)
        best = torch.argmin(costs)
        action = controls[best, 0].clamp(u_min, u_max)
        clearance = barrier_clearance(state[:, :2], obstacles.unsqueeze(0) if obstacles.ndim == 2 else obstacles[:1]).min()
        info = {
            "gamma": gamma_value,
            "min_barrier_h": float(min_h[best].detach().cpu()),
            "min_clearance": float(clearance.detach().cpu()),
            "num_barrier_violations": int(infeasible.sum().detach().cpu()),
            "infeasibility_rate": float(infeasible.float().mean().detach().cpu()),
            "correction_magnitude": float(torch.linalg.norm(action - nominal).detach().cpu()),
            "solve_time": time.perf_counter() - t0,
        }
        return action, info
