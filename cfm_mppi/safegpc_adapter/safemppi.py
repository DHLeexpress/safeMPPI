from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .barrier import (
    affine_barrier_h,
    affine_barrier_h_ho,
    affine_barrier_h_ho_all,
    barrier_clearance,
)


@dataclass
class SafeMPPIConfig:
    horizon: int = 20
    dt: float = 0.1
    num_samples: int = 128
    gamma: float = 0.5
    temperature: float = 1.0
    noise_sigma: float | Tuple[float, float] = 0.6
    u_min: Tuple[float, float] = (-2.0, -2.0)
    u_max: Tuple[float, float] = (2.0, 2.0)
    safety_margin: float = 0.5
    running_goal_weight: float = 0.25
    terminal_goal_weight: float = 80.0
    control_weight: float = 0.03
    smooth_weight: float = 0.12
    soft_clearance_weight: float = 25.0
    progress_weight: float = 2.0
    heading_weight: float = 0.4
    check_first_control_only: bool = False
    dynamics_type: str = "doubleintegrator"
    debug_max_rollouts: int = 80
    # --- Guided Safe MPPI (overnight contribution) ---
    use_ho_barrier: bool = False          # affine higher-order DCBF in (p, v)
    barrier_topk: int = 0                  # cap enforced obstacles to k nearest (0 = no cap)
    barrier_activation_radius: float = 3.5  # enforce obstacles within this current clearance (0 = all)
    eta: float = 0.6                       # velocity look-ahead (braking horizon, s)
    use_guidance: bool = False            # PSF projection of sampling mean into feasible half-space
    guidance_relax: float = 1.0           # in (0,1]: fraction of the deficit to close (1 = full projection)
    guidance_horizon: int = 12            # only project the first k nominal controls (we apply step 0 & replan)
    use_aniso_cov: bool = False           # anisotropic covariance (tangent-wide, normal-narrow)
    aniso_normal_scale: float = 0.5       # noise scale along obstacle normal
    aniso_tangent_scale: float = 1.7      # noise scale along obstacle tangent (multi-modality)
    barrier_extra_margin: float = 0.0     # buffer added to barrier radius beyond collision margin (CS-MPPI tightening)
    adaptive_gamma: bool = False          # per-step gamma schedule from distance/closing-velocity
    gamma_min: float = 0.1
    gamma_max: float = 1.0
    filter_output: bool = False           # project final applied control through one-step PSF (hard per-step guarantee)
    filter_iters: int = 3
    proposal_gaussian_mix: int = 96       # # Gaussian-around-damped-nominal samples mixed into a learned proposal (velocity regulation + coverage)
    use_sets_backup: bool = False
    sets_num_modes: int = 3
    sets_branch_scale: float = 0.85
    sets_include_cbf_backup: bool = True
    sets_cbf_push: float = 1.25
    sets_reverse_speed: float = 0.75
    sets_turn_rate: float = 1.4


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

    def _sigma(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = self.config.noise_sigma
        if isinstance(sigma, (tuple, list)):
            value = torch.tensor(sigma, dtype=dtype, device=device)
        else:
            value = torch.full((2,), float(sigma), dtype=dtype, device=device)
        return value

    def _anisotropic(self, noise: torch.Tensor, normal_axis: torch.Tensor) -> torch.Tensor:
        """Reshape isotropic noise into an anisotropic cloud: narrow along the
        obstacle normal, wide along the tangent (THEORY.md Fix 5 / covariance
        steering) to spread samples into the left/right homotopy classes."""
        H = noise.shape[1]
        n = normal_axis.view(1, H, 2)
        tang = torch.stack((-normal_axis[:, 1], normal_axis[:, 0]), dim=1).view(1, H, 2)
        cn = (noise * n).sum(dim=-1, keepdim=True)
        ct = (noise * tang).sum(dim=-1, keepdim=True)
        shaped = self.config.aniso_normal_scale * cn * n + self.config.aniso_tangent_scale * ct * tang
        # steps with no defined normal axis (e.g. past guidance horizon / open space)
        # keep isotropic noise instead of collapsing to zero.
        valid = (torch.linalg.norm(normal_axis, dim=1) > 1e-6).view(1, H, 1)
        return torch.where(valid, shaped, noise)

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

    def _linear_matrices(self, state: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(self.config.dt)
        state_dim = int(state.numel())
        device = state.device
        dtype = state.dtype
        if self.config.dynamics_type == "doubleintegrator" and state_dim >= 4:
            A = torch.eye(state_dim, dtype=dtype, device=device)
            B = torch.zeros(state_dim, 2, dtype=dtype, device=device)
            A[0, 2] = dt
            A[1, 3] = dt
            B[0, 0] = 0.5 * dt * dt
            B[1, 1] = 0.5 * dt * dt
            B[2, 0] = dt
            B[3, 1] = dt
            return A, B
        if self.config.dynamics_type == "unicycle" and state_dim >= 3:
            theta = state[2]
            v = control[0]
            c = torch.cos(theta)
            s = torch.sin(theta)
            A = torch.eye(state_dim, dtype=dtype, device=device)
            B = torch.zeros(state_dim, 2, dtype=dtype, device=device)
            A[0, 2] = -dt * v * s
            A[1, 2] = dt * v * c
            B[0, 0] = dt * c
            B[1, 0] = dt * s
            B[2, 1] = dt
            return A, B
        A = torch.eye(state_dim, dtype=dtype, device=device)
        B = torch.zeros(state_dim, 2, dtype=dtype, device=device)
        B[:2, :2] = dt * torch.eye(2, dtype=dtype, device=device)
        return A, B

    def _nominal_control(self, state: torch.Tensor, goal: torch.Tensor, horizon: int, u_min: torch.Tensor, u_max: torch.Tensor) -> torch.Tensor:
        to_goal = goal[:2].to(device=state.device, dtype=state.dtype) - state[0, :2]
        if self.config.dynamics_type == "unicycle":
            distance = torch.linalg.norm(to_goal).clamp_min(1e-6)
            desired_heading = torch.atan2(to_goal[1], to_goal[0])
            heading_error = torch.atan2(
                torch.sin(desired_heading - state[0, 2]),
                torch.cos(desired_heading - state[0, 2]),
            )
            v = torch.clamp(distance / max(horizon * self.config.dt, 1e-6), min=0.0, max=float(u_max[0]))
            omega = torch.clamp(1.5 * heading_error, min=float(u_min[1]), max=float(u_max[1]))
            return torch.stack([v, omega]).to(device=state.device, dtype=state.dtype)
        if self.config.dynamics_type == "doubleintegrator" and state.shape[1] >= 4:
            vel_err = -state[0, 2:4]
            nominal = 0.45 * to_goal + 0.8 * vel_err
            return torch.clamp(nominal, u_min, u_max)
        nominal = to_goal / max(horizon * self.config.dt, 1e-6)
        return torch.clamp(nominal, u_min, u_max)

    def _nominal_sequence(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        horizon: int,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = state[0:1].clone()
        controls = []
        states = [x[0].clone()]
        for t in range(horizon):
            remaining = max(horizon - t, 1)
            u = self._nominal_control(x, goal, remaining, u_min, u_max)
            controls.append(u)
            x = self._step(x, u.view(1, 2))
            states.append(x[0].clone())
        return torch.stack(controls, dim=0), torch.stack(states, dim=0)

    def _eta_eff(self) -> float:
        return float(self.config.eta) if self.config.use_ho_barrier else 0.0

    def _barrier_h(self, x0, x, obstacles, obstacle_velocities):
        """Dispatch: higher-order/relative-velocity barrier when enabled, else the
        original position-only affine barrier (exact backward compatibility)."""
        if self.config.use_ho_barrier:
            return affine_barrier_h_ho(x0, x, obstacles, obstacle_velocities, self._eta_eff())
        return affine_barrier_h(x0, x, obstacles)

    def _guide_nominal(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        safe_obstacles: torch.Tensor,
        obstacle_velocities: Optional[torch.Tensor],
        gamma: float,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predictive-safety-filter guidance (THEORY.md Fix 3).

        Greedily projects each nominal control onto the per-step affine half-space
        constraint  h(x_{t+1}) >= (1-gamma) h(x_t)  so the resulting reference
        sequence is feasible and the Gaussian samples centred on it are no longer
        mass-rejected. Exact for the (affine) double integrator; a first-order
        heuristic otherwise. Returns (guided_seq [H,2], normal_axis [H,2]).
        """
        H = int(self.config.horizon)
        dt = float(self.config.dt)
        eta = self._eta_eff()
        relax = float(self.config.guidance_relax)
        nominal_seq, _ = self._nominal_sequence(state, goal, H, u_min, u_max)
        x0 = state[0:1].clone()
        x = x0.clone()
        guided = []
        normal_axis = torch.zeros(H, 2, device=state.device, dtype=state.dtype)
        di = self.config.dynamics_type == "doubleintegrator"
        gh = min(H, int(self.config.guidance_horizon)) if self.config.guidance_horizon else H
        for t in range(gh):
            if obstacle_velocities is not None and safe_obstacles.numel():
                obs_t = safe_obstacles.clone()
                obs_t[..., :2] = obs_t[..., :2] + obstacle_velocities[..., :2] * (dt * t)
                obs_n = safe_obstacles.clone()
                obs_n[..., :2] = obs_n[..., :2] + obstacle_velocities[..., :2] * (dt * (t + 1))
            else:
                obs_t = obs_n = safe_obstacles
            u = nominal_seq[t].clone().view(1, 2)
            k = int(self.config.barrier_topk)
            ar = float(self.config.barrier_activation_radius)
            h_old_a, _, active = affine_barrier_h_ho_all(
                x0, x, obs_t, obstacle_velocities, eta, k, ar
            )
            x_next = self._step(x, u)
            h_new_a, grad_a, _ = affine_barrier_h_ho_all(
                x0, x_next, obs_n, obstacle_velocities, eta, k, ar
            )
            # g_j = d h_new_j / d u  (exact for double integrator; first-order else)
            scale = (0.5 * dt * dt + eta * dt) if di else dt
            g_a = grad_a * scale  # [1,N,2]
            deficit = (1.0 - gamma) * h_old_a - h_new_a  # [1,N] >0 => violates obstacle j
            deficit = torch.where(active, deficit, torch.zeros_like(deficit))
            gg = (g_a * g_a).sum(dim=2).clamp_min(1e-9)  # [1,N]
            corr = torch.clamp(deficit, min=0.0) / gg * relax  # [1,N]
            delta = (corr.unsqueeze(2) * g_a).sum(dim=1)  # [1,2] sum of per-obstacle pushes
            u = torch.clamp(u + delta, u_min, u_max)
            guided.append(u[0])
            # covariance axis = normal of the most-binding active obstacle
            masked_h = torch.where(active, h_new_a, torch.full_like(h_new_a, float("inf")))
            jstar = int(torch.argmin(masked_h[0]).item())
            gvec = grad_a[0, jstar]
            normal_axis[t] = gvec / torch.linalg.norm(gvec).clamp_min(1e-9)
            x = self._step(x, u)
        for t in range(gh, H):
            guided.append(nominal_seq[t])  # past guidance horizon: keep nominal (we replan each step)
        return torch.stack(guided, dim=0), normal_axis

    def safety_filter_action(
        self,
        state: torch.Tensor,
        obstacles: torch.Tensor,
        action: torch.Tensor,
        *,
        gamma: Optional[float] = None,
        obstacle_velocities: Optional[torch.Tensor] = None,
        iters: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Runtime predictive safety filter (THEORY.md §7): minimally project a
        single proposed ``action`` (e.g. a one-step drifting/CFM output) onto the
        intersection of the active affine HO-DCBF half-spaces so the next state
        satisfies h_j(x1) >= (1-gamma) h_j(x0) for every active obstacle. A few
        Jacobi sweeps approximate the QP projection; returns (safe_action, info)
        and gives the learned policy a hard per-step certificate at ~us cost."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        device, dtype = state.device, state.dtype
        u_min = self.u_min.to(device=device, dtype=dtype)
        u_max = self.u_max.to(device=device, dtype=dtype)
        gamma_value = float(self.config.gamma if gamma is None else gamma)
        eta = self._eta_eff()
        ar = float(self.config.barrier_activation_radius)
        k = int(self.config.barrier_topk)
        di = self.config.dynamics_type == "doubleintegrator"
        dt = float(self.config.dt)
        obs = obstacles.to(device=device, dtype=dtype)
        if obs.numel() and self.config.safety_margin:
            obs = obs.clone()
            obs[..., 2] = obs[..., 2] + float(self.config.safety_margin) + float(self.config.barrier_extra_margin)
        u = action.detach().clone().view(1, 2).to(device=device, dtype=dtype)
        x0 = state[0:1]
        obs_next = obs
        if obstacle_velocities is not None and obstacle_velocities.numel():
            obstacle_velocities = obstacle_velocities.to(device=device, dtype=dtype)
            if obs.numel():
                obs_next = obs.clone()
                obs_next[..., :2] = obs_next[..., :2] + obstacle_velocities[..., :2] * dt
        n_corr = 0
        max_deficit = 0.0
        n_active = 0
        for _ in range(max(1, iters)):
            x1 = self._step(x0, u)
            h_old, _, active = affine_barrier_h_ho_all(x0, x0, obs, obstacle_velocities, eta, k, ar)
            # check the robot's next state against the obstacle's PREDICTED next position
            h_new, grad, _ = affine_barrier_h_ho_all(x0, x1, obs_next, obstacle_velocities, eta, k, ar)
            scale = (0.5 * dt * dt + eta * dt) if di else dt
            g = grad * scale
            deficit = (1.0 - gamma_value) * h_old - h_new
            deficit = torch.where(active, deficit, torch.zeros_like(deficit))
            n_active = int(active.sum().detach().cpu())
            max_deficit = float(torch.clamp(deficit, min=0.0).max().detach().cpu())
            if max_deficit <= 1e-6:
                break
            gg = (g * g).sum(dim=2).clamp_min(1e-9)
            corr = torch.clamp(deficit, min=0.0) / gg
            delta = (corr.unsqueeze(2) * g).sum(dim=1)
            u = torch.clamp(u + delta, u_min, u_max)
            n_corr += 1
        # recompute residual deficit at the final clamped control (clamping may
        # reintroduce a deficit even if the unclamped projection was feasible)
        x1 = self._step(x0, u)
        h_old, _, active = affine_barrier_h_ho_all(x0, x0, obs, obstacle_velocities, eta, k, ar)
        h_new, _, _ = affine_barrier_h_ho_all(x0, x1, obs_next, obstacle_velocities, eta, k, ar)
        final_def = torch.where(active, (1.0 - gamma_value) * h_old - h_new, torch.zeros_like(h_new))
        max_deficit = float(torch.clamp(final_def, min=0.0).max().detach().cpu())
        info = {"filter_iters": n_corr,
                "filter_feasible": bool(max_deficit <= 1e-4),
                "filter_max_deficit": max_deficit,
                "filter_num_active": n_active,
                "correction_magnitude": float(torch.linalg.norm(u.view(-1) - action.view(-1).to(u)).detach().cpu())}
        return u.view(-1), info

    def _controllability_matrix(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        input_width: torch.Tensor,
    ) -> torch.Tensor:
        horizon = int(controls.shape[0])
        state_dim = int(states.shape[1])
        blocks = []
        suffix = torch.eye(state_dim, dtype=states.dtype, device=states.device)
        for k in reversed(range(horizon)):
            A, B = self._linear_matrices(states[k], controls[k])
            B_norm = B * input_width.view(1, 2)
            blocks.append(suffix @ B_norm)
            suffix = suffix @ A
        blocks.reverse()
        return torch.cat(blocks, dim=1)

    def _sets_backup_controls(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        safe_obstacles: torch.Tensor,
        obstacle_velocities: Optional[torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[str], list[str]]:
        horizon = int(self.config.horizon)
        if horizon <= 0:
            empty = torch.empty(0, 0, 2, dtype=state.dtype, device=state.device)
            return empty, [], []

        nominal_seq, nominal_states = self._nominal_sequence(state, goal, horizon, u_min, u_max)
        input_width = (u_max - u_min).clamp_min(1e-6)
        normalized_nominal = ((nominal_seq - u_min.view(1, 2)) / input_width.view(1, 2)).clamp(0.0, 1.0)
        cmat = self._controllability_matrix(nominal_states, nominal_seq, input_width)
        if cmat.numel() == 0:
            empty = torch.empty(0, horizon, 2, dtype=state.dtype, device=state.device)
            return empty, [], []

        gramian = cmat @ cmat.T
        eigvals, eigvecs = torch.linalg.eigh(gramian)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        pinv_c = torch.linalg.pinv(cmat)

        branches = []
        labels: list[str] = []
        kinds: list[str] = []

        def add_linear_branch(target: torch.Tensor, label: str, kind: str) -> None:
            delta_v = pinv_c @ target
            normalized = (normalized_nominal.reshape(-1) + delta_v).view(horizon, 2).clamp(0.0, 1.0)
            branches.append(u_min.view(1, 2) + normalized * input_width.view(1, 2))
            labels.append(label)
            kinds.append(kind)

        max_modes = min(int(self.config.sets_num_modes), eigvecs.shape[1])
        scale = float(self.config.sets_branch_scale)
        for mode in range(max_modes):
            if float(eigvals[mode].detach().cpu()) <= 1e-10:
                continue
            axis = torch.sqrt(eigvals[mode].clamp_min(0.0)) * eigvecs[:, mode] * scale
            add_linear_branch(axis, f"m{mode}+", "sets_mode")
            add_linear_branch(-axis, f"m{mode}-", "sets_mode")

        if self.config.sets_include_cbf_backup and safe_obstacles.numel():
            centers = safe_obstacles[:, :2]
            radii = safe_obstacles[:, 2]
            pos = state[0, :2]
            clearances = torch.linalg.norm(centers - pos.view(1, 2), dim=1) - radii
            obs_idx = int(torch.argmin(clearances).detach().cpu())
            center = centers[obs_idx]
            rel = pos - center
            rel_norm = torch.linalg.norm(rel).clamp_min(1e-6)
            away = rel / rel_norm
            tangent = torch.stack((-away[1], away[0]))
            push = float(self.config.sets_cbf_push)
            target = torch.zeros(nominal_states.shape[1], dtype=state.dtype, device=state.device)
            target[:2] = push * away
            add_linear_branch(target, "away", "cbf_backup")
            target_tan = torch.zeros_like(target)
            target_tan[:2] = 0.75 * push * tangent
            add_linear_branch(target_tan, "tan+", "cbf_backup")
            add_linear_branch(-target_tan, "tan-", "cbf_backup")

            if self.config.dynamics_type == "unicycle":
                reverse = torch.zeros(horizon, 2, dtype=state.dtype, device=state.device)
                reverse[:, 0] = -min(float(self.config.sets_reverse_speed), abs(float(u_min[0])))
                reverse[:, 1] = 0.0
                branches.append(torch.clamp(reverse, u_min.view(1, 2), u_max.view(1, 2)))
                labels.append("back")
                kinds.append("hard_backup")
                for sign, label in [(1.0, "rev+"), (-1.0, "rev-")]:
                    rev_turn = reverse.clone()
                    rev_turn[:, 1] = sign * min(float(self.config.sets_turn_rate), float(u_max[1]))
                    branches.append(torch.clamp(rev_turn, u_min.view(1, 2), u_max.view(1, 2)))
                    labels.append(label)
                    kinds.append("hard_backup")

        if not branches:
            empty = torch.empty(0, horizon, 2, dtype=state.dtype, device=state.device)
            return empty, [], []
        return torch.stack(branches, dim=0), labels, kinds

    def plan(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        *,
        gamma: Optional[float] = None,
        obstacle_velocities: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        return_rollouts: bool = False,
        proposal_controls: Optional[torch.Tensor] = None,
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
        safe_obstacles = obstacles.clone()
        if safe_obstacles.numel():
            safe_obstacles[..., 2] = safe_obstacles[..., 2] + float(self.config.safety_margin) + float(self.config.barrier_extra_margin)
        if obstacle_velocities is not None:
            obstacle_velocities = obstacle_velocities.to(device=device, dtype=dtype)
            if obstacle_velocities.ndim == 1:
                obstacle_velocities = obstacle_velocities.unsqueeze(0)
        if obstacles.ndim == 2:
            obstacles_batch0 = safe_obstacles.unsqueeze(0).expand(self.config.num_samples, -1, -1)
        else:
            obstacles_batch0 = safe_obstacles
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(int(seed))
        sigma = self._sigma(device, dtype)
        gamma_value = float(self.config.gamma if gamma is None else gamma)
        if self.config.adaptive_gamma and safe_obstacles.numel():
            from .gamma_schedule import gamma_distance_velocity
            obs2 = safe_obstacles if safe_obstacles.ndim == 2 else safe_obstacles[0]
            pos = state[0, :2]
            centers = obs2[:, :2]
            radii = obs2[:, 2]
            clr = torch.linalg.norm(centers - pos.view(1, 2), dim=1) - radii
            j = int(torch.argmin(clr).item())
            d = float(clr[j].clamp_min(0.0).item())
            dirn = (centers[j] - pos)
            dirn = dirn / torch.linalg.norm(dirn).clamp_min(1e-6)
            vel = state[0, 2:4] if state.shape[1] >= 4 else torch.zeros(2, device=device, dtype=dtype)
            if obstacle_velocities is not None and obstacle_velocities.numel():
                vrel = vel - obstacle_velocities[min(j, obstacle_velocities.shape[0] - 1)]
            else:
                vrel = vel
            v_proj = float(torch.sum(vrel * dirn).clamp_min(0.0).item())  # closing rate (>0 approaching)
            gamma_value = gamma_distance_velocity(
                d, v_proj, g_min=float(self.config.gamma_min), g_max=float(self.config.gamma_max)
            )
        if proposal_controls is not None:
            # Learned-proposal mode (THEORY §10): use externally-supplied control
            # sequences (e.g. from a gamma-conditioned flow) as the MPPI proposal;
            # the DCBF rejection + averaging + output filter remain the certificate.
            controls = torch.clamp(
                proposal_controls.to(device=device, dtype=dtype), u_min, u_max
            )
            H = int(self.config.horizon)
            if controls.shape[1] > H:
                controls = controls[:, :H]
            elif controls.shape[1] < H:
                pad = controls[:, -1:].expand(-1, H - controls.shape[1], -1)
                controls = torch.cat([controls, pad], dim=1)
            # Mix in Gaussian samples around the velocity-damped nominal so MPPI
            # can pick braking near the goal (the learned proposal alone has no
            # velocity regulation and overshoots/diverges past the goal).
            nominal_seq, _ = self._nominal_sequence(state, goal, H, u_min, u_max)
            nmix = int(self.config.proposal_gaussian_mix)
            if nmix > 0:
                gnoise = torch.randn(nmix, H, 2, generator=gen, device=device, dtype=dtype) * sigma.view(1, 1, 2)
                gmix = torch.clamp(gnoise + nominal_seq.unsqueeze(0), u_min, u_max)
                controls = torch.cat([controls, gmix], dim=0)
        else:
            if self.config.use_guidance:
                nominal_seq, normal_axis = self._guide_nominal(
                    state, goal, safe_obstacles, obstacle_velocities, gamma_value, u_min, u_max
                )
            else:
                nominal_seq, _ = self._nominal_sequence(state, goal, self.config.horizon, u_min, u_max)
                normal_axis = None
            noise = torch.randn(
                self.config.num_samples,
                self.config.horizon,
                2,
                generator=gen,
                device=device,
                dtype=dtype,
            ) * sigma.view(1, 1, 2)
            if self.config.use_aniso_cov and normal_axis is not None:
                noise = self._anisotropic(noise, normal_axis)
            controls = torch.clamp(noise + nominal_seq.unsqueeze(0), u_min, u_max)
        branch_labels: list[str] = []
        branch_kinds: list[str] = []
        branch_indices = torch.empty(0, dtype=torch.long, device=device)
        if self.config.use_sets_backup:
            branch_controls, branch_labels, branch_kinds = self._sets_backup_controls(
                state,
                goal,
                safe_obstacles,
                obstacle_velocities,
                u_min,
                u_max,
            )
            if branch_controls.numel():
                branch_indices = torch.arange(controls.shape[0], controls.shape[0] + branch_controls.shape[0], device=device)
                controls = torch.cat([controls, branch_controls], dim=0)
        sample_count = int(controls.shape[0])
        nominal = nominal_seq[0]
        if obstacles.ndim == 2:
            obstacles_batch0 = safe_obstacles.unsqueeze(0).expand(sample_count, -1, -1)
        else:
            obstacles_batch0 = safe_obstacles
        x0 = state[0].unsqueeze(0).expand(sample_count, -1)
        x = x0.clone()
        state_seq = [x.clone()]
        costs = torch.zeros(sample_count, device=device, dtype=dtype)
        infeasible = torch.zeros(sample_count, device=device, dtype=torch.bool)
        min_h = torch.full((sample_count,), float("inf"), device=device, dtype=dtype)
        initial_goal_distance = torch.linalg.norm(x[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1)
        prev_action = torch.zeros_like(controls[:, 0])
        for t in range(self.config.horizon):
            if obstacle_velocities is not None and safe_obstacles.numel():
                obs_t = safe_obstacles.clone()
                obs_t[..., :2] = obs_t[..., :2] + obstacle_velocities[..., :2] * (self.config.dt * t)
                obs_next = safe_obstacles.clone()
                obs_next[..., :2] = obs_next[..., :2] + obstacle_velocities[..., :2] * (self.config.dt * (t + 1))
                obstacles_batch = obs_t.unsqueeze(0).expand(sample_count, -1, -1) if obs_t.ndim == 2 else obs_t
                obstacles_batch_next = obs_next.unsqueeze(0).expand(sample_count, -1, -1) if obs_next.ndim == 2 else obs_next
            else:
                obstacles_batch = obstacles_batch0
                obstacles_batch_next = obstacles_batch0
            x_next = self._step(x, controls[:, t])
            if self.config.use_ho_barrier:
                eta_eff = self._eta_eff()
                k = int(self.config.barrier_topk)
                ar = float(self.config.barrier_activation_radius)
                h_old_a, _, active = affine_barrier_h_ho_all(
                    x0, x, obstacles_batch, obstacle_velocities, eta_eff, k, ar
                )
                h_new_a, _, _ = affine_barrier_h_ho_all(
                    x0, x_next, obstacles_batch_next, obstacle_velocities, eta_eff, k, ar
                )
                viol_j = (h_new_a < (1.0 - gamma_value) * h_old_a) & active
                violation = viol_j.any(dim=1)
                min_h = torch.minimum(min_h, h_new_a.min(dim=1).values)
            else:
                h_old = self._barrier_h(x0, x, obstacles_batch, obstacle_velocities)
                h_new = self._barrier_h(x0, x_next, obstacles_batch_next, obstacle_velocities)
                min_h = torch.minimum(min_h, h_new)
                violation = h_new < (1.0 - gamma_value) * h_old
            if self.config.check_first_control_only:
                if t == 0:
                    infeasible |= violation
            else:
                infeasible |= violation
            goal_distance = torch.linalg.norm(x_next[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1)
            goal_cost = self.config.running_goal_weight * goal_distance**2
            effort = self.config.control_weight * torch.sum(controls[:, t] ** 2, dim=1)
            smooth = self.config.smooth_weight * torch.sum((controls[:, t] - prev_action) ** 2, dim=1)
            progress = -self.config.progress_weight * (initial_goal_distance - goal_distance)
            clearance = barrier_clearance(x_next[:, :2], obstacles_batch_next)
            soft_clearance = self.config.soft_clearance_weight * torch.relu(-clearance) ** 2
            if self.config.dynamics_type == "unicycle":
                to_goal_next = goal[:2].to(device=device, dtype=dtype) - x_next[:, :2]
                desired_heading = torch.atan2(to_goal_next[:, 1], to_goal_next[:, 0])
                heading_error = torch.atan2(torch.sin(desired_heading - x_next[:, 2]), torch.cos(desired_heading - x_next[:, 2]))
                heading_cost = self.config.heading_weight * heading_error**2
            else:
                heading_cost = 0.0
            costs += goal_cost + effort + smooth + soft_clearance + progress + heading_cost
            x = x_next
            prev_action = controls[:, t]
            state_seq.append(x.clone())
        terminal_goal = torch.linalg.norm(x[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1)
        costs = costs + self.config.terminal_goal_weight * terminal_goal**2
        raw_costs = costs.clone()
        costs = torch.where(infeasible, torch.full_like(costs, float("inf")), costs)
        if torch.isinf(costs).all():
            costs = raw_costs + 1e4 * infeasible.float()
        best = torch.argmin(costs)
        action = controls[best, 0].clamp(u_min, u_max)
        filt_info = None
        if self.config.filter_output and self.config.use_ho_barrier and obstacles.numel():
            # PSF guarantee on the APPLIED control: even if every sample was
            # rejected (degenerate), project onto the active half-spaces so the
            # executed action provably satisfies the DCBF (THEORY §4/§7).
            action, filt_info = self.safety_filter_action(
                state[0], obstacles, action, gamma=gamma_value,
                obstacle_velocities=obstacle_velocities, iters=int(self.config.filter_iters),
            )
            action = action.clamp(u_min, u_max)
        clearance = barrier_clearance(state[:, :2], safe_obstacles.unsqueeze(0) if safe_obstacles.ndim == 2 else safe_obstacles[:1]).min()
        info = {
            "gamma": gamma_value,
            "min_barrier_h": float(min_h[best].detach().cpu()),
            "min_clearance": float(clearance.detach().cpu()),
            "num_barrier_violations": int(infeasible.sum().detach().cpu()),
            "infeasibility_rate": float(infeasible.float().mean().detach().cpu()),
            "correction_magnitude": float(torch.linalg.norm(action - nominal).detach().cpu()),
            "num_backup_branches": int(branch_indices.numel()),
            "selected_backup_branch": None,
            "solve_time": time.perf_counter() - t0,
        }
        if filt_info is not None:
            info["filter_feasible"] = filt_info["filter_feasible"]
            info["filter_max_deficit"] = filt_info["filter_max_deficit"]
            info["filter_infeasible"] = (not filt_info["filter_feasible"])
        if branch_indices.numel():
            branch_hit = torch.nonzero(branch_indices == best, as_tuple=False).flatten()
            if branch_hit.numel():
                info["selected_backup_branch"] = branch_labels[int(branch_hit[0].detach().cpu())]
        if return_rollouts:
            state_seq_t = torch.stack(state_seq, dim=1).detach().cpu()
            feasible = (~infeasible).detach().cpu()
            max_rollouts = max(1, int(self.config.debug_max_rollouts))
            branch_indices_cpu = branch_indices.detach().cpu()
            sample_mask = torch.ones(state_seq_t.shape[0], dtype=torch.bool)
            if branch_indices_cpu.numel():
                sample_mask[branch_indices_cpu] = False
            if state_seq_t.shape[0] > max_rollouts:
                accept_idx = torch.nonzero(feasible & sample_mask, as_tuple=False).flatten()[: max_rollouts // 2]
                reject_idx = torch.nonzero((~feasible) & sample_mask, as_tuple=False).flatten()[: max_rollouts - accept_idx.numel()]
                draw_idx = torch.cat([accept_idx, reject_idx], dim=0)
                if draw_idx.numel() == 0:
                    draw_idx = torch.arange(min(max_rollouts, state_seq_t.shape[0]))
            else:
                draw_idx = torch.nonzero(sample_mask, as_tuple=False).flatten()
            info["debug_rollouts"] = {
                "states": state_seq_t[draw_idx].numpy(),
                "feasible": feasible[draw_idx].numpy(),
                "best_state": state_seq_t[best].numpy(),
            }
            if branch_indices_cpu.numel():
                info["debug_rollouts"]["branch_states"] = state_seq_t[branch_indices_cpu].numpy()
                info["debug_rollouts"]["branch_feasible"] = feasible[branch_indices_cpu].numpy()
                info["debug_rollouts"]["branch_labels"] = branch_labels
                info["debug_rollouts"]["branch_kinds"] = branch_kinds
        return action, info
