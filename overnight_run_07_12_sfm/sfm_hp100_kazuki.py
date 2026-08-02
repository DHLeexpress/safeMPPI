"""Locked Kazuki generate--guide--refine comparator for SFM HP100.

This is an additive comparator: it uses the same HP100 checkpoint and scene as
the raw policy, but applies the existing Kazuki ODE reward guidance and MPPI
refinement with the historically locked coefficients.  No shield, template,
privileged simulator lookahead, or fallback is enabled.  Unlike the legacy
Hp10 comparator, every robot rollout and executed transition obeys the HP100
componentwise acceleration and velocity caps.
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_hp100_history as HPH
import sfm_kazuki as BASE
import sfm_scene as SS


VERSION = "sfm_hp100_kazuki_locked_clipped_v1"
SAFE_COEF = 0.3
GOAL_COEF = 0.5
SAMPLE_SEED = 700_000


def locked_config() -> BASE.KazukiConfig:
    """Return and audit the immutable, non-privileged comparator recipe."""
    config = BASE.KazukiConfig(
        safe_coefs=(SAFE_COEF,),
        goal_coef=GOAL_COEF,
    ).validate()
    disabled = {
        "output_filter": config.output_filter,
        "exact_sfm_step_filter": config.exact_sfm_step_filter,
        "hard_clearance_select": config.hard_clearance_select,
        "safe_coef_gamma_span": config.safe_coef_gamma_span,
        "goal_coef_gamma_span": config.goal_coef_gamma_span,
        "controller_gammas": config.controller_gammas,
    }
    if any(bool(value) for value in disabled.values()):
        raise RuntimeError(f"locked HP100 Kazuki comparator enabled an extra mechanism: {disabled}")
    return config


def clipped_di_rollout_t(state, controls, dt=DYN.DT):
    """Differentiable batched rollout under the exact HP100 cap ordering."""
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError(f"expected controls [B,H,2], got {tuple(controls.shape)}")
    batch, horizon, _ = controls.shape
    current = torch.as_tensor(
        state, dtype=controls.dtype, device=controls.device,
    ).reshape(1, 4).expand(batch, 4).clone()
    positions, velocities = [], []
    for step in range(horizon):
        current = DYN.step_torch(current, controls[:, step], dt=float(dt))
        positions.append(current[:, :2])
        velocities.append(current[:, 2:4])
    return torch.stack(positions, dim=1), torch.stack(velocities, dim=1)


def _obstacles(pedestrian_xy) -> np.ndarray:
    pedestrian_xy = np.asarray(pedestrian_xy, np.float32).reshape(-1, 2)
    return np.concatenate(
        [pedestrian_xy, np.full((len(pedestrian_xy), 1), SS.R_PED, np.float32)],
        axis=1,
    )


def _clearance(state, pedestrian_xy) -> float:
    pedestrian_xy = np.asarray(pedestrian_xy, np.float32).reshape(-1, 2)
    if not len(pedestrian_xy):
        return float("inf")
    return float(
        np.linalg.norm(pedestrian_xy - np.asarray(state[:2])[None], axis=1).min()
        - SS.R_PED
    )


def kazuki_hp100_deploy(
    policy,
    episode,
    gamma,
    *,
    scene_profile,
    T=180,
    reach=0.5,
    device="cpu",
    sample_seed=SAMPLE_SEED,
    collect_diagnostics=False,
):
    """Deploy the locked comparator on one deterministic scenario.

    ``scene_profile`` is required so the caller cannot silently evaluate a
    different pedestrian bank than the paired raw HP100 policy.
    """
    config = locked_config()
    environment = SS.scene_profile(scene_profile)
    humans = SS.make_humans(
        int(episode), seed=0, n_ped=int(environment["n_ped"]),
        speed_range=tuple(environment["ped_speed_range"]),
    )
    state = np.zeros(4, np.float32)
    goal = torch.as_tensor(SS.GOAL, dtype=torch.float32, device=device)
    horizon, latent_dim = int(policy.H_pred), int(policy.d)
    hp_history = HPH.Hp100History()
    control_history = []
    previous_latent = previous_window = None
    states, controls, pedestrian_positions, pedestrian_velocities = [state.copy()], [], [], []
    trace = []
    reached = collision = False
    minimum_clearance = float("inf")

    for step in range(int(T)):
        pedestrian_xy, pedestrian_velocity = SS.collect_humans(humans)
        pedestrian_xy = np.asarray(pedestrian_xy, np.float32)
        pedestrian_velocity = np.asarray(pedestrian_velocity, np.float32)
        clearance = _clearance(state, pedestrian_xy)
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            reached = True
            break

        hp_frame = HPF.hp100_frame(
            state[:2], _obstacles(pedestrian_xy), sensing=SS.R_SENSE,
            n_base=HPF.POLYTOPE_N_BASE,
            obstacle_velocities=pedestrian_velocity,
            robot_velocity=state[2:4],
            predict_gain=HPF.PREDICT_GAIN,
            predict_tau=HPF.PREDICT_TAU,
        )
        hp_tensor = hp_history.append(hp_frame).to(device)
        low_tensor = torch.as_tensor(
            HPF.low5(state, SS.GOAL, gamma), device=device,
        )
        history_tensor = torch.as_tensor(
            HPF.hist_pad(control_history[-HPF.K_HIST:]), device=device,
        )
        context = policy.ctx_from(
            hp_tensor[None], low_tensor[None], history_tensor[None],
        ).squeeze(0)

        torch.manual_seed(int(sample_seed) + int(episode) * 1000 + step)
        if previous_latent is None:
            latent = torch.randn(config.n_sample, latent_dim, device=device)
            ode_times = config.ode_times
        else:
            latent = float(config.warm_s) * previous_latent[None].expand(
                config.n_sample, latent_dim,
            ) + (1.0 - float(config.warm_s)) * torch.randn(
                config.n_sample, latent_dim, device=device,
            )
            ode_times = tuple(
                value for value in config.ode_times
                if value >= float(config.warm_s) - 1.0e-12
            )

        pedestrian_prediction = BASE.predict_pedestrians_t(
            pedestrian_xy, pedestrian_velocity, horizon, DYN.DT,
            device, latent.dtype,
        )
        pedestrian_velocity_tensor = torch.as_tensor(
            pedestrian_velocity, dtype=latent.dtype, device=device,
        )
        generated, guidance_diagnostics, unguided = BASE.guided_generate(
            policy, context, state, goal, pedestrian_prediction,
            pedestrian_velocity_tensor, SS.R_PED + config.collision_margin,
            latent, ode_times, config,
            collect_diagnostics=collect_diagnostics,
            rollout_fn=clipped_di_rollout_t,
        )
        generated_windows = DYN.clip_action_torch(
            generated.reshape(config.n_sample, horizon, 2) * float(policy.u_max)
        )
        selected_window, refine_diagnostics = BASE.flow_mppi_refine(
            policy, state, goal, pedestrian_prediction,
            SS.R_PED + config.collision_margin, generated_windows,
            previous_window, config, collect_diagnostics=collect_diagnostics,
            rollout_fn=clipped_di_rollout_t,
        )
        action = DYN.clip_action_numpy(
            selected_window[0].detach().cpu().numpy()
        ).astype(np.float32, copy=False)

        pedestrian_positions.append(pedestrian_xy.copy())
        pedestrian_velocities.append(pedestrian_velocity.copy())
        controls.append(action.copy())
        control_history.append(action.copy())
        previous_state = state.copy()
        state = DYN.step_numpy(state, action).astype(np.float32, copy=False)
        states.append(state.copy())
        SS.advance_humans(humans, state)

        if collect_diagnostics:
            selected_positions, _ = clipped_di_rollout_t(
                previous_state, selected_window[None], DYN.DT,
            )
            trace.append(dict(
                step=int(step), state=previous_state, action=action.copy(),
                pedestrian_xy=pedestrian_xy.copy(),
                pedestrian_velocity=pedestrian_velocity.copy(),
                guidance=guidance_diagnostics, refinement=refine_diagnostics,
                selected_plan_positions=np.concatenate([
                    previous_state[:2][None],
                    selected_positions[0].detach().cpu().numpy(),
                ], axis=0).astype(np.float32),
                unguided_available=unguided is not None,
            ))

        shifted = torch.cat([selected_window[1:], selected_window[-1:]], dim=0)
        previous_window = shifted.detach()
        previous_latent = (
            shifted / float(policy.u_max)
        ).reshape(-1).detach()

    if not reached and not collision:
        pedestrian_xy, _ = SS.collect_humans(humans)
        clearance = _clearance(state, pedestrian_xy)
        minimum_clearance = min(minimum_clearance, clearance)
        collision = clearance < 0.0
        reached = (
            not collision
            and float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach)
        )

    return dict(
        states=np.asarray(states, np.float32),
        controls=np.asarray(controls, np.float32).reshape(-1, 2),
        peds=np.asarray(pedestrian_positions, np.float32),
        ped_vels=np.asarray(pedestrian_velocities, np.float32),
        path=np.asarray(states, np.float32)[:, :2],
        success=bool(reached and not collision), collision=bool(collision),
        reached=bool(reached), steps=len(controls),
        min_clear=float(minimum_clearance), gamma=float(gamma),
        episode=int(episode), scene=environment,
        config=config.to_dict(), dynamics=DYN.contract(),
        trace=trace if collect_diagnostics else None,
    )

