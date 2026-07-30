"""Same-latent locked-Kazuki guidance for B1 acquisition repair.

This module deliberately implements only the locked Kazuki *guidance field*.
It does not add Kazuki's 200-sample generator, warm start, MPPI refinement,
output filter, privileged SFM lookahead, or any new proposal template.

For each selected B candidate, the original Gaussian base ``x0`` is reused and
integrated through the current policy with the fixed goal/CBF coefficients.
The result is therefore a guided regeneration of the same latent lineage, not
an action-space edit of the already generated control window.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

import sfm_b1_full_episode_audit as FA
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_scene as SS


LOCKED_SAFE_COEF = 0.3
LOCKED_GOAL_COEF = 0.5
TRAP_HORIZON = 10
TRAP_DISPLACEMENT = 0.2
TRAP_PATIENCE = 3


def locked_guidance_config(gamma, *, n_sample):
    """Materialize the immutable Kazuki guidance coefficients for one B pool."""
    base = KZ.KazukiConfig(
        safe_coefs=(LOCKED_SAFE_COEF,),
        goal_coef=LOCKED_GOAL_COEF,
        safe_coef_gamma_span=0.0,
        goal_coef_gamma_span=0.0,
    ).validate()
    controller = KZ._gamma_controller_config(base, float(gamma)).validate()
    guidance = KZ._gamma_guidance_config(controller, float(gamma))
    return replace(guidance, n_sample=int(n_sample)).validate()


def collector_ode_times(nfe):
    """The same Euler knot schedule used by the unguided B1 K proposals."""
    nfe = int(nfe)
    if nfe < 1:
        raise ValueError("nfe must be positive")
    return tuple(index / nfe for index in range(nfe + 1))


def same_latent_guided_controls(
    policy,
    context,
    state,
    ped_xy,
    ped_vel,
    gamma,
    x0,
    *,
    nfe=8,
    collect_diagnostics=True,
):
    """Regenerate selected B latent lineages under fixed Kazuki guidance."""
    x0 = torch.as_tensor(
        x0, device=context.device, dtype=context.dtype,
    )
    if x0.ndim != 2 or x0.shape[1] != int(policy.d):
        raise ValueError(f"x0 must have shape [B,{policy.d}], got {tuple(x0.shape)}")
    count = int(len(x0))
    if count < 1:
        raise ValueError("at least one selected latent is required")
    cfg = locked_guidance_config(gamma, n_sample=count)
    goal = torch.as_tensor(
        SS.GOAL, device=context.device, dtype=context.dtype,
    )
    ped_prediction = KZ.predict_pedestrians_t(
        ped_xy, ped_vel, int(policy.H_pred), SS.DT,
        context.device, context.dtype,
    )
    ped_velocity = torch.as_tensor(
        ped_vel, device=context.device, dtype=context.dtype,
    )
    guided, ode_trace, unguided = KZ.guided_generate(
        policy,
        context,
        np.asarray(state, np.float32),
        goal,
        ped_prediction,
        ped_velocity,
        SS.R_PED + float(cfg.collision_margin),
        x0,
        collector_ode_times(nfe),
        cfg,
        collect_diagnostics=bool(collect_diagnostics),
    )
    guided_controls = torch.clamp(
        guided.reshape(count, policy.H_pred, 2) * float(policy.u_max),
        -float(policy.u_max),
        float(policy.u_max),
    )
    diagnostics = dict(
        operator="same_latent_locked_kazuki_guidance",
        candidate_count=count,
        safe_coef=LOCKED_SAFE_COEF,
        goal_coef=LOCKED_GOAL_COEF,
        nfe=int(nfe),
        ode_times=collector_ode_times(nfe),
        no_mppi_refinement=True,
        no_new_latents=True,
    )
    if collect_diagnostics:
        unguided_controls = torch.clamp(
            unguided.reshape(count, policy.H_pred, 2) * float(policy.u_max),
            -float(policy.u_max),
            float(policy.u_max),
        )
        terminal = ode_trace[-1]
        diagnostics.update(
            unguided_controls=unguided_controls.detach().cpu().numpy(),
            net_first_action=(
                guided_controls[:, 0] - unguided_controls[:, 0]
            ).detach().cpu().numpy(),
            goal_first_action=(
                terminal["integrated_goal_guidance"]
                .reshape(count, policy.H_pred, 2)[:, 0]
                * float(policy.u_max)
            ).detach().cpu().numpy(),
            safety_first_action=(
                terminal["integrated_safety_guidance"]
                .reshape(count, policy.H_pred, 2)[:, 0]
                * float(policy.u_max)
            ).detach().cpu().numpy(),
            component_semantics=terminal["component_semantics"],
        )
    return guided_controls, diagnostics


def predicted_trap(states, first_action, *, horizon=TRAP_HORIZON,
                   displacement=TRAP_DISPLACEMENT):
    """Whether executing ``first_action`` makes the declared trap predicate true."""
    states = [np.asarray(value, np.float32) for value in states]
    if len(states) < int(horizon):
        return False
    current = states[-1]
    action = np.asarray(first_action, np.float32)
    next_position = (
        current[:2] + SS.DT * current[2:4]
        + 0.5 * SS.DT ** 2 * action
    )
    return bool(
        np.linalg.norm(next_position - states[-int(horizon)][:2])
        < float(displacement)
    )


def post_action_trap_matches(states, first_action):
    """Audit that the predictive predicate equals the existing post-action test."""
    states = [np.asarray(value, np.float32) for value in states]
    current = states[-1]
    action = np.asarray(first_action, np.float32)
    next_position = SM.rollout_positions(current, action[None])[-1]
    next_state = np.concatenate([
        next_position,
        current[2:4] + SS.DT * action,
    ]).astype(np.float32)
    expected = predicted_trap(states, action)
    observed = FA._trap(states + [next_state])
    if bool(expected) != bool(observed):
        raise AssertionError("predictive and post-action trap predicates disagree")
    return bool(expected)


def next_trap_streak(current_streak, trap_event):
    """Update the per-lineage streak and report the fail-closed boundary."""
    streak = int(current_streak) + 1 if bool(trap_event) else 0
    return streak, bool(streak >= TRAP_PATIENCE)


def manifest():
    return dict(
        operator="same-latent B4 Kazuki guidance repair",
        safe_coef=LOCKED_SAFE_COEF,
        goal_coef=LOCKED_GOAL_COEF,
        trap_horizon=TRAP_HORIZON,
        trap_displacement=TRAP_DISPLACEMENT,
        trap_patience=TRAP_PATIENCE,
        exclusions=(
            "no privileged MPC; no independent raw fallback; no new latent; "
            "no Kazuki MPPI refinement; no output shield"
        ),
    )
