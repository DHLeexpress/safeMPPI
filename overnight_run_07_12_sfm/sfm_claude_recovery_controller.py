"""Deterministic recovery candidates from the Claude offline augmentation.

This additive module isolates the two declared candidate families from the
offline replay experiment and gives them an honest receding-horizon controller
interpretation.  It does not alter B1 acquisition, replay, or raw evaluation.

The controller is deliberately fail-closed: at each state it enumerates one
fixed finite family, applies the cheap task-space/constant-velocity collision
prefilter, ranks the survivors, submits at most 24 plans to the canonical
full-H=10 moving-pedestrian verifier, and executes the first action of the
best certified plan.  If none is certified, the rollout terminates as NVP.

Important limitations:

* These are hand-crafted deterministic templates, not samples learned by the
  flow model.
* Certification predicts pedestrians with constant velocity over H=10.  It is
  not a recursive-feasibility guarantee.
* The pure recovery rollout below is a diagnostic controller.  It is not the
  replay-data intervention from which the families originated.
"""
from __future__ import annotations

import json
import math

import numpy as np

import _paths  # noqa: F401
import sfm_metrics2 as SM
import sfm_scene as SS


HORIZON = 10
BRAKE_STEPS = (0, 2, 4)
K_DIR = 16
ACCELS = (0.7, 1.4, 2.0)
PREVERIFY_CAP = 24
RECOVERY_KEEP = 2

DODGE_STEPS = (0, 2, 3)
CRUISE_SPEEDS = (1.0, 1.5)
KP_CRUISE = 4.0
V2_ACCELS = (1.4, 2.0)


def controller_manifest():
    """Return the fixed recovery-controller contract."""
    return dict(
        controller="claude_deterministic_recovery",
        horizon=HORIZON,
        preverify_cap=PREVERIFY_CAP,
        keep_per_context=RECOVERY_KEEP,
        exact_gate=(
            "SM.verify_query: resolved and y=1 and full_h=True "
            "and terminal_step=10"
        ),
        execution="first action of the lowest-objective certified candidate",
        no_certified_candidate="fail-closed NVP",
        v1=dict(
            generated_candidates=145,
            brake_steps=list(BRAKE_STEPS),
            directions=K_DIR,
            accelerations=list(ACCELS),
            objective="final goal distance",
        ),
        v2=dict(
            generated_candidates=130,
            dodge_steps=list(DODGE_STEPS),
            directions=K_DIR,
            accelerations=list(V2_ACCELS),
            cruise_speeds=list(CRUISE_SPEEDS),
            kp_cruise=KP_CRUISE,
            objective=(
                "final goal distance - 0.5 * terminal toward-goal speed"
            ),
        ),
        caveats=[
            "finite hand-crafted candidate family; not a learned proposal",
            "H=10 constant-velocity pedestrian certificate",
            "no recursive-feasibility guarantee",
        ],
    )


def _closed_loop_brake(state, steps):
    """Deterministic max-effort braking for ``steps`` transitions."""
    velocity = np.asarray(state, np.float32).reshape(4)[2:4].copy()
    controls = []
    for _ in range(int(steps)):
        action = np.clip(
            -velocity / SS.DT, -SS.U_MAX, SS.U_MAX,
        ).astype(np.float32)
        controls.append(action)
        velocity = velocity + SS.DT * action
    return controls


def recovery_candidates_v1(state):
    """Return the exact 145-candidate brake/bang-bang family."""
    candidates = [(
        np.asarray(_closed_loop_brake(state, HORIZON), np.float32),
        dict(kind="brake10", brake=HORIZON, theta=None, accel=None),
    )]
    for brake in BRAKE_STEPS:
        prefix = _closed_loop_brake(state, brake)
        remaining = HORIZON - brake
        forward = math.ceil(remaining / 2)
        for direction_index in range(K_DIR):
            theta = 2.0 * math.pi * direction_index / K_DIR
            direction = np.array(
                [math.cos(theta), math.sin(theta)], np.float32,
            )
            for accel in ACCELS:
                steer = [accel * direction] * forward
                steer += [-accel * direction] * (remaining - forward)
                controls = np.asarray(prefix + steer, np.float32)
                if controls.shape != (HORIZON, 2):
                    raise AssertionError("v1 recovery candidate must be H=10")
                candidates.append((
                    np.clip(controls, -SS.U_MAX, SS.U_MAX),
                    dict(
                        kind="brake_steer",
                        brake=int(brake),
                        theta=round(theta, 6),
                        accel=float(accel),
                    ),
                ))
    return candidates


def _cruise_controls(position, velocity, steps, v_cruise):
    """Saturated velocity servo toward the goal."""
    position = np.asarray(position, np.float32).copy()
    velocity = np.asarray(velocity, np.float32).copy()
    controls = []
    for _ in range(int(steps)):
        offset = SS.GOAL - position
        norm = float(np.linalg.norm(offset))
        desired = (
            float(v_cruise) * offset / norm
            if norm > 1.0e-6 else np.zeros(2, np.float32)
        )
        action = np.clip(
            KP_CRUISE * (desired - velocity),
            -SS.U_MAX,
            SS.U_MAX,
        ).astype(np.float32)
        controls.append(action)
        position = (
            position + SS.DT * velocity
            + 0.5 * SS.DT ** 2 * action
        )
        velocity = velocity + SS.DT * action
    return controls


def recovery_candidates_v2(state):
    """Return the exact 130-candidate dodge-then-cruise family."""
    state = np.asarray(state, np.float32).reshape(4)
    candidates = []
    for v_cruise in CRUISE_SPEEDS:
        candidates.append((
            np.asarray(_cruise_controls(
                state[:2], state[2:4], HORIZON, v_cruise,
            ), np.float32),
            dict(
                kind="cruise",
                dodge=0,
                theta=None,
                accel=None,
                v_cruise=float(v_cruise),
            ),
        ))
    for dodge in DODGE_STEPS:
        if dodge == 0:
            continue
        for direction_index in range(K_DIR):
            theta = 2.0 * math.pi * direction_index / K_DIR
            direction = np.array(
                [math.cos(theta), math.sin(theta)], np.float32,
            )
            for accel in V2_ACCELS:
                prefix = [
                    np.clip(
                        accel * direction, -SS.U_MAX, SS.U_MAX,
                    ).astype(np.float32)
                ] * dodge
                position = state[:2].copy()
                velocity = state[2:4].copy()
                for action in prefix:
                    position = (
                        position + SS.DT * velocity
                        + 0.5 * SS.DT ** 2 * action
                    )
                    velocity = velocity + SS.DT * action
                for v_cruise in CRUISE_SPEEDS:
                    controls = np.asarray(
                        prefix + _cruise_controls(
                            position, velocity, HORIZON - dodge, v_cruise,
                        ),
                        np.float32,
                    )
                    if controls.shape != (HORIZON, 2):
                        raise AssertionError(
                            "v2 recovery candidate must be H=10"
                        )
                    candidates.append((
                        controls,
                        dict(
                            kind="dodge_cruise",
                            dodge=int(dodge),
                            theta=round(theta, 6),
                            accel=float(accel),
                            v_cruise=float(v_cruise),
                        ),
                    ))
    return candidates


def _v2_objective(state, controls):
    """Declared v2 objective, including terminal toward-goal velocity."""
    segment = SM.rollout_positions(state, controls)
    velocity = np.asarray(state, np.float32).reshape(4)[2:4].copy()
    for action in np.asarray(controls, np.float32):
        velocity = velocity + SS.DT * action
    offset = SS.GOAL - segment[-1]
    distance = float(np.linalg.norm(offset))
    toward = (
        float(velocity @ (offset / distance))
        if distance > 1.0e-6 else 0.0
    )
    return distance - 0.5 * toward


def _prefilter(state, controls, ped_xy, ped_vel):
    """Cheap task-space/CV-collision filter and final goal distance."""
    segment = SM.rollout_positions(state, controls)
    if not SM.taskspace_ok(segment):
        return None
    prediction = SM.predict_pedestrians(
        ped_xy, ped_vel, H=len(controls),
    )
    if not SM.collision_free_time_indexed(segment, prediction):
        return None
    return float(np.linalg.norm(segment[-1] - SS.GOAL))


def _verify_payloads(payloads, executor):
    if executor is None:
        return list(map(SM.verify_in_worker, payloads))
    return list(executor.map(SM.verify_in_worker, payloads))


def evaluate_recovery_pool(
    state,
    ped_xy,
    ped_vel,
    gamma,
    family="v2",
    executor=None,
):
    """Prefilter, rank, and exact-verify one deterministic recovery pool.

    Returns diagnostics for every candidate actually submitted to the exact
    verifier.  ``certified`` contains at most the first two certified rows in
    objective order; ``best`` is its first row or ``None``.
    """
    if family not in ("v1", "v2"):
        raise ValueError("family must be 'v1' or 'v2'")
    state = np.asarray(state, np.float32).reshape(4)
    ped_xy = np.asarray(ped_xy, np.float32).reshape(-1, 2)
    ped_vel = np.asarray(ped_vel, np.float32).reshape(-1, 2)
    if ped_xy.shape != ped_vel.shape:
        raise ValueError("pedestrian positions and velocities do not align")

    generator = (
        recovery_candidates_v1
        if family == "v1" else recovery_candidates_v2
    )
    generated = generator(state)
    scored = []
    for controls, provenance in generated:
        goal_distance = _prefilter(
            state, controls, ped_xy, ped_vel,
        )
        if goal_distance is None:
            continue
        objective = (
            goal_distance
            if family == "v1" else _v2_objective(state, controls)
        )
        scored.append((
            float(objective),
            json.dumps(provenance, sort_keys=True),
            controls,
            provenance,
            float(goal_distance),
        ))
    scored.sort(key=lambda row: (row[0], row[1]))
    queried = scored[:PREVERIFY_CAP]
    payloads = [
        (
            0,
            rank,
            state,
            controls,
            ped_xy,
            ped_vel,
            float(gamma),
        )
        for rank, (_, _, controls, _, _) in enumerate(queried)
    ]
    results = _verify_payloads(payloads, executor)
    rows = []
    certified = []
    for (
        context_id,
        rank,
        result,
    ), (
        objective,
        _,
        controls,
        provenance,
        goal_distance,
    ) in zip(results, queried):
        if int(context_id) != 0 or int(rank) != len(rows):
            raise RuntimeError("verifier result order/identity mismatch")
        is_certified = bool(
            result.get("resolved")
            and int(result.get("y", 0)) == 1
            and bool(result.get("full_h"))
            and int(result.get("terminal_step", -1)) == HORIZON
        )
        row = dict(
            query_rank=int(rank),
            objective=float(objective),
            prefilter_goal_distance=float(goal_distance),
            controls=np.asarray(controls, np.float32),
            provenance=dict(provenance),
            verifier=result,
            certified=is_certified,
        )
        rows.append(row)
        if is_certified and len(certified) < RECOVERY_KEEP:
            certified.append(row)
    return dict(
        family=str(family),
        generated_count=len(generated),
        prefiltered_count=len(scored),
        queried_count=len(rows),
        queried=rows,
        certified=certified,
        best=certified[0] if certified else None,
        manifest=controller_manifest(),
    )


def _di_step(state, action):
    state = np.asarray(state, np.float32).reshape(4)
    action = np.asarray(action, np.float32).reshape(2)
    next_state = state.copy()
    next_state[:2] = (
        state[:2] + SS.DT * state[2:4]
        + 0.5 * SS.DT ** 2 * action
    )
    next_state[2:4] = state[2:4] + SS.DT * action
    return next_state


def rollout_recovery_controller(
    scenario_id,
    gamma,
    *,
    family="v2",
    scene_profile="double_density_velocity_ood",
    T=180,
    reach=0.5,
    scene_seed=0,
    executor=None,
):
    """Run the pure deterministic recovery controller receding-horizon.

    This function never samples or calls the flow model.  It exists only to
    test whether the declared deterministic family itself can dodge and make
    progress under the same SFM scene dynamics.
    """
    profile = SS.scene_profile(scene_profile)
    humans = SS.make_humans(
        int(scenario_id),
        int(scene_seed),
        int(profile["n_ped"]),
        tuple(profile["ped_speed_range"]),
    )
    state = np.zeros(4, np.float32)
    states = [state.copy()]
    controls = []
    trace = []
    minimum_clearance = float("inf")
    status = None

    for step in range(int(T)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        clearance = (
            float(np.linalg.norm(
                ped_xy - state[:2][None], axis=1,
            ).min() - SS.R_PED)
            if len(ped_xy) else float("inf")
        )
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            status = "collision"
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            status = "success"
            break

        pool = evaluate_recovery_pool(
            state,
            ped_xy,
            ped_vel,
            gamma,
            family=family,
            executor=executor,
        )
        row = dict(
            step=int(step),
            state=state.copy(),
            ped_xy=ped_xy.copy(),
            ped_vel=ped_vel.copy(),
            pool=pool,
            selected_action=None,
        )
        if pool["best"] is None:
            status = "nvp"
            trace.append(row)
            break

        action = np.asarray(
            pool["best"]["controls"][0], np.float32,
        )
        row["selected_action"] = action.copy()
        trace.append(row)
        controls.append(action.copy())
        state = _di_step(state, action)
        states.append(state.copy())
        SS.advance_humans(humans, state)

        post_xy, _ = SS.collect_humans(humans)
        post_clearance = (
            float(np.linalg.norm(
                post_xy - state[:2][None], axis=1,
            ).min() - SS.R_PED)
            if len(post_xy) else float("inf")
        )
        minimum_clearance = min(minimum_clearance, post_clearance)
        if post_clearance < 0.0:
            status = "collision"
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            status = "success"
            break

    if status is None:
        status = "timeout"
    return dict(
        controller="claude_deterministic_recovery",
        family=str(family),
        scenario_id=int(scenario_id),
        gamma=float(gamma),
        scene_profile=str(scene_profile),
        status=status,
        success=status == "success",
        collision=status == "collision",
        nvp=status == "nvp",
        timeout=status == "timeout",
        steps=len(controls),
        time_to_goal=(
            len(controls) * SS.DT if status == "success" else None
        ),
        minimum_clearance=float(minimum_clearance),
        path=np.asarray(states, np.float32)[:, :2],
        states=np.asarray(states, np.float32),
        controls=np.asarray(controls, np.float32).reshape(-1, 2),
        trace=trace,
        manifest=controller_manifest(),
    )
