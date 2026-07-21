"""Canonical exact moving-pedestrian full-H verifier for every B1 query.

For each real pedestrian and each artificial sensing-boundary disk, the face
normal is solved exactly in two-dimensional angle space.  There is no angular
grid.  The outer sensing boundary always uses exactly 16 canonical anchors.
Visualization imports this module, so rendered certificates and training
labels cannot silently use different solvers.
"""
from __future__ import annotations

import math
import numpy as np

import _paths  # noqa: F401
import verifier_polytope as VP
import sfm_scene as SS


ARTIFICIAL_FACES = 16
ANGLE_TOL = 2.0e-10


def verifier_manifest():
    return dict(
        solver="exact_2d_angular_interval_socp", angular_grid=False,
        K_artificial=ARTIFICIAL_FACES, horizon=10, rho_art=0.16,
        m_min=1.0e-4, r_pad=1.3,
        effective_radius="max(sensing_radius, r_pad * max candidate displacement)",
        pedestrian_prediction="constant velocity over t=0..H",
    )


def predict_pedestrians(ped_xy, ped_vel, H=10, dt=SS.DT):
    xy = np.asarray(ped_xy, np.float32).reshape(-1, 2)
    velocity = np.asarray(ped_vel, np.float32).reshape(-1, 2)
    if xy.shape != velocity.shape:
        raise ValueError("pedestrian positions and velocities do not align")
    time = np.arange(int(H) + 1, dtype=np.float32)[:, None, None] * float(dt)
    return xy[None] + time * velocity[None]


def rollout_positions(state, controls, dt=SS.DT):
    current = np.asarray(state, np.float32).reshape(4).copy()
    positions = [current[:2].copy()]
    for action in np.asarray(controls, np.float32).reshape(-1, 2):
        current[:2] += float(dt) * current[2:4] + 0.5 * float(dt) ** 2 * action
        current[2:4] += float(dt) * action
        positions.append(current[:2].copy())
    return np.asarray(positions, np.float32)


def taskspace_ok(segment, lo=SS.TASK_LO, hi=SS.TASK_HI):
    value = np.asarray(segment, float)
    return bool(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all()
                and (value >= float(lo)).all() and (value <= float(hi)).all())


def collision_free_time_indexed(segment, pedestrians, radius=SS.R_PED):
    robot = np.asarray(segment, float)
    peds = np.asarray(pedestrians, float)
    if len(robot) != len(peds):
        raise ValueError("robot/pedestrian horizon mismatch")
    if peds.shape[1] == 0:
        return True
    distance = np.linalg.norm(robot[:, None, :] - peds, axis=2)
    return bool(float(distance.min()) >= float(radius) - 1.0e-9)


def _wrap(theta):
    return float(theta) % (2.0 * math.pi)


def _angular_constraint(vector, threshold):
    """Closed arc endpoints for ``unit(theta)^T vector >= threshold``."""
    vector = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    threshold = float(threshold)
    if norm <= ANGLE_TOL:
        return () if threshold <= ANGLE_TOL else None
    if threshold > norm + ANGLE_TOL:
        return None
    if threshold <= -norm + ANGLE_TOL:
        return ()
    halfwidth = math.acos(float(np.clip(threshold / norm, -1.0, 1.0)))
    center = math.atan2(float(vector[1]), float(vector[0]))
    return _wrap(center - halfwidth), _wrap(center + halfwidth)


def _is_feasible(theta, inequalities, tol=2.0e-9):
    normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    return all(float(normal @ vector) >= float(threshold) - float(tol)
               for vector, threshold in inequalities)


def solve_moving_face(robot_centered, pedestrian_centered, radius, beta, label,
                      *, m_min=1.0e-4, kind="real-moving"):
    """Solve one independent positive max-margin moving-disk SOCP block."""
    robot = np.asarray(robot_centered, dtype=float).reshape(-1, 2)
    pedestrian = np.asarray(pedestrian_centered, dtype=float).reshape(-1, 2)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if len(robot) != len(pedestrian) or len(robot) != len(beta):
        raise ValueError("moving face horizons do not align")
    if len(robot) < 2 or np.any(beta[1:] <= 0.0):
        raise ValueError("moving face needs at least one positive-beta horizon")

    inequalities = [(center, float(radius) + float(m_min)) for center in pedestrian]
    for horizon in range(1, len(robot)):
        for center in pedestrian:
            inequalities.append((
                float(beta[horizon]) * center - robot[horizon],
                float(beta[horizon]) * float(radius),
            ))

    endpoints = []
    for vector, threshold in inequalities:
        arc = _angular_constraint(vector, threshold)
        if arc is None:
            return VP.Face(np.array([1.0, 0.0]), 0.0, kind, label, feasible=False)
        endpoints.extend(arc)

    candidates = list(endpoints)
    ordered = sorted(set([0.0] + [_wrap(value) for value in endpoints]))
    cyclic = ordered + [ordered[0] + 2.0 * math.pi]
    candidates.extend(_wrap(0.5 * (left + right)) for left, right in zip(cyclic, cyclic[1:]))
    for center in pedestrian:
        angle = math.atan2(float(center[1]), float(center[0]))
        candidates.extend((_wrap(angle), _wrap(angle + math.pi)))
    for first in range(len(pedestrian)):
        for second in range(first + 1, len(pedestrian)):
            delta = pedestrian[first] - pedestrian[second]
            if float(np.linalg.norm(delta)) <= ANGLE_TOL:
                continue
            angle = math.atan2(float(delta[1]), float(delta[0])) + 0.5 * math.pi
            candidates.extend((_wrap(angle), _wrap(angle + math.pi)))

    feasible = []
    for theta in candidates:
        theta = _wrap(theta)
        if _is_feasible(theta, inequalities):
            normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
            margin = float(np.min(pedestrian @ normal) - float(radius))
            feasible.append((margin, theta, normal))
    if not feasible:
        return VP.Face(np.array([1.0, 0.0]), 0.0, kind, label, feasible=False)
    margin, theta, normal = max(feasible, key=lambda row: (row[0], -row[1]))
    return VP.Face(
        normal, float(margin), kind, label, coefficient=1.0,
        feasible=bool(margin >= float(m_min) - 2.0e-8), interval=None,
    )


def certify_moving_window(segment, pedestrians, gamma, *, K=ARTIFICIAL_FACES,
                          rho_art=0.16, m_min=1.0e-4, r_pad=1.3):
    if int(K) != ARTIFICIAL_FACES:
        raise ValueError(f"faithful B1 verifier requires K={ARTIFICIAL_FACES}")
    robot = np.asarray(segment, float)
    peds = np.asarray(pedestrians, float)
    if robot.ndim != 2 or robot.shape[1] != 2 or peds.ndim != 3 or peds.shape[2] != 2:
        raise ValueError("invalid moving-window shapes")
    if len(robot) != len(peds):
        raise ValueError("moving-window horizons do not align")
    center = robot[0]
    robot_c = robot - center
    radius = max(float(SS.R_SENSE), float(r_pad) * float(np.linalg.norm(robot_c, axis=1).max()))
    if len(robot) == 1:
        # Reaching the goal at the current state creates an absorbing empty
        # verification horizon.  It is valid but never replay-eligible.
        return True, [], dict(
            solver="exact_2d_angular_interval_socp", angular_grid=False,
            slack=float("inf"), worst_t=0, R_eff=float(radius),
            n_real=0, n_real_feasible=0, n_artificial=0,
            n_artificial_feasible=0, K_artificial=ARTIFICIAL_FACES,
            empty_terminal_prefix=True,
        )
    alpha = (1.0 - float(gamma)) ** np.arange(len(robot), dtype=float)
    beta = 1.0 - alpha
    faces = []
    for index in range(peds.shape[1]):
        ped_c = peds[:, index] - center
        if float((np.linalg.norm(ped_c, axis=1) - SS.R_PED).min()) <= radius:
            faces.append(solve_moving_face(
                robot_c, ped_c, SS.R_PED, beta, f"ped{index}", m_min=m_min,
            ))
    for index, (x, y, obstacle_radius) in enumerate(
            VP.artificial_obstacles(radius, ARTIFICIAL_FACES, float(rho_art))):
        repeated = np.repeat(np.array([[x, y]], dtype=float), len(robot_c), axis=0)
        faces.append(solve_moving_face(
            robot_c, repeated, obstacle_radius, beta, f"art{index}",
            m_min=m_min, kind="artificial",
        ))
    ok, slack, worst_t = VP.check_certificate(faces, robot_c, alpha, include_start=False)
    real = [face for face in faces if face.kind == "real-moving"]
    artificial = [face for face in faces if face.kind == "artificial"]
    return bool(ok), faces, dict(
        solver="exact_2d_angular_interval_socp", angular_grid=False,
        slack=float(slack), worst_t=int(worst_t), R_eff=float(radius),
        n_real=len(real), n_real_feasible=sum(bool(face.feasible) for face in real),
        n_artificial=len(artificial),
        n_artificial_feasible=sum(bool(face.feasible) for face in artificial),
        K_artificial=ARTIFICIAL_FACES,
    )


def verify_query(state, controls, ped_xy, ped_vel, gamma, *, reach=0.5):
    """Resolve y without performance/cost terms; errors are explicit and non-storable."""
    try:
        controls = np.asarray(controls, np.float32).reshape(-1, 2)
        if len(controls) != 10:
            raise ValueError("B1 verifier requires H=10")
        robot = rollout_positions(state, controls)
        pedestrian = predict_pedestrians(ped_xy, ped_vel, H=len(controls))
        goal_distance = np.linalg.norm(robot - SS.GOAL[None], axis=1)
        reached = np.flatnonzero(goal_distance < float(reach))
        terminal_step = int(reached[0]) if len(reached) else len(controls)
        # A goal hit defines an absorbing terminal prefix. Post-goal repeats are not verified or replayed.
        prefix_robot = robot[:terminal_step + 1]
        prefix_pedestrian = pedestrian[:terminal_step + 1]
        task = taskspace_ok(prefix_robot)
        collision = collision_free_time_indexed(prefix_robot, prefix_pedestrian)
        certificate, faces, diagnostics = certify_moving_window(
            prefix_robot, prefix_pedestrian, gamma,
        )
        y = bool(task and collision and certificate)
        return dict(
            resolved=True, error=None, y=int(y), taskspace=bool(task),
            collision_free=bool(collision), certificate=bool(certificate),
            full_h=bool(terminal_step == len(controls)), terminal_step=terminal_step,
            train_eligible=bool(y and terminal_step == len(controls)),
            segment=robot, pedestrian_prediction=pedestrian, faces=faces,
            diagnostics=diagnostics,
        )
    except Exception as error:  # worker boundary: a failed solver/query enters no store.
        return dict(resolved=False, error=f"{type(error).__name__}: {error}")


def verify_in_worker(payload):
    context_id, candidate_id, state, controls, ped_xy, ped_vel, gamma = payload
    result = verify_query(state, controls, ped_xy, ped_vel, gamma)
    return int(context_id), int(candidate_id), result
