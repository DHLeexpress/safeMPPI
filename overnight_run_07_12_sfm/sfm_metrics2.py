"""Moving-pedestrian full-H verifier used by every selected B1 query."""
from __future__ import annotations

import math
import numpy as np

import _paths  # noqa: F401
import verifier_polytope as VP
import sfm_scene as SS


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


def _moving_face(robot_centered, ped_centered, radius, beta, label, n_theta=180, m_min=1.0e-4):
    theta = np.linspace(-math.pi, math.pi, int(n_theta), endpoint=False)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    upper = (normals @ ped_centered.T).min(axis=1) - float(radius)
    if len(robot_centered) > 1:
        lower = ((normals @ robot_centered[1:].T) / beta[1:][None]).max(axis=1)
    else:
        lower = np.full(len(normals), -np.inf)
    lower = np.maximum(lower, float(m_min))
    feasible = lower <= upper + 1.0e-8
    if not feasible.any():
        return VP.Face(np.array([1.0, 0.0]), 0.0, "real-moving", label, feasible=False)
    index = int(np.argmax(np.where(feasible, upper, -np.inf)))
    return VP.Face(normals[index], float(upper[index]), "real-moving", label, feasible=True)


def certify_moving_window(segment, pedestrians, gamma, *, n_theta=180, K=12,
                          rho_art=0.16, m_min=1.0e-4, r_pad=1.3):
    robot = np.asarray(segment, float)
    peds = np.asarray(pedestrians, float)
    if robot.ndim != 2 or robot.shape[1] != 2 or peds.ndim != 3 or peds.shape[2] != 2:
        raise ValueError("invalid moving-window shapes")
    if len(robot) != len(peds):
        raise ValueError("moving-window horizons do not align")
    center = robot[0]
    robot_c = robot - center
    radius = max(float(SS.R_SENSE), float(r_pad) * float(np.linalg.norm(robot_c, axis=1).max()))
    alpha = (1.0 - float(gamma)) ** np.arange(len(robot), dtype=float)
    beta = 1.0 - alpha
    faces = []
    for index in range(peds.shape[1]):
        ped_c = peds[:, index] - center
        if float((np.linalg.norm(ped_c, axis=1) - SS.R_PED).min()) <= radius:
            faces.append(_moving_face(robot_c, ped_c, SS.R_PED, beta, f"ped{index}", n_theta, m_min))
    artificial, _ = VP.build_faces(
        robot_c, [], float(gamma), R=radius, K=K, rho_art=rho_art,
        m_min=m_min, n_theta=n_theta,
    )
    faces.extend(artificial)
    ok, slack, worst_t = VP.check_certificate(faces, robot_c, alpha, include_start=False)
    real = [face for face in faces if face.kind == "real-moving"]
    return bool(ok), faces, dict(
        slack=float(slack), worst_t=int(worst_t), R_eff=radius,
        n_real=len(real), n_real_feasible=sum(bool(face.feasible) for face in real),
    )


def verify_query(state, controls, ped_xy, ped_vel, gamma, *, reach=0.5, n_theta=180):
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
            prefix_robot, prefix_pedestrian, gamma, n_theta=n_theta,
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
    context_id, candidate_id, state, controls, ped_xy, ped_vel, gamma, n_theta = payload
    result = verify_query(state, controls, ped_xy, ped_vel, gamma, n_theta=n_theta)
    return int(context_id), int(candidate_id), result
