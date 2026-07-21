"""Exact 2-D moving-pedestrian SOCP used only by paper visualizations.

The expansion checkpoints were trained with :mod:`sfm_metrics2`, whose legacy
moving-face implementation searches a finite angular grid and uses 12
artificial anchors.  This module does not change those checkpoints.  It solves
the same positive max-margin face problem exactly in angle space and fixes the
visualization contract to 16 artificial sensing-boundary anchors.

For one pedestrian with predicted centers ``d_t`` (robot-centered), the SOCP
block is

    maximize m
    s.t. a^T q_h <= beta_h m,
         r ||a|| <= a^T d_t - m   for every h,t,
         ||a|| <= 1, m >= m_min.

Positive max-margin makes ``||a||=1`` at the optimum.  Each remaining
constraint is therefore a closed angular interval.  We intersect those
intervals analytically and evaluate every possible stationary/switch/boundary
optimum; there is no theta grid.
"""
from __future__ import annotations

import math

import numpy as np

import _paths  # noqa: F401
import verifier_polytope as VP
import sfm_metrics2 as LEGACY
import sfm_scene as SS


ARTIFICIAL_FACES = 16
ANGLE_TOL = 2.0e-10


def _wrap(theta):
    return float(theta) % (2.0 * math.pi)


def _angular_constraint(vector, threshold):
    """Return the endpoints of ``unit(theta)^T vector >= threshold``.

    ``None`` means the constraint is infeasible; an empty tuple means it is
    vacuous.  Otherwise the returned pair are the two closed arc endpoints.
    """
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
    return (_wrap(center - halfwidth), _wrap(center + halfwidth))


def _is_feasible(theta, inequalities, tol=2.0e-9):
    normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    return all(float(normal @ vector) >= float(threshold) - float(tol)
               for vector, threshold in inequalities)


def solve_moving_face(robot_centered, pedestrian_centered, radius, beta, label,
                      *, m_min=1.0e-4, kind="real-moving"):
    """Solve one independent moving-disk positive max-margin SOCP block."""
    robot = np.asarray(robot_centered, dtype=float).reshape(-1, 2)
    pedestrian = np.asarray(pedestrian_centered, dtype=float).reshape(-1, 2)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if len(robot) != len(pedestrian) or len(robot) != len(beta):
        raise ValueError("moving face horizons do not align")
    if len(robot) < 2 or np.any(beta[1:] <= 0.0):
        raise ValueError("moving face needs at least one positive-beta horizon")

    inequalities = []
    # m >= m_min and m <= a^T d_t-r||a||, with ||a||=1.
    for center in pedestrian:
        inequalities.append((center, float(radius) + float(m_min)))
    # Eliminate m between every trajectory and moving-disk constraint.
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
    # A feasible-set component can span theta=0; midpoints make its existence
    # explicit without a dense scan and are also useful for numerical ties.
    ordered = sorted(set([0.0] + [_wrap(value) for value in endpoints]))
    if not ordered:
        ordered = [0.0]
    cyclic = ordered + [ordered[0] + 2.0 * math.pi]
    candidates.extend(_wrap(0.5 * (left + right)) for left, right in zip(cyclic, cyclic[1:]))

    # On each feasible interval m(theta)=min_t a(theta)^T d_t-r.  Its maximum
    # occurs at an interval boundary, a stationary direction of one branch, or
    # a switch between two branches.
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
    """Solve all moving-pedestrian and K=16 artificial SOCP face blocks."""
    if int(K) != ARTIFICIAL_FACES:
        raise ValueError(f"faithful visualization requires K={ARTIFICIAL_FACES}")
    robot = np.asarray(segment, dtype=float)
    peds = np.asarray(pedestrians, dtype=float)
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
            faces.append(solve_moving_face(
                robot_c, ped_c, SS.R_PED, beta, f"ped{index}", m_min=m_min,
            ))

    # Canonical artificial anchors: M_K=R cos(pi/K), center=(M_K+rho)n.
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
    """Faithful visualization-only query verification with exact K=16 faces."""
    try:
        controls = np.asarray(controls, np.float32).reshape(-1, 2)
        if len(controls) != 10:
            raise ValueError("B1 verifier requires H=10")
        robot = LEGACY.rollout_positions(state, controls)
        pedestrian = LEGACY.predict_pedestrians(ped_xy, ped_vel, H=len(controls))
        goal_distance = np.linalg.norm(robot - SS.GOAL[None], axis=1)
        reached = np.flatnonzero(goal_distance < float(reach))
        terminal_step = int(reached[0]) if len(reached) else len(controls)
        prefix_robot = robot[:terminal_step + 1]
        prefix_pedestrian = pedestrian[:terminal_step + 1]
        task = LEGACY.taskspace_ok(prefix_robot)
        collision = LEGACY.collision_free_time_indexed(prefix_robot, prefix_pedestrian)
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
    except Exception as error:
        return dict(resolved=False, error=f"{type(error).__name__}: {error}")


def verify_in_worker(payload):
    """ProcessPool-compatible worker for diagnostic-only faithful reruns."""
    context_id, candidate_id, state, controls, ped_xy, ped_vel, gamma, _legacy_n_theta = payload
    result = verify_query(state, controls, ped_xy, ped_vel, gamma)
    return int(context_id), int(candidate_id), result
