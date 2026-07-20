"""Deterministic moving-pedestrian SFM scene used by Hp10/B1."""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from cfm_mppi.utils import HumanAgent
from cfm_mppi.evaluation.render_sfm_kazuki_policy import _advance_humans, _collect_humans

GOAL = np.array([6.0, 6.0], dtype=np.float32)
R_PED = 0.2
U_MAX = 2.0
DT = 0.1
R_SENSE = 2.0
TASK_LO = -0.5
TASK_HI = 6.5
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
LEGACY_PED_SPEED_RANGE = (0.8, 1.3)
ID_PED_SPEED_RANGE = (0.5, 1.0)
OOD_PED_SPEED_RANGE = (1.0, 1.5)


def _validate_speed_range(speed_range):
    lo, hi = map(float, speed_range)
    if not (0.0 <= lo < hi):
        raise ValueError(f"invalid pedestrian speed range: {speed_range}")
    return lo, hi


def _remap_human_speed(human, speed_range):
    lo, hi = _validate_speed_range(speed_range)
    q = (float(human.sfm_des_speed) - LEGACY_PED_SPEED_RANGE[0]) / (
        LEGACY_PED_SPEED_RANGE[1] - LEGACY_PED_SPEED_RANGE[0]
    )
    human.sfm_des_speed = float(lo + np.clip(q, 0.0, 1.0) * (hi - lo))
    return human


def make_humans(scenario_id, seed=0, n_ped=20, speed_range=OOD_PED_SPEED_RANGE):
    """A scenario ID fixes starts, goals, and pedestrian speed quantiles."""
    rng = np.random.RandomState(int(seed) + int(scenario_id))
    return [_remap_human_speed(HumanAgent(GOAL, random_generator=rng), speed_range) for _ in range(n_ped)]


def collect_humans(humans):
    return _collect_humans(humans)


def advance_humans(humans, robot_state):
    _advance_humans(
        humans,
        robot_xy=np.asarray(robot_state[:2], np.float32).copy(),
        robot_control_si=np.asarray(robot_state[2:4], np.float32).copy(),
    )


def scenario_snapshot(scenario_ids, *, n_ped=20, seed=0, speed_range=OOD_PED_SPEED_RANGE):
    rows = []
    for scenario_id in map(int, scenario_ids):
        humans = make_humans(scenario_id, seed, n_ped, speed_range)
        xy, vel = collect_humans(humans)
        rows.append(dict(scenario_id=scenario_id, ped_xy=xy.tolist(), ped_vel=vel.tolist()))
    return dict(seed=int(seed), n_ped=int(n_ped), speed_range=list(map(float, speed_range)), rows=rows)
