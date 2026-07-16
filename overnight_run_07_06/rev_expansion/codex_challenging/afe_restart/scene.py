"""Exact ID/OOD stadium construction and policy contexts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import gen_uniform_data as seed_geometry
import grid_feats as grid_features

from .schemas import QueryContext


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
START = np.asarray((0.5, 0.5), dtype=np.float32)
GOAL = np.asarray((4.5, 4.5), dtype=np.float32)
GIANT_CENTER = np.asarray((2.5, 2.5), dtype=np.float32)
CENTRAL_CENTERS = np.asarray(
    ((2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)), dtype=np.float32,
)


def _set_endpoints(env, start: np.ndarray = START, goal: np.ndarray = GOAL):
    env.x0 = torch.as_tensor(
        [float(start[0]), float(start[1]), 0.0, 0.0], dtype=env.x0.dtype, device=env.x0.device,
    )
    env.goal = torch.as_tensor(goal, dtype=env.goal.dtype, device=env.goal.device)
    return env


def make_id_scene(*, start: np.ndarray = START, goal: np.ndarray = GOAL):
    """Ordinary symmetric 4x4 ID stadium with the established eight plugs."""
    return _set_endpoints(seed_geometry.make_walled_env(8), start, goal)


def make_ood_scene(
    radius: float = 1.2, *, start: np.ndarray = START, goal: np.ndarray = GOAL,
):
    """Replace exactly the four central ID circles by one giant obstacle."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    env = make_id_scene(start=start, goal=goal)
    obstacles = env.obstacles.detach().cpu().numpy()
    central = np.zeros(len(obstacles), dtype=bool)
    for center in CENTRAL_CENTERS:
        central |= np.all(np.isclose(obstacles[:, :2], center[None], atol=1e-7), axis=1)
    if int(central.sum()) != 4:
        raise RuntimeError(f"expected exactly four central obstacles, got {int(central.sum())}")
    giant = np.asarray([[*GIANT_CENTER, float(radius)]], dtype=np.float32)
    replaced = np.concatenate((obstacles[~central], giant), axis=0)
    env.obstacles = torch.as_tensor(replaced, dtype=env.obstacles.dtype, device=env.obstacles.device)
    env.obs_vel = torch.zeros(len(replaced), 2, dtype=env.obstacles.dtype, device=env.obstacles.device)
    return env


def context_from_state(
    state: np.ndarray,
    goal: np.ndarray,
    gamma: float,
    executed_controls: list[np.ndarray] | np.ndarray,
    env,
) -> QueryContext:
    """Build the original endpoint-free ``low5 + E(H_P)`` context."""
    state = np.asarray(state, dtype=np.float32)
    controls = np.asarray(executed_controls, dtype=np.float32).reshape(-1, 2)
    obstacle_array = env.obstacles.detach().cpu().numpy()
    grid, low5, history = grid_features.featurize(
        state,
        np.asarray(goal, dtype=np.float32),
        float(gamma),
        controls,
        obstacle_array,
        float(env.r_robot),
        K=grid_features.K_HIST,
    )
    return QueryContext(grid=grid, low5=low5, hist=history)


def minimum_endpoint_clearance(env) -> dict[str, float]:
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)

    def clearance(point: np.ndarray) -> float:
        return float(
            (np.linalg.norm(obstacles[:, :2] - point[None], axis=1) - obstacles[:, 2] - rr).min()
        )

    return {"start": clearance(START), "goal": clearance(GOAL)}

