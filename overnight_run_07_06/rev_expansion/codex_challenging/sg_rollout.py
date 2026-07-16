#!/usr/bin/env python3
"""Plain receding-horizon rollout for the explicit start+goal policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _paths  # noqa: F401,E402
import grid_feats as GF  # noqa: E402
from di_grid_viz import di_step  # noqa: E402

import gen_uniform_data as SEEDS  # noqa: E402


def classify_path(path: np.ndarray, goal: np.ndarray, env, reach: float) -> dict:
    positions = np.asarray(path, dtype=np.float32)
    obstacles = env.obstacles.detach().cpu().numpy()
    clearance = (
        np.linalg.norm(positions[:, None] - obstacles[None, :, :2], axis=2)
        - obstacles[None, :, 2]
        - float(env.r_robot)
    )
    min_clearance = float(clearance.min())
    collision = bool((clearance.min(axis=1) < 0.0).any())
    in_taskspace = bool(((positions >= 0.0) & (positions <= 5.0)).all())
    endpoint_distance = float(np.linalg.norm(positions[-1] - goal))
    reached = endpoint_distance < reach
    return {
        "success": bool(reached and not collision and in_taskspace),
        "reached": bool(reached),
        "collision": collision,
        "in_taskspace": in_taskspace,
        "endpoint_distance": endpoint_distance,
        "min_clearance": min_clearance,
    }


@torch.inference_mode()
def rollout_sg(
    policy,
    start: np.ndarray,
    goal: np.ndarray,
    gamma: float,
    *,
    seed: int,
    T: int = 250,
    reach: float = 0.2,
    nfe: int = 12,
    temp: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict:
    """Deploy one unguided policy sample stream on an arbitrary pair."""
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    np.random.seed(seed % (2**32))

    env = SEEDS.make_walled_env(8)
    obstacles = env.obstacles.detach().cpu().numpy()
    start = np.asarray(start, dtype=np.float32)[:2]
    goal = np.asarray(goal, dtype=np.float32)[:2]
    state = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float32)
    history: list[np.ndarray] = []
    path = [state[:2].copy()]
    controls = []
    dead_reason = None

    for _step in range(T):
        grid = torch.tensor(
            GF.axis_grid(state[:2], obstacles, float(env.r_robot)),
            dtype=torch.float32,
            device=device,
        )
        low5 = torch.tensor(GF.low5(state, goal, gamma), dtype=torch.float32, device=device)
        hist = torch.tensor(
            GF.hist_pad(
                np.asarray(history[-GF.K_HIST:], dtype=np.float32).reshape(-1, 2)
                if history
                else np.empty((0, 2), dtype=np.float32),
                GF.K_HIST,
            ),
            dtype=torch.float32,
            device=device,
        )
        window = policy.sample_window(
            grid,
            low5,
            hist,
            n=1,
            temp=temp,
            nfe=nfe,
        )[0]
        action = window[0].detach().cpu().numpy().astype(np.float32)
        state = di_step(state, action, dt=float(env.dt)).astype(np.float32)
        history.append(action)
        controls.append(action)
        path.append(state[:2].copy())

        if np.linalg.norm(state[:2] - goal) < reach:
            break
        if (state[:2] < 0.0).any() or (state[:2] > 5.0).any():
            dead_reason = "out_of_bounds"
            break
        instantaneous_clearance = (
            np.linalg.norm(state[:2][None] - obstacles[:, :2], axis=1)
            - obstacles[:, 2]
            - float(env.r_robot)
        ).min()
        if instantaneous_clearance < 0.0:
            dead_reason = "collision"
            break

    path_array = np.asarray(path, dtype=np.float32)
    status = classify_path(path_array, goal, env, reach)
    if dead_reason is None and not status["reached"] and len(controls) >= T:
        dead_reason = "timeout"
    return {
        **status,
        "path": path_array,
        "controls": np.asarray(controls, dtype=np.float32),
        "steps": len(controls),
        "dead_reason": dead_reason,
        "seed": int(seed),
        "gamma": float(gamma),
        "start": start,
        "goal": goal,
    }
