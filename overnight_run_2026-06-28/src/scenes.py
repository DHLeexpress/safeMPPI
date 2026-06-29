"""The two fixed double-integrator scenes + helpers (single obstacle; two-obstacle narrow gap)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scene:
    name: str
    x0: np.ndarray            # [4] (px,py,vx,vy)
    goal: np.ndarray          # [2]
    obstacles: np.ndarray     # [N,3] (cx,cy,radius)
    T: int = 40
    dt: float = 0.12
    u_max: float = 3.5
    r_robot: float = 0.2
    sensing_range: float = 3.0   # robot-centered ball radius R
    xlim: tuple = (-1.0, 7.0)
    ylim: tuple = (-3.2, 3.2)


def make_scene(name: str) -> Scene:
    if name == "single":
        return Scene("single", np.array([0., 0., 0., 0.]), np.array([6., 0.]),
                     np.array([[3.0, 0.0, 0.8]]), ylim=(-3.0, 3.0))
    if name == "gap":
        return Scene("gap", np.array([0., 0., 0., 0.]), np.array([6., 0.]),
                     np.array([[3.0, 1.0, 0.6], [3.0, -1.0, 0.6]]), ylim=(-3.2, 3.2))
    if name == "clutter":
        # two staggered "gates" between start and goal -> many homotopy classes (around / through gaps)
        obs = np.array([[2.5, 1.3, 0.55], [2.5, -1.3, 0.55],          # gate 1: gap at y=0, or around
                        [5.0, 0.0, 0.55], [5.0, 2.1, 0.55], [5.0, -2.1, 0.55]])  # gate 2: gaps at y≈±1.05
        return Scene("clutter", np.array([0., 0., 0., 0.]), np.array([8., 0.]),
                     obs, T=50, sensing_range=3.0, xlim=(-1.0, 9.0), ylim=(-3.3, 3.3))
    raise ValueError(name)


def di_step(state, u, dt):
    p, v = state[:2], state[2:4]
    return np.concatenate([p + dt * v + 0.5 * dt * dt * u, v + dt * u])


def make_trajectory(scene: Scene, lateral: float, kp=6.0, kd=4.0, sigma=0.0, seed=0) -> np.ndarray:
    """Deterministic-ish PD-to-waypoint rollout: straight start->goal + lateral sine bump.
    lateral>0 bends to +y (left), <0 to -y (right), ~0 goes straight (through the gap). -> states [T+1,4]."""
    p0, g = scene.x0[:2], scene.goal
    d = (g - p0); d = d / (np.linalg.norm(d) + 1e-9)
    e = np.array([-d[1], d[0]])
    s = np.linspace(0, 1, scene.T + 1)
    base = p0[None] + s[:, None] * (g - p0)[None]
    p_des = base + lateral * np.sin(np.pi * s)[:, None] * e[None]
    v_des = np.zeros_like(p_des); v_des[:-1] = (p_des[1:] - p_des[:-1]) / scene.dt
    rng = np.random.default_rng(seed)
    x = scene.x0.copy().astype(float)
    out = [x.copy()]
    for t in range(scene.T):
        u = kp * (p_des[t] - x[:2]) + kd * (v_des[t] - x[2:4])
        if sigma:
            u = u + sigma * rng.standard_normal(2)
        u = np.clip(u, -scene.u_max, scene.u_max)
        x = di_step(x, u, scene.dt)
        out.append(x.copy())
    return np.stack(out)


def clearance(states_xy: np.ndarray, scene: Scene) -> np.ndarray:
    """Min signed clearance over obstacles per state. states_xy [...,2] -> [...]."""
    o = scene.obstacles
    d = np.linalg.norm(states_xy[..., None, :] - o[:, :2], axis=-1) - (o[:, 2] + scene.r_robot)
    return d.min(-1)


def demo_trajectories(scene: Scene):
    """A few clean example trajectories per scene (for the verifier demo)."""
    if scene.name == "single":
        return {"right": make_trajectory(scene, -1.5), "left": make_trajectory(scene, 1.5)}
    return {"gap": make_trajectory(scene, 0.0),
            "left": make_trajectory(scene, 2.0), "right": make_trajectory(scene, -2.0)}
