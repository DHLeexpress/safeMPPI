"""Static cluttered environments for the Safe-Flow-Expansion smoke test (Pillar 1).

Adapts the safeGPC cluttered-obstacle field (``random_spheres_2d``) to the frozen
``best_area_mode4`` planner's meter length-scale: a ~5.6 m horizontal workspace with ~10
static circular obstacles.  The double-integrator params (dt, u_max, r_robot) MATCH the
planner so an open-loop rollout of the SafeMPPI-executed controls reproduces its states.

``Env`` is reused verbatim from ``overnight_run_today/src/dynamics.py`` (the same dataclass
the safe-flow loop, verifier and descriptors already consume).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

import _paths  # noqa: F401  (sets sys.path)
from dynamics import Env  # from overnight_run_today/src


# --------------------------------------------------------------------------- clutter gen
# Copied verbatim from /home/dohyun/projects/safeGPC/utils/obstacle_generation_2D.py so this
# folder stays self-contained (numpy-only, no cross-repo path dependency).
def random_spheres_2d(
    n_circles: int,
    box_min: float = -5.0,
    box_max: float = 5.0,
    r_min: float = 0.5,
    r_max: float = 1.5,
    clearance: float = 0.5,
    start: Tuple[float, float] = (-5.0, -5.0),
    goal: Tuple[float, float] = (5.0, 5.0),
    max_tries: int = 10000,
) -> np.ndarray:
    """Return array (n,3): [cx, cy, r]. Pairwise non-overlap + clearance from endpoints."""
    centers: List[Tuple[float, float, float]] = []
    tries = 0
    start = np.asarray(start, dtype=float)[:2]
    goal = np.asarray(goal, dtype=float)[:2]
    while len(centers) < n_circles and tries < max_tries:
        tries += 1
        r = np.random.uniform(r_min, r_max)
        cx = np.random.uniform(box_min + r, box_max - r)
        cy = np.random.uniform(box_min + r, box_max - r)
        c = np.array([cx, cy])
        ok = True
        if np.linalg.norm(c - start) < (r + clearance):
            ok = False
        if np.linalg.norm(c - goal) < (r + clearance):
            ok = False
        if ok:
            for (px, py, pr) in centers:
                if np.linalg.norm(c - np.array([px, py])) < (r + pr + clearance):
                    ok = False
                    break
        if ok:
            centers.append((cx, cy, r))
    if len(centers) < n_circles:
        raise RuntimeError("Could not place all obstacles with given constraints.")
    return np.array(centers, dtype=float)


# --------------------------------------------------------------------------- env factory
def make_clutter_env(
    seed: int,
    n_obs: int = 10,
    box: float = 2.8,
    r_min: float = 0.25,
    r_max: float = 0.5,
    clearance: float = 0.4,
    T_exec: int = 80,
    dt: float = 0.1,
    u_max: float = 2.0,
    r_robot: float = 0.2,
    device: str = "cpu",
) -> Env:
    """One static cluttered scene, horizontal start->goal chord.

    Chord is horizontal (start left, goal right) so ``Env.ylim`` serves double duty as the
    lateral coverage range for ``descriptors`` (matches the single/gap toy convention).
    """
    rng = np.random.RandomState(seed)
    state = np.random.get_state()
    np.random.set_state(rng.get_state())  # random_spheres_2d uses the global np.random
    try:
        start = (-box + 0.2, 0.0)
        goal = (box - 0.2, 0.0)
        obs = random_spheres_2d(
            n_obs, box_min=-box, box_max=box, r_min=r_min, r_max=r_max,
            clearance=clearance, start=start, goal=goal,
        )
    finally:
        np.random.set_state(state)

    env = Env(
        name="clutter",
        x0=torch.tensor([start[0], start[1], 0.0, 0.0], dtype=torch.float32),
        goal=torch.tensor([goal[0], goal[1]], dtype=torch.float32),
        obstacles=torch.tensor(obs, dtype=torch.float32),
        obs_vel=torch.zeros(n_obs, 2, dtype=torch.float32),  # STATIC
        T=T_exec, dt=dt, u_max=u_max, r_robot=r_robot,
        xlim=(-box - 0.3, box + 0.3), ylim=(-box - 0.1, box + 0.1),
    )
    return env.to(device)


def scene_bank(n_scenes: int, base_seed: int = 100, device: str = "cpu", **kwargs) -> List[Env]:
    """A bank of distinct static cluttered scenes for multi-scene pretraining."""
    return [make_clutter_env(base_seed + i, device=device, **kwargs) for i in range(n_scenes)]


def env_from_obstacles(obstacles, start, goal, T: int = 80, dt: float = 0.1,
                       u_max: float = 2.0, r_robot: float = 0.2, box: float = 2.8,
                       device: str = "cpu") -> Env:
    """Rebuild an Env from stored obstacles/start/goal (so pretrain/expand reuse the exact scene)."""
    obstacles = torch.as_tensor(obstacles, dtype=torch.float32)
    start = torch.as_tensor(start, dtype=torch.float32)
    goal = torch.as_tensor(goal, dtype=torch.float32)
    env = Env(
        name="clutter",
        x0=torch.tensor([float(start[0]), float(start[1]), 0.0, 0.0], dtype=torch.float32),
        goal=goal.clone(),
        obstacles=obstacles.clone(),
        obs_vel=torch.zeros(obstacles.shape[0], 2, dtype=torch.float32),
        T=T, dt=dt, u_max=u_max, r_robot=r_robot,
        xlim=(-box - 0.3, box + 0.3), ylim=(-box - 0.1, box + 0.1),
    )
    return env.to(device)


# --------------------------------------------------------------------------- helpers
def clearance_field(env: Env, GX: np.ndarray, GY: np.ndarray) -> np.ndarray:
    """min_j (||x - o_j|| - r_j) over obstacles -> signed clearance grid (h=0 is the boundary)."""
    obs = env.obstacles.detach().cpu().numpy()
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)          # [P,2]
    d = np.linalg.norm(pts[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2]
    return d.min(axis=1).reshape(GX.shape)


def di_step(state: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Double-integrator step, identical to SafeMPPIAdapter._step / di_gap.di_step."""
    s = np.asarray(state, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    return np.array([
        s[0] + dt * s[2] + 0.5 * dt * dt * u[0],
        s[1] + dt * s[3] + 0.5 * dt * dt * u[1],
        s[2] + dt * u[0],
        s[3] + dt * u[1],
    ], dtype=np.float32)


if __name__ == "__main__":
    # quick self-test
    e = make_clutter_env(0)
    print("env:", e.name, "obs", tuple(e.obstacles.shape), "start", e.x0[:2].tolist(),
          "goal", e.goal.tolist(), "T", e.T, "dt", e.dt, "u_max", e.u_max)
    o = e.obstacles.numpy()
    # pairwise non-overlap check
    bad = 0
    for i in range(len(o)):
        for j in range(i + 1, len(o)):
            if np.linalg.norm(o[i, :2] - o[j, :2]) < o[i, 2] + o[j, 2]:
                bad += 1
    print("overlapping pairs:", bad)
    bank = scene_bank(4)
    print("bank scenes:", len(bank), "obs counts", [b.n_obs for b in bank])
