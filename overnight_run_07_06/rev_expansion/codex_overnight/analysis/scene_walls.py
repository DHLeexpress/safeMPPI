"""Wall-plug scene variant (user 2026-07-11): extend the existing perimeter by ONE lattice circle at each
corner opening — bottom row toward the origin, top row toward the goal — so the task boundary the policy
currently CANNOT perceive (it senses only obstacles) becomes visible in the H_P channel exactly where the
two failure strata exit. Local module only; the shared grid_scene.py is untouched.

Lattice continuation (same y rows, same radius 0.2, same 0.385 spacing):
  origin plug: (0.3846, -0.2, 0.2)   — bottom wall previously started at x=0.769; seed-12 exits at x~0.15
  goal plug:   (4.6154,  5.2, 0.2)   — top wall previously ended at x=4.231; overshoots exit at x~4.85-4.96
Safety: start (0,0) clearance to origin plug = 0.233 m; goal (5,5) clearance to goal plug = 0.235 m.
Residual openings (still open, by design so the corners stay livable): bottom x<0.185, top x>4.815.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent.parent)]

import grid_scene as GS  # noqa: E402

STEP = 5.0 / 13.0                               # exact lattice step 0.38461...
ORIGIN_PLUG = (STEP, -0.2, 0.2)                 # bottom row toward origin (was open x<0.769)
GOAL_PLUG = (5.0 - STEP, 5.2, 0.2)              # top row toward goal (was open x>4.231)
LEFT_PLUG = (-0.2, STEP, 0.2)                   # left col toward origin (was open y<0.769)
RIGHT_PLUG = (5.2, 5.0 - STEP, 0.2)             # right col toward goal (was open y>4.431) —
                                                # zero-shot M25 with 2 plugs: ALL new failures exited here


def make_grid_walls(n_plugs=4):
    """n_plugs=2: bottom+top only (first proposal); n_plugs=4: one per side (user's original phrasing)."""
    env = GS.make_grid()
    plugs = [ORIGIN_PLUG, GOAL_PLUG] if n_plugs == 2 else [ORIGIN_PLUG, GOAL_PLUG, LEFT_PLUG, RIGHT_PLUG]
    extra = torch.tensor(plugs, dtype=env.obstacles.dtype)
    env.obstacles = torch.cat([env.obstacles, extra], dim=0)
    return env


if __name__ == "__main__":
    env = make_grid_walls()
    print("obstacles:", env.obstacles.shape, "| last two:", env.obstacles[-2:].tolist())
