# Stage 1 report — geometry and matched-seed SafeMPPI sweep

**Status:** complete; awaiting geometry approval. No Stage 2 data generation has started.

## Locked task

- ID stadium: the existing 8-plug scene with 72 circular obstacles.
- Fixed start: `(0.6464456, 0.6464456)`.
- Fixed goal: `(4.3535542, 4.3535542)`.
- Start and goal obstacle clearance: `0.3000012 m` each.
- OOD change: remove exactly `(2,2), (2,3), (3,2), (3,3)` and add one circle at
  `(2.5,2.5)`. Each OOD scene therefore has 69 obstacles.
- Safety levels: `gamma = 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0`.
- The planner-noise seed is matched across gamma within each scene so gamma is not confounded with
  sampling noise.

## Sweep result

Successful-rollout statistics below use only successful paths for the mean step and path-length columns.
The minimum-clearance column includes every rollout.

| Scene | Nearest surface gap [m] | SafeMPPI success | Mean steps | Mean path length [m] | Minimum clearance [m] | Detour modes (upper/lower) |
|---|---:|---:|---:|---:|---:|---:|
| ID | — | 7/7 | 109.0 | 5.893 | 0.016 | 4 / 3 |
| radius 0.90 | 0.481 | 7/7 | 116.6 | 6.310 | 0.018 | 4 / 3 |
| radius 1.00 | 0.381 | 7/7 | 123.1 | 6.443 | 0.021 | 1 / 6 |
| radius 1.10 | 0.281 | 7/7 | 136.0 | 6.564 | 0.009 | 4 / 3 |
| **radius 1.20** | **0.181** | **7/7** | **154.9** | **6.613** | **0.007** | **2 / 5** |
| radius 1.28 | 0.101 | 6/7 | 143.8 | 6.485 | 0.010 | 4 / 3 |

At radius 1.28, `gamma=0.1` remains collision-free but times out at 250 steps, 3.900 m from the goal.
This is the first observed expert-feasibility failure. It is not counted as a valid trajectory.

For radius 1.20, the matched-seed paths show a useful safety contrast: `gamma=0.1` takes 232 steps with
0.065 m minimum clearance, while `gamma=1.0` takes 134 steps with 0.007 m minimum clearance. Both
upper-left and lower-right detours occur across the gamma sweep.

## Recommendation

Approve **radius 1.20 m** for the first benchmark attempt. It is the largest tested radius with all seven
SafeMPPI trajectories valid, leaves only 0.181 m to the nearest surrounding obstacle surfaces, and raises
expert traversal from 109.0 to 154.9 mean steps. Radius 1.28 is a useful upper stress bound, but it violates
the requirement that the expert reliably demonstrate every gamma.

This sweep is a deterministic feasibility probe, not a statistical success estimate. Stage 2 will generate
repeated stochastic ID demonstrations and must verify high per-gamma success before pretraining. If the
frozen pretrained policy later solves radius 1.20 too easily, the plan returns to this gate and tests a
finer radius between 1.20 and 1.28 rather than changing the endpoints.

## Artifacts and integrity

- Figure: `viz/giant_radius_sweep.png` and `viz/giant_radius_sweep.pdf`
- Exact paths and controls: `data/expert_radius_sweep_paths.npz`
- Machine-readable metrics: `logs/stage1_geometry_summary.json`
- NPZ SHA-256: `98e212c81e2ab93bde0a244379f4940f94cc8bd51d403270c95cdf51ee90c1df`
- JSON SHA-256: `b6af65ffebd9817ee42d373a30145678e09a02820cbdc5003ff82316ce26cd9c`
- PNG SHA-256: `f899a03c60bbee716db196f2a892f7c8d7f7d045df4301cb04705a85db1429de`

