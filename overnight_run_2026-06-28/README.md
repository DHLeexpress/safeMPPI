# overnight_run_2026-06-28 — polytope_v1 + verifier + Safe Flow Expansion

Realizes the SafeFlow Exploration loop with a **polytope-based** safety object (not the earlier
distance-circles). Two fixed double-integrator scenes: `single` (one obstacle → left/right) and
`gap` (two stacked obstacles, narrow passable slot → left/gap/right). Defaults: **γ=0.5**, **zero gap**
(polytope flush to the actual obstacle), level sets in the `visualize_di_gamma` format.

## New modules (separate, by design)
- `cfm_mppi/safegpc_adapter/polytope_v1.py`
  - `build_candidate_v1(pos, nominal_last, obstacles, sensing_range)` — the DETERMINISTIC, less-restrictive
    candidate: a robot-centered sensing-ball box biased toward the nominal control's last state + ONE
    tangent separating **hyperplane** per obstacle on the corridor side (faces = 4+N, zero gap). Reaches
    *past* obstacles toward free space (gap corridor / one-side go-around). Replaces the "6:2:4:4 box".
  - `build_rectangle_polytope(...)` — pure max-rectangle in the ball (the verifier's building block).
  - `normalized_barrier(poly, pts)` — H=1 at ref, 0 on boundary; nested `{H≥(1-γ)^i}` level sets.
- `cfm_mppi/safegpc_adapter/rectangle_verifier.py`
  - `certify(seg, ...)` / `verify_trajectory(states, ...)` — the SEPARATE existential verifier: *does there
    exist a rectangle in the ball, separating obstacles, s.t. every state satisfies `H(x_i)≥(1-γ)^i`?*
    Per-inference-step, outermost-state-seeded orientation search, **warm-started**; gates FM updates.

## Steps (run from repo root with the libstdc++ preload)
```
LD_PRELOAD=/home/dohyun/miniforge3/lib/libstdc++.so.6 python overnight_run_2026-06-28/step0_polytope_v1_viz.py --gamma 0.5
LD_PRELOAD=/home/dohyun/miniforge3/lib/libstdc++.so.6 python overnight_run_2026-06-28/step1_safemppi_propagate.py --gamma 0.5
LD_PRELOAD=/home/dohyun/miniforge3/lib/libstdc++.so.6 python overnight_run_2026-06-28/step2_vanilla_cfm.py --device cuda
LD_PRELOAD=/home/dohyun/miniforge3/lib/libstdc++.so.6 python overnight_run_2026-06-28/step3_safeflow_expansion.py --device cuda --gamma 0.5
```

- **Step 0** (`figures/<scene>_stage0_*`): the deterministic `polytope_v1` (gap corridor / tilted go-around,
  zero gap, nested level sets) + the per-step verifier GIF (rectangle reorienting, warm-started).
- **Step 1** (`<scene>_stage1_safemppi.gif`): SafeMPPI sample-then-reject driven by the `polytope_v1` ruler
  (γ=0.5); both scenes reach the goal; healthy accept/reject (e.g. gap accept 293/7 entering the gap).
- **Step 2** (`<scene>_stage2_vanilla_cfm.png`): vanilla CFM (lightened Mizuta backbone, GPU), no MPPI/
  rejection → single bimodal, gap trimodal.
- **Step 3** (`<scene>_stage3_compare.png`, `<scene>_stage3_expansion.gif`): Safe Flow Expansion — the
  per-step verifier gates which generated samples feed the FM update; compare vs the conservative
  polytope-induced policy.

## Key result (step 3, gap)
| policy | modes | median min-clearance |
|---|---|---|
| polytope.py-induced (conservative) | LEFT/RIGHT (GAP:0) | 0.48 |
| **verifier-EXPANDED (polytope_v1)** | **LEFT/GAP/RIGHT (balanced)** | **0.39** |

→ the verifier-expanded policy is **more diverse** (discovers the narrow-gap mode) and **less conservative**
(smaller clearance) than the conservative candidate, while every retained sample is verifier-certified safe.
(single has no extra mode to discover, so its gain is only the small clearance reduction 0.71→0.69.)

## Notes / next
- Strengthening the (sparse) gap-mode mass is a *learning-distribution* problem (mode-balanced replay) —
  deferred per the plan. γ, sensing R, and the zero-gap margin are all easy knobs.
- Next phase: pedestrian data + Mizuta comparison, same four stages.
