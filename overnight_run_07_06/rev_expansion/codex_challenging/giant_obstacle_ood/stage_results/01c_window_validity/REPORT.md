# Stage 1C report — window valid2 and moving certificates

**Status:** complete; awaiting approval. No Stage 2 generation or training has started.

## What was measured

The source is the exact 14 Stage 1B trajectories: two matched seeds for every gamma. Every source
trajectory reaches `(4.5,4.5)`, stays inside the task, and has no executed collision.

For each executed control step, this audit emits the same H=10 target shape as
`stage2_grid_data.windows_from`. The last nine samples of each trajectory repeat their final available
control, so results are reported both with and without those terminal-padded samples.

Window valid2 is:

`task space AND net progress AND fitted-verifier SOCP`.

- Task space uses the authoritative 0.12 m tolerance.
- Progress requires at least 0.10 m net goal progress over ten controls; a window beginning within 0.45 m
  of the goal auto-passes.
- SOCP uses `verifier_polytope.certify_window`, `R=2.5`, and `n_theta=180`.

## How much valid data is inside the successful trajectories?

| Population | Windows | Task | Progress | SOCP | Joint valid2 |
|---|---:|---:|---:|---:|---:|
| All training-style samples | 3,323 | 3,322 (99.97%) | 2,076 (62.47%) | 3,293 (99.10%) | **2,047 (61.60%)** |
| Executed-only full H=10 windows | 3,197 | 3,197 (100%) | 1,950 (60.99%) | 3,173 (99.25%) | **1,927 (60.28%)** |
| Terminal-padded windows | 126 | 125 (99.21%) | 126 (100%) | 120 (95.24%) | 120 (95.24%) |

The scientifically conservative headline is **1,927/3,197 = 60.3% valid2-positive executed windows**.
Whole-trajectory valid2 is zero because it is an all-windows conjunction: one failing window invalidates
the entire trajectory. That does not make every sample from that trajectory negative.

The padded subset should not be used as a safety headline. Its progress rate is artificially high because
all starts lie inside the 0.45 m arrival auto-pass region, while repeating the last control creates three
predicted collisions and one task-space exit that were never executed.

## Per-gamma training-style samples

| gamma | Samples | Progress | SOCP | Joint valid2 | Executed-full joint | Nominal schedule |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 874 | 44.1% | 99.4% | 381 (43.6%) | 367/856 (42.9%) | 87.3% |
| 0.2 | 514 | 64.0% | 100% | 329 (64.0%) | 311/496 (62.7%) | 76.3% |
| 0.3 | 397 | 75.1% | 100% | 298 (75.1%) | 280/379 (73.9%) | 68.8% |
| 0.4 | 347 | 75.5% | 100% | 262 (75.5%) | 244/329 (74.2%) | 62.5% |
| 0.5 | 351 | 72.6% | 100% | 255 (72.6%) | 237/333 (71.2%) | 67.8% |
| 0.7 | 435 | 61.1% | 97.9% | 257 (59.1%) | 239/417 (57.3%) | 68.3% |
| 1.0 | 405 | 69.4% | 96.0% | 265 (65.4%) | 249/387 (64.3%) | 62.5% |

The bottleneck is progress, especially for conservative gamma 0.1 while it follows the obstacle boundary.
SOCP accepts nearly all local windows even though the all-windows trajectory test fails.

## Nominal-polytope existence versus verifier existence

The literal statements must be separated:

1. **A geometric nominal polytope exists:** true for 3,323/3,323 windows. This only says the current robot
   has a nonempty local safe region.
2. **The next executed H=10 window obeys that nominal polytope's gamma level-set schedule:** true for
   2,433/3,323 windows.
3. **A fitted verifier polytope certifies the executed window:** true for 3,293/3,323 windows.

Nominal geometric existence alone is not a certificate for an arbitrary future executed window: the
nominal region exists in all 30 windows where the fitted verifier fails. However, the stronger implication
requested by the method is perfectly supported in this dataset:

`nominal schedule certificate => fitted verifier certificate`: **2,433/2,433, with zero counterexamples**.

Moreover, the fitted verifier certifies `3,293 - 2,433 = 860` additional windows that do not satisfy the
fixed nominal schedule. On executed-only full windows the corresponding expansion is 859 windows. This is
the concrete window-level evidence that the green fitted verifier is less conservative than the blue
nominal certificate.

## Animation

The synchronized GIF uses matched seed 65100 and all seven gamma values:

- Blue solid: moving nominal polytope.
- Blue dashed: nominal H-step gamma level set.
- Green solid: fitted verifier polytope when the SOCP succeeds.
- Green dashed: verifier H-step gamma level set.
- Black points: the next executed window.
- A red `verifier infeasible` label replaces green geometry on failed solves.

It contains 115 frames, advances 0.4 s of simulation per frame, displays each for 250 ms, and therefore
plays for 28.75 s. Panels share real simulation time; completed trajectories freeze at their goal while
gamma 0.1 continues.

## Artifacts

- `viz/nominal_blue_verifier_green_all_gamma.gif` — requested slow synchronized animation.
- `viz/nominal_verifier_poster.png` — 10 s animation frame at publication resolution.
- `viz/window_validity_by_gamma.png` and `.pdf` — window validity and nominal/verifier rates.
- `tables/window_validity.md` — concise per-gamma table.
- `logs/window_validity_summary.json` — exact counts and definitions.
- `logs/polytope_gif_summary.json` — animation provenance.
- `data/window_records.csv` and `data/window_masks.npz` — every window label.

Integrity hashes:

- Window summary JSON: `4cbed54327e6b9dab7fc90331fd79de37fead42a4c44af2a6e72c3953b74845f`
- Window masks NPZ: `3c674bbcf5606ce3edd65bab470039dc5e25cfe9bba687fcb94557d58e02d179`
- Window figure: `0f7b04c3da2ea42c1aabc1093acf9d7cee19d2435347d03269ac8a4800804634`
- GIF: `7b6aa2b39250d9e1d15b698004515e51f92f5322500cb7d3b8ed53fbb7c87e9d`
