# Stage 2A — soft exponential anti-retreat expert

## Gate result

**Recommended recipe:** keep the Stage-1B `smooth_weight=8` expert and add

\[
J_{\mathrm{retreat}}
= 1.0\,\operatorname{expm1}\!\left(
\min\left(\frac{[d_{t+1}-d_t]_+}{0.05\ \mathrm{m}}, 6\right)
\right).
\]

This is a soft rollout-cost term, not a hard monotonic-progress constraint. It leaves tangential motion
around the giant obstacle available. The implementation is disabled by default
(`goal_retreat_exp_weight=0`) and therefore does not alter any existing planner configuration.

Stage 2B data generation and pretraining have **not** started. They remain behind the user approval gate.

## Protocol

- Scene: radius-1.20 giant obstacle, start `(0.5,0.5)`, goal `(4.5,4.5)`.
- Safety levels: `0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0`.
- Long horizon: 800 controls / 80 s, reach radius 0.15 m.
- Planner: Stage-1B SafeMPPI mode-1 configuration with `smooth_weight=8`.
- Tune: one matched seed per gamma for weights `0, .01, .03, .10, .30, 1, 3, 10`.
- Candidate audit: M=2 matched seeds per gamma, 14 candidate and 14 locked baseline trajectories.
- Hardware: physical NVIDIA H100 GPU 2 (exposed to the process as `cuda:0`).

Executed retreat is measured independently of the planner cost as
`sum(max(d[t+1]-d[t], 0))`. A retreating step increases goal distance by more than 1 mm; a radial direction
switch ignores changes below 5 mm.

## Tuning result

All weights retained 7/7 matched-seed physical success, but the response was deliberately not assumed to
be monotone: changing the running cost changes which MPPI samples and homotopies are selected.

| retreat weight | success | mean retreat [m] | direction switches | time [s] | path [m] |
|---:|---:|---:|---:|---:|---:|
| 0 | 7/7 | 0.267 | 9.7 | 21.7 | 8.02 |
| 0.01 | 7/7 | 0.252 | 9.1 | 22.2 | 8.07 |
| 0.03 | 7/7 | 0.339 | 12.6 | 24.1 | 8.36 |
| 0.10 | 7/7 | 0.331 | 12.3 | 25.1 | 8.29 |
| 0.30 | 7/7 | 0.413 | 13.7 | 27.9 | 8.46 |
| **1.0** | **7/7** | **0.138** | **6.0** | **19.5** | **7.46** |
| 3.0 | 7/7 | 0.453 | 14.6 | 32.1 | 9.06 |
| 10.0 | 7/7 | 0.267 | 10.3 | 24.5 | 8.22 |

Weights 3 and 10 bracket the selected value and show that 1.0 is not merely the best endpoint of the
initial sweep. Stronger penalties create new stalls/mode switches through the sampling dynamics.

## M=2 validation against the locked expert

| metric (mean over 14) | no penalty | selected | relative change |
|---|---:|---:|---:|
| physical success | 14/14 | 14/14 | unchanged |
| collision | 0/14 | 0/14 | unchanged |
| radial retreat | 0.312 m | 0.224 m | **-28.2%** |
| retreating-step fraction | 13.65% | 12.88% | -5.6% |
| radial direction switches | 11.57 | 9.43 | **-18.5%** |
| time to goal | 23.74 s | 21.92 s | **-7.6%** |
| path length | 8.21 m | 7.98 m | **-2.7%** |
| mean nearest-obstacle clearance | 0.159 m | 0.170 m | +6.6% |
| global homotopy set | upper-left + lower-right | upper-left + lower-right | preserved |

The improvement is aggregate rather than pointwise. The two-seed mean retreat decreases for gamma
`0.1, 0.2, 0.7, 1.0` and increases for `0.3, 0.4, 0.5`. This is acceptable for a soft bias, but it is why
Stage 2B must oversample and then quota geometric modes rather than trusting raw rollout frequency.

## Validity caveat

All 14 selected trajectories physically reach the goal without collision or task-space exit. Whole-path
valid2 remains 0/14 because every path contains at least one H=10 window rejected by the existing strict
net-progress criterion; fitted SOCP passes 12/14. The two fitted-SOCP failures are gamma 0.7 and 1.0 for
seed 65100. This does not redefine valid2 and is not reported as valid2 success. For expert distillation,
physical expert trajectories will be stored with window-level criterion masks so demo use and verifier
evaluation remain separate and auditable.

## Compatibility and artifacts

The new zero-weight path exactly reproduces the pre-feature Stage-1B gamma-0.3/seed-65100 trajectory:

`16283c4c6ff93dfe3de722cfe2fb5c25fd9ddda4f6abb14400004c2ee522455e`

Both the float32 states and controls are byte-identical.

- Primary trajectory comparison: `viz/selected_m2_by_gamma.png`
- Goal-distance audit: `viz/selected_m2_goal_distance.png`
- Metric comparison: `viz/selected_vs_baseline_m2.png`
- Full tuning trajectories: `viz/tuning_paths.png`
- Tuning goal-distance audit: `viz/tuning_goal_distance.png`
- Strong-weight bracket: `strong_extension/viz/tuning_paths.png`
- Machine-readable validation: `logs/selected_m2_summary.json`
- Per-gamma comparison: `tables/selected_vs_baseline_m2.csv`
- Candidate arrays: `data/selected_m2_paths.npz`

Key artifact SHA-256 values:

- `selected_m2_summary.json`: `00529e083cfba52e81cdba1a9ca0d6882576371b90382fde6f9a9c03b3d4ddc7`
- `selected_m2_paths.npz`: `e4150411ca596ef81f4d5207ccfb2a65d4b4e0cdfeb7889dd24f420a706a6df4`
- `selected_m2_by_gamma.png`: `76df4b411de1638daaa8ea216bba433b88476d047a3c79b9c787e4a6af994d03`

## Proposed next gate (not executed)

After approval, Stage 2B will generate the **ordinary 4x4 ID stadium** dataset with the fixed start and
goal. It will:

1. use the selected anti-retreat recipe for expert rollout candidates;
2. classify successful paths by monotone right/up geometric crossing word (four right and four up
   crossings, at most 70 signatures) and detour side;
3. oversample planner seeds, then accept a uniform quota over observed geometric signatures separately
   for every gamma instead of preserving SafeMPPI's raw mode frequency;
4. give every accepted trajectory equal weight before H=10 window slicing, so long/stalled paths do not
   dominate the pretrained loss;
5. store unbalanced candidates, accepted balanced trajectories, rejected reasons, per-gamma quotas, and
   window-level masks separately.

This is the geometric balancing needed to make an up/right pretrained policy neutral with respect to
which side of the grid obstacles it uses.
