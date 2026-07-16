# Stage 2B report — balanced fixed-pair ID demonstrations

## Decision

Stage 2B is complete and ready for the approval gate.  This is the intended **ordinary 4x4 ID
stadium**, not the giant-obstacle OOD deployment scene.  Start `(0.5,0.5)`, goal `(4.5,4.5)`, the
endpoint-free model inputs, and the approved Stage-2A SafeMPPI recipe are unchanged.

The training set contains 168 real, physically successful SafeMPPI trajectories and 10,752 complete
executed H=10 windows.  It contains no synthetic reflected trajectories and no terminal padding.

## What “balanced” means here

The 672-rollout raw census is deliberately not used at its natural frequency.  For each gamma, the
builder retains three R/U reflection-pairs of four-right/four-up crossing words and exactly four real
trajectories per word:

- 24 trajectories per gamma: 12 R-first and 12 U-first;
- 64 windows per trajectory: 768 R-first and 768 U-first windows per gamma;
- 84 R-first and 84 U-first trajectories globally;
- 5,376 R-first and 5,376 U-first windows globally;
- equal 1,536-window loss mass for every gamma;
- exact mirror-count residual zero at trajectory and window level.

The balance is not merely categorical.  Across the seven gamma datasets, the largest relative gap
between mean absolute x/y control magnitude is 2.18%, the largest x-vs-y target-distribution quantile
MAE is 0.017, and the largest mean x/y context-position gap is 0.0249 m.  Thus the data have no material
upward sampling bias.  Whether training preserves this symmetry remains a Stage-3 rollout criterion;
it is not inferred from the data audit alone.

## Independent integrity audit

`validate_stage2b.py` reloads every tensor independently and traces every row back to the retained raw
candidate control sequence.  It passed all checks:

- all 10,752 targets and context positions match the expert source exactly (maximum absolute error 0);
- 10,752/10,752 reconstructed windows are physically collision-free and remain in task space;
- 0/10,752 windows are terminal-padded;
- all numeric tensors are finite;
- all selected sources are real, eligible, successful expert trajectories with zero collisions;
- all shapes, endpoint constants, gamma labels, seed counts, signature IDs, and equal-mass quotas match.

The earlier provisional build had 11 unsafe synthetic continuations, all caused by repeating terminal
controls.  It was overwritten.  The final `v2_full_horizon` tensors sample only starts satisfying
`step + H <= number of executed controls`; none of those 11 rows remain.

## Per-gamma expert and window audit

| gamma | expert SR | CR | time (s) | path (m) | mean clearance (m) | retreat (m) | switches | window valid2 | physical windows | minimum window clearance (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100% | 0% | 19.79 | 7.263 | 0.285 | 0.054 | 2.33 | 89.3% | 100% | 0.0977 |
| 0.2 | 100% | 0% | 16.01 | 6.808 | 0.222 | 0.057 | 2.92 | 92.6% | 100% | 0.0326 |
| 0.3 | 100% | 0% | 14.00 | 6.548 | 0.230 | 0.016 | 0.92 | 96.7% | 100% | 0.0197 |
| 0.4 | 100% | 0% | 13.11 | 6.418 | 0.227 | 0.015 | 1.00 | 95.3% | 100% | 0.0082 |
| 0.5 | 100% | 0% | 12.27 | 6.449 | 0.244 | 0.009 | 0.67 | 96.9% | 100% | 0.0096 |
| 0.7 | 100% | 0% | 12.37 | 6.533 | 0.261 | 0.013 | 0.58 | 98.1% | 100% | 0.0102 |
| 1.0 | 100% | 0% | 12.36 | 6.481 | 0.258 | 0.014 | 0.58 | 96.0% | 100% | 0.0001 |

`valid2` is a stricter per-window task-space/progress/fitted-SOCP conjunction.  It is reported as a
diagnostic and is not used to relabel physically successful demonstrations.  In particular, a valid2
rate below 100% does not mean that the corresponding expert trajectory collided or failed to reach the
goal.

## Approval artifacts

- `viz/balanced_id_overlay_all_gamma.png`: all selected paths, gamma colors, and per-gamma thumbnails.
- `viz/balanced_id_paths_by_gamma.png`: solid/dashed R/U reflection modes for each gamma.
- `viz/signature_balance_raw_vs_selected.png`: natural SafeMPPI bias versus exact retained quotas.
- `viz/window_validity.png`: physical, progress, SOCP, joint-valid2, and nominal-schedule diagnostics.
- `logs/independent_audit.json`: full machine-readable source and symmetry audit with SHA-256 hashes.
- `logs/balanced_dataset_summary.json`: selection, expert metrics, and validity summary.
- `data/balanced_id_windows_g*.pt`: seven final model-ready datasets.

## Gate

Stage 3 has not started.  On approval, Stage 3 will train a fresh original endpoint-free model only on
these ID tensors, then evaluate all gammas in this same ID stadium.  The gate requires both successful
diagonal navigation and visibly/numerically balanced right/up deployment before the giant obstacle is
introduced.
