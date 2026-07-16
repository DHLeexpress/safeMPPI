# Stage 1B report — smooth long-horizon SafeMPPI expert

**Status:** complete; awaiting approval. Stage 2 has not started.

## Locked experiment

- Start `(0.5,0.5)`, goal `(4.5,4.5)`; endpoint clearance `0.507 m`.
- Giant center `(2.5,2.5)`, radius `1.20 m`; nearest surrounding surface gap `0.181 m`.
- SafeMPPI mode-1 configuration unchanged except for `smooth_weight`.
- Maximum 800 controls at `dt=0.1 s` (`80 s` simulated time); goal reach `0.15 m`.
- Final evaluation uses `M=2` seeds per gamma. Replicate seeds are matched across gamma.

Metric e is adapted to this local-geometry task: it counts the two collision-free homotopies around the
giant obstacle, upper-left and lower-right. Its maximum is 2. The old 252-staircase identifier assumes a
goal at `(5,5)` and is not applicable to this exact task.

## Smoothness tuning

| Smooth weight | Pilot success | Collision | Mean executed control delta | Mean jerk RMS | Mean time [s] |
|---:|---:|---:|---:|---:|---:|
| 0.12 (default) | 7/7 | 0 | 1.284 | 14.919 | 13.79 |
| 0.50 | 6/7 | 1 | 1.259 | 14.549 | 15.53 |
| 1.00 | 7/7 | 0 | 1.230 | 14.199 | 15.00 |
| 2.00 | 7/7 | 0 | 1.139 | 13.084 | 18.84 |
| 4.00 | 7/7 | 0 | 1.077 | 12.565 | 19.77 |
| **8.00** | **7/7** | **0** | **1.068** | **12.329** | **21.74** |
| 16.0 | 3/7 | 0 | — | — | — |
| 32.0 | 1/7 | 0 | — | — | — |

Weight 8 is the highest all-gamma feasible value. Relative to default in the matched-seed pilot, it lowers
mean executed control variation by 16.8% and jerk RMS by 17.4%, at a 57.7% mean time cost. Weights 16 and
32 time out for four and six gamma values, respectively, even with the 80 s ceiling.

## Final expert a--e, M=2 per gamma

Here c is the authoritative episode mean of nearest-obstacle clearance on successful paths. Minimum
clearance is included because it exposes the gamma safety ordering particularly clearly.

| gamma | a: SR | b: CR | c: clearance [m] | Minimum clearance [m] | d: time [s] | e: modes [/2] |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100% | 0% | 0.176 ± 0.004 | 0.064 ± 0.005 | 43.70 ± 1.60 | 1/2 |
| 0.2 | 100% | 0% | 0.175 ± 0.000 | 0.052 ± 0.003 | 25.70 ± 5.50 | 2/2 |
| 0.3 | 100% | 0% | 0.150 ± 0.024 | 0.034 ± 0.008 | 19.85 ± 2.45 | 1/2 |
| 0.4 | 100% | 0% | 0.159 ± 0.012 | 0.020 ± 0.002 | 17.35 ± 2.15 | 1/2 |
| 0.5 | 100% | 0% | 0.171 ± 0.008 | 0.012 ± 0.006 | 17.55 ± 0.65 | 2/2 |
| 0.7 | 100% | 0% | 0.144 ± 0.013 | 0.008 ± 0.000 | 21.75 ± 7.35 | 2/2 |
| 1.0 | 100% | 0% | 0.138 ± 0.006 | 0.003 ± 0.001 | 20.25 ± 1.25 | 2/2 |

### Gamma-intuition verdict

- a/b pass cleanly: all 14 paths succeed and none collide.
- c broadly passes: low gamma has the largest mean clearance; gamma versus mean clearance correlation is
  `-0.811`. More strongly, minimum clearance decreases strictly at every gamma step from `0.064 m` to
  `0.003 m`.
- d broadly passes: gamma 0.1 is the slowest and a medium gamma is fastest; gamma versus time correlation
  is `-0.541`. With only two samples, the high-gamma times are not strictly monotone.
- e shows both local homotopies at gamma 0.2, 0.5, 0.7, and 1.0. Gamma 0.1, 0.3, and 0.4 cover one side
  in M=2; this sample size is a visualization probe, not a coverage estimate.

## Validity audit — important caveat

Physical a--e success and `valid2` are not the same result.

| gamma | Mean fraction of 10-step windows failing net progress | External fitted-SOCP pass rate | Whole-path valid2 |
|---:|---:|---:|---:|
| 0.1 | 57.7% | 100% | 0% |
| 0.2 | 39.7% | 100% | 0% |
| 0.3 | 26.3% | 100% | 0% |
| 0.4 | 25.4% | 100% | 0% |
| 0.5 | 28.7% | 100% | 0% |
| 0.7 | 34.0% | 50% | 0% |
| 1.0 | 32.0% | 0% | 0% |

The progress failures are expected from the current rule: it demands at least 0.10 m net Euclidean goal
progress in every one-second window, whereas following the giant boundary requires tangential motion and
occasionally slight retreat. The three external SOCP failures are tight high-gamma windows, not physical
collisions; increasing the verifier angular resolution from 180 to 720 does not change them. No validity
criterion was weakened or redefined.

Therefore this run establishes a physically successful, gamma-conditioned expert ceiling and a smoothness
setting, but it does **not** establish a whole-trajectory `valid2` expert. That mismatch must remain visible
when deciding how later expansion samples are accepted.

Stage 1C subsequently measured each H=10 sample independently: 61.6% of all training-style samples and
60.3% of executed-full windows pass joint valid2. See `../01c_window_validity/REPORT.md` for the complete
window-level breakdown and moving-certificate GIF.

## Artifacts

- `viz/expert_m2_by_gamma.png`: two trajectories in each gamma panel.
- `viz/expert_m2_overlay.png`: all 14 trajectories overlaid.
- `viz/expert_ae_m2.png`: a--e and smoothness summary.
- `viz/smoothness_tuning.png`: weight-selection sweep.
- `tables/expert_ae_m2.csv`: exact a--e rows.
- `logs/expert_m2_summary.json`: per-path metrics and audit telemetry.
- `data/expert_m2_paths.npz`: exact states, controls, paths, seeds, and task metadata.

Integrity hashes:

- NPZ: `2b85232e562f2614af014ecea16427e42a49d666a126730624173e599ff3b312`
- JSON: `e3dd51f70c1e1ff014e8711be4d33e55b901ca3f4c81211ee46fa5aa0b6067d9`
- CSV: `0b43f9a18bf2c8e7f9df86456424954bac3d0bf3b0e6e383df0fa763abdced62`
- Main figure: `3d4abd068db1422a80c3a98b86938f11702dc535557ede3368cf8febe7736849`
