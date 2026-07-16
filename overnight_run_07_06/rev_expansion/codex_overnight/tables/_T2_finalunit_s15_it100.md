# P2 selected q=.5 K=14 seed15 iteration 100 (M=100 audit)

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flow-expanded-q50-k14-it100 | 0.1 | 91.0% | 0.0% | 0.308 ± 0.014 | 1.515 ± 0.045 | 18.46 ± 1.85 | 9 | 91 | 100 | — |
| Flow-expanded-q50-k14-it100 | 0.2 | 96.0% | 0.0% | 0.296 ± 0.013 | 1.583 ± 0.028 | 14.00 ± 1.07 | 7 | 96 | 100 | — |
| Flow-expanded-q50-k14-it100 | 0.3 | 95.0% | 2.0% | 0.297 ± 0.015 | 1.604 ± 0.021 | 12.41 ± 0.76 | 5 | 95 | 100 | — |
| Flow-expanded-q50-k14-it100 | 0.4 | 95.0% | 1.0% | 0.299 ± 0.014 | 1.609 ± 0.018 | 12.01 ± 0.64 | 4 | 95 | 100 | — |
| Flow-expanded-q50-k14-it100 | 0.5 | 94.0% | 1.0% | 0.299 ± 0.015 | 1.607 ± 0.018 | 12.10 ± 0.79 | 4 | 94 | 100 | — |
| Flow-expanded-q50-k14-it100 | 0.7 | 91.0% | 1.0% | 0.299 ± 0.015 | 1.601 ± 0.018 | 12.58 ± 0.75 | 6 | 91 | 100 | — |
| Flow-expanded-q50-k14-it100 | 1.0 | 93.0% | 0.0% | 0.300 ± 0.013 | 1.595 ± 0.021 | 12.81 ± 0.79 | 5 | 93 | 100 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
