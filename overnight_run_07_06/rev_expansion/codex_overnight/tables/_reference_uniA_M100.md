# Reference only — prior ad-hoc uni_A best

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flow-expanded | 0.1 | 88.0% | 0.0% | 0.310 ± 0.011 | 1.511 ± 0.061 | 18.71 ± 1.52 | 13 | 88 | 100 | — |
| Flow-expanded | 0.2 | 93.0% | 1.0% | 0.296 ± 0.011 | 1.590 ± 0.025 | 13.62 ± 0.96 | 6 | 93 | 100 | — |
| Flow-expanded | 0.3 | 93.0% | 2.0% | 0.295 ± 0.011 | 1.610 ± 0.018 | 12.13 ± 0.63 | 5 | 93 | 100 | — |
| Flow-expanded | 0.4 | 88.0% | 5.0% | 0.296 ± 0.011 | 1.616 ± 0.015 | 11.70 ± 0.55 | 3 | 88 | 100 | — |
| Flow-expanded | 0.5 | 92.0% | 2.0% | 0.297 ± 0.011 | 1.614 ± 0.016 | 11.77 ± 0.57 | 4 | 92 | 100 | — |
| Flow-expanded | 0.7 | 94.0% | 2.0% | 0.298 ± 0.011 | 1.606 ± 0.017 | 12.12 ± 0.59 | 5 | 94 | 100 | — |
| Flow-expanded | 1.0 | 95.0% | 1.0% | 0.301 ± 0.012 | 1.598 ± 0.022 | 12.26 ± 0.70 | 5 | 95 | 100 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
