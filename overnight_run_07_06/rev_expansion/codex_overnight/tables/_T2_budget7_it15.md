# P2 corrected fixed-quantile expansion, iteration 15 (M=100 diagnostic)

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flow-expanded-budget7-it15 | 0.1 | 85.0% | 0.0% | 0.321 ± 0.015 | 1.483 ± 0.058 | 17.94 ± 1.48 | 13 | 85 | 100 | — |
| Flow-expanded-budget7-it15 | 0.2 | 87.0% | 0.0% | 0.304 ± 0.017 | 1.561 ± 0.036 | 14.19 ± 1.36 | 8 | 87 | 100 | — |
| Flow-expanded-budget7-it15 | 0.3 | 85.0% | 2.0% | 0.304 ± 0.019 | 1.589 ± 0.027 | 12.62 ± 1.01 | 6 | 85 | 100 | — |
| Flow-expanded-budget7-it15 | 0.4 | 83.0% | 4.0% | 0.305 ± 0.018 | 1.595 ± 0.026 | 12.26 ± 0.91 | 6 | 83 | 100 | — |
| Flow-expanded-budget7-it15 | 0.5 | 79.0% | 4.0% | 0.303 ± 0.017 | 1.595 ± 0.026 | 12.24 ± 0.86 | 5 | 79 | 100 | — |
| Flow-expanded-budget7-it15 | 0.7 | 79.0% | 2.0% | 0.305 ± 0.016 | 1.584 ± 0.028 | 12.71 ± 0.84 | 5 | 79 | 100 | — |
| Flow-expanded-budget7-it15 | 1.0 | 83.0% | 0.0% | 0.306 ± 0.016 | 1.573 ± 0.029 | 13.03 ± 0.96 | 7 | 83 | 100 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
