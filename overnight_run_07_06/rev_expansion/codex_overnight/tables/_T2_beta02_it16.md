# P2 corrected fixed-quantile expansion, beta=.2 iteration 16 (M=100 diagnostic)

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flow-expanded-beta02-it16 | 0.1 | 86.0% | 0.0% | 0.318 ± 0.015 | 1.497 ± 0.056 | 17.65 ± 1.84 | 14 | 86 | 100 | — |
| Flow-expanded-beta02-it16 | 0.2 | 89.0% | 1.0% | 0.304 ± 0.015 | 1.569 ± 0.035 | 13.69 ± 1.08 | 8 | 89 | 100 | — |
| Flow-expanded-beta02-it16 | 0.3 | 83.0% | 5.0% | 0.304 ± 0.017 | 1.595 ± 0.027 | 12.30 ± 0.88 | 7 | 83 | 100 | — |
| Flow-expanded-beta02-it16 | 0.4 | 85.0% | 4.0% | 0.305 ± 0.018 | 1.600 ± 0.026 | 11.89 ± 0.86 | 7 | 85 | 100 | — |
| Flow-expanded-beta02-it16 | 0.5 | 85.0% | 4.0% | 0.303 ± 0.018 | 1.597 ± 0.026 | 12.04 ± 0.98 | 7 | 85 | 100 | — |
| Flow-expanded-beta02-it16 | 0.7 | 79.0% | 2.0% | 0.304 ± 0.017 | 1.590 ± 0.026 | 12.44 ± 0.94 | 7 | 79 | 100 | — |
| Flow-expanded-beta02-it16 | 1.0 | 81.0% | 4.0% | 0.306 ± 0.016 | 1.578 ± 0.037 | 12.58 ± 0.89 | 9 | 81 | 100 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
