# T1 — SafeMPPI expert ground truth

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SafeMPPI | 0.1 | 100.0% | 0.0% | 0.333 ± 0.011 | 1.555 ± 0.020 | 15.13 ± 0.83 | 8 | 100 | 100 | — |
| SafeMPPI | 0.2 | 100.0% | 0.0% | 0.290 ± 0.013 | 1.592 ± 0.021 | 11.53 ± 0.92 | 6 | 100 | 100 | — |
| SafeMPPI | 0.3 | 100.0% | 0.0% | 0.281 ± 0.015 | 1.596 ± 0.021 | 10.99 ± 0.83 | 9 | 100 | 100 | — |
| SafeMPPI | 0.4 | 100.0% | 0.0% | 0.282 ± 0.015 | 1.597 ± 0.018 | 10.68 ± 0.70 | 7 | 100 | 100 | — |
| SafeMPPI | 0.5 | 100.0% | 0.0% | 0.285 ± 0.015 | 1.597 ± 0.019 | 10.54 ± 0.70 | 6 | 100 | 100 | — |
| SafeMPPI | 0.7 | 100.0% | 0.0% | 0.287 ± 0.014 | 1.594 ± 0.017 | 10.58 ± 0.76 | 6 | 100 | 100 | — |
| SafeMPPI | 1.0 | 100.0% | 0.0% | 0.294 ± 0.013 | 1.584 ± 0.022 | 10.76 ± 0.69 | 11 | 100 | 100 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
