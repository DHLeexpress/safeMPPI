# Kazuki mixed_M20

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kazuki-guidance | 0.1 | 100.0% | 0.0% | 0.374 ± 0.003 | 1.296 ± 0.030 | 10.04 ± 1.20 | 5 | 20 | 20 | — |
| Kazuki-guidance | 0.2 | 100.0% | 0.0% | 0.374 ± 0.004 | 1.296 ± 0.017 | 9.22 ± 0.84 | 4 | 20 | 20 | — |
| Kazuki-guidance | 0.3 | 100.0% | 0.0% | 0.375 ± 0.004 | 1.300 ± 0.024 | 9.04 ± 0.68 | 5 | 20 | 20 | — |
| Kazuki-guidance | 0.4 | 100.0% | 0.0% | 0.376 ± 0.004 | 1.293 ± 0.018 | 9.11 ± 0.45 | 4 | 20 | 20 | — |
| Kazuki-guidance | 0.5 | 100.0% | 0.0% | 0.374 ± 0.003 | 1.302 ± 0.019 | 8.96 ± 0.71 | 5 | 20 | 20 | — |
| Kazuki-guidance | 0.7 | 100.0% | 0.0% | 0.373 ± 0.004 | 1.299 ± 0.021 | 9.18 ± 0.87 | 5 | 20 | 20 | — |
| Kazuki-guidance | 1.0 | 100.0% | 0.0% | 0.374 ± 0.004 | 1.292 ± 0.015 | 9.37 ± 0.92 | 6 | 20 | 20 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
