# T3 — Tuned Kazuki guidance baseline

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kazuki-guidance | 0.1 | 100.0% | 0.0% | 0.372 ± 0.004 | 1.299 ± 0.023 | 10.47 ± 1.07 | 5 | 200 | 200 | — |
| Kazuki-guidance | 0.2 | 100.0% | 0.0% | 0.375 ± 0.004 | 1.299 ± 0.021 | 9.45 ± 0.91 | 6 | 200 | 200 | — |
| Kazuki-guidance | 0.3 | 100.0% | 0.0% | 0.375 ± 0.003 | 1.298 ± 0.021 | 9.09 ± 0.71 | 6 | 200 | 200 | — |
| Kazuki-guidance | 0.4 | 100.0% | 0.0% | 0.375 ± 0.003 | 1.299 ± 0.019 | 8.97 ± 0.65 | 7 | 200 | 200 | — |
| Kazuki-guidance | 0.5 | 100.0% | 0.0% | 0.375 ± 0.003 | 1.299 ± 0.019 | 8.96 ± 0.67 | 7 | 200 | 200 | — |
| Kazuki-guidance | 0.7 | 100.0% | 0.0% | 0.375 ± 0.004 | 1.300 ± 0.018 | 9.06 ± 0.73 | 8 | 200 | 200 | — |
| Kazuki-guidance | 1.0 | 100.0% | 0.0% | 0.375 ± 0.004 | 1.300 ± 0.018 | 9.04 ± 0.73 | 7 | 200 | 200 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
