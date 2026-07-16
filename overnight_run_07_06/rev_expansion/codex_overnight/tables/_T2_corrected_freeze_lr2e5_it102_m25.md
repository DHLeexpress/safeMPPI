# Corrected coherent frozen it102 lr2e-5 M25

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected-freeze-lr2e5-it102 | 0.1 | 92.0% | 0.0% | 0.309 ± 0.013 | 1.511 ± 0.047 | 18.09 ± 1.58 | 6 | 23 | 25 | — |
| Corrected-freeze-lr2e5-it102 | 0.2 | 96.0% | 0.0% | 0.293 ± 0.017 | 1.579 ± 0.027 | 14.17 ± 1.38 | 4 | 24 | 25 | — |
| Corrected-freeze-lr2e5-it102 | 0.3 | 88.0% | 0.0% | 0.296 ± 0.013 | 1.606 ± 0.013 | 12.34 ± 0.65 | 2 | 22 | 25 | — |
| Corrected-freeze-lr2e5-it102 | 0.4 | 84.0% | 0.0% | 0.300 ± 0.010 | 1.612 ± 0.012 | 11.81 ± 0.40 | 2 | 21 | 25 | — |
| Corrected-freeze-lr2e5-it102 | 0.5 | 88.0% | 0.0% | 0.298 ± 0.014 | 1.608 ± 0.018 | 11.99 ± 0.66 | 2 | 22 | 25 | — |
| Corrected-freeze-lr2e5-it102 | 0.7 | 88.0% | 0.0% | 0.297 ± 0.014 | 1.603 ± 0.013 | 12.50 ± 0.60 | 2 | 22 | 25 | — |
| Corrected-freeze-lr2e5-it102 | 1.0 | 92.0% | 0.0% | 0.297 ± 0.015 | 1.596 ± 0.014 | 12.91 ± 0.87 | 2 | 23 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
