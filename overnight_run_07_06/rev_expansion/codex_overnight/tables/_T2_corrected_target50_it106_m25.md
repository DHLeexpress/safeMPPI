# Corrected targeted coherent it106 M25

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected-target50-it106 | 0.1 | 84.0% | 0.0% | 0.303 ± 0.014 | 1.515 ± 0.047 | 17.64 ± 1.40 | 5 | 21 | 25 | — |
| Corrected-target50-it106 | 0.2 | 92.0% | 0.0% | 0.289 ± 0.019 | 1.580 ± 0.031 | 14.14 ± 1.57 | 4 | 23 | 25 | — |
| Corrected-target50-it106 | 0.3 | 92.0% | 0.0% | 0.293 ± 0.015 | 1.603 ± 0.022 | 12.18 ± 0.83 | 3 | 23 | 25 | — |
| Corrected-target50-it106 | 0.4 | 92.0% | 0.0% | 0.296 ± 0.012 | 1.613 ± 0.015 | 11.70 ± 0.64 | 2 | 23 | 25 | — |
| Corrected-target50-it106 | 0.5 | 88.0% | 0.0% | 0.296 ± 0.013 | 1.608 ± 0.019 | 11.83 ± 0.70 | 2 | 22 | 25 | — |
| Corrected-target50-it106 | 0.7 | 88.0% | 0.0% | 0.297 ± 0.012 | 1.605 ± 0.013 | 12.17 ± 0.54 | 2 | 22 | 25 | — |
| Corrected-target50-it106 | 1.0 | 88.0% | 0.0% | 0.296 ± 0.014 | 1.599 ± 0.013 | 12.59 ± 0.79 | 2 | 22 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
