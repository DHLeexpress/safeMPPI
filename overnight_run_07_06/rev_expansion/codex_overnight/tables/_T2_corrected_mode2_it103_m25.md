# Corrected mode2 targeted it103 M25

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected-mode2-target50-it103 | 0.1 | 92.0% | 0.0% | 0.307 ± 0.013 | 1.516 ± 0.046 | 17.88 ± 1.31 | 6 | 23 | 25 | — |
| Corrected-mode2-target50-it103 | 0.2 | 96.0% | 0.0% | 0.294 ± 0.014 | 1.585 ± 0.020 | 13.88 ± 1.07 | 3 | 24 | 25 | — |
| Corrected-mode2-target50-it103 | 0.3 | 96.0% | 0.0% | 0.296 ± 0.012 | 1.606 ± 0.018 | 12.15 ± 0.49 | 3 | 24 | 25 | — |
| Corrected-mode2-target50-it103 | 0.4 | 92.0% | 0.0% | 0.297 ± 0.013 | 1.607 ± 0.021 | 11.82 ± 0.64 | 3 | 23 | 25 | — |
| Corrected-mode2-target50-it103 | 0.5 | 92.0% | 0.0% | 0.298 ± 0.011 | 1.610 ± 0.017 | 11.77 ± 0.41 | 3 | 23 | 25 | — |
| Corrected-mode2-target50-it103 | 0.7 | 92.0% | 0.0% | 0.297 ± 0.013 | 1.604 ± 0.012 | 12.35 ± 0.58 | 2 | 23 | 25 | — |
| Corrected-mode2-target50-it103 | 1.0 | 96.0% | 0.0% | 0.298 ± 0.013 | 1.595 ± 0.016 | 12.67 ± 0.64 | 3 | 24 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
