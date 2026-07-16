# Corrected mode2 targeted it102 M25

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected-mode2-target50-it102 | 0.1 | 88.0% | 0.0% | 0.307 ± 0.015 | 1.513 ± 0.047 | 18.28 ± 1.87 | 6 | 22 | 25 | — |
| Corrected-mode2-target50-it102 | 0.2 | 96.0% | 0.0% | 0.295 ± 0.014 | 1.583 ± 0.022 | 13.93 ± 1.11 | 3 | 24 | 25 | — |
| Corrected-mode2-target50-it102 | 0.3 | 96.0% | 0.0% | 0.296 ± 0.012 | 1.605 ± 0.018 | 12.24 ± 0.54 | 3 | 24 | 25 | — |
| Corrected-mode2-target50-it102 | 0.4 | 92.0% | 0.0% | 0.298 ± 0.010 | 1.611 ± 0.017 | 11.76 ± 0.40 | 3 | 23 | 25 | — |
| Corrected-mode2-target50-it102 | 0.5 | 88.0% | 0.0% | 0.300 ± 0.010 | 1.609 ± 0.016 | 11.83 ± 0.41 | 3 | 22 | 25 | — |
| Corrected-mode2-target50-it102 | 0.7 | 92.0% | 0.0% | 0.298 ± 0.014 | 1.604 ± 0.012 | 12.42 ± 0.56 | 2 | 23 | 25 | — |
| Corrected-mode2-target50-it102 | 1.0 | 92.0% | 0.0% | 0.299 ± 0.013 | 1.598 ± 0.013 | 12.69 ± 0.62 | 2 | 23 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
