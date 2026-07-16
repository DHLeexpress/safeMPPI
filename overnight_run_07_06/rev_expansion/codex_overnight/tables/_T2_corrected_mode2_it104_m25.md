# Corrected mode-quota targeted checkpoint iteration 104 (M=25)

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected-mode2-target50-it104 | 0.1 | 92.0% | 0.0% | 0.305 ± 0.015 | 1.515 ± 0.046 | 18.20 ± 1.92 | 6 | 23 | 25 | — |
| Corrected-mode2-target50-it104 | 0.2 | 96.0% | 0.0% | 0.294 ± 0.014 | 1.585 ± 0.021 | 13.80 ± 1.01 | 4 | 24 | 25 | — |
| Corrected-mode2-target50-it104 | 0.3 | 96.0% | 0.0% | 0.295 ± 0.012 | 1.606 ± 0.018 | 12.10 ± 0.53 | 3 | 24 | 25 | — |
| Corrected-mode2-target50-it104 | 0.4 | 92.0% | 0.0% | 0.297 ± 0.012 | 1.609 ± 0.017 | 11.78 ± 0.56 | 3 | 23 | 25 | — |
| Corrected-mode2-target50-it104 | 0.5 | 92.0% | 0.0% | 0.298 ± 0.011 | 1.611 ± 0.017 | 11.73 ± 0.40 | 3 | 23 | 25 | — |
| Corrected-mode2-target50-it104 | 0.7 | 92.0% | 0.0% | 0.296 ± 0.013 | 1.603 ± 0.016 | 12.32 ± 0.68 | 2 | 23 | 25 | — |
| Corrected-mode2-target50-it104 | 1.0 | 96.0% | 0.0% | 0.297 ± 0.014 | 1.594 ± 0.018 | 12.65 ± 0.68 | 3 | 24 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
