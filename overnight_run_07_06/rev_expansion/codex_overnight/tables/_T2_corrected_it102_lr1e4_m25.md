# Corrected coherent it102 lr1e-4 M25

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected-coherent-it102 | 0.1 | 92.0% | 0.0% | 0.309 ± 0.014 | 1.508 ± 0.048 | 18.71 ± 1.84 | 6 | 23 | 25 | — |
| Corrected-coherent-it102 | 0.2 | 80.0% | 0.0% | 0.297 ± 0.011 | 1.574 ± 0.039 | 14.18 ± 1.02 | 4 | 20 | 25 | — |
| Corrected-coherent-it102 | 0.3 | 84.0% | 0.0% | 0.296 ± 0.012 | 1.603 ± 0.016 | 12.50 ± 0.65 | 2 | 21 | 25 | — |
| Corrected-coherent-it102 | 0.4 | 80.0% | 0.0% | 0.295 ± 0.017 | 1.605 ± 0.017 | 12.22 ± 0.87 | 2 | 20 | 25 | — |
| Corrected-coherent-it102 | 0.5 | 76.0% | 0.0% | 0.297 ± 0.015 | 1.604 ± 0.017 | 12.22 ± 0.80 | 2 | 19 | 25 | — |
| Corrected-coherent-it102 | 0.7 | 84.0% | 0.0% | 0.295 ± 0.016 | 1.600 ± 0.013 | 12.78 ± 0.88 | 2 | 21 | 25 | — |
| Corrected-coherent-it102 | 1.0 | 88.0% | 0.0% | 0.296 ± 0.013 | 1.590 ± 0.017 | 13.19 ± 0.78 | 2 | 22 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
