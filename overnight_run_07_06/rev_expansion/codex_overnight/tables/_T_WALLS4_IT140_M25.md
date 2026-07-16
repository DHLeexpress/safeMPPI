# WALLS-4 iteration-140 decision gate (M25)

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full pipeline it140 M25 | 0.1 | 64.0% | 8.0% | 0.226 ± 0.010 | 1.527 ± 0.020 | 12.41 ± 0.94 | 3 | 16 | 25 | — |
| Full pipeline it140 M25 | 0.2 | 76.0% | 16.0% | 0.223 ± 0.006 | 1.556 ± 0.009 | 9.31 ± 0.35 | 2 | 19 | 25 | — |
| Full pipeline it140 M25 | 0.3 | 76.0% | 20.0% | 0.224 ± 0.005 | 1.562 ± 0.006 | 8.42 ± 0.27 | 2 | 19 | 25 | — |
| Full pipeline it140 M25 | 0.4 | 80.0% | 20.0% | 0.223 ± 0.005 | 1.563 ± 0.006 | 8.16 ± 0.24 | 2 | 20 | 25 | — |
| Full pipeline it140 M25 | 0.5 | 76.0% | 20.0% | 0.223 ± 0.005 | 1.563 ± 0.006 | 8.13 ± 0.25 | 2 | 19 | 25 | — |
| Full pipeline it140 M25 | 0.7 | 76.0% | 20.0% | 0.225 ± 0.005 | 1.561 ± 0.006 | 8.35 ± 0.25 | 2 | 19 | 25 | — |
| Full pipeline it140 M25 | 1.0 | 64.0% | 20.0% | 0.228 ± 0.005 | 1.560 ± 0.006 | 8.51 ± 0.29 | 2 | 16 | 25 | — |
| No curriculum it140 M25 | 0.1 | 80.0% | 4.0% | 0.237 ± 0.009 | 1.454 ± 0.032 | 17.45 ± 1.46 | 5 | 20 | 25 | — |
| No curriculum it140 M25 | 0.2 | 72.0% | 4.0% | 0.224 ± 0.011 | 1.508 ± 0.031 | 12.30 ± 1.29 | 5 | 18 | 25 | — |
| No curriculum it140 M25 | 0.3 | 48.0% | 32.0% | 0.224 ± 0.010 | 1.540 ± 0.019 | 10.17 ± 0.43 | 3 | 12 | 25 | — |
| No curriculum it140 M25 | 0.4 | 52.0% | 32.0% | 0.223 ± 0.010 | 1.548 ± 0.013 | 9.65 ± 0.40 | 3 | 13 | 25 | — |
| No curriculum it140 M25 | 0.5 | 48.0% | 32.0% | 0.225 ± 0.008 | 1.547 ± 0.013 | 9.66 ± 0.46 | 3 | 12 | 25 | — |
| No curriculum it140 M25 | 0.7 | 60.0% | 16.0% | 0.225 ± 0.010 | 1.533 ± 0.022 | 10.31 ± 0.76 | 4 | 15 | 25 | — |
| No curriculum it140 M25 | 1.0 | 48.0% | 28.0% | 0.228 ± 0.009 | 1.527 ± 0.024 | 10.55 ± 0.56 | 4 | 12 | 25 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
