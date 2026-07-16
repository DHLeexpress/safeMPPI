# SafeMPPI canonical sanity, M=6

| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SafeMPPI-sanity | 0.1 | 100.0% | 0.0% | 0.265 ± 0.007 | 1.373 ± 0.027 | 20.48 ± 1.57 | 3 | 6 | 6 | — |
| SafeMPPI-sanity | 0.2 | 100.0% | 0.0% | 0.241 ± 0.008 | 1.434 ± 0.007 | 11.80 ± 0.39 | 2 | 6 | 6 | — |
| SafeMPPI-sanity | 0.3 | 100.0% | 0.0% | 0.220 ± 0.009 | 1.424 ± 0.016 | 11.55 ± 0.54 | 3 | 6 | 6 | — |
| SafeMPPI-sanity | 0.4 | 100.0% | 0.0% | 0.240 ± 0.004 | 1.435 ± 0.016 | 10.57 ± 0.49 | 3 | 6 | 6 | — |
| SafeMPPI-sanity | 0.5 | 100.0% | 0.0% | 0.234 ± 0.013 | 1.432 ± 0.005 | 10.78 ± 0.60 | 2 | 6 | 6 | — |
| SafeMPPI-sanity | 0.7 | 100.0% | 0.0% | 0.245 ± 0.008 | 1.431 ± 0.006 | 10.78 ± 0.30 | 2 | 6 | 6 | — |
| SafeMPPI-sanity | 1.0 | 100.0% | 0.0% | 0.252 ± 0.008 | 1.426 ± 0.014 | 10.52 ± 0.50 | 2 | 6 | 6 | — |

Clearance is the successful-episode mean over time of the nearest-obstacle clearance; 'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).
Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.
