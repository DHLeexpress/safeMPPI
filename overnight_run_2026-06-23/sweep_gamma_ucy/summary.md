# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=384 horizon=40

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 0.954/0.602 | 0.06 | 10.5 | 106.1 |
| safemppi_gamma | 0.3 | 40.0 [20.0,60.0] | 60.0 [40.0,80.0] | 0.393/-0.102 | 0.17 | 10.2 | 36.3 |
| safemppi_gamma | 0.5 | 40.0 [20.0,60.0] | 60.0 [40.0,80.0] | 0.391/-0.102 | 0.14 | 10.2 | 36.3 |
| safemppi_gamma | 0.7 | 40.0 [20.0,60.0] | 60.0 [40.0,80.0] | 0.393/-0.102 | 0.13 | 10.2 | 36.2 |
| safemppi_gamma | 1 | 40.0 [20.0,60.0] | 60.0 [40.0,80.0] | 0.404/-0.076 | 0.16 | 10.2 | 36.1 |
| guided_safemppi | 0.3 | 80.0 [64.0,96.0] | 8.0 [0.0,20.0] | 0.738/0.449 | 0.37 | 10.1 | 77.1 |
| guided_safemppi | 0.5 | 64.0 [44.0,84.0] | 28.0 [12.0,48.0] | 0.600/0.348 | 0.41 | 10.2 | 77.1 |
| guided_safemppi | 0.7 | 68.0 [51.9,84.0] | 20.0 [8.0,36.0] | 0.586/0.269 | 0.39 | 10.3 | 77.1 |
| guided_safemppi | 1 | 84.0 [68.0,96.0] | 12.0 [0.0,24.0] | 0.627/0.384 | 0.24 | 10.3 | 77.0 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| safemppi_gamma | 0.3 | -0.600 | 0/15 p=6.1e-05 | -0.562 p=0.00014 |
| safemppi_gamma | 0.5 | -0.600 | 0/15 p=6.1e-05 | -0.562 p=0.00014 |
| safemppi_gamma | 0.7 | -0.600 | 0/15 p=6.1e-05 | -0.562 p=0.00014 |
| safemppi_gamma | 1 | -0.600 | 0/15 p=6.1e-05 | -0.562 p=0.00014 |
| guided_safemppi | 0.3 | -0.200 | 0/5 p=0.0625 | -0.142 p=0.00942 |
| guided_safemppi | 0.5 | -0.360 | 0/9 p=0.00391 | -0.187 p=0.00189 |
| guided_safemppi | 0.7 | -0.320 | 0/8 p=0.00781 | -0.296 p=0.000733 |
| guided_safemppi | 1 | -0.160 | 0/4 p=0.125 | -0.142 p=0.00322 |
