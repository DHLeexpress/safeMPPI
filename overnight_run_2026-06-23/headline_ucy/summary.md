# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=100 samples=384 horizon=40

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 99.0 [97.0,100.0] | 0.0 [0.0,0.0] | 0.973/0.734 | 0.06 | 10.4 | 89.5 |
| safemppi_gamma | 0.2 | 39.0 [30.0,49.0] | 61.0 [51.0,70.0] | 0.255/-0.141 | 0.14 | 9.9 | 35.8 |
| safemppi_gamma | 0.3 | 40.0 [31.0,50.0] | 60.0 [50.0,69.0] | 0.258/-0.127 | 0.14 | 10.0 | 35.8 |
| guided_safemppi | 0.2 | 74.0 [66.0,82.0] | 12.0 [6.0,18.0] | 0.750/0.637 | 0.71 | 9.7 | 51.0 |
| guided_safemppi | 0.3 | 70.0 [61.0,78.0] | 12.0 [6.0,18.0] | 0.700/0.556 | 0.66 | 9.8 | 51.0 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| safemppi_gamma | 0.2 | -0.600 | 0/60 p=1.73e-18 | -0.653 p=2.5e-16 |
| safemppi_gamma | 0.3 | -0.590 | 0/59 p=3.47e-18 | -0.662 p=2.16e-16 |
| guided_safemppi | 0.2 | -0.250 | 1/26 p=4.17e-07 | -0.120 p=2.31e-05 |
| guided_safemppi | 0.3 | -0.290 | 1/30 p=2.98e-08 | -0.146 p=9.65e-08 |
