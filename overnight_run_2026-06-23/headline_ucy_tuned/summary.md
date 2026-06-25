# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=100 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 98.0 [95.0,100.0] | 1.0 [0.0,3.0] | 0.998/0.818 | 0.06 | 10.0 | 103.8 |
| safemppi_gamma | 0.2 | 50.0 [40.0,60.0] | 50.0 [40.0,60.0] | 0.283/0.009 | 0.05 | 9.5 | 29.1 |
| guided_safemppi | 0.2 | 75.0 [66.0,83.0] | 7.0 [3.0,12.0] | 0.752/0.646 | 0.76 | 9.2 | 46.5 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| safemppi_gamma | 0.2 | -0.480 | 1/49 p=9.06e-14 | -0.697 p=2.17e-16 |
| guided_safemppi | 0.2 | -0.230 | 1/24 p=1.55e-06 | -0.193 p=8.18e-08 |
