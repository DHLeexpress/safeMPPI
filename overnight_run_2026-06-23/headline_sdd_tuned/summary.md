# Pedestrian benchmark — moving obstacles

dataset=sdd dynamics=doubleintegrator episodes=100 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.150/0.922 | 0.05 | 9.7 | 95.5 |
| safemppi_gamma | 0.2 | 82.0 [74.0,89.0] | 18.0 [11.0,26.0] | 0.795/0.686 | 0.05 | 9.6 | 27.3 |
| guided_safemppi | 0.2 | 91.0 [85.0,96.0] | 1.0 [0.0,3.0] | 1.052/0.892 | 0.19 | 10.0 | 42.4 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| safemppi_gamma | 0.2 | -0.180 | 0/18 p=7.63e-06 | -0.217 p=3.98e-11 |
| guided_safemppi | 0.2 | -0.090 | 0/9 p=0.00391 | -0.009 p=0.0454 |
