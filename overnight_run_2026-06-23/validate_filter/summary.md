# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=40 samples=384 horizon=40

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 97.5 [92.5,100.0] | 2.5 [0.0,7.5] | 0.873/0.616 | 0.07 | 10.3 | 96.1 |
| guided_safemppi | 0.2 | 77.5 [65.0,90.0] | 7.5 [0.0,15.0] | 0.838/0.717 | 0.81 | 9.7 | 52.8 |
| guided_safemppi | 0.3 | 75.0 [62.5,87.5] | 5.0 [0.0,12.5] | 0.789/0.613 | 0.77 | 9.8 | 52.8 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.200 | 0/8 p=0.00781 | +0.011 p=0.967 |
| guided_safemppi | 0.3 | -0.225 | 0/9 p=0.00391 | -0.050 p=0.176 |
