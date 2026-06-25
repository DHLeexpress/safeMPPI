# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=40 samples=384 horizon=40

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 97.5 [92.5,100.0] | 0.0 [0.0,0.0] | 0.882/0.606 | 0.08 | 10.6 | 96.8 |
| guided_safemppi | 0.2 | 77.5 [65.0,90.0] | 12.5 [5.0,22.5] | 0.815/0.669 | 0.80 | 9.8 | 52.6 |
| guided_safemppi | 0.3 | 72.5 [57.5,87.5] | 5.0 [0.0,12.5] | 0.779/0.632 | 0.66 | 9.9 | 52.5 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.200 | 1/9 p=0.0215 | +0.020 p=0.738 |
| guided_safemppi | 0.3 | -0.250 | 0/10 p=0.00195 | -0.032 p=0.171 |
