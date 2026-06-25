# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=384 horizon=40

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 0.955/0.566 | 0.08 | 10.6 | 102.1 |
| guided_safemppi | 0.3 | 76.0 [60.0,92.0] | 4.0 [0.0,12.0] | 0.747/0.513 | 0.40 | 10.2 | 79.5 |
| guided_safemppi | 0.5 | 76.0 [60.0,92.0] | 12.0 [0.0,28.0] | 0.693/0.414 | 0.36 | 10.2 | 79.8 |
| guided_safemppi | 1 | 88.0 [72.0,100.0] | 8.0 [0.0,20.0] | 0.660/0.359 | 0.26 | 10.3 | 79.6 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.3 | -0.240 | 0/6 p=0.0312 | -0.103 p=0.0693 |
| guided_safemppi | 0.5 | -0.240 | 0/6 p=0.0312 | -0.100 p=0.0199 |
| guided_safemppi | 1 | -0.120 | 0/3 p=0.25 | -0.158 p=0.00942 |
