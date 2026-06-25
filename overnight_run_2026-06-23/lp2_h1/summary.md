# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.086/0.827 | 0.04 | 9.9 | 96.3 |
| guided_safemppi | 0.2 | 84.0 [68.0,96.0] | 4.0 [0.0,12.0] | 0.803/0.648 | 0.36 | 9.7 | 43.7 |
| cfm_proposal_mppi | 0.2 | 80.0 [64.0,96.0] | 4.0 [0.0,12.0] | 0.920/0.766 | 0.88 | 10.3 | 110.0 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.160 | 0/4 p=0.125 | -0.163 p=0.00453 |
| cfm_proposal_mppi | 0.2 | -0.200 | 0/5 p=0.0625 | -0.074 p=0.078 |
