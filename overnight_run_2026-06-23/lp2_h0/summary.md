# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 92.0 [80.0,100.0] | 4.0 [0.0,12.0] | 0.825/0.697 | 0.09 | 9.4 | 96.7 |
| guided_safemppi | 0.2 | 64.0 [44.0,80.0] | 8.0 [0.0,20.0] | 0.629/0.470 | 0.94 | 8.6 | 43.6 |
| cfm_proposal_mppi | 0.2 | 68.0 [48.0,84.0] | 8.0 [0.0,20.0] | 0.631/0.486 | 1.21 | 9.3 | 110.3 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.280 | 2/9 p=0.0654 | -0.240 p=0.0247 |
| cfm_proposal_mppi | 0.2 | -0.240 | 2/8 p=0.109 | -0.166 p=0.0303 |
