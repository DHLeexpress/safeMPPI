# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 1.187/0.901 | 0.06 | 10.4 | 97.7 |
| guided_safemppi | 0.2 | 76.0 [60.0,92.0] | 8.0 [0.0,20.0] | 0.775/0.748 | 0.63 | 9.5 | 43.5 |
| cfm_proposal_mppi | 0.2 | 76.0 [60.0,92.0] | 4.0 [0.0,12.0] | 0.868/0.804 | 0.67 | 9.7 | 110.9 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.200 | 0/5 p=0.0625 | -0.246 p=0.00189 |
| cfm_proposal_mppi | 0.2 | -0.200 | 0/5 p=0.0625 | -0.292 p=0.0149 |
