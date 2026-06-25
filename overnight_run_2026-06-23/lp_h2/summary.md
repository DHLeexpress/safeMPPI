# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 0.978/0.896 | 0.08 | 10.5 | 95.2 |
| guided_safemppi | 0.2 | 76.0 [60.0,92.0] | 8.0 [0.0,20.0] | 0.775/0.748 | 0.63 | 9.5 | 43.6 |
| cfm_proposal_mppi | 0.2 | 0.0 [0.0,0.0] | 0.0 [0.0,0.0] | 1.428/1.319 | 52.43 | 60.2 | 104.3 |
| guided_drifting | - | 0.0 [0.0,0.0] | 8.0 [0.0,20.0] | 0.825/0.747 | 44.81 | 53.0 | 4.0 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.200 | 0/5 p=0.0625 | -0.115 p=0.0173 |
| cfm_proposal_mppi | 0.2 | -0.960 | 0/24 p=1.19e-07 | +0.379 p=0.00416 |
| guided_drifting | - | -0.960 | 0/24 p=1.19e-07 | -0.131 p=0.221 |
