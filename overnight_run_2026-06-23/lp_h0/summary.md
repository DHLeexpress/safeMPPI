# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 92.0 [80.0,100.0] | 4.0 [0.0,12.0] | 0.789/0.463 | 0.11 | 9.6 | 98.3 |
| guided_safemppi | 0.2 | 64.0 [44.0,80.0] | 8.0 [0.0,20.0] | 0.629/0.470 | 0.94 | 8.6 | 43.4 |
| cfm_proposal_mppi | 0.2 | 0.0 [0.0,0.0] | 16.0 [4.0,32.0] | 0.693/0.758 | 37.16 | 44.3 | 104.2 |
| guided_drifting | - | 0.0 [0.0,0.0] | 8.0 [0.0,20.0] | 0.799/0.808 | 43.83 | 51.7 | 3.9 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.280 | 2/9 p=0.0654 | -0.125 p=0.037 |
| cfm_proposal_mppi | 0.2 | -0.920 | 0/23 p=2.38e-07 | +0.007 p=0.946 |
| guided_drifting | - | -0.920 | 0/23 p=2.38e-07 | +0.000 p=0.976 |
