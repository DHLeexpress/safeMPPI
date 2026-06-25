# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=30 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 96.7 [90.0,100.0] | 0.0 [0.0,0.0] | 0.888/0.699 | 0.09 | 9.4 | 97.5 |
| guided_safemppi | 0.2 | 63.3 [46.7,80.0] | 10.0 [0.0,23.3] | 0.579/0.468 | 0.90 | 8.8 | 42.6 |
| cfm_proposal_mppi | 0.2 | 0.0 [0.0,0.0] | 6.7 [0.0,16.7] | 1.219/0.759 | 59.98 | 64.6 | 102.8 |
| guided_drifting | - | 0.0 [0.0,0.0] | 13.3 [3.3,26.7] | 1.095/0.664 | 62.92 | 67.7 | 3.2 |
| safe_cfm | 0.2 | 0.0 [0.0,0.0] | 46.7 [30.0,63.3] | 0.854/0.376 | 85.61 | 90.4 | 19.0 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.333 | 1/11 p=0.00635 | -0.196 p=0.00042 |
| cfm_proposal_mppi | 0.2 | -0.967 | 0/29 p=3.73e-09 | +0.127 p=0.191 |
| guided_drifting | - | -0.967 | 0/29 p=3.73e-09 | +0.044 p=0.721 |
| safe_cfm | 0.2 | -0.967 | 0/29 p=3.73e-09 | -0.554 p=0.347 |
