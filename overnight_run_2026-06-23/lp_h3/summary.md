# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.118/1.131 | 0.04 | 9.8 | 99.9 |
| guided_safemppi | 0.2 | 76.0 [56.0,92.0] | 8.0 [0.0,20.0] | 0.801/0.866 | 1.09 | 9.1 | 43.4 |
| cfm_proposal_mppi | 0.2 | 0.0 [0.0,0.0] | 8.0 [0.0,20.0] | 0.925/0.874 | 42.00 | 48.9 | 146.4 |
| guided_drifting | - | 0.0 [0.0,0.0] | 4.0 [0.0,12.0] | 0.983/0.812 | 51.92 | 60.2 | 3.8 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.240 | 0/6 p=0.0312 | -0.311 p=0.000255 |
| cfm_proposal_mppi | 0.2 | -1.000 | 0/25 p=5.96e-08 | -0.172 p=0.145 |
| guided_drifting | - | -1.000 | 0/25 p=5.96e-08 | +0.000 p=0.605 |
