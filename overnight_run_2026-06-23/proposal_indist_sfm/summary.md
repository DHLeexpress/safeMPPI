# Pedestrian benchmark — moving obstacles

dataset=sfm dynamics=doubleintegrator episodes=20 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 65.0 [45.0,85.0] | 0.0 [0.0,0.0] | 0.273/0.246 | 0.84 | 12.1 | 98.3 |
| guided_safemppi | 0.2 | 0.0 [0.0,0.0] | 75.0 [55.0,90.0] | -0.120/-0.224 | 4.36 | 6.1 | 43.4 |
| cfm_proposal_mppi | 0.2 | 0.0 [0.0,0.0] | 55.0 [35.0,75.0] | -0.063/-0.035 | 6.37 | 8.2 | 103.7 |
| guided_drifting | - | 0.0 [0.0,0.0] | 60.0 [35.0,80.0] | -0.022/-0.087 | 7.17 | 11.5 | 4.5 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.650 | 0/13 p=0.000244 | -0.436 p=0.00039 |
| cfm_proposal_mppi | 0.2 | -0.650 | 0/13 p=0.000244 | -0.345 p=0.00039 |
| guided_drifting | - | -0.650 | 0/13 p=0.000244 | -0.297 p=0.00039 |
