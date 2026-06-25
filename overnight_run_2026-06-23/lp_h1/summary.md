# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.108/0.734 | 0.04 | 10.1 | 95.0 |
| guided_safemppi | 0.2 | 84.0 [68.0,96.0] | 4.0 [0.0,12.0] | 0.803/0.648 | 0.36 | 9.7 | 43.4 |
| cfm_proposal_mppi | 0.2 | 0.0 [0.0,0.0] | 0.0 [0.0,0.0] | 1.114/0.822 | 45.75 | 53.6 | 104.0 |
| guided_drifting | - | 0.0 [0.0,0.0] | 12.0 [0.0,24.0] | 0.797/0.671 | 41.47 | 49.7 | 4.0 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.160 | 0/4 p=0.125 | -0.154 p=0.0283 |
| cfm_proposal_mppi | 0.2 | -1.000 | 0/25 p=5.96e-08 | -0.017 p=0.989 |
| guided_drifting | - | -1.000 | 0/25 p=5.96e-08 | -0.002 p=0.264 |
