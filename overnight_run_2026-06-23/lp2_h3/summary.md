# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.196/1.158 | 0.04 | 9.9 | 99.8 |
| guided_safemppi | 0.2 | 76.0 [56.0,92.0] | 8.0 [0.0,20.0] | 0.801/0.866 | 1.09 | 9.1 | 46.2 |
| cfm_proposal_mppi | 0.2 | 80.0 [64.0,96.0] | 8.0 [0.0,20.0] | 0.790/0.834 | 1.01 | 9.1 | 112.4 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.2 | -0.240 | 0/6 p=0.0312 | -0.271 p=4.03e-05 |
| cfm_proposal_mppi | 0.2 | -0.200 | 0/5 p=0.0625 | -0.330 p=6.77e-05 |
