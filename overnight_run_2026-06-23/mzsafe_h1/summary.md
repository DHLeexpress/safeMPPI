# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.055/0.733 | 0.04 | 10.2 | 95.8 |
| mizuta_safe | 0.1 | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 1.312/1.050 | 0.25 | 10.3 | 97.3 |
| mizuta_safe | 0.2 | 92.0 [80.0,100.0] | 0.0 [0.0,0.0] | 1.214/0.934 | 0.19 | 10.0 | 96.9 |
| mizuta_safe | 0.4 | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 1.204/0.915 | 0.07 | 10.1 | 97.0 |
| mizuta_safe | 0.8 | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.132/0.841 | 0.06 | 10.0 | 96.6 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| mizuta_safe | 0.1 | -0.040 | 0/1 p=1 | +0.031 p=0.00564 |
| mizuta_safe | 0.2 | -0.080 | 0/2 p=0.5 | +0.040 p=0.00511 |
| mizuta_safe | 0.4 | -0.040 | 0/1 p=1 | +0.032 p=0.0207 |
| mizuta_safe | 0.8 | +0.000 | 0/0 p=1 | +0.031 p=0.153 |
