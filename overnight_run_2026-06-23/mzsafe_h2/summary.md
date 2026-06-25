# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 92.0 [80.0,100.0] | 0.0 [0.0,0.0] | 1.159/0.886 | 0.09 | 10.7 | 95.8 |
| mizuta_safe | 0.1 | 88.0 [76.0,100.0] | 0.0 [0.0,0.0] | 1.285/1.228 | 0.37 | 10.8 | 97.1 |
| mizuta_safe | 0.2 | 88.0 [76.0,100.0] | 4.0 [0.0,12.0] | 1.187/1.051 | 0.28 | 10.2 | 97.3 |
| mizuta_safe | 0.4 | 92.0 [80.0,100.0] | 4.0 [0.0,12.0] | 1.176/0.892 | 0.36 | 10.3 | 96.6 |
| mizuta_safe | 0.8 | 92.0 [80.0,100.0] | 4.0 [0.0,12.0] | 1.150/0.895 | 0.08 | 10.3 | 96.7 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| mizuta_safe | 0.1 | -0.040 | 0/1 p=1 | +0.049 p=0.0615 |
| mizuta_safe | 0.2 | -0.040 | 0/1 p=1 | +0.031 p=0.264 |
| mizuta_safe | 0.4 | +0.000 | 0/0 p=1 | -0.007 p=0.946 |
| mizuta_safe | 0.8 | +0.000 | 0/0 p=1 | -0.011 p=0.882 |
