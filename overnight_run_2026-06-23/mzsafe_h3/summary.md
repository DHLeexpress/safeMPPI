# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.034/0.990 | 0.03 | 9.8 | 97.4 |
| mizuta_safe | 0.1 | 88.0 [75.9,100.0] | 0.0 [0.0,0.0] | 1.370/1.254 | 0.24 | 9.9 | 98.9 |
| mizuta_safe | 0.2 | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 1.270/1.196 | 0.16 | 9.9 | 98.0 |
| mizuta_safe | 0.4 | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 1.118/1.174 | 0.06 | 9.7 | 98.2 |
| mizuta_safe | 0.8 | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 1.164/1.062 | 0.06 | 9.8 | 98.5 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| mizuta_safe | 0.1 | -0.120 | 0/3 p=0.25 | +0.117 p=0.000615 |
| mizuta_safe | 0.2 | -0.040 | 0/1 p=1 | +0.019 p=0.0186 |
| mizuta_safe | 0.4 | -0.040 | 0/1 p=1 | +0.000 p=0.394 |
| mizuta_safe | 0.8 | -0.040 | 0/1 p=1 | +0.017 p=0.00973 |
