# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=512 horizon=30

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 96.0 [88.0,100.0] | 0.0 [0.0,0.0] | 0.794/0.574 | 0.10 | 9.4 | 96.2 |
| mizuta_safe | 0.1 | 88.0 [76.0,100.0] | 4.0 [0.0,12.0] | 1.034/0.788 | 0.62 | 9.4 | 96.9 |
| mizuta_safe | 0.2 | 88.0 [72.0,100.0] | 0.0 [0.0,0.0] | 0.958/0.765 | 0.25 | 9.6 | 96.6 |
| mizuta_safe | 0.4 | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 0.828/0.678 | 0.08 | 9.3 | 96.9 |
| mizuta_safe | 0.8 | 92.0 [80.0,100.0] | 4.0 [0.0,12.0] | 0.901/0.763 | 0.15 | 9.7 | 96.6 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| mizuta_safe | 0.1 | -0.080 | 0/2 p=0.5 | +0.076 p=0.000607 |
| mizuta_safe | 0.2 | -0.080 | 0/2 p=0.5 | +0.136 p=0.00468 |
| mizuta_safe | 0.4 | +0.040 | 1/0 p=1 | +0.038 p=0.361 |
| mizuta_safe | 0.8 | -0.040 | 0/1 p=1 | +0.053 p=0.0177 |
