# Pedestrian benchmark — moving obstacles

dataset=ucy dynamics=doubleintegrator episodes=25 samples=384 horizon=40

| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |
|---|---|---|---|---|---|---|---|
| mizuta_cfm_mppi | - | 100.0 [100.0,100.0] | 0.0 [0.0,0.0] | 1.026/0.597 | 0.08 | 10.5 | 96.3 |
| guided_safemppi | 0.3 | 84.0 [68.0,96.0] | 0.0 [0.0,0.0] | 0.859/0.578 | 0.44 | 10.2 | 78.5 |
| guided_safemppi | 0.5 | 68.0 [48.0,84.0] | 12.0 [0.0,28.0] | 0.746/0.509 | 0.37 | 10.3 | 78.5 |
| guided_safemppi | 1 | 72.0 [56.0,88.0] | 16.0 [4.0,32.0] | 0.683/0.482 | 0.66 | 10.0 | 78.5 |
| guided_adaptive | - | 68.0 [48.0,84.0] | 20.0 [8.0,36.0] | 0.680/0.448 | 0.44 | 10.3 | 78.7 |

## Paired vs Mizuta (same episodes)
| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |
|---|---|---|---|---|
| guided_safemppi | 0.3 | -0.160 | 0/4 p=0.125 | -0.077 p=0.15 |
| guided_safemppi | 0.5 | -0.320 | 0/8 p=0.00781 | -0.105 p=0.0264 |
| guided_safemppi | 1 | -0.280 | 0/7 p=0.0156 | -0.184 p=0.0119 |
| guided_adaptive | - | -0.320 | 0/8 p=0.00781 | -0.103 p=0.0102 |
