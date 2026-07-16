# T4 — mode discovery beyond the teacher (per-γ M100 deployed staircase modes)

Expert = SafeMPPI P1 (`results/expert_gt`, M=100/γ): the SAME controller that produced every
demonstration the policy was ever pretrained on. A NEW mode = a staircase word the policy
deploys successfully (faithful temp=1/NFE8/reach=.1, collision-free) that the expert never
deployed at that γ in its own 100 trials.

## s792_M100  (`results/p2/eval_s792_m100`)

| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 10 | 4 | **6** | 4 | RRURUUURRU, RRUURURURU, RRUURUURRU, RRUUURURRU, RURUURRURU, RUURURURRU |
| 0.2 | 6 | 5 | 4 | **1** | 2 | RRURUURURU |
| 0.3 | 9 | 4 | 3 | **1** | 6 | RRURUURURU |
| 0.4 | 7 | 4 | 2 | **2** | 5 | RRURUURURU, RURURUURRU |
| 0.5 | 6 | 5 | 3 | **2** | 3 | RRURUURURU, RURURUURRU |
| 0.7 | 6 | 4 | 2 | **2** | 4 | RRURUURURU, RUURURURRU |
| 1.0 | 11 | 3 | 3 | **0** | 8 | — |

Total NEW modes across γ: **14**

| γ | SR | CR | clearance vs P1 | time vs P1 | coverage vs P1 |
|---|---|---|---|---|---|
| 0.1 | 100% | 0% | 0.307 vs 0.333 | 17.86 vs 15.13 | 10 vs 8 |
| 0.2 | 99% | 0% | 0.293 vs 0.290 | 13.55 vs 11.53 | 5 vs 6 |
| 0.3 | 97% | 2% | 0.296 vs 0.281 | 11.92 vs 10.99 | 4 vs 9 |
| 0.4 | 97% | 3% | 0.296 vs 0.282 | 11.56 vs 10.68 | 4 vs 7 |
| 0.5 | 97% | 2% | 0.296 vs 0.285 | 11.63 vs 10.54 | 5 vs 6 |
| 0.7 | 97% | 1% | 0.297 vs 0.287 | 12.12 vs 10.58 | 4 vs 6 |
| 1.0 | 97% | 0% | 0.297 vs 0.294 | 12.31 vs 10.76 | 3 vs 11 |

## t104_M100  (`results/p2/eval_corrected_mode2_it104_m100`)

| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 10 | 3 | **7** | 5 | RRURUURURU, RRUURURURU, RRUURUURRU, RRUUURURRU, RURUURRURU, RURUUURRRU, RUURURURRU |
| 0.2 | 6 | 7 | 4 | **3** | 2 | RRURURUURU, RRURUURURU, RRUURURURU |
| 0.3 | 9 | 4 | 3 | **1** | 6 | RRURUURURU |
| 0.4 | 7 | 4 | 1 | **3** | 6 | RRURUURURU, RURURUURRU, RUURURURRU |
| 0.5 | 6 | 4 | 2 | **2** | 4 | RRURUURURU, RURURUURRU |
| 0.7 | 6 | 5 | 2 | **3** | 4 | RRURUURURU, RURUURURRU, RUURURURRU |
| 1.0 | 11 | 6 | 6 | **0** | 5 | — |

Total NEW modes across γ: **19**

| γ | SR | CR | clearance vs P1 | time vs P1 | coverage vs P1 |
|---|---|---|---|---|---|
| 0.1 | 94% | 0% | 0.305 vs 0.333 | 18.23 vs 15.13 | 10 vs 8 |
| 0.2 | 96% | 1% | 0.295 vs 0.290 | 13.76 vs 11.53 | 7 vs 6 |
| 0.3 | 97% | 1% | 0.297 vs 0.281 | 12.18 vs 10.99 | 4 vs 9 |
| 0.4 | 96% | 0% | 0.299 vs 0.282 | 11.83 vs 10.68 | 4 vs 7 |
| 0.5 | 97% | 0% | 0.299 vs 0.285 | 11.89 vs 10.54 | 4 vs 6 |
| 0.7 | 96% | 0% | 0.298 vs 0.287 | 12.41 vs 10.58 | 5 vs 6 |
| 1.0 | 97% | 0% | 0.299 vs 0.294 | 12.62 vs 10.76 | 6 vs 11 |

