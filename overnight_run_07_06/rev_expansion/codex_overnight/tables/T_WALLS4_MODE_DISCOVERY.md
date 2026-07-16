# T4 — mode discovery beyond the teacher (per-γ M100 deployed staircase modes)

Expert = SafeMPPI (`results/expert_gt_walls4`, M=100/γ): the SAME controller that produced every
demonstration the policy was ever pretrained on. A NEW mode = a staircase word the policy
deploys successfully (faithful temp=1/NFE8/reach=.1, collision-free) that the expert never
deployed at that γ in its own 100 trials.

## full_it100  (`results/p2/eval_walls4_base_it100_m100`)

| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 8 | 7 | **1** | 1 | RURUURURRU |
| 0.2 | 5 | 3 | 3 | **0** | 2 | — |
| 0.3 | 9 | 2 | 2 | **0** | 7 | — |
| 0.4 | 7 | 2 | 2 | **0** | 5 | — |
| 0.5 | 6 | 2 | 2 | **0** | 4 | — |
| 0.7 | 7 | 2 | 2 | **0** | 5 | — |
| 1.0 | 7 | 2 | 2 | **0** | 5 | — |

Total NEW modes across γ: **1**

| γ | SR | CR | clearance vs expert | time vs expert | coverage vs expert |
|---|---|---|---|---|---|
| 0.1 | 71% | 3% | 0.232 vs 0.299 | 15.45 vs 15.95 | 8 vs 8 |
| 0.2 | 75% | 13% | 0.229 vs 0.248 | 10.72 vs 11.82 | 3 vs 5 |
| 0.3 | 70% | 24% | 0.227 vs 0.240 | 9.50 vs 11.08 | 2 vs 9 |
| 0.4 | 68% | 24% | 0.226 vs 0.240 | 9.13 vs 10.87 | 2 vs 7 |
| 0.5 | 69% | 23% | 0.226 vs 0.243 | 9.11 vs 10.78 | 2 vs 6 |
| 0.7 | 76% | 16% | 0.228 vs 0.247 | 9.43 vs 10.70 | 2 vs 7 |
| 1.0 | 70% | 16% | 0.230 vs 0.254 | 9.65 vs 10.87 | 2 vs 7 |

## no_curriculum_it100  (`results/p2/eval_walls4_nocur_it100_m100`)

| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 14 | 6 | **8** | 2 | RRURUURURU, RRUURURURU, RRUURUURRU, RRUUUURRRU, RURUURURRU, RUURRUURRU, RUURURRURU, URRURUURRU |
| 0.2 | 5 | 7 | 5 | **2** | 0 | RUURURURRU, URURURRURU |
| 0.3 | 9 | 4 | 4 | **0** | 5 | — |
| 0.4 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |
| 0.5 | 6 | 4 | 3 | **1** | 3 | RURUURURRU |
| 0.7 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |
| 1.0 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |

Total NEW modes across γ: **14**

| γ | SR | CR | clearance vs expert | time vs expert | coverage vs expert |
|---|---|---|---|---|---|
| 0.1 | 56% | 17% | 0.254 vs 0.299 | 20.87 vs 15.95 | 14 vs 8 |
| 0.2 | 71% | 1% | 0.229 vs 0.248 | 13.34 vs 11.82 | 7 vs 5 |
| 0.3 | 79% | 7% | 0.229 vs 0.240 | 11.19 vs 11.08 | 4 vs 9 |
| 0.4 | 74% | 14% | 0.229 vs 0.240 | 10.63 vs 10.87 | 4 vs 7 |
| 0.5 | 76% | 12% | 0.230 vs 0.243 | 10.62 vs 10.78 | 4 vs 6 |
| 0.7 | 76% | 6% | 0.231 vs 0.247 | 11.21 vs 10.70 | 4 vs 7 |
| 1.0 | 71% | 6% | 0.233 vs 0.254 | 11.71 vs 10.87 | 4 vs 7 |

## no_SOCP_it100  (`results/p2/eval_walls4_nosocp_it100_m100`)

| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 6 | 4 | **2** | 4 | RURUURURRU, RURUUURRRU |
| 0.2 | 5 | 2 | 2 | **0** | 3 | — |
| 0.3 | 9 | 3 | 3 | **0** | 6 | — |
| 0.4 | 7 | 2 | 2 | **0** | 5 | — |
| 0.5 | 6 | 3 | 2 | **1** | 4 | RURUURURRU |
| 0.7 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |
| 1.0 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |

Total NEW modes across γ: **5**

| γ | SR | CR | clearance vs expert | time vs expert | coverage vs expert |
|---|---|---|---|---|---|
| 0.1 | 54% | 25% | 0.222 vs 0.299 | 12.45 vs 15.95 | 6 vs 8 |
| 0.2 | 33% | 54% | 0.226 vs 0.248 | 9.64 vs 11.82 | 2 vs 5 |
| 0.3 | 38% | 46% | 0.223 vs 0.240 | 8.86 vs 11.08 | 3 vs 9 |
| 0.4 | 39% | 45% | 0.224 vs 0.240 | 8.62 vs 10.87 | 2 vs 7 |
| 0.5 | 38% | 44% | 0.224 vs 0.243 | 8.67 vs 10.78 | 3 vs 6 |
| 0.7 | 40% | 49% | 0.225 vs 0.247 | 8.96 vs 10.70 | 4 vs 7 |
| 1.0 | 37% | 46% | 0.228 vs 0.254 | 9.13 vs 10.87 | 4 vs 7 |

## no_progress_it100  (`results/p2/eval_walls4_noprog_it100_m100`)

| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 6 | 4 | **2** | 4 | RURUURURRU, RUURURRURU |
| 0.2 | 5 | 4 | 4 | **0** | 1 | — |
| 0.3 | 9 | 4 | 4 | **0** | 5 | — |
| 0.4 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |
| 0.5 | 6 | 4 | 3 | **1** | 3 | RURUURURRU |
| 0.7 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |
| 1.0 | 7 | 4 | 3 | **1** | 4 | RURURUURRU |

Total NEW modes across γ: **6**

| γ | SR | CR | clearance vs expert | time vs expert | coverage vs expert |
|---|---|---|---|---|---|
| 0.1 | 70% | 4% | 0.237 vs 0.299 | 16.10 vs 15.95 | 6 vs 8 |
| 0.2 | 67% | 15% | 0.229 vs 0.248 | 10.94 vs 11.82 | 4 vs 5 |
| 0.3 | 67% | 24% | 0.227 vs 0.240 | 9.60 vs 11.08 | 4 vs 9 |
| 0.4 | 69% | 21% | 0.225 vs 0.240 | 9.20 vs 10.87 | 4 vs 7 |
| 0.5 | 67% | 22% | 0.226 vs 0.243 | 9.16 vs 10.78 | 4 vs 6 |
| 0.7 | 69% | 19% | 0.227 vs 0.247 | 9.48 vs 10.70 | 4 vs 7 |
| 1.0 | 58% | 23% | 0.230 vs 0.254 | 9.75 vs 10.87 | 4 vs 7 |

