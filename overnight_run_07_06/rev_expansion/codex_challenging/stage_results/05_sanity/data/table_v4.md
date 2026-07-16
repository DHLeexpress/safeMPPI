# Challenging sanity comparison (v4 style)

| γ | Method | SR% | CR% | Clearance (m) | Time (s) | Coverage |
|---:|---|---:|---:|---:|---:|---:|
| 0.1 | Demo expert (SafeMPPI) | 100 | 0 | 0.265 | 20.48 | 3 |
| 0.1 | Our approach | 0 | 100 | — | — | 0 |
| 0.1 | Pretrained | 0 | 100 | — | — | 0 |
| 0.1 | CFM-MPPI* (low guidance) | 0 | 100 | — | — | 0 |
| 0.1 | NO safety validity check | 17 | 83 | 0.196 | 10.80 | 1 |
| 0.1 | NO progress check | 0 | 100 | — | — | 0 |
| 0.1 | NO curriculum | 0 | 100 | — | — | 0 |
| 0.5 | Demo expert (SafeMPPI) | 100 | 0 | 0.234 | 10.78 | 2 |
| 0.5 | Our approach | 0 | 100 | — | — | 0 |
| 0.5 | Pretrained | 0 | 100 | — | — | 0 |
| 0.5 | CFM-MPPI* (low guidance) | 0 | 100 | — | — | 0 |
| 0.5 | NO safety validity check | 0 | 100 | — | — | 0 |
| 0.5 | NO progress check | 0 | 100 | — | — | 0 |
| 0.5 | NO curriculum | 0 | 100 | — | — | 0 |
| 1.0 | Demo expert (SafeMPPI) | 100 | 0 | 0.252 | 10.52 | 2 |
| 1.0 | Our approach | 0 | 100 | — | — | 0 |
| 1.0 | Pretrained | 0 | 100 | — | — | 0 |
| 1.0 | CFM-MPPI* (low guidance) | 17 | 83 | 0.216 | 4.30 | 1 |
| 1.0 | NO safety validity check | 0 | 100 | — | — | 0 |
| 1.0 | NO progress check | 0 | 100 | — | — | 0 |
| 1.0 | NO curriculum | 0 | 100 | — | — | 0 |

Sanity sample size is M=6 per gamma; this table is diagnostic, not a final claim.
