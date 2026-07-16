# Origin-window / faithful-failure diagnostic

This is a read-only causal triage. `control condition` is the ratio of the two singular values of each centered 10×2 target-control matrix; it is an explicit numerical proxy, not a synonym for poor closed-loop behavior.

## Accepted training windows

| Snapshot | Windows | near origin | easy near / away | σ near / away | progress near / away | condition median near / away |
|---|---:|---:|---:|---:|---:|---:|
| mode2_it102 (it102) | 1783 | 20.2% | 89.5% / 87.8% | 0.765 / 0.451 | 0.522 / 0.619 | 1.67 / 1.57 |
| mode2_it103 (it103) | 2849 | 21.1% | 92.0% / 87.4% | 0.745 / 0.456 | 0.517 / 0.636 | 1.62 / 1.58 |
| mode2_it104 (it104) | 2540 | 21.3% | 90.8% / 90.6% | 0.766 / 0.442 | 0.502 / 0.634 | 1.54 / 1.59 |
| mode2_it105 (it105) | 2190 | 21.2% | 94.2% / 88.2% | 0.772 / 0.443 | 0.479 / 0.607 | 1.59 / 1.61 |

## Faithful evaluation failure taxonomy

| Evaluation | success | origin-boundary OOB | near-goal OOB | other OOB/nonreach | repeated origin seed(s) |
|---|---:|---:|---:|---:|---|
| rollback_it100 | 655 | 8 | 31 | 6 | 1:γ=[0.7], 12:γ=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] |
| mode2_it103 | 164 | 7 | 4 | 0 | 12:γ=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] |
| mode2_it104 | 164 | 7 | 4 | 0 | 12:γ=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] |

## Interpretation

- Near-origin accepted windows are high-uncertainty and mostly land in the easy pool because the frontier is a three-way AND cell. The table quantifies their share rather than calling all of them bad.
- The near-origin target-control condition number must be compared with the away group. Similar values argue against a numerical low-rank-window explanation.
- Repeated origin-boundary OOB for the same latent seed across γ is direct evidence of a faithful-flow tail failure at the boundary. Near-goal OOB is a separate overshoot mechanism and needs separate treatment.
- Do not loosen Valid2 or add an inference safety filter to make the metric pass. Diagnose and correct the generative tail while retaining faithful temp=1 evaluation.

