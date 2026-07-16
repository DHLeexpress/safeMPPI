# Origin-window / faithful-failure diagnostic

This is a read-only causal triage. `control condition` is the ratio of the two singular values of each centered 10×2 target-control matrix; it is an explicit numerical proxy, not a synonym for poor closed-loop behavior.

## Accepted training windows

| Snapshot | Windows | near origin | easy near / away | σ near / away | progress near / away | condition median near / away |
|---|---:|---:|---:|---:|---:|---:|

## Faithful evaluation failure taxonomy

| Evaluation | success | origin-boundary OOB | near-goal OOB | other OOB/nonreach | repeated origin seed(s) |
|---|---:|---:|---:|---:|---|
| mode2_it104_m100 | 673 | 7 | 18 | 2 | 12:γ=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] |

## Interpretation

- Near-origin accepted windows are high-uncertainty and mostly land in the easy pool because the frontier is a three-way AND cell. The table quantifies their share rather than calling all of them bad.
- The near-origin target-control condition number must be compared with the away group. Similar values argue against a numerical low-rank-window explanation.
- Repeated origin-boundary OOB for the same latent seed across γ is direct evidence of a faithful-flow tail failure at the boundary. Near-goal OOB is a separate overshoot mechanism and needs separate treatment.
- Do not loosen Valid2 or add an inference safety filter to make the metric pass. Diagnose and correct the generative tail while retaining faithful temp=1 evaluation.

