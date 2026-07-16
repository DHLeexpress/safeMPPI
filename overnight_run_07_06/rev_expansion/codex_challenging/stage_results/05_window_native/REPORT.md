# Window-native flow-expansion sanity

This is a **six-iteration semantic sanity** on the cleared 8-plug stadium task `(0.3,0.3) -> (4.7,4.7)`, not the giant-obstacle benchmark and not a 50-iteration result.

## Handoff alignment

- Aggregation unit: coherent executed H=10 window; whole-trajectory `traj_ok`, later collision, and goal reach are audit-only.
- Full / -Curriculum: task-space + progress + SOCP.
- -SOCP: task-space + progress + positive geometric clearance; SOCP is not called.
- -Progress: task-space + SOCP; progress is not called.
- All seven gamma values; no emergent gamma, recovery, hard quota, targeted proposal, demo, or LwF.
- The handoff prose says GP buffer 500, while its cited `faithful_g47/recipe.json` records 200/200. This sanity used 200/200 and is therefore comparable to that archived lineage on this knob.

## Results

| arm | accepted windows | contributors | whole-valid2 rollouts | progress evals | SOCP evals | M6 SR | M6 CR | goal dist [m] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 3378 | 42 | 0 | 6474 | 3653 | 0.000 | 1.000 | 4.286 |
| $-$SOCP | 3492 | 42 | 15 | 6806 | 0 | 0.000 | 0.976 | 4.300 |
| $-$Progress | 5003 | 43 | 8 | 0 | 5370 | 0.000 | 1.000 | 4.310 |
| $-$Curriculum | 3378 | 51 | 0 | 7963 | 4045 | 0.000 | 0.976 | 4.349 |

Full accepted-window counts by iteration: `437, 695, 554, 585, 410, 697`.

Controlled -Curriculum count match: **True**. It used a true single-class 16+0 batch; Full used 6+10. All updates had finite loss, nonzero functional step, and zero rollbacks.

The central finding is sample availability: Full collected thousands of locally certified windows even though no queried rollout was whole-valid2. This fixes the starvation mechanism. The M=6 deployment is still not successful after only six iterations, so this is evidence for the gather correction—not a performance claim.

## Artifacts

- `viz/prelim_rollouts.png` — Full and all three No brothers.
- `viz/prelim_training.png` — accepted counts, update size/loss, and M=6 deployment.
- `data/eval_m6/*/scorecard.json` and `paths_g*.npz` — matched faithful rollouts.
- `../../reference/analysis/test_window_expand.json` — 9/9 predicate and control checks.

**Gate:** pause before a 50-iteration run or resuming the giant-obstacle pipeline.
