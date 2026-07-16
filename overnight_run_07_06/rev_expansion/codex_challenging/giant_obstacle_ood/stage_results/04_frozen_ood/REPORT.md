# Stage 4 — frozen giant-obstacle OOD baselines

## Outcome

The radius-1.2 scene remains expert-feasible: **42/42 SafeMPPI successes with 0 collisions**. The frozen pretrained policy obtains **0/112 successes and 112 collisions**. The selected bounded low-guidance CFM-MPPI* setting obtains **0/42 successes and 0 collisions**, with 100.0% explicitly classified local-minimum timeouts.

| method | M | a: SR | b: CR | c: clearance (success) | d: time (success) | e: homotopies | local-min |
|---|---:|---:|---:|---:|---:|---:|---:|
| SafeMPPI expert | 42 | 100.0% | 0.0% | 0.168 m | 22.19 s | 2/2 | 0.0% |
| Frozen pretrained (T=0.1) | 112 | 0.0% | 100.0% | — | — | 0/2 | 0.0% |
| CFM-MPPI* low guidance | 42 | 0.0% | 0.0% | — | — | 0/2 | 100.0% |

## Low-guidance Mizuta selection

Selected `lg040`: `w_safe=0.04`, `coll_w=4.0`, `goal_w=2.0`, `goal_coef=0.2`. The bounded sweep promoted performance first, then collision rate, goal progress, inherited pre-obstacle diagonal behavior, and finally lower guidance. It did not reward failure or trapping. Obstacle radii are modeled per obstacle, so the 1.2 m circle is not collapsed to the mean small-obstacle radius.

## Why the pretrained paths look smooth

The dominant cause is the approved source temperature 0.1: it suppresses high-variance acceleration tails before each H=10 flow window is decoded. In these exact raw rollouts, mean action-to-action change is 0.154 at temperature 0.1 versus 0.907 at temperature 1.0 (83.0% lower). The teacher windows also came from the smooth-weight-8 expert recipe, while CFM training denoises their conditional structure. The reflection penalty balances R/U modes; it is not itself a temporal smoother.

This is a controller property, not plotting post-processing: deployment uses H-exec=1 and the saved paths are the raw integrated states. The faithful temperature-1.0 diagnostic is retained next to the approved temperature-0.1 result so the variance reduction is visible and disclosed.

## Artifacts

- `viz/rollouts_and_local_minimum.png`: exact executed paths and entry-pocket zooms.
- `viz/failure_taxonomy.png`: failure classes and final goal distance.
- `viz/mizuta_tuning.png`: bounded tuning outcomes and behavior-retention diagnostic.
- `viz/pretrained_temperature_diagnostic.png`: T=0.1 vs faithful T=1.0.
- `data/*.npz`: exact paths, controls, seeds, and serialized per-episode metrics.
- `logs/stage4_summary.json` and `tables/`: full provenance and a–e tables.
- `logs/independent_audit.json`: dynamics, geometry, label, count, and checkpoint-hash audit.

## Independent audit

**PASS.** Every saved control sequence re-integrates to its stored path within 8.3e-6 m; all collision labels recompute using true per-obstacle radii; all gamma counts and checkpoint invariants match. The 42 CFM-MPPI* endpoints move at most 0.0248 m over their final 30 controls and make at most 0.0141 m recent goal progress, independently confirming the stall.

## Gate

No learning occurred. Stage 5 remains approval-gated: proceed only if this expert-feasible / frozen-baseline-hard scene and the selected low-guidance presentation are accepted.
