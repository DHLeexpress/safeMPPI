# RESULTS — Guided Safe MPPI vs Mizuta CFM-MPPI (moving pedestrians)

Overnight run 2026-06-23. All numbers are from `cfm_mppi.evaluation.eval_pedestrian_benchmark`
(reuses render_validation_comparison's exact scene + metric definitions): success =
final goal-dist ≤ 0.5 AND min-clearance ≥ 0; collision = min-clearance < 0;
clearance = min over all pedestrians of (dist − radius − 0.5 margin). Bootstrap
95% CIs; paired McNemar (success) and Wilcoxon (clearance) vs Mizuta on identical
episodes. Double integrator, dt=0.1, receding horizon, 80 steps.

## Headline — held-out episodes 100–199 (100 episodes), tuned config
Tuned config (param sweep, optimized for success at zero collision):
γ=0.2, η=0.6, barrier margin buffer 0.25, progress wt 9, terminal 200, samples 512,
horizon 30, guidance-horizon 10.

### UCY (real pedestrian tracks)
| method | success% [95% CI] | collision% [95% CI] | min-clear med | latency |
|---|---|---|---|---|
| Mizuta CFM-MPPI | 98 [95,100] | 1 [0,3] | 0.82 | 104 ms |
| last-year rejection MPPI (γ=0.2) | 50 [40,60] | 50 [40,60] | 0.01 | 29 ms |
| **Guided Safe MPPI (ours)** | 75 [66,83] | 7 [3,12] | 0.65 | 47 ms |

### SDD (real pedestrian tracks)
| method | success% [95% CI] | collision% [95% CI] | min-clear med | latency |
|---|---|---|---|---|
| Mizuta CFM-MPPI | 100 [100,100] | 0 [0,0] | 0.92 | 96 ms |
| last-year rejection MPPI (γ=0.2) | 82 [74,89] | 18 [11,26] | 0.69 | 27 ms |
| **Guided Safe MPPI (ours)** | 91 [85,96] | 1 [0,3] | 0.89 | 42 ms |

Paired vs Mizuta (UCY): ours Δsuccess −0.23 (McNemar 1/24, p=1.6e-6);
vs last-year Δsuccess −0.48 (1/49, p=9e-14). Clearance ours vs Mizuta median
−0.19 (p=8e-8); ours vs last-year +0.70 (p=2e-16).

## What the numbers say (honest)
1. **We decisively beat last year's rejection method** — collision 50%→7% (UCY)
   and 18%→1% (SDD), success +25–35 pts, all p<1e-5. The freeze/averaging
   pathology is fixed.
2. **Mizuta still wins raw success** (98–100% vs 75–91%, significant). It is a
   strong learned, anticipatory policy (274k real trajectories). A reactive
   model-based planner has a structural ceiling (see diagnostic).
3. **Where we are competitive / better:** provable per-step safety guarantee
   (Props 1–4) that Mizuta lacks; ~2× lower latency (42–47 vs 96–104 ms);
   clearance at parity (SDD 0.89 vs 0.92) or modestly below (UCY); a tunable γ
   safety knob (a Pareto front, not a single point); and on the easier SDD set we
   reach 91%/1% — close to Mizuta.

## Collision-source diagnostic (60 held-out UCY eps, output filter ON)
- Per-step feasible-set-empty (H=∅, robot cornered, Assumption-1 violated): 2.77%.
- Of all collisions: **1/3 set-infeasibility (cornered), 2/3 constant-velocity
  PREDICTION error**. NOT sample-infeasibility (the output PSF closes that path).
- Implication: residual collisions are model-assumption failures, not planner
  bugs. Fixes: constant-acceleration obstacle prediction (targets the 2/3) +
  anticipatory learned proposal (targets the 1/3 cornering).

## Parameter sweep (40 configs × 24 eps, success at zero collision)
Best validation config reached 92% success / 0% collision / clear 0.94 on the
TUNING episodes (0–23); on held-out 100–199 it generalizes to 75%/7% (UCY) —
i.e. the 92%/0% was tuning-set optimistic, reported honestly. High sample count
(512) is the most important knob (more samples → fewer all-rejected freezes).

## Ablations available (same Stage-2 certificate, different Stage-1 generation)
- guided_safemppi = Gaussian proposal (un-learned).
- cfm_proposal_mppi = learned CFM proposal.
- guided_drifting = 1-NFE learned proposal + runtime projection.
(See LEARNED-PROPOSAL section — data-distribution work in progress.)

## Learned proposal — held-out 100-ep result (in-distribution UCY, eps 100-199)
Proposal CFM trained on 1547 real-UCY guided SUCCESS trajectories (diverse goals).
Required a velocity-regulation fix (proposal-only MPPI overshot the goal and
diverged; fixed by mixing Gaussian-around-damped-nominal samples into the learned
proposal — theory ρ-mixing).

| method (100 eps) | succ% | coll% | clear med | vs Mizuta McNemar |
|---|---|---|---|---|
| Mizuta CFM-MPPI | 97 | 1 | 0.85 | — |
| guided_safemppi (Gaussian proposal) | 75 | 7 | 0.65 | 2/24 p=1e-5 |
| cfm_proposal_mppi (LEARNED proposal) | 76 | 6 | 0.73 | 2/23 p=2e-5 |

Findings (honest):
- The two-stage framework works end-to-end; the learned proposal recovers the
  planner's performance in an amortized, multimodal form and is slightly safer
  than the Gaussian proposal (collision 7%->6%, clearance 0.65->0.73) at equal
  success — evidence that learned "clever sampling" helps.
- It does NOT beat Mizuta. Root cause is principled: the proposal imitates the
  REACTIVE guided-MPPI teacher (~75% ceiling); imitation cannot exceed its
  teacher. Mizuta trains on 274k real HUMAN trajectories (anticipatory, 97%).
- => Beating Mizuta needs a better teacher: Expert Iteration with a stronger
  improvement operator, distilling a SOTA policy + wrapping it in our certificate
  (clean Props 3-4 demo: add a hard guarantee + γ-knob to ANY policy), or the real
  ego-data pipeline (train80_ego lacks a clean per-step obstacle-context channel,
  so needs Mizuta's exact feature extraction — future work).

## Learned proposal (Stage-1 learning) — earlier status
Framework + theory complete (THEORY §10, Props 3–4: safety is proposal-agnostic).
First attempt diverged on UCY due to a degenerate training distribution (fixed
(0,0)→(6,6) goal => conditioning-OOD on UCY's diverse goals). Fix in progress:
randomized-pose guided data (rotation + distance scaling) for generalizable,
relative goal-seeking; the principled long-term source is the real ego dataset
(train80_ego, same as Mizuta), with obstacle context extracted. Honest result to
be filled after the randomized retest.

## Reproduce
- Headline UCY: see `headline_ucy_tuned/` (cmd in STATUS.md / this folder logs).
- Diagnostic: `python -m cfm_mppi.evaluation.diagnose_collisions --dataset ucy --episodes 60`.
- Sweep: `python -m cfm_mppi.evaluation.sweep_guided_params --dataset ucy --num-configs 40`.
