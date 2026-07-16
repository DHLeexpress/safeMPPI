# Pure AFE-minimal Safe Flow Expansion — findings (2026-07-16)

Trainer: `grid_expand_afe.py` (+ `afe_core.py`), built to the minimal redesign spec after the 9-fault
critique. One object identity (planned window = queried = FULL-SOCP-verified **before** execution =
stored = trained), frozen-φ⁰ cumulative 32×32 A_n over every verified query (pos+neg), Gibbs
acquisition π∝e^((σ−σmax)/β) with B=8 verifier budget/step, certified SafeMPPI fallback, uniform
replay over cumulative D⁺ under the single proximal objective ℓ_CFM + (1/2η)‖θ−θₙ‖².
**PURE per user directive: no demo replay, no LwF, no anchoring, no encoder freezing, no curriculum.**
The prox term is the only regularizer; all parameters train.

Rollback point: git tag `pre-afe-2026-07-16` (original code), commits `6a5312b`/`83c6033` (AFE).
Arms (100 rounds × 8 episodes, ~590–710k fully-verified queries each, GPU 3):
`pure_s910/911/912` (λ=10), `pure_lam001_s910` (λ=0.01 reference), `pure_pi_s910` (execute by
sampling ∝π among verified-safe instead of argmax-progress).

## Calibration (measured, not assumed)
- SOCP full-window verify: **2.3 ms/plan** (n_theta 180 ≈ 120) → B=8/step affordable; SafeMPPI
  fallback 17 ms/step.
- η=0.01 makes the prox self-limiting: the round update plateaus at ~0.02–0.03 relative field
  displacement regardless of step count (solver bound, not a tuned step knob).
- Rest-state audits sit at a 99% ceiling → added the **adverse-velocity audit slice** (moving toward
  the nearest obstacle): pretrained V̂_adverse = 0.44 pooled, **0.00 at γ0.1** — the axis with headroom.
- hp-channel-only context storage verified exact (ctx diff 0.0).

## Results (consistent across 3 seeds + both λ + both exec rules)

**1. Runtime safety by construction: confirmed.** 0 episode deaths in ~3,200 shielded episodes/arm
(~590k executed steps); gather-time collisions = 0. Dithering never appeared (max 0.7% of new D⁺;
the feared safe-dithering spiral did not materialize despite progress being removed from the label).

**2. Stability without anchors: confirmed (no spiral), with a mid-run dip.** Closed-loop SR (bare
policy) dips 0.91→~0.77–0.86 around rounds 10–50, then recovers to 0.93–1.00 with CR 0.00–0.07 by
round 100 (M=8). V̂_rest rises 0.994→1.00 (no forgetting on easy contexts). M=40 final evals:
SR 0.90–0.94, CR 0.06–0.10, clearance ≈0.247, covΣ 26–30, a-d 5–11/28.

**3. NO validity expansion where validity was missing.** V̂_adverse is flat at 0.44 for 100 rounds in
every arm; γ0.1-adverse stays 0.00; pooled fallback rate never decays (~4–8%).

**4. WHERE expansion did/didn't happen — the shield-moral-hazard decomposition.** Fallback location
(pure_s910): round 5 = 31 near-goal + 12 mid-route; round 100 = 40 near-goal + 3 mid-route. The
method **did** expand the verified set along the route (mid-route fallback −75%), and **stalled
permanently at the goal corner**: plans there keep failing certification (goal (4.7,4.7) against the
wall plugs is OOD vs the (5,5)-goal pretraining demos; overshoot plans hit the plugs), so no
positives ever enter D⁺ there (visible hole in the trained-samples map), so the flow never learns to
finish — the certified fallback bridges the last ~10 steps of essentially every episode, forever.
**The shield removes exactly the failure pressure that would force expansion at hard contexts.**

**5. Mode sharpening returns — but it is NOT the old prior collapse, and the EXECUTION RULE is the
diversity lever.** Under argmax-progress execution, per-γ coverage falls from 2–4 modes (round 0) to
**1 dominant mode for every γ≥0.3 by round 20** at M=8 (γ0.1 keeps 4–7), and stays low even at M=40
(covΣ 26–30 vs pretrained 34 — a diversity LOSS). Yet the raw first-window up-fraction at the start
context is UNCHANGED (0.137–0.153 vs pretrained 0.145) — unlike the measured un-anchored curriculum
collapse (0.14→0.73): the sharpening lives in the visited-context + replay amplification loop, not
the prior. **exec-rule=pi (execute a verified-safe plan sampled ∝π instead of argmax-progress)
fixes the realized diversity: covΣ 52 at M=40 — equal to the curriculum+anchor recipe (51), near
expert (56) — with SR 0.92/CR 0.08.** At M=8 its per-γ draw still shows the dominant mode (1/γ),
i.e. π-execution preserves a fat tail of secondary routes rather than equalizing them. No anchor,
no curriculum — the diversity was recoverable inside the method's own vocabulary.

**6. σ saturates regardless of λ; acquisition ≈ uniform after ~round 10.** Drawn-candidate σ:
0.19→0.02 by round 20 at λ=10 (λ=0.01: ≈0 after round 1). The λ-study's surviving high-σ
eigendirections are directions p_θ never proposes: candidates always live in the already-queried
feature subspace. **Plan-level uncertainty tilting at policy-visited states cannot steer which
states get visited** — the context distribution, not the plan distribution, is the binding
constraint on expansion.

## A/B/C comparison (M=40/γ, T=350, cleared goal, 8 wall plugs, vs expert_g47)

| arm | SR | CR | clearance | time | covΣ | a-d |
|---|---|---|---|---|---|---|
| pure_s910 (λ10) | 0.91 | 0.09 | 0.247 | 10.55 | 28 | 5/28 |
| pure_s911 (λ10) | 0.94 | 0.06 | 0.248 | 10.49 | 30 | 11/28 |
| pure_s912 (λ10) | 0.90 | 0.10 | 0.245 | 10.31 | 26 | 6/28 |
| pure_lam001_s910 | 0.94 | 0.06 | 0.249 | 10.46 | 29 | 11/28 |
| **pure_pi_s910 (λ10, exec∝π)** | **0.92** | **0.08** | **0.257** | **10.82** | **52** | **9/28** |
| pretrained baseline | 0.95 | 0.05 | 0.253 | 11.26 | 34 | 7/28 |
| C: faithful_div (curriculum+anchor) | 0.94 | 0.06 | 0.255 | 11.08 | 51 | 8/28 |
| expert (SafeMPPI) | 1.00 | 0.00 | — | — | 56 | — |

Read against the strong T=350 pretrained baseline (SR 0.95, covΣ 34): every arm stays near baseline
SR/CR; argmax arms LOSE coverage (26–30); **exec∝π matches the curriculum+anchor recipe's coverage
(52 vs 51) with zero stabilizers** and the best clearance of the pure arms. The differentiators of
the method are the gathering-time guarantees (0 collisions, every trained window certified) and the
π-execution diversity — not raw SR on this task, where the baseline is already near-saturated.

## The honest scoped claim this supports
The deterministic full-window verifier + certified fallback carry runtime safety completely
(0 deaths / 0 collisions while gathering, at every γ, from round 0). Uncertainty-tilted querying
expands the verified-plan set **only along the executed distribution's own support**: mid-route
holes close (fallback −75% there), but contexts the shielded process never visits (adverse
velocities, the OOD goal corner) never expand. Realized route diversity is governed by the
execution rule among verified-safe plans: argmax-progress sharpens to one mode per γ; sampling ∝π
retains curriculum-recipe-level coverage with no stabilizers. The curriculum-era mechanisms
(frontier replay, recovery starts, demo/LwF anchor) were ad-hoc compensations for holes the minimal
method now makes *measurable*; one of the three (diversity) closes inside the method itself, the
other two (adverse-context validity, the goal-corner hole) are properties of the fixed context
process ρ(c), not of the learner.

Within the method's own vocabulary the remaining lever is ρ(c): the audit already samples the hard
contexts, so an *episode-start distribution matching ρ_eval* (start some episodes at adverse/corner
states) would create verified queries exactly where validity is missing — the certified-recovery
analogue of what the strip diagnosis recommended — but that is a deliberate design change to the
context process, to be decided, not smuggled in.

## Artifacts
- Trainer/lib: `grid_expand_afe.py`, `afe_core.py`; drivers `analysis/afe_driver.sh`,
  `analysis/afe_assemble.sh`, `analysis/afe_lam_study.py`.
- Tracking: `results/afe/<arm>/probe.jsonl` (per-round â, per-γ acceptance/fallback, V̂ per γ
  rest/adverse, dither, σ, solver stats), `history.json`, `viz_db/round*.pt`, `dstore.pt` (~600k
  verified queries), ckpts every 10 rounds + `final.pt` (eval-compatible).
- Viz: `paper_results/afe_validity_v1.png` (5-arm validity tracking + collapse audit);
  `paper_results/afe_pure_*_expansion.mp4` (per-round: green/red verified plan fans, σ dots,
  executed paths + fallback steps, trained-samples map with round-of-origin, validity curves).
- Evals: `results/p2/eval_afe_pure_*` (+ pi + pretrained pending), logs `logs/eval_afe_*.log`.
