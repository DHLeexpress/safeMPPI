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

**5. Mode sharpening returns — NOT the old prior collapse; the execution rule is the (partial)
diversity lever.** [CORRECTED 2026-07-16b: an earlier draft mis-attributed the pretrained's pooled
line to the π arm.] The pretrained base is highly diverse: **covΣ 52** at M=40. Every pure arm
loses diversity: argmax-progress execution collapses per-γ coverage to 1 dominant mode (γ≥0.3, by
round 20 at M=8; covΣ 26–30 at M=40), while **exec-rule=pi retains covΣ 34 with the best shielded
closed-loop numbers (SR 0.954 / CR 0.046)** — better than argmax, but well below the
curriculum+anchor recipe, which preserved the base's diversity (51). The raw first-window
up-fraction at the start stays at the pretrained 0.10–0.15 in all arms (the old prior collapse
0.14→0.73 does NOT reappear): the sharpening lives in the visited-context + replay amplification
loop. Verdict: π-execution mitigates, does not solve; full diversity preservation was what the
anchor bought.

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
| **pure_pi_s910 (λ10, exec∝π)** | **0.954** | **0.046** | 0.253 | 11.26 | **34** | 7/28 |
| −Verifier brother | 0.954 | 0.046 | 0.249 | 10.73 | 34 | 8/28 |
| −Fallback brother | 0.946 | 0.054 | 0.251 | 11.09 | 33 | 6/28 |
| −Prox brother | 1.000 | 0.000 | 0.262 | 10.91 | 18 | 24/28 |
| pretrained baseline | 0.918 | 0.082 | 0.257 | 10.82 | **52** | 9/28 |
| C: faithful_div (curriculum+anchor) | 0.943 | 0.057 | 0.255 | 11.08 | 51 | 8/28 |
| expert (SafeMPPI) | 1.000 | 0.000 | 0.259 | 10.80 | 56 | — |

[CORRECTED 2026-07-16b — all numbers now pooled from row_g*.json, one consistent source.]
Read against the T=350 pretrained baseline (SR 0.918, covΣ 52): Ours improves reliability
(CR 0.082→0.046, SR +3.6 pts) with zero gather-time collisions and a fully certified training set,
at the cost of diversity (52→34; argmax exec loses more, 26–30). The gate ablations locate each
piece's value at GATHER time or in the audit, not eval CR: −Verifier kills 8.5% of gathering
episodes and admits uncertified training plans; −Fallback kills 4.9%; −Prox posts the best
closed-loop line (SR 1.0, a-d 24/28) while collapsing routes (covΣ 18) and ERODING audit validity
(V̂_adv 0.438→0.393, the only arm to move it down) — the prox term is a validity/diversity
preserver, not an SR maximizer. See paper_results/table_v6.md.

## The honest scoped claim this supports
The deterministic full-window verifier + certified fallback carry runtime safety completely
(0 deaths / 0 collisions while gathering, at every γ, from round 0 — vs 8.5%/4.9% dead episodes
without the verifier/fallback). Uncertainty-tilted querying expands the verified-plan set **only
along the executed distribution's own support**: mid-route holes close (fallback −75% there), but
contexts the shielded process never visits (adverse velocities, the OOD goal corner) never expand.
Realized route diversity is governed by the execution rule among verified-safe plans:
argmax-progress sharpens to one mode per γ (covΣ 26–30); sampling ∝π mitigates (34) but nothing in
the pure method preserves the base's 52 — that preservation is what the curriculum-era anchor
bought (51). The prox term's measured role is validity/diversity preservation: removing it
maximizes closed-loop SR while eroding the untilted audit and collapsing routes. The curriculum-era
mechanisms (frontier replay, recovery starts, demo/LwF anchor) were ad-hoc compensations for holes
the minimal method now makes *measurable*; the remaining two structural holes (adverse-context
validity, the goal-corner hole) are properties of the fixed context process ρ(c), not of the
learner.

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
