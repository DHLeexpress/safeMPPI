# Fast-lab report: neutral-expansion signal hunt (frozen base 6c99a05)

Session date: 2026-07-31 evening. Compute: GPU 3 + light GPU 1 only.
Archive root: `/data3/research1/claude_sfm_neutral_6c99a05/fastlab/`.
Figures (cmd-clickable copies): `/home/dohyun/projects/cfm_mppi/claude_fastlab_6c99a05/`.
Commits on `agent/claude-neutral-expansion-20260731`:
`1272f1e` (ID-anchor arm + pre-registration), `2cd89f5` (guided-positive
collector + pilot), `1a87f18`, `1ace169` (pilot fixes).

## Scoreboard (four ideas, ~4 h wall, all verdicts at raw temp-1)

| # | idea | verdict | decisive number |
|---|---|---|---|
| 1 | Kazuki's success rides on unbounded velocity | **REFUTED** | clipping to our p95 (1.99 m/s) *raises* its SR .749→.771, CR .251→.229; only time degrades 4.39→5.19 s |
| 2 | Early NVP is an acquisition-budget artifact (B=4 too small) | **REFUTED** | verifying all K=16 rescued only 26/290 D0 (−9%); true NVP contexts have a fully uncertifiable policy pool |
| 3 | External-controller certified harvest at early steps | **collection VALIDATED / training NULL** | kazuki_full teacher certifies 44.1% in t<50 (vs 1.2% weak guidance) — but 3 rounds × 4 passes @3e-5 leaves M20 flat/negative |
| 4 | Massive updates on certified windows move the raw policy | **REFUTED at tested doses** | D++G+: SR .707→.679, CR .286→.321, Val .609→.584; G+-only: SR/CR exactly flat, Val .609→.545 |

## Mechanism findings (Track B, authenticated gathers, rounds 1–50)

- NVP/D0 is a **transit-band** phenomenon: trigger rate 0.281 (t<10) → **0.605
  peak at t∈[10,20)** → 0.065 (t≥60); 96.3% of triggers are `finite_B_NVP`.
  In the severe-OOD bundle the band shifts to t∈[30,40) with zero triggers at
  t<10 (the robot needs 1–3 s to reach the crowd).
- The SFM crowd **stops**: mean ped speed peaks 1.21 m/s at t∈[10,20), median
  is exactly 0.000 m/s from t≥50. Late-episode "ease" is verifier acceptance
  (pool certification 0.305 in the band → 0.98 late), not window geometry
  (late D+ actually turn slightly more than early D+).
- Early-NVP persists through round 50 of the canonical loop (rate ratio never
  below 1.4×): 50 rounds of self-imitation did not teach early dodging.
- The current guided-repair operator (same-latent, no MPPI refinement)
  certifies at **1.2%** in t<40 (0% in the severe bundle) — it is structurally
  a D0 generator, exactly as observed (`D0_actual_repairs=0` every round).

## Velocity autopsy (Track A, fresh bank ep0=830000, M25/γ)

| arm | SR | CR | Validity | succ. clr [m] | succ. ttg [s] |
|---|---|---|---|---|---|
| Kazuki locked | .749 | .251 | .322 | .149 | 4.39 |
| Kazuki clip p99 (2.33 m/s) | .731 | .269 | .348 | .160 | 4.80 |
| **Kazuki clip p95 (1.99 m/s)** | **.771** | **.229** | .350 | **.171** | 5.19 |
| Pretrained raw temp-1 | .714 | .286 | **.617** | .139 | 8.33 |

Recipe confirmed at file:line (goal .5 / safe .3 / 200 samples / 10 elites /
200 perturbations; accel clamp ±2; velocity provably unbounded in planner and
env). Clip applied consistently in both. Our policy's speed: median 1.04,
p99 2.33 m/s; locked Kazuki median 2.13, max 4.01 m/s.
**The new bar ("conquer clipped Kazuki"): SR .77 / CR .23 / clr .17 / ttg 5.2,
while preserving our Validity lead (.62 vs .35).**
Caveat: the exact verifier integrates the nominal (unclipped) DI, so clipped
arms' Validity is validity of the nominal window.

## Guided-positive pilots (Track C)

Iteration 1 (1 round, u50, topk2, P=4 @3e-5, D++G+): teacher yield 567/1286 =
44.1%, γ-balanced, covers t<10 (188) through t<50; base path byte-identical to
reference round 1 (D+ 657 / D0 290). Paired M10: pooled null.
Crunch full-pool arm: 7,416 extra exact queries, 44.5% positive, D0 290→264
only. Paired M10: pooled null.

Iteration 2 (3 chained rounds, fresh scenario pairs 260002–07, M20 paired):

| chain | r0 (pretrained) | r3 | read |
|---|---|---|---|
| D+ + G+ (`kfull_x3_P4_v2`) | SR .707 / CR .286 / Val .609 / clr .126 | SR .679 / CR .321 / Val .584 / clr .128 | uniformly slightly negative |
| G+-only (`kfull_gonly_x3_P4_v2`) | same r0 | SR .714 / CR .286 / **Val .545** / clr .132 | SR/CR exactly flat; Validity −.064 |

Within-loop D0 (scenario-confounded): 220/135/340 (D++G+) and 220/180/408
(G+-only) — no dodging improvement; G+-only trends worse than D++G+ on the
same scenarios.

## Consolidated conclusion

Certified teacher windows are **harvestable at scale** (44%) exactly where the
policy's own pool is empty — but **window-level CFM imitation does not convert
them into closed-loop improvement** at 12× canonical dose × 3 rounds; G+-only
training actively erodes the policy's one real advantage (Validity). Together
with the canonical 50-round null (all four paired M50 CIs straddle zero) and
the mode-level NVP finding, the evidence converges on: **the bottleneck is the
imitation channel/objective, not the data supply.**

## Branch options (decision needed)

1. **Pivot the evaluated object to verifier-gated deployment** (the loop's
   closed-loop SR with repair is where certified windows demonstrably help;
   prior work: 3% coll / 97% SR at 6 ms). Raw-policy distillation stays a
   diagnostic, not the target.
2. **Change the learning signal, not the data**: signed negatives (α>0 on
   collided windows), or DAgger-style execution of teacher actions during
   collection (state-distribution shift rather than more windows), or
   context-weighted loss concentrated on crunch-band contexts.
3. **Teacher-matched variants** (velocity-clipped kazuki_full teacher,
   higher topk, P≫4): low expected value — same imitation channel that just
   returned null twice.
4. **Stop the neutral-expansion line** and consolidate the negative result +
   the clipped-Kazuki benchmark as the paper's honest comparison section.

## Figure index (cmd-clickable)

- `/home/dohyun/projects/cfm_mppi/claude_fastlab_6c99a05/trackA_kazuki_clip_fig1_speed_distributions.png`
- `/home/dohyun/projects/cfm_mppi/claude_fastlab_6c99a05/trackA_kazuki_clip_fig2_per_gamma_locked_vs_clipped.png`
- `/home/dohyun/projects/cfm_mppi/claude_fastlab_6c99a05/early_nvp_step_structure.png`
- `/home/dohyun/projects/cfm_mppi/claude_fastlab_6c99a05/rerendered_Dplus_D0_ess_comparison.png`

Numeric dumps: `fastlab/early_nvp/early_nvp_summary.json`,
`fastlab/kazuki_clip/results/tables.txt`, pilot roots
`fastlab/guided_positive/{kfull_u50_topk2_P4,crunch_u50_P4,kfull_x3_P4_v2,kfull_gonly_x3_P4_v2}/`
(each with PILOT_COMPLETE.json or surviving gathers + paired_eval/).

Held ready but not launched (pre-registered, committed): the 4-arm 50-round
campaign (control / pgm / noD0 / ID-anchor) from
`CLAUDE_NEUTRAL_PREREGISTRATION_6c99a05.md` — superseded by the fast-lab
findings unless the direction call revives it.
