# Claude SFM Safe Flow Expansion — best fixed recipe study

- Private worktree: `/home/dohyun/projects/safeMPPI-claude-sfm-recipe-f06e8dd`
- Branch: `agent/claude-sfm-best-recipe-20260726` from `f06e8ddc11fc7a1f5bada2cb2a587bff3ba4e424` (origin/agent/sfm-b1-offline-eval-funnel)
- Output root: `/data3/research1/claude_sfm_best_recipe_f06e8dd`
- r0 checkpoint: `/home/dohyun/projects/sfm_hp10_b1_runs/103476d/pretrained_hp10.pt`
  - SHA-256 verified `1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215` (matches `safe_flow_expansion_SFM@491e478:checkpoints/hp10_pretrained_r0.pt` blob hash)
- Plotting contract checkout: `/home/dohyun/projects/safe_flow_expansion-claude-plot-87063d3` @ 87063d3 (read-only)
- GPUs: physical 1 and 3 (verified idle at claim time: 18–19 MiB, 0% util). Exact SOCP verifier stays CPU.
- Python: conda env `cfm_mppi` (Python 3.11).

## 2026-07-26 02:30 — starting state inherited from Codex

Prior completed work (all on `double_density_velocity_ood`, 40 peds, 1.0–2.0 m/s):

- 27 trained arms (3 selectors × α∈{0,.01,.1} × exposure∈{1,10,100}), rounds 0–10, lr 1e-4, ESS 0.5, batch 128, seed 20260724, expansion bank ep 20000+:
  - margin: `/data3/research1/sfm_b1_offline_exec_9arm_f36393b`
  - safemppi_cost: `/data3/research1/sfm_b1_offline_cost_9arm_db80dfc`
  - balanced_rank: `/data3/research1/sfm_b1_offline_balanced_9arm_97630c6`
- Funnel + final M100 (completed 2026-07-26 05:49 UTC): `/data3/research1/sfm_b1_offline_final_funnel_f06e8dd`
  - M100 (ep0 280000, seed 20260725): r0 SR .660 CR .337 V .595 clr .118 t 8.65
  - global winner (cost α.01 exp10 r8): SR .690 CR .310 V .531 clr .117 t 7.52 — CR −2.7pt but Validity −6.4pt: **not** a genuine fixed-recipe win.

### Key measured phenomenon driving this study

Per-round fixed-raw M50 curves (margin 9-arm factorial CSV) show **SR collapse to 0.00 by round 3–4 in every margin arm** while Validity rises to ~.85 — a monotone slowdown (successful time 8.7→10.7→13→16.4 s) until nothing reaches the goal inside T=180. Balanced_rank arms collapse identically (M10 screening). Cost-selector arms avoid the collapse but mostly degrade (CR rises to .4–.6 in 6/9 arms). Only round-1/2 checkpoints ever beat r0, modestly (best M50: margin α.01 exp100 r1 = SR .694, CR .28, V .74, clr .128, t 10.7).

Working hypothesis (to be tested in Stage A): replay trains the flow toward the *gathering controller's* executed-window distribution, which is verifier-gated and conservative; each round compounds the slowdown. The failure is data composition, not just step size.

## Plan

Stages A–E per task spec; banks in `EPISODE_BANKS.json`. Interventions implemented as opt-in modules, default OFF, original behavior preserved with existing tests.

## 2026-07-26 03:20 — modules, tests, baselines launched

- New additive modules (commit 779d2de): `claude_offline_aug.py` (declared pop-A/B rules + deterministic certified-recovery generator; family = 145 candidates/context, prefilter cap 24, keep ≤2, exact `SM.verify_query` gate), `claude_offline_exec_ext.py` (opt-in lr/ESS/rounds/replay-mode; immutable core reused), `claude_stageA.py` (mine/branch/update/compare), `claude_kazuki_eval.py` (locked comparator + executed-window Validity), `claude_paper_trends.py` (evaluator → paper plot contract).
- Existing tests: 43 passed. New tests: 6 passed, incl. bitwise default-OFF equivalence of `replay_with_mode("original")` vs `OR.replay`, no-relabel guarantee, and independent exact recertification of every synthetic recovery record.
- Baselines launched on the pre-registered codex M100 bank (ep0 280000, seed 20260725): raw r0 + margin winner (α.01 exp100 r1) + cost winner (α.01 exp10 r8) on GPU1; locked Kazuki on GPU3.
- Stage A mining (codex round-1 shards): margin arm — 1034/4014 NVP contexts, 22 collision windows, 310 trap windows; cost arm — 1102 NVP, 19 collisions, only 32 traps. D+ plan-geometry drift r1→r3 (median displacement 0.95→0.70 m) supports the composition-drift hypothesis.
- Stage A single-update candidates U1–U8 launched (one replay round on r0 from archived round-1 shards; margin + cost shards × {control, lr1e-5, exp1, hard, hard_recovery, lowdose-hardrec}).

## PREDECLARED Stage-B qualification rule (written before any Stage-B run)

- Bank: qualification raw bank ep0 310000, noise seed 20260728, M=25/γ, temperature 1.0, via `sfm_b1_offline_eval.py` only. No gathering-controller SR, no training loss.
- Every candidate arm trains rounds 1–4 from the exact r0 checkpoint (seed 20260724, expansion bank ep 20000+, identical gather semantics).
- Eligibility per round r ∈ {1..4}: SR(r) ≥ SR(r0) − 0.02 AND timeout(r) ≤ timeout(r0) + 0.05 on the qualification bank (liveness gate; r0 evaluated on the same bank/noise).
- Arm score = its best eligible round ordered by (min CR, then max Validity, then max successful clearance, then min successful time-to-goal). Arms with no eligible round are disqualified (collapse).
- Stability tie-break: among arms whose best-round CR are within 0.03 of the leader, prefer the arm whose round-4 checkpoint is still eligible; among those, the better round-4 CR. Rationale: the frozen Stage-C/D recipe must hold 10 round-invariant macro-rounds.
- The frozen recipe = the winning arm's knobs verbatim; final study rounds fixed at 10; Stage-D checkpoint selection governed solely by SELECTION_RULE.json on the disjoint M50 bank (ep0 320000).

## 2026-07-26 05:30 — Stage A results (complete)

**Mechanism (r0 branch trace, 24 gathering lineages, all-K exact verification):** 2117 contexts, 456 NVP (21.5%); only 49 (2.3%) had a positive in K that B missed → the flow itself lacks certifiable support at hard contexts; B=4 is not binding. NVP concentrates at episode start (43–51% in steps 0–39 → ~0 after step 60): origin-corner congestion with 40 fast pedestrians. Every collision episode dies through a terminal run of K+=0 NVP contexts with negative predicted clearance (uncertified raw execution). Selector disagreement at 59% of contexts. Certified deterministic escapes exist at ~53% of hard contexts (269/506 margin shard, 202/501 cost shard).

**Single-update raw reads (diag M8 + anchor M12, n=140/ckpt):** r0 SR .614 / CR .386 / V .571.
- U6 cost-shard original: SR .714 / CR .286 / V .566, faster — only arm improving SR/CR/time; Validity flat.
- U1–U5 margin-shard arms: Validity +.08–.13 (U5 hard+recovery best, .698) but SR −.03–.07 and slower — conservative drift visible after ONE update.
- U7 cost-shard hard-only: WORSE than U6 across the board — discarding the goal-directed mass hurts.
- U8 lowdose hardrec: mild moves, dose too small per round.

**Local repair (before/after branch traces, identical keyed latents):** U5 hard+recovery cut NVP 456→341 (−25%), raised B-positive fraction .712→.844, repaired the three hardest collision lineages (s20005 γ.1/γ.5, s20004 γ.5 → success), introduced slowdown timeouts elsewhere. U6 sped the policy up but *lowered* gathering certifiability (B+ frac .592, NVP 483). Conclusion: recovery data provides certifiable support exactly where the flow lacks it; the composition question (keep goal-seeking mass + add recovery) is what Stage B arms B4/B5/B8 test (`orig_plus_recovery`, declared before evaluation).

**Kazuki locked baseline (M100 ep0 280000):** SR .779 / CR .217 / **Validity .350** / clearance .181 / time 4.36 s — fast and lower-CR than r0 but far below r0 on exact-certificate Validity (.35 vs .59).

## Baseline reproduction complete (M100, ep0 280000, seed 20260725)

| method | SR | CR | timeout | Validity | succ. clearance | succ. time |
|---|---:|---:|---:|---:|---:|---:|
| r0 raw (repro) | .6586 | .3386 | .0029 | .5944 | .1176 | 8.664 |
| r0 raw (codex funnel) | .6600 | .3371 | .0029 | .5945 | .1179 | 8.646 |
| B1 control: margin α.01 e100 **r1** | .7314 | .2371 | .0314 | .7390 | .1336 | 10.733 |
| B1 control: cost α.01 e10 **r8** (codex global winner) | .6900 | .3100 | .0000 | .5311 | .1169 | 7.520 |
| locked Kazuki (.3/.5) | .7786 | .2171 | .0043 | .3495 | .1809 | 4.355 |

r0 reproduces codex within 1–2 flipped episodes (GPU FP nondeterminism). The margin-r1 control dominates the codex-selected cost-r8 on this bank — but it is a pre-collapse snapshot (SR→0 by r3–4 in that arm). Bar for the new fixed recipe: margin-r1-level CR/Validity gains with multi-round stability.

## Stage B launched 05:20 — 8 arms × 4 rounds from exact r0

B1 cost/original, B2 cost/hard, B3 cost/hard_recovery, B4 cost/orig_plus_recovery (all α.01 e10 lr1e-4 ess.5); B5 cost/orig_plus_recovery lowdose (e1 lr1e-5); B6 margin/hard_recovery lowdose; B7 cost/original ess.3; B8 cost/orig_plus_recovery ess.3. Qualification: predeclared rule on M25 bank ep0 310000 (see above).

### Stage B qualification (M25, ep0 310000; r0 = SR .669 / CR .320 / V .576 / clr .106 / t 8.58)

Per arm r1..r4 (SR/CR/V):
- B1 cost orig: .60/.39/.54, .67/.33/.53, .64/.35/.53, .64/.36/.51 — no gain, V drifts down
- B2 cost hard: .71/.29/.59 then degrades to .62/.38/.52
- B3 cost hardrec: .68/.32/.60 then degrades to .55/.45/.55
- B4 cost origrec: ≈flat (.60–.66 SR, V .57–.59)
- B5 cost origrec lowdose: flat
- B6 margin hardrec lowdose: V .58→.65 climbing, CR .34–.40 (no CR gain), t 8.9→10.2 — slow conservative drift
- B7 cost orig ess.3: worse than B1 (lower-ESS acquisition does not help)
- B8 cost origrec ess.3: ≈B4
Verdict: no cost-selector composition materially improves CR or Validity on this bank; the lower ESS target (0.3) is not beneficial. The strongest known pattern (margin/original/e100 — codex arm, r1 CR .237/V .739 on the M100 280k baseline) was absent from the set.

## 2026-07-26 13:00–15:00 — Stage D result, mechanism deep-dive, iteration 2

**Stage D (frozen cost/hard recipe, M50 bank 320000): honest null.** Rule-selected r7: CR .263 vs r0 .269 but Validity .571 vs .638. Full 11-round curve + paper-contract plot at `stageD/paper_trends/`. Recorded in STAGE_D_SELECTION.json with Pareto frontier.

**Mechanism figures** (user request; artifact https://claude.ai/code/artifact/3c16fe2a-1450-4a8c-addf-466beaea1111, PNGs in `~/claude_sfm_figs/` and `mechanism/`): the closing certifiability window quantified on stored contexts — onset−2: 10–13/16 flow candidates certify, escapes 14–24/24; onset: flow 1–2/16, escapes 5–15/24; onset+4–7: 0 everywhere including both deterministic families. Closed-loop replays: B9-r1 converts both collision lineages to successes (s20005 γ.1 clearance .236; s20004 γ.5 clearance .139).

**Recovery family v2** (user hypothesis: v1 escapes too conservative): dodge-then-cruise family implemented + declared; single-update head-to-head at matched dose (n=140): U10 v2 SR .679/CR .264/V .723/t 11.22 vs U12 v1 SR .614/CR .350/V .706/t 10.92. v2 did NOT remove the slowdown (the declared signature), so per the pre-registered criterion iteration 2 froze R_A (v1); the v2 SR/CR edge (within noise) is documented as follow-up.

**Iteration 2 (pre-registered):** recipe = margin/orig_plus_recovery-v1/α.01/e100/lr1e-4/ess.5/rounds4 (B9; checkpoints trained from exact r0 before any selection-bank read). Fresh M50 selection bank 340000: r0 = SR .563/CR .434/V .574; **r1 = only eligible round: SR .654 (+.091), CR .311 (−.123), V .715 (+.141), clearance flat, time +2.36 s**; r2 fails gate (timeout .163), r3–r4 collapse. Selected checkpoint: B9 round_01.pt. Population counts + per-row certificate audits in `iteration2/population_counts_B9.json` and `iteration2/recovery_certificate_audit_B9.json` (280–407 exact-certified recovery positives/round from 8.3–12.3k exact queries).

**Stage E launched** on untouched M100 bank 330000: r0 + selected + locked Kazuki.

## 2026-07-26 18:20 — FINAL: Stage E confirmation + delivery

M100 confirmation (untouched bank 330000, 700 CRN rollouts/method): r0 SR .643/CR .353/V .603/clr .108/t 8.76 → **selected (margin+orig∪recovery-v1, e100, round 1): SR .723 / CR .253 / V .736 / clr .122 / t 10.84**; locked Kazuki SR .827/CR .173/**V .353**/clr .168/t 4.17. Paired scenario-cluster 95% CIs (selected − r0): ΔCR −.100 [−.157,−.043], ΔV +.133 [+.114,+.153], Δclr +.014 [+.005,+.023], Δt +2.08 [1.84,2.32] — all exclude zero. γ=0.1 keeps the largest clearance (.148) and longest time (13.4 s). Stability honestly reported: round-1 phenomenon; r2+ collapse. Full record in DELIVERY_COMPLETE.json; 33-artifact SHA manifest.

## 2026-07-26 16:20+ — MPC distillation follow-up study (separate branch)

Pre-registered in MPC_STUDY_PREREGISTRATION.json (banks M10 350000 / M50 360000 / M100 370000). Smoke round: 172 D_MPC+ records; dedicated block doubled the local SOCP-positive rate at MPC contexts (.161→.313) — a visible raw-policy change; M10 CR moved adversely at that dose (smoke bank). 4-arm sweep (lr_d × epochs_d) running.

**COMPLETE (fail-closed at M10), 21:35** — Answer to the primary question: **YES, D_MPC+ distillation visibly changes the raw policy** (SOCP-positive rate at MPC contexts up in 15/16 blocks, up to .106→.482; target-recovery RMSE down in 16/16; movement present even when the block's CFM loss rose) — **but no swept dose is globally beneficial**: 0/32 pre/post cells pass the pre-registered liveness gate on the M10 bank (r0 SR .657; post-block round-1 SR .243–.600), the base margin/e10 recipe collapses by r2–3 with or without distillation (no-distill control confirms), and per the pre-registration the M50/M100 confirmations are not triggered. Full record: MPC_STUDY_DELIVERY.json, mpc_distill/MPC_M10_SELECTION.json (all 37 cells), per-block audits in mpc_distill/M*/metrics.jsonl, every pre/post checkpoint kept. Synthesis: across deterministic v1/v2 escapes and the privileged MPC pool alike, hard-context-only distillation trades global liveness for local certifiable support; the confirmed winning recipe worked by EMBEDDING certified targets in the full replay mixture (~5% mass) — composition, not target quality, is the binding constraint.

### Stage B extension (declared 07:55 before reading its results)

- B0: codex margin/original/α.01/e100 checkpoints r1–r4 evaluated on the SAME M25 qual bank (matched-round control; identical recipe lineage, same commit and seeds).
- B9: margin/orig_plus_recovery/α.01/e100/lr1e-4/ess.5, rounds 1–4 from exact r0 — tests the marginal contribution of certified recovery positives ON TOP of the strongest known recipe. Comparison B0 vs B9 at matched rounds on the same bank is the pre-registered arm-2/arm-3 style contrast for the final freeze decision; freeze criterion remains the predeclared Stage-B rule.
