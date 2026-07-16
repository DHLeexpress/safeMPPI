# WALLS-4 FROM-SCRATCH SUITE — codex takeover (fine-tune / diagnose / end-to-end viz / publishable)

## WHEN TO START (verify with one command)

Start as soon as ALL of these exist — expected within ~1–3 h of this file's creation
(base finishing ~it100; walled-expert M100 is the slow one):

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
ls results/p2/walls4_scratch_base_s820/final.pt results/p2/walls4_scratch_nocur_s821/final.pt \
   results/p2/walls4_scratch_nosocp_s822/final.pt results/p2/walls4_scratch_noprog_s823/final.pt \
   results/expert_gt_walls4/EVAL_DONE 2>/dev/null | wc -l   # start when this prints 5
```

If the walled expert is still running, you may begin everything except the (c)/(d)/(e) comparisons.
GPUs: check `nvidia-smi`; GPU3 was ours. No wandb, no push, PROGRESS.md append-only, work only here.

## MOTIVATION (the user's framing — this is the paper angle)

The task boundary was invisible to the policy's perception (it senses only obstacles), which produced the
ill-conditioned boundary behaviors — the seed-12 saturated fiber, mm-grazings above the goal — that
previously required HAND-CRAFTED PLUMBING (boundary adapters, preservation replays, oracle probes, per-seed
surgery). The 4-plug walled scene makes the boundary perceptible, so the CURRENT PIPELINE ALONE, from
scratch, should reach the full goals with no manual seeding — and the ablation arms isolate exactly why
each pipeline component matters. Reduce ill-conditioned behaviors ⇒ eliminate hand-crafted plumbing.

## USER REQUESTS (VERBATIM — the deliverable bar)

Suite definition: faithful comparison with 4 plugs, every gamma, internals + curriculum video each:
> (1) Without curriculum learning (meaning there are no difference of easy and frontiers - if validated
> (if shown in panel A,B,C) then treat those as equal samples (Expected: metrics are poor and gradients
> are unstable), (2) the case where verifier drops multi-step safety verifying condition (not SOCP, just
> checking each states are in free space (only obstacle avoidance). In this case, still checking goal
> progress, expecting the safety metric (mean distance) is poor. (3) the case where verifier drops
> performance condition (no goal progress - just checking only SOCP in the validity). (2)(3) is still
> doing curriculum learning. So basically (1)(2)(3) is for every gamma, will be a valuable assets to argue
> about our current pipeline is superior.

Baseline target (VERBATIM):
> baseline method should be a. SR 100% b. CR 0%, c. within-onlysuccessfulepisode-average-min-clearance
> mean + std: SIMILAR TREND across gamma WITH DEMO, but must be safer across gamma (BECAUSE VERIFIER
> POLYTOPE stresses overly conservative NOMINAL POLYTOPE, so aggressive and safe generative policy can
> sometime be verified where demo cannot generate) d. average-time-completion mean + std: SIMILAR TREND
> across gamma WITH DEMO, but must be faster across gamma. e. Coverage: Close to 16 must be lot then demo.
> (with added 4 plugs shouldn't matter because starting from the origin was originally out of the
> distribution - the scene without plugs)

Also: the from-scratch test itself —
> This is the test to see how our current recipe (without manual seeding, etc) works from the early
> beginning of iteration and see if it figures out the stable learning (without illconditioned and
> grazing). Concatenate the result with previous best it 100 and report … You must be able to produce a
> similar plot with full_history_curriculum.mp4 (shows iterations evolving).

## STATE AT HANDOFF (2026-07-11 ~22:00)

| Arm | Status | Last M5 (walled) | Known verdict |
|---|---|---|---|
| BASE `walls4_scratch_base_s820` | ~it88, finishing | SR .69 / CR .17–.29 | NOT at goals; collisions localized at INTERIOR diagonal pinches (1,1),(3,3),(4,4) — 13/75 probe, ZERO on walls/plugs → frontier (low-margin) pressure front-loads pinch data a from-scratch policy can't execute: **curriculum needs competence first** (matches 2D micro20 warm-up-noise lesson) |
| (1) NOCUR `_nocur_s821` | FINISHED it100 | SR .80 / CR .06 | Best EARLY (uniform sampling avoids pinch pressure). 2D history + mode-concentration diagnosis predict plateau/oscillation with depth — that is its expected failure, test it in the continuation |
| (2) NOSOCP `_nosocp_s822` | FINISHED it100 | SR .37 / CR .37 | M100 done: SR 33–54%, **CR 25–54%**, clearance collapsed to .22, times "beat the expert" by dying — the money ablation row (`results/p2/eval_walls4_nosocp_it100_m100`) |
| (3) NOPROG `_noprog_s823` | FINISHED it100 | SR .71 / CR .20 | M100 pending |
| Walled expert reference | `results/expert_gt_walls4` running (M100×7γ, `--wall-plugs 4`) | — | THE bar for (c)/(d)/(e) — never compare against open-scene T1 |

Zero-shot rows for the tables: pretrained on walls M5 ≈ .69; s792 zero-shot walled M25 66.3%/CR 1.7% (4 plugs).

## YOUR TASKS (in order)

1. **Finish the 0–100 exhibit**: M100 walled evals for base/nocur/noprog (`eval_ae.py policy-worker
   --wall-plugs 4 --M 100 --seed0 0` per γ, outdirs `results/p2/eval_walls4_<arm>_it100_m100`), assemble
   `tables/T_WALLS4_SUITE.md` = per-γ a–e mean±std for 4 arms + walled expert + zero-shot rows. Render the
   4 curriculum videos (`video_curriculum_fixed.py --run <arm dir> --iters 0,1,2,3,4,5,10,20,...,100` —
   the module now AUTO-DRAWS the plugs from recipe.json) and `analysis/suite_internals.py` (auto-includes
   all four + the previous open-scene lineage overlay).
2. **Continuation toward the a–e bar**: exact stateful resume of BASE (and NOCUR as the foil) from
   `final.pt` — identical recipe/signature, `--iters 120` more (the built-in mid-phase engages at it100:
   inner-steps 2). Gate every ~10 iters: M25 walled; decide on M100 only. β→0.2 ONLY if coverage stalls
   (the locked-recipe rule). If BASE's pinch-collision pattern persists past ~it140, the pre-authorized fix
   direction is PHASED CURRICULUM (uniform first, frontier on once M5 SR ≥ ~.85 sustained) implemented as
   a flag — log it as the recipe's from-scratch amendment, run as a NEW arm, keep the pure arms intact.
3. **Diagnosis discipline** (use, don't reinvent): collision-location classifier (inline pattern in
   PROGRESS ~21:0x entry), `analysis/trio_probe.py` pattern for marginal seeds, `analysis/seed12_tail_trace.py`
   machinery for latent probes, `analysis/latent_support_map.py` for support maps. Fixed eval seeds 0–24/0–99;
   training/replay seeds 100+; never train on evaluation seeds.
4. **Coverage to ~16** (goal e): targeted proposals alone historically produce ZERO exact hits — if
   coverage < ~8 by it160, implement the mode-hit gate + absolute mode schedule (2→4→8→12→14) from
   `analysis/coverage_iteration_diagnosis.md` §next-phase; rare-mode retention via the escape-replay
   mechanism (certified rare-mode windows as immutable replay — builder pattern in
   `analysis/build_escape_replay.py` + certify filter `analysis/audit_replay_certify.py`).
5. **Publishable package**: T_WALLS4_SUITE (+CSV), suite internals, 4+1 videos (arms + baseline
   full-history with the 1,2,3,4,5,10,20,… cadence), the curriculum-needs-competence finding with its
   evidence probe, clearance/time distribution plots vs walled expert per γ (the (c)/(d) "similar trend,
   safer & faster" claim needs mean±std plots, not just tables), T4-style mode-discovery vs walled expert,
   and a METHODS.md section: walls remove boundary-invisibility ⇒ no hand-crafted plumbing needed.

## RULES LEARNED THIS WEEK (violating these wasted hours — don't)

- M5 (35 rollouts) is telemetry, NOT evidence; M25 misses rare CR; decide on M100 only.
- From-scratch: `--min-modes-per-gamma 0` (tracking-only; the quota is a coin-flip gate and stalls arms),
  `--gather-attempt-cap 600` (γ0.1 starves first), trust-anchor rollback OFF (fine-tuning device only).
- The single-class ablation needs the `_fresh_batch_plan` bypass (already in the trainer, flag-gated).
- `--wall-plugs 4` EVERYWHERE (trainer, every eval_ae call, every probe env) — a missing flag silently
  evaluates the wrong scene.
- pkill self-match: always `pkill -f "grid_expand_hardtail.*<tag>"`, never a bare folder-name pattern.
- 18-gate harness (`analysis/test_hardtail_trainer.py`) must pass after ANY trainer edit.
- Every run: `--viz-db-every 1 --log-comp-every 1` or the videos/internals cannot be produced later.
- Report at round boundaries; extras are proposals. Mizuta/Kazuki stays frozen benchmark-only.

## 6. ADAPTIVE-γ DEPLOYMENT (user addition — the capstone experiment; VERBATIM below)

> deploying adaptive policy (at the deployment level after flow expansion) by scheduling gamma to actually
> prove our adaptive policy (maybe one heuristic schedule of gamma depends on the proximity to nearby
> obstacles / gradient based optimization of gamma in [0, 1]) to accomplish the safety and performance
> superior to baseline discrete gamma cases.

Run this on the MATURED baseline checkpoint (after the continuation approaches the a–e bar) — it is pure
deployment logic, no retraining. γ is already a continuous conditioning input (low5[4]); the policy was
trained at 7 discrete values and interpolates.

Two adaptive modes to implement (deploy-time only, in a copy of the deploy path):

1. **Heuristic proximity schedule**: per replan step, γ(t) = clip(γ_min + (γ_max−γ_min) ·
   (d_min(t) − d_lo)/(d_hi − d_lo), γ_min, γ_max) with d_min = min clearance to obstacles (plugs included),
   start with (d_lo, d_hi, γ_min, γ_max) = (0.3, 1.0, 0.1, 1.0). Far from obstacles → aggressive; near →
   cautious. One tuning sweep over (d_lo, d_hi) is allowed and must be reported as such.
2. **Verifier-guided γ selection** (the stronger claim; do after 1 works): per replan step, evaluate k γ
   values {0.1..1.0} with the SAME latent, score each generated window by the VERIFIER (exact certificate
   pass + face margin + window progress), execute the best-certified window's first action. Optionally the
   gradient variant: γ* by a few ascent steps on margin+progress through the differentiable NFE map.
   NOTE: this is a NEW DEPLOYMENT MODE evaluated as its own method — it must NEVER contaminate the fixed-γ
   a–e tables (those stay faithful temp=1/NFE8/fixed-γ, no selection). Two separate result sections.

Evaluation protocol (M100, walled scene, same seeds/metrics as everything else):
- Rows: adaptive-heuristic, adaptive-verifier, each fixed γ ∈ {.1,.2,.3,.4,.5,.7,1.0}, random-γ-schedule
  control, walled expert.
- The CLAIM to test: adaptive sits ABOVE the fixed-γ Pareto front — safety (clearance/CR) ≥ the safest
  fixed γ while time ≤ the fastest fixed γ (or dominates on one without losing the other).
- Deliverables: `tables/T_ADAPTIVE_GAMMA.md` (a–e mean±std per row); Pareto figure clearance-vs-time with
  the 7 fixed-γ points + adaptive points (std ellipses, SR/CR annotations); a γ(t)-trace figure over
  representative rollouts (γ dipping at pinches, rising in open space); one video with the trajectory
  COLORED BY γ(t) (viridis; reuse the video tooling — this is the paper's adaptive-policy money shot).
