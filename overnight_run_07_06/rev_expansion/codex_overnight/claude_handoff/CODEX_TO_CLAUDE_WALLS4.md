# Codex to Claude: WALLS-4 deliverables and exact stopping point

## Read this first

This is the plain-English handoff for the WALLS-4 experiment. Start here, then use
`claude_handoff/WALLS4_SUITE_HANDOFF.md` for the original detailed specification and
`analysis/WALLS4_COMPLETION_AUDIT.md` for the checklist.

Important: the checklist was written before the later `walls4_phased_s830` run and is stale for that one
row. It is still correct that the overall job is **not complete**. No experiment process was running when
this handoff was written on 2026-07-12.

## What the user wanted, in easy English

The old navigation scene had an invisible outer boundary. The policy could sense obstacles, but it could
not sense that boundary. This caused strange edge behavior and forced earlier work to use manual fixes.
The user asked us to add four small obstacle "plugs" so the boundary becomes visible to the policy, then
test whether the normal learning recipe can solve the task from scratch without hand-made repairs.

The requested study has these parts:

1. Train and compare the full method and three ablations at every gamma value
   (`0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0`): no curriculum, no multi-step SOCP safety check, and no progress
   check. Show metrics, training internals, and curriculum videos.
2. Mature the full method toward the hard target: 100% success, 0% collision, safer and faster than the
   walled expert, and close to 16 distinct path modes per gamma.
3. Diagnose failures instead of hiding them. Evaluation seeds are fixed and must never be used for
   training or replay.
4. After a strong fixed-gamma model exists, test deployment-only adaptive gamma scheduling. Compare a
   distance heuristic and same-latent verifier selection against all seven fixed gammas, a random-gamma
   control, and the expert. The desired claim is that adaptive gamma lies above the fixed-gamma Pareto
   front, but that claim must be tested, not assumed.

In one sentence: make the boundary observable, prove what each training component contributes, get a
safe/fast/diverse model from scratch, and only then test adaptive gamma as the paper's capstone result.

## What Codex completed

### Corrected and completed the iteration-100 suite

- Fixed `eval_ae.py`: the expert worker used `wall_plugs`, but its command-line parser did not accept
  `--wall-plugs`. The walled expert was rerun correctly with four plugs.
- Completed M100 evaluation for BASE, NOCUR, NOSOCP, and NOPROG at all seven gamma values.
- Completed the correct walled-expert M100 reference and the requested zero-shot context rows.
- Produced the main suite table/CSV, mode-discovery table, collision audit, expert comparison plot,
  internals figure, four curriculum videos, and concatenated history video.
- Updated the methods draft with the visible-boundary motivation.

The important iteration-100 result is negative but useful:

| Method | M100 success range | M100 collision range | Main finding |
|---|---:|---:|---|
| BASE | 68–76% | 3–24% | The full curriculum starts stressing hard pinches before the new model is competent. |
| NOCUR | 56–79% | 1–17% | Uniform sampling is better early, but remains unstable and incomplete. |
| NOSOCP | 33–54% | 25–54% | Removing multi-step safety verification badly damages safety. |
| NOPROG | 58–70% | 4–24% | Safety alone does not produce a reliable task-solving policy. |
| Walled expert | 100% | 0% | Correct reference; clearance 0.240–0.299 m and coverage 5–9. |

### Diagnosed why BASE was failing

- Classified all M100 collision locations. BASE had 119 collision episodes out of 700, versus 63 for
  NOCUR. For BASE, 116/119 were at interior obstacles, 70 at `(1,1)`, and only 3 were at wall plugs.
  Therefore the new walls fixed the old invisible-boundary failure, but did not solve learning order.
- Ran a fixed pinch trio and a 4,096-latent probe at iterations 100 and 104. The bad pinch is a broad,
  absorbing failure basin, not one rare random latent. At the final pre-collision context, 100% of sampled
  plans immediately collided.
- The resulting amendment was: learn basic competence with uniform sampling first, then turn on frontier
  curriculum only after success is sustained. The threshold was calibrated to aggregate M5 SR >= 0.85
  for two consecutive gates.

### Continued the pure BASE and NOCUR controls to iteration 140

Both exact full-state continuations finished and the external M25 gate finished. No process is still live.

| Pure control at it140 | Aggregate SR | Aggregate CR | Per-gamma coverage | Verdict |
|---|---:|---:|---:|---|
| BASE | 73% | 18% | 2–3 | Not promotable; interior pinch collisions persist. |
| NOCUR | 58% | 21% | 3–5 | Not promotable; the expected deeper-training instability is visible. |

The exact per-gamma table is `tables/_T_WALLS4_IT140_M25.md`. BASE had 31/175 collision episodes in the
saved paths, all at interior obstacles. NOCUR had 37/175, including only two wall-plug collisions. The
header in `analysis/walls4_collision_locations_it140_m25.md` incorrectly says `/700`; seven gammas times
M25 is 175. Correct that label before reusing the audit in a report.

### Implemented and tested the next mechanisms, without treating them as final evidence

- Added irreversible phased curriculum flags to `grid_expand_hardtail.py`.
- Added a mode-hit gate and absolute mode schedule `2 -> 4 -> 8 -> 12 -> 14`.
- Added `--target-perp-brake`, which points toward the requested mode boundary while braking perpendicular
  momentum. The Valid2/SOCP acceptance rule is unchanged. In a small controlled probe it produced 3/10
  exact certified target hits versus 0/10 before.
- Extended rare-mode replay building and certification to record/check the four-plug scene and mode
  provenance. Only a compatibility smoke replay exists; the real WALLS-4 rare-mode replay is not built.
- Added deployment-only adaptive-gamma evaluation, tuning, reporting, Pareto, trace, and video tooling.
  Smoke tests pass and same-latent selection was checked, but no mature-checkpoint M100 adaptive result
  exists.
- The final trainer regression harness passes 20/20:
  `analysis/test_hardtail_trainer.cpu.final.json`.

## Later Claude work already present in the workspace

The append-only `PROGRESS.md` shows that a later Claude session launched
`results/p2/walls4_phased_s830` and evaluated it at iteration 140 with M100. That arm enabled both phased
curriculum **and** `--target-perp-brake`.

Its M100 result is encouraging:

- SR by gamma: `{74, 74, 77, 75, 76, 72, 73}%`.
- CR by gamma: `{9, 3, 1, 1, 1, 1, 2}%`.
- Clearance: `0.238–0.271 m`.
- Successful-episode mean time: `12.4–21.0 s`, still slower than the expert.
- Coverage: `{20, 5, 5, 5, 5, 5, 5}`. Gamma 0.1 is highly diverse, but the other gammas are not near the
  requested coverage target.
- The phase switched at iteration 17; training logged 267 exact target hits and a zero-collision SR50
  tail according to the progress entry.

Do **not** say that phased curriculum alone caused these gains: this arm changed two mechanisms at once.
Also do not say that it met the final target. Its success is only 72–77%, gamma 0.1 still has 9%
collisions, completion is slower than the expert, and coverage is unbalanced.

## Deliverables and where they are

### Primary reports

- Original full request: `claude_handoff/WALLS4_SUITE_HANDOFF.md`
- This takeover note: `claude_handoff/CODEX_TO_CLAUDE_WALLS4.md`
- Completion checklist: `analysis/WALLS4_COMPLETION_AUDIT.md`
- Append-only chronological log: `PROGRESS.md`
- Methods draft: `claude_handoff/METHODS.md`

### Iteration-100 paper assets

- Main results: `tables/T_WALLS4_SUITE.md` and `tables/T_WALLS4_SUITE.csv`
- Mode discovery: `tables/T_WALLS4_MODE_DISCOVERY.md`
- Collision diagnosis: `analysis/walls4_collision_locations.md` and `.json`
- Clearance/time plot: `figures/walls4_clearance_time_vs_expert.png`
- Suite internals: `figures/internals_suite_walls4.png`
- Four arm videos: `video/walls4_scratch_{base,nocur,nosocp,noprog}_curriculum.mp4`
- Old history: `video/full_history_curriculum.mp4`
- Concatenation: `video/previous_best_plus_walls4_it100_curriculum.mp4`
- Correct expert: `results/expert_gt_walls4/`
- Authoritative arm evaluations: `results/p2/eval_walls4_{base,nocur,nosocp,noprog}_it100_m100/`

### Iteration-140 pure-control gate

- Table/CSV: `tables/_T_WALLS4_IT140_M25.md` and `.csv`
- Collision audit: `analysis/walls4_collision_locations_it140_m25.md` and `.json`
- Completion marker: `analysis/WALLS4_IT140_GATE_READY`
- Checkpoints: `results/p2/walls4_scratch_base_s820/` and
  `results/p2/walls4_scratch_nocur_s821/`

### Diagnosis and mechanism evidence

- Pinch trio: `analysis/walls4_pinch_trio_it100_it104.txt`
- Latent probe code/results:
  `analysis/walls4_pinch_latent_probe.py`,
  `analysis/walls4_pinch_latent_offset10_it100_it104.json`, and
  `figures/walls4_pinch_latent_offset10_it100_it104.png`
- Target proposal evidence:
  `analysis/targeted_coverage_walls4_nocur104_sharp.json` and
  `analysis/targeted_coverage_walls4_nocur104_perpbrake_defaultknobs.json`
- Clean phased-only launch script, not yet run: `analysis/run_walls4_phased.sh`
- Coverage/replay branch launcher: `analysis/run_walls4_coverage_branch.sh`
- Replay tools: `analysis/build_escape_replay.py` and `analysis/audit_replay_certify.py`
- Adaptive tools: `analysis/adaptive_gamma_eval.py`, `analysis/adaptive_gamma_tune.py`, and
  `analysis/adaptive_gamma_report.py`
- Adaptive smoke: `analysis/test_adaptive_gamma.json`

### Later combined arm

- Checkpoint/history: `results/p2/walls4_phased_s830/`
- M100 rows and paths: `results/p2/eval_walls4_phased_it140_m100/`
- Video: `video/walls4_phased_curriculum.mp4`
- `figures/internals_suite_walls4.png` was refreshed for this arm, but
  `tables/T_WALLS4_SUITE.md` was not updated with its row.

## What remains; do not claim completion yet

1. Reconcile the stale audit and add the combined it140 row to a clearly labeled table. Keep the original
   four-arm iteration-100 table scientifically intact or make a separate amendment table.
2. Preserve the clean ablation question. `s830` combines phased ordering and perpendicular braking. If a
   causal phased-only claim is needed, run the prepared clean phased-only arm separately. If the goal is
   only the best recipe, continue the combined lineage but label it honestly.
3. Finish the requested pure BASE/NOCUR control continuation from iteration 140 to 220, unless the user
   explicitly releases that requirement. These controls are evidence, not promotion candidates.
4. Mature the promising lineage. The likely branch is a new checkpoint-derived coverage branch with the
   mode-hit gate, absolute mode schedule, certified rare-mode replay, and beta 0.2. Never overwrite the
   pure or `s830` directories.
5. Use M25 only as a gate. Promotion and paper claims require M100 at all seven gammas.
6. Build/certify the actual WALLS-4 rare-mode replay from training seeds 100+, then audit that it contains
   no evaluation seed and `wall_plugs == 4` before training with it.
7. Only after a fixed-gamma checkpoint meets or closely approaches the a–e target, run the adaptive-gamma
   tuning/evaluation/report pipeline. Produce `T_ADAPTIVE_GAMMA`, the Pareto plot, gamma traces, and the
   gamma-colored video. Keep adaptive rows out of fixed-gamma tables.
8. Refresh the final methods text and all mature comparison figures. State negative findings plainly if
   the requested bar is not reached.

## Safe first commands for Claude

These inspect the state; they do not launch anything:

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
sed -n '1,260p' claude_handoff/CODEX_TO_CLAUDE_WALLS4.md
sed -n '1,260p' analysis/WALLS4_COMPLETION_AUDIT.md
tail -n 100 PROGRESS.md
ps -eo pid,etime,cmd | rg 'grid_expand_hardtail|eval_ae|adaptive_gamma|walls4'
```

Before any new trainer launch, rerun the regression harness:

```bash
LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib python analysis/test_hardtail_trainer.py \
  --out analysis/test_hardtail_trainer.cpu.final.json
```

The prepared clean phased-only launch is:

```bash
bash analysis/run_walls4_phased.sh
```

Do not run that blindly: `s830` already supplies a combined-arm result, and the next scientific decision
is whether the user wants causal phased-only evidence or wants to spend compute maturing the best combined
recipe.

## Rules that must not be broken

- Pass `--wall-plugs 4` to every trainer, evaluator, expert, and diagnostic environment.
- Evaluation uses seeds 0–24 for M25 and 0–99 for M100. Training and replay use seeds 100+ only.
- Never train on evaluation failures or use them as replay.
- Do not overwrite or silently alter the four pure arm directories.
- Keep adaptive-gamma deployment results separate from fixed-gamma scientific tables.
- Do not call M5 or M25 paper evidence; M100 is the decision standard.
- From scratch, keep `--min-modes-per-gamma 0`, `--gather-attempt-cap 600`, and trust rollback disabled.
- Preserve exact Valid2/SOCP acceptance when using targeted proposals.
- `PROGRESS.md` is append-only.
- No Git push was requested or performed.

## Bottom line

The four visible wall plugs solved the old boundary-observability problem, but the original curriculum
still introduced difficult interior pinch cases too early. The pure BASE and NOCUR controls do not mature
by iteration 140. The strongest current checkpoint is the later combined phased+perpendicular-brake arm:
it greatly improves safety and discovers many modes at gamma 0.1, but it is not yet reliable, fast, or
balanced enough for the user's final claim. The infrastructure for the next coverage branch and adaptive
gamma experiment is ready; the final training and M100 evidence are not.
