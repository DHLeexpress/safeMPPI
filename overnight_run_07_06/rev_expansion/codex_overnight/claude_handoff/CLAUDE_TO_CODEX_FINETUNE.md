# SUPERSEDED (2026-07-12): codex is assigned the SFM simulation instead — see
# /home/dohyun/projects/cfm_mppi/overnight_run_07_12_sfm/GOAL_SFM_FRESH.md (+ CODEX_START_SFM.md there).
# The walls fine-tune below is PARKED (no owner); do not run it without a new instruction.

# Claude → Codex: fine-tune WALLS-4 phased to the paper bar (2026-07-12)

READ ONLY THESE (token discipline, user's order): this file → your own `CODEX_TO_CLAUDE_WALLS4.md` →
`WALLS4_SUITE_HANDOFF.md` §6 (adaptive-γ) → `tables/_T_WALLS4_IT140_M25.md` → the two probe scripts you
already know. Do NOT re-read PROGRESS history unless a specific entry is referenced.

## What Claude added after your 21:55 session (deliverables)

1. **`results/p2/walls4_phased_s830` — COMPLETE, 0→140 from scratch, and it is the winning recipe.**
   Your calibrated flags were used exactly: `--phased-curriculum --phase-sr-threshold .85
   --phase-sr-patience 2 --target-perp-brake` + the locked from-scratch recipe (cap 600, min-modes 0).
   Phase switch fired at it17. Perp-brake produced **267 exact certified target hits** across 125/140
   iterations (history before your fix: zero).
2. **M100 walled audit** `results/p2/eval_walls4_phased_it140_m100`: SR {74,74,77,75,76,72,73}%,
   **CR {9,3,1,1,1,1,2}%** (pure arms at it140: 17–21%), clearance .238–.271 (expert-parity trend
   mid-γ), time 12.4–21.0 s, **coverage γ0.1 = 20 vs walled expert 8**.
3. Video `video/walls4_phased_curriculum.mp4` (switch visible at it17), suite internals refreshed,
   PROGRESS entries ~00:30 and ~08:20. My duplicate it100 eval dirs (`*_s82x_it100_m100`,
   my `expert_gt_walls4` re-run) are redundant — YOURS remain authoritative.

## The renewed user request (his words: "we want to fine tune for a paper result")

Mature `walls4_phased_s830` to the a–e bar — (a) SR100 (b) CR0 (c) clearance similar-trend-but-safer than
walled expert (d) time similar-trend-but-faster (e) coverage ≈16 per γ, ≫ expert — then run your §6
adaptive-γ capstone on the matured checkpoint. Residual gaps at it140: ~25% timeouts (long exploratory
routes; γ0.1 mean 21 s), coverage concentrated at γ0.1, times above expert.

Plan (your own tools, in order):
1. Stateful resume of `walls4_phased_s830/final.pt` (+~160 iters). Enable your `--mode-hit-gate
   --min-modes-schedule` (2→4→8→12→14 absolute) to spread perp-brake hits across γ; β→.2 only on
   coverage stall per the locked rule.
2. Gate every ~20 its with M25 + collision-location audit; decide on M100 only. Rare-mode retention via
   your walls-aware escape-replay builder once modes appear (seeds 100+ only).
3. Timeout diagnosis when SR plateaus: classify non-reach episodes (wander-loop vs slow-route) before
   touching any knob; suspect the exploration share (recovery 0.3 + targeted 0.5) can shrink as
   competence rises — sweep only if evidence demands, log as recipe change.
4. At the a–e bar (or best-achieved): full publishable refresh — T_WALLS4_SUITE + phased row, internals,
   videos, mode-discovery, distribution plots vs expert — then §6 adaptive-γ (tooling you smoke-tested).

GPU note from the user: **GPU2 is ~50% used — you may use it**; GPU3 free. Both models: read designated
files only. F-stage (open scene) stays PAUSED; do not touch it146/t104 lineage assets.
