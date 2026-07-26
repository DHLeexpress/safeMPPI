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
