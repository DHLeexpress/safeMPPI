# Pure AFE-style Safe Flow Expansion (codex_overnight, 2026-07-16)

This folder contains the complete implementation, runs, evaluations, and paper figures of the
**minimal AFE-style Safe Flow Expansion** — the redesign that replaced the curriculum trainer after
the 9-fault critique ("Faults in Claude's implementation"). The whole design reduces to one rule:

> **The generated, uncertainty-scored, verified, stored, and trained object is the SAME planned
> action window.** Runtime safety is carried by a deterministic full-window verifier + a certified
> fallback; σ is used exactly once (acquisition); progress only ranks already-safe plans; there is
> no curriculum and there are no stabilizers (no demo replay, no LwF, no anchoring, no encoder
> freezing — user directive). The only regularizer is a proximal term.

Rollback: git tag `pre-afe-2026-07-16` restores all pre-refactor code.

Two studies live here: **Study 1** (§1–§8): the pure AFE-minimal method, 100 rounds, certified
SafeMPPI fallback, 5 arms + 3 gate-brothers. **Study 2 — AFE2** (§9): the corrected two-arm
10-round study (evolving representation, rebuilt A, EXPERT-FREE verify-or-terminate). §10 is the
complete manifest of every file added/modified in this work.

---

## 1. Notation (what you see in the figures/video)

| symbol | meaning |
|---|---|
| `c_t = (grid_t, low5_t, hist_t)` | the context at control step t: a robot-centered 32×32 polar map whose H_P channel encodes the local SafeMPPI polytope (the only grid channel this model reads); `low5 = [relgoal_x, relgoal_y, v_x, v_y, γ]` (normalized); `hist` = last 16 executed actions. |
| `U_t ∈ R^{10×2}` | ONE planned action window: 10 future accelerations sampled by the flow model, conditioned on `c_t`. Only `U_t[0]` is ever executed (receding horizon). |
| `y = v_safe(U,c) ∈ {0,1}` | the FULL verifier, run BEFORE execution: DI-roll the whole plan; y=1 iff every predicted position is in the task box AND the SOCP polytope certificate holds at this γ (contraction weights α_k=(1−γ)^k). |
| `m, r` | stored SEPARATELY from y: `m` = the verifier's face margin; `r` = net goal-approach of the plan (d₀−d₁₀). Progress is never part of the safety label. |
| `φ⁰_s(U,c)` | the 32-d penultimate feature of the **frozen copy of the pretrained flow** at noise level s=0.9 (superscript 0 = never updated during expansion). `z = φ⁰_s/‖φ⁰_s‖`. |
| `A_n = I + λ⁻¹ Σ z_i z_iᵀ` | cumulative 32×32 design matrix over EVERY window ever submitted to the verifier (positives AND negatives). No cap, no eviction, no decimation ⇒ uncertainty shrinks monotonically. |
| `σ_n(U,c) = √(zᵀA_n⁻¹z)` | Bayesian-linear-regression posterior std = "how unlike everything already verified is this plan's feature". Empty A → σ≡1. |
| `π_j ∝ exp((σ_j−σ_max)/β)` | the acquisition distribution over the K sampled candidate plans (Gibbs solution of max_q E_q[σ] − β·KL(q‖p_θ)). B indices are drawn from π WITHOUT replacement and each is fully verified. |
| `D_n / D_n⁺` | the append-only store of all verified queries `(c,U,y,m,r)` / its verified-safe subset. D⁺ is the ONLY training set; replay over it is UNIFORM. |
| prox objective | `θ_{n+1} = argmin_θ  mean_{D⁺} ℓ_CFM(θ;c,U) + ‖θ−θ_n‖²/(2η)`. Bounded continued pretraining; the Adam step count is a solver detail (we stop at a functional-displacement bound or 40 steps and report which fired). |
| `â_n` | acceptance of the σ-TILTED queries of round n (verifier-query efficiency — NOT model validity). |
| `V̂_n, V̂ᵖʳᵒᵍ` | model validity: sample plans UNTILTED at a FIXED held-out context set ρ_eval, fully verify, report per γ; V̂ᵖʳᵒᵍ additionally requires the calibrated net-progress bar r ≥ min(0.10, 0.5·d₀). Audit samples never enter D or A. |
| fallback | when none of the B drawn plans verifies safe, the step is executed by the certified SafeMPPI backup (the same controller that generated the pretraining demos). Its per-γ rate and LOCATION are reported — fallback decay = expansion of the verified set under the executed distribution. |

## 2. The loop (grid_expand_afe.py)

Per round n (100 rounds): run 8 shielded episodes (γ round-robin over {0.1,…,1.0}, fixed start
(0.3,0.3), goal (4.7,4.7), 8-plug walled scene, T=300 steps, reach 0.15). At every control step:
sample K=64 plans from the current flow → σ from A_n on frozen φ⁰ → draw B=8 by π → FULL-verify
each (all 8 go into D and A_n) → execute one verified-safe plan (`--exec-rule pi`: sample ∝π among
safe; spec default `progress`: argmax r) → else certified SafeMPPI fallback. After the episodes:
one proximal update on uniform D⁺ replay (batch 128, Adam lr 2e-5, η=0.01, stop at relative field
displacement ≥0.03 vs round start or 40 steps). Every 5 rounds: the untilted audit. Every 10:
closed-loop SR/CR/coverage (M=8/γ, T=350) + checkpoint. Every round: viz_db (all queries, executed
paths+fallback mask, the exact D⁺ rows drawn by the update, A_n snapshot) + probe.jsonl.

## 3. Fault → fix map (how each of the 9 critiques is answered in code)

| fault | this implementation |
|---|---|
| 1. selected plan ≠ verifier query | every plan that enters ANY buffer was itself fully SOCP-verified BEFORE its first action could be executed (`afe_core.verify_plan` in the acquisition loop). |
| 2. σ = plan novelty, not verifier uncertainty | σ counts ONLY fully-verified queries (A_n updated exactly when verify_plan ran). |
| 3. one buffer, incompatible roles | one store (DStore), one role; no post-hoc executed-window rescoring exists. |
| 4. FIFO/random-eviction inconsistency | A_n is cumulative — no cap, no eviction, no [::3] decimation; DStore is append-only. |
| 5. σ used twice | σ appears once (acquisition π). Replay is uniform; no frontier quota. |
| 6. frontier = tuned AND-cell | no easy/frontier, no quantile, no mix ratio, no mode/γ balancing anywhere. |
| 7. safety/progress conflation | y is safety-only; r stored separately; r only ranks verified-safe plans at execution and defines V̂ᵖʳᵒᵍ. |
| 8. inner=4 unjustified | single explicit prox objective; step count = solver tolerance (η self-limits the round displacement to ~0.015–0.02; the 0.03 bound rarely fires). Solver ablation = the −Prox brother. |
| 9. post-hoc SOCP, undefined all-fail | verification is pre-execution; the all-fail branch is DEFINED: certified SafeMPPI backup (17 ms/step). |

## 4. Exact parameters (everything, with provenance)

**Scene/task** (`grid_scene.make_grid` + `grid_expand_hardtail._apply_wall_plugs(8)`): 5×5 m box,
16 interior obstacles r=0.2 at {1,2,3,4}², dense wall circles + 8 corner plugs; double-integrator
dynamics, dt=0.1, u_max=1.0; start (0.3,0.3) (`--start-eps 0.3`), goal (4.7,4.7) (`--goal-xy`),
reach 0.15; γ ∈ {0.1,0.2,0.3,0.4,0.5,0.7,1.0} (conditioning input, fixed rotation, never scheduled).

**Pretrained flow** (`../../results/hp_repr/pretrained_a32uni.pt`, `grid_hp_expt.GridHPFlowPolicy`):
~68k params; ctx = raw low5(5) ⊕ CNN(1→8→16, AdaptiveAvgPool 8×8 → 32) over the 1×32×32 H_P
channel; trunk 89→160→96→32 (SiLU), head 32→20; H=10, K_hist=16. Trained by `pretrain_repr.py`
(AdamW 3e-4 cosine, batch 256, 120 epochs) on 219,908 SafeMPPI windows from 566 uniform-grid
off-diagonal starts (`gen_uniform_data.py`, druni_ prefix).

**SafeMPPI (expert generator AND certified fallback)** (`cfm_mppi.safegpc_adapter.safemppi.
SafeMPPIAdapter` with `grid_scene.mode1_config()`): plain-Gaussian-sampling MPPI (mode 1, no
polytope-area importance sampling), barrier activation radius 2.0 m, u ∈ [−1,1]², noise σ ≈ 0.87
per axis (0.5·√3), planning obstacles inflated by r_robot + margin (`planner_obstacles`). One
`plan()` call per fallback step (~17 ms).

**Verifier** (`grid_metrics2.window_socp_stats`, resolves `verifier_polytope` from
`overnight_run_2026-07-01/` — the same verifier every previous run used): polytope fit with
n_theta=180, α_k=(1−γ)^k certificate over the DI-rolled plan; in-bounds via `GM.in_taskspace`.
2.3 ms/plan (measured). Verifier RuntimeErrors count as y=0 (conservative; counter in probe.jsonl).

**Acquisition**: K=64 (matches the deployment sampler), B=8 (verifier budget; affordable at 2.3 ms),
β=0.2 (carried from the previous recipe; with σ∈(0,1] this is a strong early tilt), s=0.9,
λ=10 for the main arms — **measured choice**: the GP-era λ=0.01 makes a single query kill a feature
direction, saturating the cumulative A_n within ~2 rounds (σ-spread study in
`analysis/afe_lam_study.py`); λ=0.01 kept as a reference arm (result: no practical difference —
candidates live in the queried subspace either way).

**Update**: uniform D⁺ replay, batch 128, Adam lr 2e-5 (all parameters — encoder included),
η=0.01 (probe-calibrated: self-limits the per-round relative field displacement to ~0.015–0.02),
stop at fstep ≥ 0.03 vs round start or max 40 steps, grad-clip 1.0.

**Audit ρ_eval**: 12 fixed free-space positions (position 0 = the episode start; rng seed 20260716,
clearance > 0.05) × 2 velocity conditions (rest AND adverse = 0.65 m/s toward the nearest obstacle
— added after measuring that rest-only audits sit at a 99% ceiling) × 7 γ = 168 contexts; 4 untilted
plans each, fully verified, every 5 rounds; never buffered.

**Closed-loop eval** (`analysis/report_at.py`): M=40/γ, T=350, bare policy (no tilt, no verifier,
no fallback), vs `results/expert_g47`; coverage = distinct goal-relative staircase words
(`GM2.staircase_id_goal`).

## 5. Assumptions / deviations from the written spec (state these when distributing)

1. **Execution rule**: the spec says argmax-progress among verified-safe. Both rules are
   implemented; the recommended arm uses `--exec-rule pi` (sample ∝π among verified-safe) because
   argmax-progress measurably collapses per-γ route diversity via visited-context replay
   amplification (covΣ 26–30 vs 34; base 52). This is a deviation, flagged, with the spec-default
   available.
2. **ρ(c)**: "γ sampled from a fixed ρ(c)" is realized as a deterministic round-robin (fixed
   marginal) over the 7 γ; episodes always start at the same cleared start state. The context
   distribution is therefore ENDOGENOUS beyond the start — this is the binding limitation (see §6).
3. **λ** is re-calibrated (10, measured) because the spec inherited λ=1e-2 from the rolling-buffer
   GP where eviction kept σ alive; a cumulative A_n needs a larger effective prior count.
4. **Solver stop**: "optimize to a stated tolerance or update-norm bound" is realized as a
   functional-displacement bound (0.03 relative field change on a fixed probe batch) + a 40-step
   cap; η itself is the effective trust region.
5. **Storage**: contexts are stored per control step (the B queries of a step share one context);
   the grid is stored as the H_P channel only (float16) — verified to reconstruct `ctx_from`
   EXACTLY (this model reads only channel 2).
6. **Robustness**: verifier internal errors → y=0 (counted); fallback exceptions → braking action
   (never observed); episodes also end on reach/OOB/collision (OOB/collision never occurred in any
   shielded arm: 0/800 episodes per arm).
7. **No train-state resume**: AFE runs are short (~1 h); checkpoints are model-only.
8. **The audit's "coverage with CIs over seeds"**: 3 seeds were run for the argmax arms (910/911/
   912); π/λ-reference/brothers are single-seed (910). Extend before publishing CIs.

## 6. Findings (full detail: paper_results/AFE_FINDINGS.md, table_v6.md)

- Shielded gathering is perfectly safe (0/800 episode deaths per arm; −Verifier: 68/800 die,
  −Fallback: 39/800 die). Dithering never materialized (<1.4%) despite progress-free labels.
- Stability without anchors holds (SR dips then recovers to 0.95–1.00; V̂_rest → 1.0), BUT
  validity does NOT expand where it is missing: V̂_adverse flat at 0.44 (γ0.1: 0.00) for 100
  rounds; fallback never decays — mid-route fallbacks are learned away (12→3) while the goal-corner
  ones persist (31→40): **the shield removes exactly the failure pressure that would force
  expansion** (no certifiable plan exists at the OOD goal corner to learn from).
- σ saturates for every λ: candidates always lie in the already-queried feature subspace —
  plan-level tilting cannot steer WHICH STATES are visited. **ρ(c) is the expansion boundary**; to
  expand adverse-context validity, episode starts must cover those contexts (a deliberate ρ(c)
  design change — the certified analogue of recovery starts — NOT smuggled in here).
- Diversity: base covΣ 52 → argmax arms 26–30, π arm 34, curriculum+anchor recipe 51. The −Prox
  brother posts the best closed-loop line (SR 1.0, CR 0, a-d 24/28) while collapsing routes
  (covΣ 18) and eroding audit validity (V̂_adv −4.5 pts): the prox term is a validity/diversity
  preserver, not an SR maximizer.
- The old prior-collapse (raw up-frac 0.14→0.73) does NOT reappear in any arm (0.10–0.15).
- Validity-mass guarantee: NOT claimed — it would need a separate density-floor/EBM assumption on
  top of the CFM update.

## 7. Code map (what to read, in order)

- `afe_core.py` — BLRSigma (A_n, Sherman-Morrison), DStore, verify_plan, SafeMPPIFallback,
  build_audit_contexts / run_audit.
- `grid_expand_afe.py` — config, gather_round, prox_update, run_afe, component_probe (--probe),
  CLI; brothers via `--ablate-verifier` / `--ablate-fallback` / (`--eta 1e18 --fstep-stop 999`).
- `analysis/afe_driver.sh` (launch), `analysis/afe_assemble.sh` (evals+report+videos),
  `analysis/afe_lam_study.py` (λ calibration), `analysis/afe_report.py` (validity tracking +
  up-frac collapse audit).
- `video_afe.py` — per-round expansion video (green/red verified plan fans, σ dots, executed paths
  + fallback steps, the exact trained-on D⁺ rows, validity curves; notation footer).
- `paper_results/{rollouts,scatter,internals}_v6.py`, `table_v6.md`, `AFE_FINDINGS.md`.
- Shared (imported, NOT modified): `grid_scene.py`, `grid_feats.py`, `grid_rollout.py`
  (window_positions/di_rollout), `grid_metrics(.2).py`, `grid_hp_expt.py`, `pretrain_repr.py`,
  `verifier_polytope.py` (⚠ resolves from `overnight_run_2026-07-01/` via sys.path — never edit a
  same-named local copy without checking `module.__file__`).

## 8. Reproduce

```bash
cd overnight_run_07_06/rev_expansion/codex_overnight
python grid_expand_afe.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt \
  --outdir results/afe/probe --probe                     # component checks (SOCP ms, σ separation, η)
bash analysis/afe_driver.sh                              # main arms (GPU 3)
bash analysis/afe_assemble.sh                            # evals + validity report + videos
```
Artifacts per run: `probe.jsonl` (per-round â, per-γ acceptance/fallback, V̂, σ, solver, dither),
`viz_db/round*.pt`, `dstore.pt` (~600k verified queries), `ckpt_*.pt`/`final.pt` (report_at-
compatible), `history.json`.

---

## 9. Historical Claude AFE2: fixed-H two-arm study (spec 2026-07-16b)

This section records the result produced at `e97eead`. The current trainer preserves its acquisition
and update values but applies the explicit absorbing-goal correction in §11.

- **Evolving representation.** σ features come from the CURRENT policy φ_s⁽ⁿ⁾ (initialized at the
  pretrained φ_s⁽⁰⁾); encoder + trunk + head all train. The raw query archive D_n stays cumulative;
  at the START of every round all stored queries are re-embedded under φ_s⁽ⁿ⁾ and
  A = I + λ⁻¹Σzzᵀ is REBUILT from scratch (never carried across a representation update). θ and φ
  are held fixed during the round's gathering while A updates per completed verifier query.
  The implementation that produced the reported result computes one pre-step `sigma/pi` vector and
  draws all B indices without replacement before those updates; it does not sequentially re-score
  the remaining candidates. `socp_error` is not stored in D and does not update A.
- **Historical fixed-H execution.** No SafeMPPI or fallback action. If none of
  the B queried plans is SOCP-positive the rollout TERMINATES with status `NO_VERIFIED_POSITIVE`.
  Execution among positives = argmax progress (fixed nominal J_exec for this study).
- **Complete γ sweep**: one episode per γ, all seven γ, fixed order, every round; 10 rounds.
- **β fixed by ESS calibration** (`--calibrate`): dry round-0 pass over all 7 γ; choose
  β ∈ {0.01, 0.02, 0.05} whose median acquisition ESS/K ∈ [0.25, 0.5] (never from σ's absolute
  magnitude). Measured: 0.01→0.036, 0.02→0.125, 0.05→0.626 — NONE in band; rule-based fallback
  (nearest band midpoint) picked **β=0.02** over 0.05 by 0.250 vs 0.251 (`results/afe2/calib/
  beta_calibration.json`). Both arms share it.
- **Two matched arms**, sharing code/configuration, initial checkpoint, execution rule, budget, and
  common-random-number streams; their learned representations, plans, contexts, archives, and A
  matrices diverge after their different updates:
  `--arm prox` (control: batch 128, lr 2e-5, η 0.01, stop fstep ≥ 0.03 or 40 steps) vs
  `--arm afe` (uniform cumulative D⁺ replay, batch 128, lr 1e-4, 250 steps, NO prox).
  No curriculum, expert replay, anchors, easy/frontier, or automatic collapse rollback.
- **Diagnostics per round** (probe.jsonl): all-K and selected-B σ quantiles, ESS, normalized
  acquisition entropy, selected-vs-pool σ uplift, eigen-spectrum + effective rank of A, total CFM
  loss, encoder/trunk/head gradient norms, relative per-module parameter change, fixed-probe
  representation cosine drift, per-γ query/positive/distinct-trained counts, untilted raw validity
  (audit), dither share.
- **Evaluation** (round 0 = pretrained, then every round): (i) untilted generator audit;
  (ii) expert-free verified controller with fixed-index equal-count rollouts (M=8/γ, the SAME
  rollout seeds at every round → paired across rounds), reporting SR / CR / NVP rate / true min
  clearance / time-to-goal; Wilson CIs in the report.
- **Video** (`video_afe2.py`): every round, all seven γ panels; gray = every K=64 generated plan
  at every executed step, orange = B full-verifier query objects, green = full-H SOCP-positive,
  red = full-H rejected, blue/thick = cost-selected
  plan + executed path, X = NO_VERIFIED_POSITIVE; text = positive count, min SOCP margin, raw
  untilted validity, termination timestep.

**Hard baseline fact**: the round-0 expert-free verified controller of the pretrained scores
SR 0.00 / NVP 1.00 at EVERY γ (CR 0.00) — in Study 1 the certified fallback was completing every
episode. Any SR > 0 after training is unambiguous expert-free expansion of the verified set.

Runs: `results/afe2/{prox_s910,afe_s910}` (+ `calib/`). Report: `analysis/afe2_report.py` →
`paper_results/afe2_report_v1.png`; videos → `paper_results/afe2_{prox,afe}.mp4`.

---

## 10. File manifest — everything added/modified in this work

**New — Study 1 (pure AFE-minimal):**

| file | what it is |
|---|---|
| `afe_core.py` | BLRSigma (cumulative 32×32 A_n, Sherman-Morrison), DStore (append-only verified-query archive with float32 embedding inputs), `verify_plan` (legacy-tolerance task box + SOCP + separate m, r), isolated diagnostic RNG, SafeMPPIFallback, and fixed ρ_eval audit helpers. |
| `grid_expand_afe.py` | Study-1 trainer: σ-tilted B-budget acquisition, verified-before-execution, certified fallback, uniform cumulative D⁺ replay + prox objective, untilted audit, per-round viz_db/probe.jsonl; `--probe` component checks; brothers via `--ablate-verifier` / `--ablate-fallback` (+ noprox via `--eta 1e18 --fstep-stop 999`). History: demo-replay arm and encoder freezing existed briefly and were **removed entirely** (user: no ad-hoc stabilizers). |
| `video_afe.py` | Study-1 per-round expansion video (verified plan fans, σ dots, executed paths + fallback steps, trained-on D⁺ rows by round-of-origin, validity curves; notation footer). |
| `analysis/afe_driver.sh` | launches the pure arms (λ10 × seeds 910/911/912 + λ0.01 reference). |
| `analysis/afe_assemble.sh` | end-of-run evals (report_at vs expert), validity report, videos. |
| `analysis/afe_lam_study.py` | measured λ choice: σ-spread under candidate λ from a saved A_n. |
| `analysis/afe_report.py` | multi-arm validity tracking figure + raw-up-frac collapse audit. |
| `paper_results/rollouts_v6.py` | paper rollout gallery for the new paradigm: Ours vs the 3 method-gate brothers (panels annotated with measured costs: −verifier 8.5% gather deaths, −fallback 4.9%, −prox covΣ 18 + audit erosion) + safety-calibrated Kazuki + expert + pretrained. |
| `paper_results/scatter_v6.py` | SR-CR and clearance-time phase planes, all methods (plasma_trunc = γ; viridis reserved for σ). |
| `paper_results/internals_v6.py` | training-internals figure: per-γ fallback + LOCATION split (shield moral hazard), D/D⁺ growth, prox solver, untilted-audit-vs-acceptance, σ decay + dither. |
| `paper_results/table_v6.md` | all methods, one row-file-consistent source, incl. gather-time deaths and V̂_adverse. |
| `paper_results/AFE_FINDINGS.md` | the full findings write-up (includes the 2026-07-16b coverage-number correction). |

**New — Study 2 (AFE2):**

| file | what it is |
|---|---|
| `grid_expand_afe2.py` | shared two-arm trainer: e97 acquisition/update values plus the §11 absorbing-goal correction, evolving φ_s⁽ⁿ⁾, per-round A rebuild, expert-free verify-or-terminate, full 7-γ sweep, diagnostics, and fixed-index controller evaluation. |
| `video_afe2.py` | the 7-γ-panel spec-color video (gray/orange/green/red/blue/X + per-panel text). |
| `analysis/afe2_report.py` | two-arm diagnostics figure (controller SR/NVP/CR, per-γ SR, per-γ raw validity, CFM + per-module grads, rep cosine drift + Δθ/θ, ESS/entropy/uplift with the calibration band, σ all-K vs selected + A effective rank, final per-γ Wilson-CI table). |
| `AFE2_HANDOFF.md` | the RESULT story with final numbers (prox frozen SR 0; afe SR 0→0.34→0.16 oscillation + audit erosion −7.5 pts adverse; both walls located; σ blind effR≈1.1), exact file:line pinpoints of every mechanism/knob, and the prioritized recipe matrix for the next arms — the document to give a continuing agent. |
| `afe2_scene_profiles.py` | explicit `claude_grid_v1` and `codex_radius1_v1` task adapters plus immutable scene snapshots/fingerprints. |
| `afe2_calibration.py` | shared fail-closed radius-1 beta-calibration contract used both before arm 1 and at pair promotion. |
| `run_afe2_radius1_pair.sh` | sequential calibration→prox→afe launcher that locks the declared non-beta Claude recipe, absorbing-goal contract, scene profile, and supplied checkpoint hash. |
| `analysis/validate_afe2_pair.py` | completion, seven-gamma/K/B semantic, report-decode, and ten-frame-video gate; emits a hash manifest only for a matched rounds-0--10 pair. |
| `README.md` | this document. |

**Modified (existing files touched by this work):**

| file | change |
|---|---|
| `grid_expand_afe.py`, `afe_core.py` | (listed above as new; noted here because they were edited across the session: sys.path precedence fix — local copies must win or `grid_metrics2` resolves to the older rev_expansion copy; adverse-velocity audit slice added after the rest-only audit measured at a 99% ceiling; demo/freeze purge; ablation flags.) |
| `paper_results/internals_v6.py` | coverage annotation corrected (M=40 covΣ = 34 vs base 52; an earlier draft mis-attributed the pretrained's pooled line to the π arm). |
| `analysis/afe_report.py` | line-style cycling fix for ≥5 arms. |
| `video_afe.py` | ffmpeg even-dimension scale filter; notation footer. |

**NOT modified** (imported as-is; other agents must not edit local same-named copies without
checking `module.__file__`): `grid_scene.py`, `grid_feats.py`, `grid_rollout.py`,
`grid_metrics.py`, `grid_metrics2.py` (codex_overnight copy), `grid_hp_expt.py`,
`grid_policy2.py`, `pretrain_repr.py`, `gen_uniform_data.py`, `verifier_polytope.py`
(resolves from `overnight_run_2026-07-01/`), `sr_cr_eval.py`, `eval_ae.py`,
`analysis/report_at.py`, `grid_expand_hardtail.py` (the superseded curriculum trainer — kept
untouched as the museum piece; `_apply_wall_plugs`/`_save_hp_atomic` are reused from it).

**Git waypoints**: `pre-afe-2026-07-16` (tag, pre-refactor code) → `6a5312b` (Study-1 trainer) →
`83c6033` (pure purge) → `5dc25b7` (findings) → `2333b71` (v6 viz) → `4a1e665` (brother evals +
coverage correction) → `14afca5` (README) → `13dad74` (AFE2). Branch `codex/safe-mppi-publish`,
mirrored to `master`, pushed to origin.

---

## 11. Codex radius 1 with an absorbing goal set

The Claude and Codex tasks now call the same `grid_expand_afe2.py`; there is no copied trainer to
drift. The Codex run preserves Claude's K/B, lambda, acquisition, and two update recipes, with
one declared shared correction: the unchanged radius-0.15 goal set is absorbing. A full-H rejected
plan can be executed only when its prefix through the first goal hit is certified; that plan keeps
its full-H negative label and never enters D+. Thus the safety claim is through the goal hitting
time, not after termination. Execution progress is truncated at that same first hit, while the
full-H progress remains separately logged for data analysis. Here B is the candidate-query budget; terminal rechecks make the
SOCP-solve count variable, so solver count/time are logged separately. `--scene-profile codex_radius1_v1` replaces exactly the four center disks by one
disk at `(2.5,2.5)` with physical radius `1.0`, retains the remaining obstacles/walls/plugs, and
sets `(0.5,0.5) -> (4.5,4.5)`. The pretrained checkpoint is supplied explicitly and hash-recorded.
Before either arm, the launcher performs one beta-neutral radius-1 round-0 ESS calibration over
`{0.01,0.02,0.05}` and hash-binds its selected beta to both arms; it does not assume Claude's beta
transfers across scenes. Archive embedding inputs remain float32, and named RNG streams isolate
gathering from update/audit/evaluation randomness. The inherited task-box check is also explicit:
it preserves Claude's legacy `[-0.12,5.12]` tolerance rather than silently claiming exact `[0,5]`
containment. Evaluation uses M=8 fixed-index rollouts per gamma and is labeled as a pilot;
the Wilson/bootstrap intervals are conditional on fixed contexts/episode indices and expose that
limited power rather than claiming across-seed uncertainty or a safety guarantee.

Run the two original arms sequentially, with no knob changes:

```bash
./run_afe2_radius1_pair.sh \
  /absolute/path/to/codex_pretrained_32d.pt \
  EXPECTED_CHECKPOINT_FILE_SHA256 \
  /absolute/path/to/output/afe2_radius1
```

The launcher uses `--lock-reference-recipe`, so changing K/B, lambda, update rates/steps,
horizon, evaluation count, or either arm's update rule is an error. It refuses existing arm
directories, requires the shared beta-calibration artifact, validates the pair before rendering,
and writes `DELIVERY_COMPLETE.json` only after hashing the calibration, report, and both videos.
The complete continuation
contract and completion gates are in `../codex_challenging/afe_restart/AFE2_RADIUS1_HANDOFF.md`.
