# goal_07_02_Hp — H_P inductive-bias reduced model: capacity → then sweep (2026-07-05)

The 0704 plan (beat Kazuki) is TERMINATED (9 tasks: 7 done; Stage-E head-to-head closed at "compute 15× + succ
+6 n.s., collision 9.7 vs 4.0 open"; sanity-3 folded into validity2 work). Focus = HP on the 0702 chessboard.
Figures for testing → `figures/hp_test/`.
Current 20k iter basline -> 

## CURRENT STATUS — necessary & sufficient config snapshot

**Model (`grid_hp_expt.GridHPFlowPolicy`, ckpt `results/hp_chessboard/pretrained_hp.pt`):**
- ctx = raw low5(5)=[relgoal2, vel2, γ] ⊕ **E_hp**: H_P channel [1,16,12] → Conv(1→8,3×3) SiLU → Conv(8→16,3×3)
  SiLU → AdaptiveAvgPool(4,3) → Flatten → Linear(192→32) SiLU. NO E_l / hist / GRU. 101.4k params (E_hp 7.4k).
- velocity trunk: **[U(20) + ctx(37) + fourier-t(32)] = 89 → 256 → 256 (SiLU, depth 2) → head 20**; u_max=1.0.
- pretrain: 52,201 windows (450 trajs, 150/γ, SafeMPPI mode-1 expert seeds 0-149), AdamW+warmup+cosine 120 ep,
  val-cfm 0.977. validity2@iter0 (n=25): **γ0.5 92% · γ1.0 56% · γ0.1 0%**.

**Expansion defaults (`grid_expand2.SFG2Config`, 0702):** temp=**1.3** (exploration sampling — wide ON PURPOSE to
generate data; measurement uses temp_measure=**1.0**) · β=0.1 · s=0.9 · GP-RBF ell=0.2 λ=1e-2 gp_buf=384 · N=64 ·
lr=2e-4 Adam+cosine · α=0 (positive-only) · inner_steps=12 · batch=128 · enc_lr_mult=1.0 · T=250 · n_measure=**25**
eps/γ · caps pos 60k (FIFO) · γ **log-column order (0.5, 1.0, 0.1)** · gate = validity2 (approach ∧ taskspace ∧ SOCP)
· coverage = 252 staircases (cum + final).

**20k full-run verdict (Part 5 of GOAL_07_02.md):** coverage 100/252 (39.8%, decelerating); no collapse in 20k
iters; γ-selective consolidation (γ1.0 64-84 / γ0.5 ~44 / γ0.1 starved 0-12 — the rich-get-richer certification
loop); drift 0.757 flat from it3000; demoCFM →4.0.

## KEY OBSERVATION driving this phase (user, from `figures/expand2_multimodal_g0.5.gif`)
At the FIRST obstacle encounter the pretrained policy is **unimodal → unsafe**. Capacity problem: the vector
field must produce complex/multimodal landscapes BEFORE σ-dispersion or β-tilt can mean anything. Track the FM
**policy distribution itself** (not only σ): plot first-control u₀ distributions at obstacle-encounter states;
measure left/right peak splitting.

## PLAN

**Step 0 — model capacity (BEFORE any sweep):**
- 0.1 MORE DATA: `gen_more_data.py` (RUNNING) seeds 150-599 → target ~1800 trajs / ~200k windows total
  (was 450/52k). Backup of old shards: `dataset/backup_450traj/`.
- 0.2 ARCH SWEEP on the enlarged data: trunk ∈ {depth-2 256 (current), depth-3 256, **ResNet-MLP** 2 and 3
  pre-LN residual blocks (the proven 0704 cooked-trunk recipe), width 384 variant}. Keep 89-in → 20-out family.
- 0.3 AUTO-SELECT: train each (~8 min), rank by val-cfm → for the top models measure validity2@iter0 (n=25/γ,
  the standard) + the **multimodality score**: at obstacle-encounter states sample 256 windows, KDE/split of u₀
  (and next-position) left vs right — a model must place peaks BOTH sides. Best model = new pretrained.
- Deliverables: train/val curves, validity2 table, u₀-distribution snapshots (`figures/hp_test/`), overlay viz of
  generative policies.

**Step 1 — the 5-arm sweep** (2000 iters/arm, one knob, all else defaults, compare blocks 0-2000), FROM the new
pretrained: enc_lr_mult=0 (freeze E_hp) · lr 1e-5 · inner_steps 4 · α 0.005 · **β 0.2 (KEPT — user: flat σ may be
a symptom of unimodal candidates; re-judge β only after capacity is fixed)**.

**Step 2 —** per-γ fixes for the certification loop (γ0.1 starvation): per-γ harvest quotas / temp↓ for tight γ /
balanced batches. Then long run with the winning recipe.

## Q&A (user 2026-07-05)
1. **enc_lr_mult=0 ≡ freeze E_hp — the code** (grid_expand2.run_expand2): parameters are split into two optimizer
   groups — `groups = [{field(trunk+head), lr}, {enc(E_hp), lr × enc_lr_mult}]`. Learning rate 0 ⇒ Adam's step
   size for those weights is 0 ⇒ they never change = frozen. **Random-freeze p=0.95** ≈ in expectation a 0.05×
   slowdown, but with Adam it becomes sporadic FULL-size steps (jumpier than lr×0.05). The literature versions of
   "slow encoder": layer-wise lr decay (BERT fine-tuning), surgical fine-tuning (Lee et al. '22), and — closest —
   **momentum/target encoders** (BYOL, RL target networks): enc follows an EMA of itself. Our enc_lr_mult∈{0, 0.05,
   0.1} IS the standard knob; EMA-enc is the upgrade if partial freezing wins. OOD corners (e.g. bottom-right)
   argue for small-but-nonzero mult or EMA rather than hard freeze — the sweep tests the extreme first.

3. α>0: in the sweep. 4. β: kept, deferred judgment until multimodality is fixed (your causal point accepted:
   candidate diversity is upstream of σ dispersion).

## TREE VIZ MODULE (user spec 2026-07-05, `hp_tree_viz.py`) — the standing per-arm report
For an arm at Nk iters → **N+1 rows** (row 0 = pretrained "just sampling"; row k = ckpt_k000 with the arm's own
temp/β): each row is one TREE — a σ-tilt trunk rollout where at every **1 s node (= one H=10 window)** we sample
N=64 candidates at the arm's temp (default 1.3), importance-resample **k** branches via p∝exp(σ/β) with the
decaying schedule **k = 5,4,4,3,3,2,2,1,1,1,…**, roll the k branches 1 s in PARALLEL, colored by σ (viridis);
branches failing validity are marked RED with the failure class (**✗G** goal-seeking / **✗T** taskspace /
**✗S** SOCP) and terminated at the failure point; the trunk continues via the tilt winner until goal or T=250.
σ buffer for the viz = φ_s of 512 random dataset windows under that row's policy (documented stand-in for the
run's query buffer). β-sweep arms are read by how the spray CHOICE shifts as iterations grow. REPORT FORMAT:
this tree (per arm) + the `figures/hp_full20k_trend.png`-style 4-panel trend, consistently.

## WORKFLOW (standing, per sweep round)
1. RUN: 5 arms × 2000 iters from `res2w256_ft` (one knob each; ckpt/500, measure/200, n=25/γ).
2. PER ARM on completion: (a) recursive TREE, rows = ft/ckpt_1000/ckpt_2000 at the arm's temp/β (metrics:
   branches, died-by-✗G/✗T/✗S, reached, corridors); (b) 4-panel trend (val2 per γ + jiggle amplitude, coverage,
   SOCP-viol, drift+demoCFM); (c) row in the cross-arm table.
3. SELECT: winner = holds val2 γ-mean ≥ it0 baseline (≈76%) with jiggle ↓ and acceptable coverage slope;
   tie-break on γ0.1 (the starved end).
4. NEXT ROUND: winner → long run + per-γ fixes (harvest quotas / per-γ temp) targeting coverage↑ with validity
   HELD (the rich-get-richer loop). Report each stage: tree + trend + updated goal md.

## ell CALIBRATION (2026-07-05, `hp_ell_calib.py`, `figures/hp_test/ell_calibration.png`)
Default ell=0.2 gives σ≡1 (σ-std 0.006 — the uniform-yellow tree): candidate features sit ~0.46-0.61 from the
buffer ≈ 2-3× the lengthscale → kernel ≈ 0 → prior everywhere. Sweep: **ell\* = 0.5** (σ-std 0.0514, ~9×;
median-heuristic 0.57 agrees; 0.403 tied). **ALL hp4 sweep arms run with --ell 0.5** (documented deviation from
the 0702 default, calibration-justified). Tree-viz conventions locked: recursive no-orphan branching (children
throttled, k_eff≥1 — every branch resolves to goal/fail), depth-based opacity (first branches opaque), black
dots = true branch events (k≥2), failure classes color-coded (orange G: goal-seeking · purple T: taskspace ·
red S: SOCP), ONE gold goal star, green dots = reached leaves, σ-colorbar (ell*).

## SWEEP ROUND 2 (hp5, 2026-07-05 evening) — NEW LOCKED CENTER (user)
**Defaults changed:** temp explore 1.3→**2.0** (measure stays 1.0) · enc_lr_mult 1.0→**0.5** · **ell re-calibrated
at the new center: ELL\* = 0.5 again** (σ-std 0.049 at temp 2.0; median-heuristic 0.62; 0.2 → σ≡1 dead).
Unchanged: β 0.1 · s 0.9 · N 64 · lr 2e-4 cosine · α 0 · inner 12×128 · n_measure 25/γ · traj-level validity2
gate · 252 coverage. it0 anchor (measure temp 1.0, unchanged): validity2 64/72/92 (mean 76).
Center-tree sanity (`figures/hp_test/tree_sanity_center_t2.png`): temp 2.0 → 731 branches, 99 died, 38 reached
(vs 923/55/83 at 1.3) — much wider exploration, riskier proposals, harvest slower per trajectory.
**ARM SET (one override each vs the new base):** 1) encm0.1 (enc_lr_mult 0.5→0.1) · 2) lr1e-5 · 3) inner4 ·
4) **alpha0.1** (α 0→0.1, raised from 0.005) · 5) beta0.2. Judging unchanged (hold ≥76 γ-mean, jiggle↓, cov
slope, SOCP, γ0.1 tie-break, tree metrics ft→1k→2k).

## SWEEP ROUND 3 (hp6, 2026-07-05) — separated effects at the FINAL LOCKED BASE
Base (LOCKED, final): temp **1.5** explore / 1.0 measure · ell 0.5 · enc_lr_mult 0.5 · β 0.1 · s 0.9 · N 64 ·
lr 2e-4 Adam+cosine · α 0 · inner 12×128 · measure/100 · n=25/γ. (β0.2 quota experiments showed temp 2.0 too
extreme → 1.5; γ log-column order still (0.5, 1.0, 0.1).)

**Results (1000 iters, one knob each; `results/hp_sweep6/`; tree = reached-goal leaves at γ0.5/temp1.5,
ft baseline 77/963 branches):**
| arm | val2 γ-mean it0→1000 | per-γ at 1000 | cov_cum | drift | demoCFM | tree reached ft→500→1000 | verdict |
|---|---|---|---|---|---|---|---|
| beta0.01 (greedy tilt) | 68 → **36%** | 56/32/20 | 7.4% | 0.280 | 0.79→1.20 | 77 → 1 → **0** | collapse, discovery also slow |
| enc0 (freeze E_hp) | 79 → **35%** | 48/44/12 | 12.8% | **0.000 ≡** | 0.79→1.20 | 77 → 11 → **5** | collapse WITH frozen encoder |
| lr1e-5 (rerun) | 81 → **21%** | 28/36/**0** | 13.5% | 0.140 | 0.79→1.01 | (tree rendering) | WORST arm — tiny steps, same bias; γ0.1 dead |

**The separated-effects verdict — where forgetting lives and what each knob can/cannot do:**
- **enc0 is the theorem**: drift ≡ 0.000 for 1000 iters (encoder bit-frozen) yet validity collapsed 79→35 on the
  SAME trajectory as every other arm ⇒ **forgetting lives in the trunk/head FIELD, not the encoder**. enc_lr_mult
  protects ctx geometry (σ/kernel sanity), it is NOT a retention lever. Optimize: keep 0.5; remote tests 0.1;
  EMA-encoder only if ctx drift ever bites.
- **β is a selection knob, not a retention knob**: β0.01 (greedy-σ) collapsed identically AND discovered less
  (7.4% vs 12.8): extreme tilt picks weirder candidates that fail the gate → fewer, more-biased positives.
  Optimize: β 0.1 stays; brackets 0.05/0.3 on remote judge discovery-rate-vs-validity-slope only after a hold
  mechanism exists.
- **lr is a speed knob**: prior evidence (quota-C at lr1e-5 still collapsed) says small steps walk the SAME biased
  direction, just slower. The rerun completes the table; expectation: slower slope, same sign.
- **⇒ The collapse is a DATA-COMPOSITION bias**: positive-only batches are drawn ever further from the pretraining
  distribution; no per-parameter step-size/locus knob can fix a gradient that points away from the demo manifold.
  **The real levers change the gradient DIRECTION: 2.1 demo_frac (mix old-input batches) and 2.2 LwF (anchor the
  field on old inputs)** — exactly the two-machine phase below.

## TWO-MACHINE DISTRIBUTED PHASE (2026-07-05, clean restart — tasks #51-54)
**Split (user): LOCAL = main part / aggressive search · REMOTE = fine-tuning brackets.**
- **LOCAL (GPU 0/3)**: **WAVE-1 FINALS (2k it, done 20:44)** — the mechanisms WORK where every plain knob failed:
  | arm | val2 γ-mean it0→2000 | per-γ @2k | cov_cum | drift | demoCFM | shape |
  |---|---|---|---|---|---|---|
  | **dfrac0.25** | 71 → **67%** | 68/**88**/44 | **27.4%** | 0.264 | **0.835** | dip ~55 mid → RECOVERS to 60-69 |
  | lwf0.1 | 79 → **52%** | 68/56/32 | 27.0% | 0.294 | 0.895 | holds mid-50s, slow decay |
  (plain arms 21-36% & cov 7-13% at half the iterations; both mechanism arms saturate the 60k FIFO pos-buffer —
  raise cap for long runs. dfrac0.25 = single-arm winner: better hold, better demoCFM, γ0.1 alive at 44.)
  **WAVE 2 (running since 21:00)**: dfrac0.25+lwf0.1 combined (2k, GPU0) · dfrac0.25 LONG 5k (GPU3, winner
  long-run). NEXT: quota-D discovery harvest (frozen-core Hamming + 5%-floor ban) on the winner.
- **REMOTE (ssh dohyunlee@dhcp-101-145.caltech.edu, driven by Claude; via `HP_RUNBOOK.md`)**: **plain-knob
  brackets CUT (user 2026-07-05: "the only hope is the dfrac and lwf")** → remote runs mechanism brackets only:
  dfrac{0.1,0.5}, lwf{0.01,1.0} singles, then tier-2 combos dfrac0.25+lwf1.0 & dfrac0.5+lwf0.1.
  Deliverables per run: 3-row tree + 4-panel trend + last 3 log lines. (LwF = Learning without Forgetting,
  Li & Hoiem 2017 — frozen-teacher distillation of the pretrained field on demo contexts.)
- **Judging (both machines, §5 of runbook)**: hold val2 γ-mean ≥~76 while cov_cum climbs ≥ ~13%/1k-it; jiggle↓;
  γ0.1 tie-break; tree branches/died/reached at it2000.
- **ROADMAP after the winner**: transfer the hold-while-explore recipe to SFM (`grid_expand_sfm.py` gains
  demo_frac/lwf_eta) → re-run expansion → `stage_e_benchmark` ours-side ONLY (Kazuki numbers cached) → close
  the 9.7% vs 4.0% collision gap = "beat Kazuki as the bonus of full exploration".
