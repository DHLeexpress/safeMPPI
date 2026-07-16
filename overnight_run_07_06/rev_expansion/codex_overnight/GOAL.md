# GOAL — Codex overnight run (2026-07-10): paper-ready tables for Safe Flow Expansion

**Mission.** Produce neat, concise, paper-ready tables (+ progress log + viz) for three method rows —
(1) SafeMPPI expert ground truth, (2) our flow-EXPANDED policy, (3) the Kazuki guidance baseline — with
metrics a–e per γ, following the USER INSTRUCTIONS below exactly. Priority (1)>(2)>(3).

**Hard rules.**
- Work ONLY inside this folder (`overnight_run_07_06/rev_expansion/codex_overnight/`). NEVER edit files
  outside it. To change algorithm behavior, COPY the file here and edit your copy (path-shim pattern, §C2).
- No `git push`. No wandb. GPUs **2 and 3 only** (0/1 belong to others — verify with `nvidia-smi` first).
- Cap threads: `OMP_NUM_THREADS=16` (or `torch.set_num_threads`) per process — 7 uncapped procs once spawned
  318 threads each and stalled the 256-core box.
- Log every run + decision + intermediate table in `PROGRESS.md` (append-only, timestamped).
- When confused, SEARCH the existing code (paths in §C) — do not invent assumptions.

---

## STORYLINE (user's abstract draft — read this to understand what the tables must support)

> Existing safety filters typically evaluate individual actions without accounting for long-horizon task
> progress. Generative policies provide diverse control candidates, but their deployment in safety-critical
> settings requires mechanisms that jointly account for safety, task performance, and distribution shift.
> Prior approaches often guide pretrained policies using a scalar reward or cost that ties both safety and
> task performance objectives. However, guidance-based sampling is myopic when try to optimize both
> objectives, especially in out-of-the-distribution (OOD) of the pretrained policy.
> We introduce Safe Flow Expansion, which decouples the dual objectives through independent trajectory-level
> verification and expands the policy distribution using only rollouts verified for both safety and task
> progress. First, a SafeMPPI expert generates simulated trajectories across a safety–performance trade-off
> parameterized by a conditioning variable γ for rapid parallel simulation. We then train a conditional
> flow-matching policy on these trajectories. As a result, the policy learns spatial and behavior changes
> with the desired safety–performance preference. Next, flow expansion targets regions of high uncertainty
> and updates the policy using rollouts that independently satisfy safety and progress criteria. Rather than
> relying solely on local guidance, this procedure expands the policy distribution toward verified
> trajectories that achieves dual objectives. Safety is evaluated over finite-horizon control sequences
> using a convex polytope and its levelsets, which retains multi-step safety that satisfy the horizon-level
> constraints.

Table roles: (1) expert = in-distribution ground truth; (2) expanded policy deployed FROM THE ORIGIN
(OOD — no demo starts there) must match/beat it on safety+performance and WIN on coverage; (3) guidance
baseline = the myopic alternative.

---

## USER INSTRUCTIONS (VERBATIM — do not reinterpret; execute exactly)

Priority (1)>(2)>(3). Note: don’t add extra assumptions or modify existing codes. Search the existing code
back when you are confused, use full GPUS (2 and 3) overnight to achieve this task.

(1) (Easy) For every gamma list, our safeMPPI expert’s ground truth. Rollout at least 100 times.

a. SR maybe 100%

b. CR maybe 0%,

c. within-onlysuccessfulepisode-average-min-clearance-to-goal mean + std (For each obstacles, you have
minimum distance, and you average during the time of rollout, for successful cases; expected: low gamma has
maximum clearance),

d. average-time-completion mean + std (expected: high gamma is shortest or medium gamma is).

e. Coverage (there are total 2^4 discrete staircase choices stay within that one-cell diagonal corridor.
Expected: Our expanded generative policy wins)

This gives us a baseline metric to compare against after the safe expansion is successfully done, the
storyline is that we deploy from the OOD distribution (no demo starts from the origin) how can we get
expressive (coverage) and performant (time completion,) and safe (mean min distance). For these all gammas,
we will make a long table and decide which one to put on the report. Later, I will deploy the adaptive gamma
schedule!

(2) (HARD) Do the same thing for our flow expanded model. IMPORTANT: THE RULE = iterate some recipe for
expansion (1) N % quantile for frontier (2) beta (my suggestion is use constant beta of 0.2 or 0.3) and
mixing recipe (easy and frontier; my suggestion is using quantile) UNTIL YOU REACH THIS GOALs. THIS MAYBE
TIME CONSUMING, but I am pretty sure after few iterations you will achieve these. The key is turned out that
stay tuned on illconditioned valid rollouts by actually looking at the number (CR is zero but SR is not 100
then it’s a problem). For every gamma, sample sufficient amounts (this is actually also the control variable
because you may wanna achieve high coverage) REPORT how many iterations to achieve THIS GOALS at same time.
I will report intermediate iteration’s result so that we can appeal that okay after few iterations this
behavior is achieved therefore optimizing high uncertainty, safe, performant behavior can be generated via
safe expansion WITH VERIFIER (Valid2). **Try to test in few iterations unit~ 100 or 200, compare other
sweeps, and reason, proceed using a saved updated model for additional iters**. To this end, the recipe must
be a fixed schedule (not relative w.r.t. the total iteration)

a. SR 100%

b. CR 0%,

c. within-onlysuccessfulepisode-average-min-clearance-to-goal mean + std: SIMILAR TREND across gamma WITH
DEMO, but must be safer across gamma (BECAUSE VERIFIER POLYTOPE stresses overly conservative NOMINAL
POLYTOPE, so aggressive and safe generative policy can sometime be verified where demo cannot generate)

d. average-time-completion mean + std : SIMILAR TREND across gamma WITH DEMO, but must be faster across
gamma.

e. Coverage: Close to 16 must be lot then demo.

This goals should be satisfied for all gammas.

(3) (Medium) Claude already implemented Kazuki’s (only use original code of his:
/home/dohyun/projects/cfm_mppi/external_data/kazuki_cfm_mppi. Here you actually need to make assumptions to
use his ‘idea on generating + refining via mppi’. Currently the parameters are not well-tuned because his
original pedestrian dataset and our scene is different. But the storyline is that: guidance gives you
locally best solution but the guidance (gradient of reward w.r.t. u) leads you to out of the distribution,
and optimizing relying on single cost can be challenge. I think we mainly swept w_safe term to put more /
less weights on reward / safety, you can do that. Proper viz of failure / success needed to understand.
Place a guidance vector when you are thinking of generating video. Anyways, you should fine tune the
hyperparameters like weight and stuff, reason about those parameters, to achieve : Descent amount of a~e
(>=70%) for every gamma. This goals also should be satisfied for all gammas. Since this is a baseline method
to compare, you may want to make that looks reasonable but not like perfect lol

Also, as you can see from the /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/video/
m_base_curriculum.mp4 I think the video module you just generated is very impactful. This methods +
internals plot should be the central viz tool to him. Make a subfolder that he can work, and tell me our
latest model + recipe (m_base and m_combo) for actual safe expansion algorithms. He might not copy but use
our code to save the memory, save all the results and preliminary viz that is necessary to check the
progress (e.g., the latest curriculum video)

+ m_base: Issues with illconditioned trajectories stay near origin (high uncertainty~1 but still was a valid
sample -> after some iterations those ill-conditioned trajectory update the flow matching that doesn’t
increase SR but zero CR) so as a ad hoc solution you enforce the sigma threshold to be 0.25 and so it didn’t
happened before. (m_combo) However this is not a constructive method for the paper; use fixed N% quantile
data and schedule N with respect to iteration time looks appealing to the reviewers. For example, I think we
might first try quantile based (Let’s say 50:50 then draw the plane for each axes and select 12.5% as a
frontier). Let’s fix this recipe for the experiment.

---

## C. HOW THINGS WORK — code map & mechanics (added by Claude; everything below is verified against the repo)

### C0. Environment, conventions, run pattern

- Repo root: `/home/dohyun/projects/cfm_mppi/`. Working parent: `overnight_run_07_06/` (call it `$W`);
  our algorithm folder: `$W/rev_expansion/` (call it `$R`). You are in `$R/codex_overnight/`.
- Run pattern (background-safe):
  `cd $R && CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 \
   setsid nohup python <script> <flags> > <log> 2>&1 < /dev/null &`
- Scene: `$W/grid_scene.py` → `make_grid()`: 5×5 m world, 4×4 inner obstacle discs + boundary discs,
  start (0,0) bottom-left, goal (5,5), double-integrator dynamics (`di_grid_viz.di_step`), `env.dt`,
  episode cap `T=250` steps. `mode1_config()` = the SafeMPPI expert config dict.
- γ list (training set): `0.1 0.2 0.3 0.4 0.5 0.7 1.0`.
- **SR convention**: faithful rollouts (temp=1.0, NO σ-tilt), reach = **0.1 m** to goal, ≤250 steps,
  collision-free. CR = fraction with any collision (`sr_cr_eval.path_collides`).
- ⚠ **reach gotcha**: `gen_uniform_data.rollout_from(..., reach=0.4)` defaults to 0.4 (data-gen
  convention). For ground-truth tables pass `reach=0.1` explicitly so expert and policy use the SAME
  success criterion.
- ⚠ **β semantics INVERTED**: σ-tilt weight is `w = exp((σ−σmax)/β)` (`$W/grid_rollout.py:149`) —
  **LOW β = MORE novelty exploration; HIGH β = faithful**. The user's "constant beta 0.2 or 0.3" =
  a constant strong-explore gather.
- ⚠ single-measure SR flip-flops. Never conclude from one measure; average ≥3 measures or use M≥50.

### C1. Priority (1): expert ground truth — mechanism

- Rollout helper: `$R/gen_uniform_data.py::rollout_from(env, cfg, gamma, start_xy, seed, reach)` —
  builds a `SafeMPPIAdapter` (`cfm_mppi/safegpc_adapter/safemppi.py`; expert enforces the NOMINAL-polytope
  decay `h(x_{t+1}) ≥ (1−γ)h(x_t)`, see safemppi.py:771) and rolls receding-horizon to `reach`.
  Deterministic per (start, seed). Usage example: `$R/viz_uni_trajs.py`.
- Protocol (user-confirmed 2026-07-10): start (0.0, 0.0), seeds 0..99+ per γ, `reach=0.1`, all 7 γ.
  ~0.8–1.3 s/rollout ⇒ 100×7 ≈ 15 min with 7 parallel CPU procs (cap threads!).
- Metrics from the returned `states[:, :2]` path:
  - SR: `‖p[-1] − goal‖ < 0.1` (and steps ≤ 250); CR: `sr_cr_eval.path_collides(path, env)`.
  - c (clearance): obstacles `env.obstacles` (J,3 = x,y,radius), robot radius `env.r_robot`;
    clearance series `c_t = min_j (‖p_t − o_j‖ − r_j − r_robot)`. The user's parenthetical is ambiguous
    between (i) mean over t of min-over-j and (ii) per-obstacle min over t then mean over j —
    compute BOTH, report (i) as primary, (ii) in a footnote column. Successful episodes only.
  - d (time): `(len(path)−1) × env.dt`, successful only, mean ± std.
  - e (coverage): `$W/grid_metrics.py::staircase_id(path)` returns the R/U mode word (None if not a
    monotone staircase). Coverage = #distinct ids among successful episodes. User states 2^4 = 16
    distinct choices stay within the one-cell diagonal corridor — EMPIRICALLY enumerate the distinct
    ids you observe (expert + policy) and confirm/report the reachable-id count; do not hardcode 16.
- Output: `tables/T1_expert.md` + `.csv` (rows = γ; cols = SR, CR, clearance μ±σ, time μ±σ, coverage,
  n_success), raw paths per γ in `results/expert_gt/paths_g{γ}.npz` (reuse the `viz_uni_trajs.py`
  npz pattern).

### C2. Priority (2): flow-expanded model — mechanism

**Model stack.**
- Policy loader: `$W/grid_hp_expt.py::load_hp(ckpt, device)` → `GridHPFlowPolicy` (flow-matching,
  repr32, γ-conditioned via low5/context).
- Pretrained backbone (uniform-grid data, origin in-distribution):
  `$W/results/hp_repr/pretrained_a32uni.pt` — from-origin SR 0.36 / CR 0.03 (7γ avg, M=25).
- Best expanded checkpoint so far: `$R/results/uni_expand/uni_A_b64i121/best.pt` (SR 0.93 / CR 0.00,
  7γ avg, M=25) — useful as a reference/warm start comparison, NOT the required recipe.

**Trainer** (the ONE script that does gather→label→update): `$R/grid_expand_cur_rev.py`.
- Per iter: σ-tilted gather (`GR.fm_deploy` with tilt dict) → validity gate per window
  (**valid2** = taskspace ∧ SOCP trajectory certificate `GM.socp_ok` (verifier polytope,
  `$W/../overnight_run_2026-07-01/verifier_polytope.py::certify_trajectory`) ∧ net-progress ≥
  min(0.10, 0.5·d0) ∧ `--valid-prog-floor` 0.15) → 2-class labeling (`label_fresh`/`_front_mask`) →
  dynamic batch (demo δ-anchor + LwF η) → CFM update.
- Known recipes (both start from `pretrained_a32uni.pt`; exact flag lines):
  - **m_base** (LOCKED recipe; quantile-OR frontier):
    `--ckpt ../results/hp_repr/pretrained_a32uni.pt --batch 64 --early-inner 1 --inner-steps 2
     --cooldown-inner 1 --lr 1e-4 --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05
     --easy-strict --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0`
    (β step 1.0/0.5/0.2/0.1; frontier = σ≥q.67 ∨ margin≤q.33 ∨ prog≥0.3). Fails as the user
    describes: ill-conditioned near-origin high-σ windows become "easy" → CR 0 but SR ≪ 1, jiggle.
  - **m_combo** (stability winner, but ad-hoc): m_base + `--beta-steps 0.3 0.5 0.7 1.0 --strat-rid
    --easy-sig-abs 0.25 --easy-demo-backfill --easy-skip-first 2`. Caveat: the σ<0.25 absolute gate
    leaves fresh-easy ≈ 0 all run (works, but "not a curriculum") — see
    `preliminary/m_combo_curriculum.mp4`.
- **THE RECIPE TO IMPLEMENT (user-fixed 2026-07-10, replaces both):** copy `grid_expand_cur_rev.py`
  (and `grid_metrics2.py` if needed) into THIS folder; path-shim at top:
  `sys.path.insert(0, '$R'); sys.path.insert(0, '$W')` (then your local copy wins). Edit ONLY your copy:
  1. Frontier = **AND-cell fixed-quantile rule**: quantile plane on each of the three axes
     (σ high / SOCP-margin low / net-progress high); frontier = windows beyond the plane on ALL THREE
     axes (50% planes ⇒ ≈12.5% frontier). The AND-quantile level N% is a SWEEP VARIABLE (together
     with the mixing ratio); schedule N by **absolute iteration index** (fixed table
     {it<200: N₁, it<400: N₂, …}), NOT by fraction of total iters. Also convert the phase logic
     (`early_frac`/`cooldown_frac` are relative) to fixed absolute-iteration thresholds.
  2. **NO demo backfill** (do NOT use `--easy-demo-backfill`; remove the fallback in your copy): if a
     class pool (easy or frontier) is empty at gather time, **sample MORE rollouts** until both
     classes are populated (raise the per-iter rollout budget / keep gathering to a sane attempt cap),
     never fill from demo.
  3. **Fresh sample ratio should be HIGHER**: lower `--demo-frac` from 0.25 (e.g. 0.25 → 0.125 → 0.0)
     so the batch is fresh-dominated — this is also a CONTROL VARIABLE to sweep.
  4. β = constant: start **0.3**, switch to **0.2 only if coverage stalls** (`--beta-steps b b b b`
     gives a constant — no code edit needed). One controlled comparison, logged.
  5. Keep the validity gate (valid2 + SOCP + valid_prog_floor 0.15) EXACTLY as-is.
- **The iterate-until-goals loop** (the user's RULE): run 100–200 iters → checkpoint (`ckpt_100.pt`…,
  auto every `--ckpt-every`; `--ckpt <saved>` resumes from any checkpoint) → evaluate a–e for ALL γ
  (§C4 eval spec) → if goals unmet, adjust ONLY {AND-quantile N%-schedule, mixing ratio, demo_frac
  (fresh ratio), rollout budget, β∈{0.3→0.2}, per-γ sample amount M} → resume from the saved model →
  repeat. RECORD in `PROGRESS.md`: recipe JSON, iteration count so far, full a–e table per γ at every
  checkpoint. Final deliverable includes **iterations-to-goal**.
- **Ill-conditioned-rollout watch** (the failure the user warns about): per-iter `probe.jsonl`
  (enable `--log-comp-every 1`) fields `near0_e` (frac of easy windows starting <1 m from origin),
  `sig_e` (their mean σ), `n_easy/n_frontier`, `batch_e/f/d`; per-iter M=50 instantaneous
  SR50/CR50/coverage probe: `--probe-cov 1`. If CR≈0 but SR flat and `near0_e`→1 with high `sig_e`,
  you are reinforcing origin-dither — change the N-schedule, don't loosen the validity gate.
- **Central viz tools** (run for EVERY serious training run; source runs need
  `--viz-db-every 1 --log-comp-every 1`):
  - Curriculum VIDEO: `python $R/video_curriculum.py --run <outdir> --out video/<name>.mp4
    [--sig-abs 0] [--vpf 0.15] --title "<recipe>"` → panels A(rollouts+σ-windows+easy rings),
    B(σ hist), C(3D σ-on-Z + validity/frontier planes), D(5 bins + used-arrows), traces β/counts/mix/lr.
  - Internals overlay: copy `$R/expand_report_rev.py` here, edit its WHITELIST to your arms.
  - Per-iter probe traces: parse `probe.jsonl` (see `$R/figures/micro100_internals*.png` for the look).

### C3. Priority (3): Kazuki guidance baseline — mechanism

- Faithful port (use as starting point): `$R/kazuki_baseline.py` — `guided_generate` (flow sampling
  with reward-gradient guidance v←v+λ∇R on the x1 endpoint; CBF r_safe K=5-worst + goal terms),
  `flow_mppi_refine` (top-10 elite by exp-proximity MPPI cost, β_mppi=20), `kazuki_deploy`
  (τ=0.75 warm-start). Original code: `/home/dohyun/projects/cfm_mppi/external_data/kazuki_cfm_mppi`
  (paper: UnifiedGenRefine arXiv-2508). Tunables exposed: `--coll-w --goal-w`, w_safe sweep, `--viz-out`
  records candidate/elite/selected trajectories for the failure/success viz (add the guidance-vector
  arrows in YOUR copy when making videos).
- Known state (2026-07-09, on a32uni, H=10 windows): SR 0.00 across γ{.1,.5,1.0}×w_safe{.1,.5,.9} —
  freeze, not collision: the `100·exp(−20(d−r))` proximity wall dominates the `0.1·goal` term at our
  short horizon; robot stalls ~0.45 m out. Your job = retune weights (their framework assumed 80-step
  windows) until a–e reach ≥70% of the expert values per γ; document the reasoning per parameter.

### C4. The a–e evaluation spec (same function for priorities 1/2/3)

Write ONE `eval_ae.py` here and reuse it:
- For a POLICY: `rows, agg, paths = sr_cr_eval.eval_policy(pol, env, gammas=[g], M=100, T_max=250,
  reach=0.1, temp=1.0, device=dev, keep_paths=100, log=silent)` → SR/CR from `rows[g]`; compute
  c/d/e from the returned `paths[g]` exactly as §C1 (success re-checked as ‖p[-1]−goal‖<0.1).
- For the EXPERT: same metric code over `rollout_from` paths.
- M per γ: ≥100 (user: "sample sufficient amounts" — M is a control variable for coverage; if
  coverage still grows at M=100, raise M and note it).
- Output one long combined table `tables/T_ALL.md` (+csv): rows = method × γ; cols = SR, CR,
  clearance μ±σ, time μ±σ, coverage, M, (for P2) iterations-to-goal.

### C5. Known pitfalls (each cost us hours — read once)

- `nohup ... &` inside wrapper shells detaches; verify with `pgrep -cf <tag>` after launch.
- `pkill -f <pattern>` can match your own shell → exit 144; use precise patterns, verify separately.
- Thread oversubscription (see Hard rules).
- Collapse-guard: trainer auto-terminates if SR < 0.45×best for 3 measures after it600 — for short
  100–200-iter units this never fires; don't disable it.
- `viz_db` snapshots gate on `n_valid ≥ 8`; probe.jsonl is append-only (delete before re-running into
  the same outdir).
- GPU2 vs GPU3: both free as of 2026-07-10 04:00; the tiny grid policy runs fine several-per-GPU —
  parallelize your 100–200-iter sweeps.

### C6. Deliverables checklist

- [ ] `tables/T1_expert.{md,csv}` — P1, all 7 γ, M≥100.
- [ ] `tables/T2_expanded.{md,csv}` + iterations-to-goal per γ + recipe JSON of the final schedule.
- [ ] `tables/T3_kazuki.{md,csv}` + tuned-parameter reasoning.
- [ ] `tables/T_ALL.md` — the combined long table (the paper artifact).
- [ ] `video/` — curriculum video of the FINAL P2 recipe run (+ any diagnostic videos).
- [ ] `figures/` — internals overlay + per-iter probe traces of the final run; Kazuki success/failure viz.
- [ ] `PROGRESS.md` — timestamped log: every run launched (cmd line verbatim), every intermediate a–e
      table, every recipe change and WHY.
- Reference viz of current state: `preliminary/` (m_base/m_combo/P5 curriculum videos, micro100
  internals, early-iters easy-vs-σ evidence, 566-traj dataset figure).
