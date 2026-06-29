# Mizuta CFM-MPPI — source-level analysis & our benchmark plan

Read from the ORIGINAL author code merged into our repo (Kazuki Mizuta,
github.com/m-kazuki/cfm_mppi): `cfm_mppi/example/doubleintegrator.ipynb`,
`cfm_mppi/mppi/utils.py`, `cfm_mppi/data/canonical_dataset.py`, plus direct
inspection of `dataset/{train80,eval80}*`. Paper: arXiv:2508.01192
(Unified Generation-Refinement Planning), cfm-mppi.github.io.

## 0. Mizuta DOES have a double-integrator robot → direct apples-to-apples benchmark
- `example/doubleintegrator.ipynb`: `doubleintegrator_dynamics`, `dynamics_type="doubleintegrator"`,
  `DI_KP=3`, `U_MIN/MAX=±2`, `N_SAMPLES=200`, `HORIZON=80`, `DT=0.1`, scenario idx 110.
- `mppi/utils.py:20` `doubleintegrator_dynamics(state=[x,y,vx,vy], action=[ax,ay])`.
- => We can benchmark OURS (DI + polytope DTCBF rejection) vs Mizuta CFM-MPPI (DI) on the
  identical `eval80` scenarios. This is the headline comparison.

## 1. Datasets and how raw trajectories become states/controls
Verified by loading the tensors:
- **Training** `dataset/train80_ego.pt` = **[273,989, 9, 80]** float32 — 273,989 ETH pedestrian
  trajectories (≈ paper's 276,874), 9 feature channels, 80 steps (×dt=0.1 = **8 s horizon**;
  "80" = horizon length).
- **Eval** `dataset/eval80_ego_{ucy,sdd}.pt` = **[300, 6, 80]** — 300 test scenarios each
  (matches paper's "300 test scenarios").
- **Eval obstacles** `dataset/eval80_obs_{ucy,sdd}.pkl` = python list of 300, each
  **[1, 17, 6, 80]** = up to **17 surrounding agents**, 6 channels, 80 steps, padded with NaN
  (scenario 0 had 10 real agents of 17).

Channel layout (verified): channels **0:2 = position (x,y)**, **2:4 = velocity (vx,vy)**.
The velocity IS the single-integrator control — verified numerically: `pos[t+1]-pos[t] == vx·dt`
exactly (0.09492 == 0.94923·0.1). The remaining ego channels (4:6 eval, 4:9 train) are auxiliary
features (heading/speed/goal-relative), used by the transformer condition.

How raw → canonical (documented in `canonical_dataset.py:94 build_canonical_from_mizuta`):
`pos = raw[:,0:2]`, `controls = raw[:,2:4]`, state `[x,y,vx,vy]` with velocity by finite
difference (`_safe_velocity_from_positions`), **start = pos[:,0]**, **goal = pos[:,-1]** (the
pedestrian's own endpoint is the goal). Notebook centers the ego at the origin (`start_pos=0`)
and scales controls by `SCALE=10`.

## 1b. The training set is ONE form, not two (answer to your question)
There is a **single** training tensor `train80_ego.pt`, loaded once (`train.py:116`
`LightDataset("dataset/train80_ego.pt")`), of single-integrator (pos+vel) pedestrian
trajectories. The CFM is trained ONCE on velocity sequences. The robot-dynamics mapping happens
only at **control-synthesis time** in `flowmppi.py:93 _convert_si_to_dynamics`, two ways from the
SAME CFM output:
- **Double integrator** (`flowmppi.py:113`): feedforward + proportional velocity feedback —
  `a = a_ff + a_fb`, `a_ff = (v_des[t+1] − v_des[t])/dt`, `a_fb = k_p·(v_des − v_actual)`,
  with `k_p = DI_KP = 3`. (So the CFM's velocity field is *tracked* by acceleration.)
- **Unicycle** (`flowmppi.py:105`): `v = vx·cosθ + vy·sinθ`,
  `ω = (1/d)(−vx·sinθ + vy·cosθ)`, control-point offset `d = 0.1`.
So you do NOT need two training datasets — one SI dataset serves both robots. (This is also why
their generative sampler is dynamics-agnostic: it predicts pedestrian-like velocity fields.)
Note: raw `eval80_obs_*.pkl` agents are **6-channel** `[px,py,vx,vy,·,·]` (our measurement); the
code consumes channels 0:2 (pos) and 2:4 (vel). Our `canonical_dataset` re-expresses ego as
`[N,T+1,4]=[x,y,vx,vy]` with `controls_si==controls_dyn` (identical → SI).

## 2. Their safety is SOFT (both layers) — this is our core differentiator
Mizuta has TWO safety layers and **both are soft**:
1. **MPPI soft cost** (`mppi/utils.py:69 stage_cost`): `collision = clamp(exp(−α(d−r)),1).sum`,
   weight `100·(1+0.99^t)`, α=20; goal weight only 0.1.
2. **Reward-guided CFM** (`reward.py single_cbf_reward_fn_pairwise`): the flow is nudged by the
   GRADIENT of a CBF-style barrier over the **top-k worst** obstacle constraints — a guidance
   term, not a feasibility filter.
Neither rejects or enforces invariance ⇒ no hard guarantee.
- **OURS**: hard convex-polytope DTCBF **rejection** of samples leaving the `(1−γ)^i` ruler ⇒
  for the DI (LTI) case the MPPI-averaged control stays in the convex polytope by convexity
  (safe, no extra assumption). That hard guarantee is exactly what Mizuta lacks — and the γ knob
  is a tunable *class of trajectories*, where they only have a fixed soft weight.

## 3. Observation assumption + the closest-obstacle ablation (your tweak)
Mizuta assumes the robot has **exact positions and velocities of ALL surrounding agents** (up to
17), and predicts their futures with a **constant-velocity** model
(`pos_obs_seq = pos_obs + cumsum(vel·dt)`, notebook `synthesize_control`). The CFM conditions on
all agents and `stage_cost` sums collision over all obstacles.
- **Proposed ablation**: restrict BOTH the CFM condition and the cost to the **single closest
  obstacle** (partial observability), re-evaluate Mizuta and ours, and compare. Our polytope
  already uses only near obstacles' tangents, so it should degrade gracefully; their full-scene
  CFM should degrade more. (Exact code site for the restriction: the obstacle tensor assembled in
  `synthesize_control` / `run_CFM` — to be pinpointed from `eval_utils.run_CFM` + transformer.)

## 4. OOD is built into their split — we lean on it
`train80` = ETH; `eval80_{ucy,sdd}` = UCY/SDD ⇒ **train-on-ETH, test-on-UCY/SDD is OOD by
construction** (their "distributional robustness" test; SDD also has faster agents/bicyclists,
and they add Gaussian obstacle-position noise σ=0.05 in the SFM sims). Our claim: the geometric
DTCBF ruler is **distribution-independent**, so safety holds under the shift where a learned
proposal degrades. Measure collision/success on UCY & SDD with the ETH-trained proposal.

## 5. Compute time
`eval_cfm_mppi_doubleintegrator.py`: `total_time += time_end−time_start` around
`synthesize_control` each step; reports `average_time = total_time/horizon` (per-planning-step),
`N_SAMPLES=200`, `HORIZON=80`. They compare within their own variants. We add **geometric
overhead** (per-step polytope build + sample rejection); we must measure and report our per-step
latency against theirs (CFM is <0.1 s/gen per the paper).

## 6. 2D only — no 3D
All states/controls are 2D (`mppi/utils.py`): SI `[x,y]`, DI `[x,y,vx,vy]`, unicycle `[x,y,θ]`;
controls 2D. Nothing 3D anywhere.

## Benchmark plan (ours vs Mizuta, double integrator)
1. Run **Mizuta CFM-MPPI (DI)** on `eval80_{ucy,sdd}` (300 each): collision rate, min-distance,
   distance-to-goal/success, per-step time. (Their own checkpoint `output_dir/cfm_transformer`.)
2. Run **OURS (DI + polytope DTCBF rejection + reward-tilted δU proposal)** on the SAME scenarios.
3. Headlines: (a) **success + collision** on UCY/SDD; (b) **OOD** ETH→UCY/SDD; (c) **compute
   overhead** of the geometric ruler; (d) **closest-obstacle ablation** (partial observability).
4. Safety claim scoped to DI/LTI only (no overclaim to unicycle/nonlinear).

## Resolved by the deep code trace
- **One training set**: `train.py:116` loads only `train80_ego.pt`; `controls_si==controls_dyn`.
- **SI→DI / SI→unicycle**: `flowmppi.py:93 _convert_si_to_dynamics` (formulas in §1b).
- **Metrics** (`utils.py:evaluate`): `collision = any(dist<r)` over all agents/time;
  `distance_to_goal = ‖p_T − goal‖`. Eval reports collision-rate %, mean distance, mean time
  (`eval_cfm_mppi_doubleintegrator.py`). 300 scenarios per dataset.
- **Closest-obstacle ablation site**: after obstacles are loaded/NaN-filtered in
  `eval_cfm_mppi_doubleintegrator.py` (~L89-91), insert nearest-1 selection
  (`argmin‖pos_obs(t=0)‖`) before `synthesize_control`; this restricts BOTH the CFM condition
  and the MPPI cost. Their CFM already keeps only **top-k worst** obstacle constraints
  (`reward.py`), so k=1 is the natural partial-observability knob.

## Concrete next actions
1. Wire a `mizuta_di_native` eval path that runs THEIR `FlowMPPI` + checkpoint on
   `eval80_{ucy,sdd}` and logs collision%, dist-to-goal, sec/step (their own metrics) — the
   honest baseline.
2. Run OURS on the identical 300+300 scenarios; same metrics + our per-step geometric overhead.
3. Ablations: (a) OOD already free (ETH→UCY/SDD); (b) closest-obstacle (k=1) for both.
