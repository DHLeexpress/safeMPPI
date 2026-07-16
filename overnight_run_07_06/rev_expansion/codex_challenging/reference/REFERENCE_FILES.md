# Reference files (snapshot from codex_overnight, 2026-07-14) — what to copy & change

**Import caveat:** these expect `sys.path` to include `overnight_run_07_06/` and its parents (they do
`sys.path[:0]=['.','..','../..']` or import `_paths`). From `codex_challenging/` add
`sys.path.insert(0, '/home/dohyun/projects/cfm_mppi/overnight_run_07_06')` and one level up, OR run them
with cwd = a dir two levels under `overnight_run_07_06` (like codex_overnight is). Adapt, don't fight it.

| file | role | change for THIS task |
|---|---|---|
| `grid_expand_hardtail.py` | expansion trainer (all flags, ablations, resume, 20-gate harness) | set defaults (demo 0/lwf 0/rollouts 14); confirm the policy ctx carries raw start+goal (see below) |
| `pretrain_repr.py` + `stage3_pretrain.py` | build+pretrain `GridHPFlowPolicy` (32×32 CNN/AAP encoder) | **train from scratch on w8sg varied-goal demos**; extend ctx with raw start_xy+goal_xy |
| `gen_uniform_data.py` | `uniform_starts()` = Image #10 grid (`\|y-x\|>=1`, obstacle-free) | split blue(y>x)=start / red(y<x)=goal; apply 8 plugs before the free test |
| `gen_dr_data.py` | SafeMPPI demo-window gen (patched: `--wall-plugs/--start-eps/--canonical-frac/--reach`) | **sample goal∈red per episode** (currently fixed `env.goal`); save `goals` + goal-aware `low5` |
| `eval_ae.py` | metric eval (patched `--wall-plugs/--start-eps`); `expert-worker`/`policy-worker`/`saved-worker`/`assemble` | reach 0.15; expert baseline → `results/expert_challenging` |
| `kazuki_baseline.py` | CFM-MPPI baseline (patched `--reach/--wall-plugs/--start-eps`) | rough sweep w_safe×coll_w×goal_w for a non-zero result |
| `analysis/report_at.py` | per-γ a–d scorecard vs expert (7γ, parallel) | `--expert-dir results/expert_challenging` |
| `analysis/per_gamma_valid.py` | per-γ valid2 rate ("is low γ joining?") | — |
| `analysis/faithful_taxonomy.py` | raw reach/collision/oob taxonomy | — |
| `analysis/socp_sanity.py` | nominal-polytope viz (GREEN certified) along a rollout | reuse for stage 2 |
| `analysis/new_scene_viz.py` | walled-scene geometry figure | reuse to draw the scene |
| `analysis/one_rollout_viz.py` | single rollout per γ + off-diagonal expert npz | reuse for stages 2/4 |
| `analysis/make_table.py` | IEEE .tex+.md table (demo/Kazuki/ours + brothers) | add the 3 brother rows |
| `analysis/make_scatter.py` | quick per-γ time-vs-clearance scatter | or use paper_results/scatter_v4 |
| `paper_results/scatter_v4.py` | 1×2 SR-CR & clearance-time phase planes (γ=plasma_trunc) | add −SOCP/−Progress/−Curriculum series |
| `paper_results/rollouts_v4.py` | 2×4 gallery incl. brothers + Kazuki/pretrained fail insets | re-point dirs; brothers must show collapse/unsafe |
| `paper_results/internals_v4.py` | training internals (per-γ certified-window share = curriculum) | re-point RUN |
| `video_curriculum_fixed.py` | curriculum video (σ=viridis, per-iter viz_db) | re-point --run; `EXTRA_OBS` handles 8 plugs |
| `analysis/greedy_sched_driver.sh` | greedy β/mix/gp_buf schedule search pattern | template for stage 7 |
| `analysis/test_hardtail_trainer.py` | 20-gate trainer harness | run after any trainer edit |

**Colormap discipline (hard rule):** γ = plasma/plasma_trunc; σ (uncertainty) = viridis. NEVER share.
