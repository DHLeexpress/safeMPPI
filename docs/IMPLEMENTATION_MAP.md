# Implementation Map

## Repository Paths

- CFM-MPPI benchmark repo: `/home/dohyun/projects/cfm_mppi`
  - Branch: `master`
  - Commit: `837083ecf5743d1f60d56929170d01c3246685a3`
  - Pre-existing untracked items at discovery: `Dohyun_ICRA2026_final/`, `UnifiedGenRefine_arXiv-2508.01192v3/`, `cfm_mppi/example/unicycle_executed.ipynb`
- safeGPC repo: `/home/dohyun/projects/safeGPC`
  - Branch: `main`
  - Commit: `0165acf9d95731cd4d7167ce24a3b651d5ef892d`
  - Additional local branch found: `backup/pre-clean-main-20260616-192816`

## Files Inspected

CFM-MPPI:

- `cfm_mppi/train.py`
- `cfm_mppi/train_arg_parser.py`
- `cfm_mppi/training/train_loop.py`
- `cfm_mppi/models/transformer.py`
- `cfm_mppi/models/model_configs.py`
- `cfm_mppi/mppi/flowmppi.py`
- `cfm_mppi/mppi/flowmppi_candidates.py`
- `cfm_mppi/mppi/utils.py`
- `cfm_mppi/evaluation/eval_utils.py`
- `cfm_mppi/evaluation/eval_cfm_mppi_doubleintegrator.py`
- `cfm_mppi/evaluation/eval_cfm_mppi_unicycle.py`
- `cfm_mppi/reward.py`
- `cfm_mppi/utils.py`
- `dataset/train80_ego.pt`, `dataset/eval80_ego_{ucy,sdd}.pt`, `dataset/eval80_obs_{ucy,sdd}.pkl`

safeGPC:

- `main_v4.1.ipynb`
- `main_v4.2.ipynb`
- `algs/mppi.py`
- `utils/alg_base.py`
- `tasks/doubleIntegrator.py`
- `tasks/mpc_cbf_doubleIntegrator.py`
- `collector_2d.py`
- `models/simulator.py`
- `models/flow_model.py`
- `models/flow_training.py`
- `networks/seq_vf.py`
- `artifacts/mppi_seq_newcsv_keep_nobs_20250828_174641.pkl`
- `artifacts/new_train_norms_1.json`
- `artifacts/train_norms_20250904_074856.json`

## Files Added Or Edited

- Compatibility edits: `cfm_mppi/train.py`, `cfm_mppi/train_arg_parser.py`, `cfm_mppi/training/train_loop.py`, `cfm_mppi/training/load_and_save.py`, `cfm_mppi/evaluation/eval_cfm_mppi_doubleintegrator.py`, `cfm_mppi/evaluation/eval_cfm_mppi_unicycle.py`, `cfm_mppi/utils.py`, `flow_matching/__init__.py`
- Canonical data: `cfm_mppi/data/__init__.py`, `cfm_mppi/data/canonical_dataset.py`, `cfm_mppi/data/build_canonical_dataset.py`
- Context/model code: `cfm_mppi/models/context_encoder.py`, `cfm_mppi/models/contextual_transformer.py`, `cfm_mppi/models/drifting_generator.py`
- Training: `cfm_mppi/training/train_loop_safe_cfm.py`, `cfm_mppi/training/train_safe_cfm.py`, `cfm_mppi/training/train_drifting.py`, `cfm_mppi/training/drift_loss_torch.py`
- safeGPC adapter: `cfm_mppi/safegpc_adapter/__init__.py`, `cfm_mppi/safegpc_adapter/barrier.py`, `cfm_mppi/safegpc_adapter/gamma_schedule.py`, `cfm_mppi/safegpc_adapter/safemppi.py`
- Evaluation: `cfm_mppi/evaluation/eval_benchmark.py`, `cfm_mppi/evaluation/metrics.py`, `cfm_mppi/evaluation/result_writer.py`, `cfm_mppi/evaluation/run_safe_cfm.py`, `cfm_mppi/evaluation/run_drifting.py`
- Configs: `configs/benchmark/*.yaml`
- Scripts: `scripts/*.sh`
- Tests: `tests/test_*.py`, `tests/conftest.py`
- Docs: `README_SAFE_GPC_BENCHMARK.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/CONFLICTS_AND_RESOLUTIONS.md`, `docs/DATASET_SCHEMA.md`, `docs/RESULT_SCHEMA.md`

## CFM-MPPI Baseline Shapes And Constants

- `TransformerModel` constructor defaults:
  - `in_channels=2`, `out_channels=2`, `d_model=256`, `nhead=4`, `num_layers=6`, `dim_feedforward=1024`, `dropout=0.1`, `max_len=500`
- `TransformerModel.forward(x, timesteps, start, goal)`:
  - `x`: `[B, 2, T]`
  - `timesteps`: `[B]`
  - `start`: `[B, 2]`
  - `goal`: `[B, 2]`
  - output: `[B, 2, T]`
- Original `LightDataset`: loads `dataset/train80_ego.pt`.
- Original `random_collate_fn`: picks random `L in [10,80]` and returns `[B, C, L]`.
- `dataset/train80_ego.pt`: `torch.float32`, shape `[273989, 9, 80]`.
  - Original training uses channels `0:2` as positions and `2:4` as controls.
- Evaluation constants:
  - Horizon: `80`
  - `dt`: `0.1`
  - Safe margin / agent radius: `0.5`
  - `SAFE_COEF`: `[0.1, 0.3, 0.5, 0.7, 0.9]`
  - `GOAL_COEF`: `0.1`
  - ODE first step times: `[0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]`
  - ODE receding times: `[0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]`
  - `n_sample`: `200`
  - MPPI lambda: `0.1`
  - Double-integrator sigma: `[0.4, 0.4]`
  - Unicycle sigma: `[0.3, 0.6]`
  - `u_min/u_max`: `[-2,-2]`, `[2,2]`
- Dynamics selection:
  - Double integrator uses `cfm_mppi.mppi.utils.doubleintegrator_dynamics`.
  - Unicycle uses `cfm_mppi.mppi.utils.unicycle_dynamics`.
- Costs:
  - Stage cost combines goal distance, exponential collision cost, and optional smoothness.
  - Terminal cost is `0.1 * goal distance`.
- `SAFE_COEF` in `eval_utils.run_CFM` scales normalized CBF-gradient markup during CFM ODE sampling. It is not safeGPC gamma.

## safeGPC Barrier, Gamma, And Data

- Online safeMPPI source path:
  - `safeGPC/algs/mppi.py`
  - `safeGPC/utils/alg_base.py`
  - `safeGPC/tasks/doubleIntegrator.py`
- Affine barrier:
  - `DoubleIntegrator2D.hnew` and `hnew_torch` select the nearest circle by current position and compute a scalar affine projection relative to the initial state's nearest-boundary normal.
  - `huniversal_proj` reduces to `hnew` for circle-only runs and is the v4 collection default.
- safeMPPI rejection rule:
  - `violation = h_new < (1 - gamma) * h_old`
  - v4 collection sets `check_first_control_only=False`, so rejection is checked over the rollout.
- main_v4.1 collection:
  - `gamma_logspace=(0.05, 1.0, 10)`
  - `plan_horizon=1.0`
  - `num_samples=256`
  - `noise_level=2.0`
  - `temperature=5000.0`
  - `safety_fn='huniversal_proj'`
- main_v4.2 collection:
  - `gamma_logspace=(0.1, 1.0, 10)`
  - `plan_horizon=2.0`
  - `num_samples=256`
  - `noise_level=2.0`
  - `temperature=5000.0`
  - `safety_fn='huniversal_proj'`
- main_v4.2 contextual vector field:
  - Features: `FEAT_DIM=7` as `[obs6, gamma]`
  - `obs6=[px, py, vx, vy, nearest_surface_dx, nearest_surface_dy]`
  - `TRAIN_WIN=11` in the notebook, while local saved norms show `win_len=7`.
  - Notebook model depth is GRU/MLP; the benchmark implementation uses Mizuta-depth transformer for fair CFM/Drifting comparison.
- main_v4.2 adaptive schedule:
  - `gamma(d,v_proj)=g_min+(g_max-g_min)*(1-exp(-beta*d))*exp(-alpha*max(0,v_proj))`
  - Extracted constants: `alpha=0.1541`, `beta=1.5826`, default bounds `g_min=0.1`, `g_max=1.0`.

## Proposed Model Shapes

- Canonical safe CFM target controls: `[B, T, 2]`, transposed to `[B, 2, T]` for flow matching.
- Safe contextual CFM input:
  - noisy controls `[B, 2, T]`
  - time `[B]`
  - context tokens `[B, 7, d_model]`
  - output vector field `[B, 2, T]`
- Drifting generator input:
  - noise controls `[B, 2, T]`
  - same context fields/tokens as safe CFM
  - output controls `[B, 2, T]`
  - inference model calls / NFE: `1`

## Evaluation Entry Points

- Original baseline scripts:
  - `python cfm_mppi/train.py`
  - `python cfm_mppi/evaluation/eval_cfm_mppi_doubleintegrator.py`
  - `python cfm_mppi/evaluation/eval_cfm_mppi_unicycle.py`
- New unified harness:
  - `python -m cfm_mppi.evaluation.eval_benchmark --dataset sfm --dynamics doubleintegrator --methods mizuta_cfm_mppi safemppi_gamma safe_cfm drifting --num-episodes 100 --seed 0 --output-root results/benchmark`
- Training:
  - `python -m cfm_mppi.training.train_safe_cfm`
  - `python -m cfm_mppi.training.train_drifting`

## Notes

- `cfm_mppi/evaluation/eval_benchmark.py::BenchmarkPolicies._mizuta_action` carries a per-episode receding-horizon state and calls the original `eval_utils.synthesize_control` with `FlowMPPI`. Under `--smoke`, MPPI sample count is reduced for runtime; normal runs use the original `200` samples.
