# Conflicts And Resolutions

## PyTorch CFM-MPPI vs JAX Drifting

Drifting's released implementation is JAX-oriented. The benchmark repo is PyTorch. Resolution: do not add JAX; implement a PyTorch one-step generator in `cfm_mppi/models/drifting_generator.py` and a PyTorch affinity-style objective in `cfm_mppi/training/drift_loss_torch.py`.

## Mizuta SAFE_COEF vs safeGPC Gamma

Mizuta `SAFE_COEF` scales a CBF-gradient correction inside CFM ODE sampling. safeGPC `gamma` appears in the discrete barrier condition `h_new >= (1 - gamma) h_old`. Resolution: keep them as separate result fields: `safe_coef` for Mizuta-style CFM and `gamma` for safeGPC/safeMPPI.

## Double-Integrator Theorem vs Unicycle Empirical Results

The local safeGPC theorem-relevant implementation is double-integrator based. Resolution: every double-integrator record includes `safety_guarantee_scope="linear_system_theorem_relevant"`; every unicycle record includes `safety_guarantee_scope="empirical_only_unicycle"`.

## One-Step Drifting vs Multi-Step ODE CFM Sampling

CFM samples through multiple model calls along an ODE path. Drifting is a one-step generator. Resolution: report `nfe` and `model_calls_per_step`; Drifting logs `1`, safe contextual CFM logs its configured ODE count.

## Dataset Schema Mismatch

Mizuta data is a dense tensor `[N, C, T]`; safeGPC data is per-step records with NaN gamma EOS rows. Resolution: add canonical `.pt` dictionaries with explicit `states`, `controls_dyn`, `controls_si`, context histories, gamma, dynamics type, source, and metadata.

## safeGPC Path And Branch Ambiguity

The public safeGPC remote may be empty. Resolution: only local `/home/dohyun/projects/safeGPC` was inspected. The active branch is `main`; v4.1/v4.2 logic is in notebooks, with reusable code in `algs`, `utils`, `tasks`, `models`, and `collector_2d.py`.

## main_v4.2 TRAIN_WIN vs Saved Norms

The v4.2 notebook sets `TRAIN_WIN=11`, but local saved norm JSON files in `artifacts/` show `win_len=7`. Resolution: canonical builder defaults to v4.2 `history_len=11`, while model/training CLIs expose `--history-len` and saved checkpoints preserve the actual setting.

## Drifting Negatives

safeGPC artifacts do not label unsafe/colliding negative controls. Resolution: `train_drifting.py` uses the sampled noise sequence as a documented fallback negative and `drift_loss_torch.py` keeps the negative term optional.

## Unified Mizuta Harness Runtime

The original Mizuta `synthesize_control + FlowMPPI` path is computationally heavier than the other smoke methods. Resolution: the unified harness uses the original path, but reduces MPPI samples under `--smoke`; normal runs use the original sample count.
