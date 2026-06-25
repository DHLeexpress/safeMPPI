# SafeGPC Benchmark Pipeline

This repo now contains an additive benchmark layer comparing Mizuta CFM-MPPI, online safeMPPI gamma sweeps, safe contextual CFM, and a same-depth one-step Drifting-style generator.

## Assumptions

- CFM repo: `/home/dohyun/projects/cfm_mppi`
- safeGPC repo: `/home/dohyun/projects/safeGPC`
- Original Mizuta checkpoint: `output_dir/cfm_transformer/checkpoint.pth`
- Default safeGPC artifact: `/home/dohyun/projects/safeGPC/artifacts/mppi_seq_newcsv_keep_nobs_20250828_174641.pkl`
- `WANDB_MODE=disabled` is set by scripts unless you override it.

## Dataset

Build canonical splits from local safeGPC:

```bash
scripts/build_canonical_dataset.sh
```

For a smaller smoke dataset:

```bash
MAX_EPISODES=20 scripts/build_canonical_dataset.sh
```

## Training

Safe contextual CFM:

```bash
scripts/train_safe_cfm.sh --epochs 1 --batch-size 32 --device cpu --test-run
```

Drifting-style one-step generator:

```bash
scripts/train_drifting.sh --epochs 1 --batch-size 32 --device cpu --test-run
```

Both write `args.json`, `checkpoint_latest.pth`, `checkpoint_best.pth`, and `train_log.jsonl` under `output_dir/`.

## Evaluation

Mizuta baseline:

```bash
scripts/eval_mizuta_baseline.sh
```

safeMPPI gamma schedule:

```bash
scripts/eval_safemppi_gamma.sh
```

Safe CFM:

```bash
scripts/eval_safe_cfm.sh
```

Drifting:

```bash
scripts/eval_drifting.sh
```

Full benchmark:

```bash
scripts/run_full_benchmark.sh
```

Smoke checks:

```bash
scripts/run_smoke_tests.sh
```

## Validation Comparison Videos

The project-page teaser style shows the same scene with generated/planned robot motion over obstacles. To render a side-by-side validation comparison for Mizuta CFM-MPPI, safeMPPI with varying `gamma`, and safe contextual CFM with the same `gamma` values:

```bash
python -m cfm_mppi.evaluation.render_validation_comparison \
  --dataset ucy \
  --dynamics unicycle \
  --episode 110 \
  --pedestrian-source validation \
  --steps 80 \
  --gamma-grid 0.1 0.5 1.0 \
  --device cuda \
  --draw-hyperplanes \
  --hyperplane-horizon 20 \
  --safemppi-horizon 40 \
  --safemppi-samples 512 \
  --safe-cfm-num-candidates 16 \
  --debug-rollouts 64 \
  --output results/benchmark_videos/moving_validation_comparison_gamma_v1.mp4 \
  --gif-output results/benchmark_videos/moving_validation_comparison_gamma_v1.gif
```

Quick CPU smoke render:

```bash
python -m cfm_mppi.evaluation.render_validation_comparison \
  --smoke \
  --dataset ucy \
  --dynamics unicycle \
  --episode 110 \
  --pedestrian-source validation \
  --steps 6 \
  --device cpu \
  --output results/benchmark_videos/smoke_validation_comparison.gif
```

`--pedestrian-source validation` loads the moving pedestrian trajectories from `dataset/eval80_obs_<dataset>.pkl`; for `sfm` use `--pedestrian-source sfm-social-force` to generate moving social-force pedestrians. SafeMPPI panels draw the affine DCBF half-plane sequence `h_aff(x; x0) = (1 - gamma)^i`; low `gamma` values show tightly spaced per-step planes, while high `gamma` values collapse toward the obstacle tangent faster. SafeMPPI sampled rollouts are drawn as accepted blue and rejected red trajectories. Safe CFM generated sequences are drawn in green. The renderer also writes a metrics CSV next to the video, so the visual comparison and the benchmark fields (`success`, `collision`, `final_goal_distance`, `min_clearance`, planning time) stay paired.

For the current paper-style gamma tradeoff figure, render a 4-scenario by 4-method grid with Mizuta plus SafeMPPI `gamma={0.1,0.5,1.0}` and Safe CFM disabled:

```bash
conda run --live-stream -n cfm_mppi python -m cfm_mppi.evaluation.render_validation_comparison \
  --dataset ucy \
  --dynamics unicycle \
  --pedestrian-source validation \
  --episode-list 0 41 110 123 \
  --steps 80 \
  --gamma-grid 0.1 0.5 1.0 \
  --no-safe-cfm \
  --device cuda \
  --safemppi-horizon 40 \
  --safemppi-samples 512 \
  --debug-rollouts 48 \
  --hyperplane-horizon 20 \
  --hyperplane-stride 1 \
  --output results/benchmark_videos/smoke_moving_validation_comparison_v2.mp4 \
  --gif-output results/benchmark_videos/smoke_moving_validation_comparison_v2.gif
```

The v2 renderer highlights the nearest effective safety disk in orange, draws the affine support planes tangent to that disk, and writes an adjacent `_notes.md` with the exact settings for later seed/episode sweeps.

To test SETS-style discrete backup proposals inside SafeMPPI, enable the v3 backup branch sampler:

```bash
conda run --live-stream -n cfm_mppi python -m cfm_mppi.evaluation.render_validation_comparison \
  --dataset ucy \
  --dynamics unicycle \
  --pedestrian-source validation \
  --episode-list 0 41 110 123 \
  --steps 80 \
  --gamma-grid 0.1 0.5 1.0 \
  --no-safe-cfm \
  --device cuda \
  --safemppi-horizon 40 \
  --safemppi-samples 512 \
  --safemppi-use-sets-backup \
  --safemppi-sets-num-modes 3 \
  --safemppi-sets-branch-scale 0.85 \
  --safemppi-sets-include-cbf-backup \
  --safemppi-sets-cbf-push 1.25 \
  --safemppi-sets-reverse-speed 0.75 \
  --safemppi-sets-turn-rate 1.4 \
  --debug-rollouts 48 \
  --hyperplane-horizon 20 \
  --hyperplane-stride 1 \
  --no-show-backup-labels \
  --figure-tag v3 \
  --output results/benchmark_videos/smoke_moving_validation_comparison_v3.mp4 \
  --gif-output results/benchmark_videos/smoke_moving_validation_comparison_v3.gif
```

The v3 sampler keeps Gaussian MPPI rollouts but appends deterministic backup branches from controllability Gramian eigenmodes plus nearest-halfspace away/tangent/reverse controls. In the video, random rollouts remain red/blue while backup branches use distinct brighter colors; dashed backup branches failed the affine barrier check.

## Interpreting Safety Scope

- Double-integrator results are labeled `linear_system_theorem_relevant`.
- Unicycle results are labeled `empirical_only_unicycle`.

## Outputs

Episode JSONL streams are written under:

`results/benchmark/<timestamp>/<dataset>/<dynamics>/<method>.jsonl`

Run summaries are written as `summary.csv`, `summary.json`, and `summary.md`.

## Known Limitations

`--smoke` intentionally reduces rollout/sample counts for quick verification. Use non-smoke script defaults for benchmark runs intended for reporting.
