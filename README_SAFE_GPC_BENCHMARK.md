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

## Interpreting Safety Scope

- Double-integrator results are labeled `linear_system_theorem_relevant`.
- Unicycle results are labeled `empirical_only_unicycle`.

## Outputs

Episode JSONL streams are written under:

`results/benchmark/<timestamp>/<dataset>/<dynamics>/<method>.jsonl`

Run summaries are written as `summary.csv`, `summary.json`, and `summary.md`.

## Known Limitations

`--smoke` intentionally reduces rollout/sample counts for quick verification. Use non-smoke script defaults for benchmark runs intended for reporting.
