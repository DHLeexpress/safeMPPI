# Stage results

Each experiment stage keeps generated artifacts in its own directory:

```text
stage_results/
├── 01_seeds/
│   ├── viz/       # seed geometry
│   └── logs/      # counts and geometric checks
├── 02_demos/
│   ├── viz/       # fixed-pair preview and demo visualizations
│   ├── logs/      # rollout/certificate summaries
│   └── data/      # preview paths, then approved training shards
├── 03_pretrain/{viz,logs,data}/
├── 04_deploy/{viz,logs,data}/
├── 05_expansion/{viz,logs,data}/
├── 06_baselines/{viz,logs,data}/
├── 07_iteration/{viz,logs,data}/
└── 08_final/{viz,logs,data}/
```

Generated files are reproducible from the commands recorded in `PROGRESS.md`. Gamma is always rendered
with the discrete `plasma_trunc` palette in `viz_style.py`; sigma/uncertainty is always rendered with
continuous `viridis`.

Stage 3's active checkpoint uses the original 37-dimensional `low5 + E(H_P)` context. The superseded
41-dimensional raw-start/raw-goal experiment is preserved under
`03_pretrain/rejected_raw_endpoints/` for provenance and is never loaded by the active evaluator.
