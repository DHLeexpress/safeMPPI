# Portable AFE M=20 evaluation-only screen

`paper_results/afe_m20_eval.py` accepts one scene profile and one completed,
delivered adaptive-ensemble AFE run. It does not resume training or write
inside the run. Before loading either checkpoint it verifies the delivery
manifest and the complete trainer-written artifact inventory.

The screen evaluates only authenticated `ckpt_0.pt` and the predeclared
completed/headline checkpoint (`ckpt_50.pt`), all seven gamma values, and
exactly 20 paired rollouts per cell. Raw untilted behavior and the frozen
expert-free verified controller are reported separately. The gallery always
uses archive indices `0..9`, and the all-round curves are read from the
authenticated trainer `probe.jsonl`. M=20 is screening evidence, not a final
estimate.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHON=/home/dohyun/miniforge3/envs/cfm_mppi/bin/python

./run_afe_m20_eval.sh \
  codex_radius1_v1 \
  /path/to/completed/afe/run \
  /new/output/root
```

The launcher refuses an existing output root or a physical GPU 1 with an
active compute process. The evaluator emits authenticated per-cell archives,
`metrics.jsonl`, trainer-probe curves, a fixed-index raw/verified gallery, and
`EVALUATION_COMPLETE.json`.
