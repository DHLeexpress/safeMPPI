# AFE100 M=20 evaluation-only screen

`paper_results/afe100_m20_eval.py` is an additive evaluator for the completed
RBF and deep-ensemble 100-round studies. It does not resume training or write
inside either run. Before loading a checkpoint it verifies the trainer-written
`COMPLETE.json` inventory, including every artifact hash.

The screen evaluates rounds `0,10,...,100`, all seven gamma values, and exactly
20 paired rollouts per cell for both the raw untilted generator and the frozen
expert-free verified controller. M=20 is labeled as screening evidence rather
than a final estimate.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHON=/home/dohyun/miniforge3/envs/cfm_mppi/bin/python

./run_afe100_m20_eval.sh \
  /path/to/rbf/afe_rbf_s910 \
  /path/to/ensemble/afe_ensemble_s910 \
  /new/output/root
```

The launcher refuses an existing output root or a physical GPU 1 with an
active compute process. The evaluator emits authenticated per-cell archives,
`metrics.jsonl`, checkpoint curves, a fixed-index raw-policy gallery, and
`EVALUATION_COMPLETE.json`.
