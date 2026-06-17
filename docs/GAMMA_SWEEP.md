# Gamma Sweep Workflow

This branch adds a three-panel animation comparing Mizuta CFM-MPPI with online safeMPPI over gamma in `[0, 1]`.

Mizuta's pretrained checkpoint is expected at `output_dir/cfm_transformer/checkpoint.pth`. safeMPPI is evaluated online and does not require training. To train the safe contextual CFM/GPC model before the comparison, use the overnight script.

Run the sweep:

```bash
DATASET=sfm \
DYNAMICS=doubleintegrator \
NUM_EPISODES=20 \
GAMMA_COUNT=21 \
SAFEMPPI_NUM_SAMPLES=2048 \
DEVICE=cuda \
scripts/run_gamma_sweep.sh
```

Train safe contextual CFM/GPC first, then run the same sweep:

```bash
GPC_EPOCHS=300 \
GPC_BATCH_SIZE=256 \
GPC_DEVICE=cuda \
NUM_EPISODES=20 \
GAMMA_COUNT=21 \
SAFEMPPI_NUM_SAMPLES=2048 \
scripts/train_gpc_then_gamma_sweep.sh
```

Direct module call:

```bash
python -m cfm_mppi.visualization.live_gamma_compare \
  --dataset sfm \
  --dynamics doubleintegrator \
  --num-episodes 20 \
  --gamma-count 21 \
  --safemppi-num-samples 2048 \
  --device cuda
```

Outputs are written under `results/visualization/gamma_sweep/<timestamp>/<dataset>/<dynamics>/`:

- `gamma_sweep_records.jsonl`
- `summary.csv`
- `summary.json`
- `summary.md`
- `live_gamma_sweep_last_frame.png`
- `live_gamma_sweep.mp4` or `live_gamma_sweep.gif`

The panels are trajectory comparison, success/collision over gamma, and safety/performance/compute over gamma.
