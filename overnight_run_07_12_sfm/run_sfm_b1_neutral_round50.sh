#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

CHECKPOINT=$(realpath "$1")
OUTPUT_ROOT=$(realpath -m "$2")
PYTHON=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
WORKERS=${VERIFIER_WORKERS_PER_ARM:-48}
EVAL_ROUNDS=0,1,2,5,10,20,30,40,50

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse output root: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT/logs"

names=(lr1em5_s01 lr1em5_s04 lr3em5_s01 lr3em5_s04)
lrs=(1e-5 1e-5 3e-5 3e-5)
steps=(1 4 1 4)
gpus=(1 1 3 3)
pids=()

for index in "${!names[@]}"; do
  name=${names[$index]}
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=${gpus[$index]}
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    exec "$PYTHON" overnight_run_07_12_sfm/sfm_b1_neutral_multiround.py \
      --checkpoint "$CHECKPOINT" \
      --output-root "$OUTPUT_ROOT/$name" \
      --name "$name" \
      --rounds 50 \
      --scenario-ep0 260000 \
      --eval-ep0 270000 \
      --eval-M 20 \
      --eval-rounds "$EVAL_ROUNDS" \
      --lr "${lrs[$index]}" \
      --inner-steps "${steps[$index]}" \
      --probe-per-gamma 4 \
      --device cuda \
      --workers "$WORKERS"
  ) >"$OUTPUT_ROOT/logs/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "arm failed: ${names[$index]}" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi

for name in "${names[@]}"; do
  test -f "$OUTPUT_ROOT/$name/DELIVERY_COMPLETE.json"
done
echo "SFM_B1_NEUTRAL_ROUND50_FOUR_ARM_COMPLETE $OUTPUT_ROOT"
