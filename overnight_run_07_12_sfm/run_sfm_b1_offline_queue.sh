#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT SMOKE_OUTDIR FULL_OUTDIR" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
SMOKE_OUTDIR="$(realpath -m "$2")"
FULL_OUTDIR="$(realpath -m "$3")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
EXPECTED_SHA="1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
POLL_SECONDS="${POLL_SECONDS:-20}"
IDLE_POLLS_REQUIRED="${IDLE_POLLS_REQUIRED:-3}"
LOG="${QUEUE_LOG:-${FULL_OUTDIR}.queue.log}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "$(date -Is) QUEUE_START"
echo "source=$(git -C "$HERE/.." rev-parse HEAD)"
echo "checkpoint=$CHECKPOINT"
echo "smoke_outdir=$SMOKE_OUTDIR"
echo "full_outdir=$FULL_OUTDIR"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi
if [[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" != "$EXPECTED_SHA" ]]; then
  echo "checkpoint SHA-256 mismatch" >&2
  exit 1
fi
if [[ -e "$SMOKE_OUTDIR" || -e "$FULL_OUTDIR" ]]; then
  echo "output roots must both be absent" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

idle_polls=0
while (( idle_polls < IDLE_POLLS_REQUIRED )); do
  process_count="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader |
      sed '/^[[:space:]]*$/d' | wc -l
  )"
  bad_gpu_count="$(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F, '{if ($1+0 > 1024 || $2+0 > 5) bad++} END {print bad+0}'
  )"
  gpu_count="$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l
  )"
  if [[ "$gpu_count" -eq 4 && "$process_count" -eq 0 && "$bad_gpu_count" -eq 0 ]]; then
    idle_polls=$((idle_polls + 1))
    echo "$(date -Is) IDLE_CONFIRMATION ${idle_polls}/${IDLE_POLLS_REQUIRED}"
  else
    idle_polls=0
    echo "$(date -Is) GPU_BUSY processes=$process_count bad_gpus=$bad_gpu_count"
  fi
  if (( idle_polls < IDLE_POLLS_REQUIRED )); then
    sleep "$POLL_SECONDS"
  fi
done

cd "$HERE"
echo "$(date -Is) SMOKE_START"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" sfm_b1_offline_exec.py \
  --checkpoint "$CHECKPOINT" \
  --outdir "$SMOKE_OUTDIR" \
  --alpha 0.01 \
  --exposure-epochs 1 \
  --rounds 1 \
  --verifier-workers 32 \
  --seed 20260724 \
  --device cuda:0 \
  --smoke

"$PYTHON" - "$SMOKE_OUTDIR" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
with (root / "COMPLETE.json").open() as stream:
    payload = json.load(stream)
assert payload["status"] == "SFM_B1_OFFLINE_EXEC_COMPLETE"
assert payload["source"]["tracked_worktree_clean"] is True
assert len(payload["history"]) == 1
row = payload["history"][0]
summary = row["gather"]["shard"]
assert summary["D"] == summary["contexts"]
assert summary["D"] == summary["Dplus"] + summary["Dminus"]
assert summary["errors"] == summary["unresolved_contexts"] == 0
assert row["gather"]["counts"]["B_queries"] == 4 * summary["contexts"]
assert row["replay"]["optimizer_steps"] == math.ceil(summary["D"] / 128)
assert row["replay"]["positive_total_visits"] == summary["Dplus"]
assert row["replay"]["negative_total_visits"] == summary["Dminus"]
print(json.dumps({
    "status": "SMOKE_VALIDATED",
    "contexts": summary["contexts"],
    "Dplus": summary["Dplus"],
    "Dminus": summary["Dminus"],
    "NVP": row["gather"]["counts"].get("NVP_contexts", 0),
    "optimizer_steps": row["replay"]["optimizer_steps"],
    "wall_seconds": row["wall_seconds"],
}, sort_keys=True))
PY

echo "$(date -Is) SMOKE_VALIDATED_FULL_START"
"$PYTHON" run_sfm_b1_offline_9arm.py \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$EXPECTED_SHA" \
  --outdir "$FULL_OUTDIR" \
  --gpu-indices 0,1,2,3 \
  --verifier-workers 8 \
  --seed 20260724 \
  --eval-ep0 260000 \
  --eval-noise-seed 20260723
echo "$(date -Is) FULL_DELIVERY_COMPLETE"
