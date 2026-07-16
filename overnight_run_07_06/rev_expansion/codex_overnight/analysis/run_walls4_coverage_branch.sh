#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 CHECKPOINT CERTIFIED_RARE_MODE_REPLAY OUTDIR [GPU] [ITERS]" >&2
  exit 2
fi
ckpt="$1"; replay="$2"; out="$3"; gpu="${4:-3}"; iters="${5:-100}"
[[ -f "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; exit 2; }
[[ -f "$replay" ]] || { echo "missing certified replay: $replay" >&2; exit 2; }
[[ ! -e "$out/recipe.json" ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
mkdir -p "$out" logs
LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib python - "$replay" <<'PY'
import sys,torch
r=torch.load(sys.argv[1],map_location='cpu',weights_only=False); m=r.get('metadata',{})
assert m.get('destination_gamma_certified') is True, 'replay is not destination-gamma certified'
assert int(m.get('wall_plugs',-1)) == 4, f"replay has wrong wall scene: {m.get('wall_plugs')}"
assert len(r.get('x0',[])) > 0, 'certified replay is empty'
PY
start=$(LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib python - "$ckpt" <<'PY'
import sys,torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
print(int(c.get('iter',c.get('history_tail',{}).get('iter',0))))
PY
)
log="logs/$(basename "$out").log"

CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=8 \
python grid_expand_hardtail.py \
  --ckpt "$ckpt" --drop-train-state --outdir "$out" --iters "$iters" --start-iter "$start" \
  --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 \
  --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 \
  --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
  --quantile-schedule 0:0.50 200:0.60 400:0.70 \
  --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.2 \
  --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 \
  --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt ../../results/hp_repr/pretrained_a32uni.pt \
  --lr 1e-4 --nfe-explore 8 --field-grad-clip 1.0 \
  --max-functional-step 999 --max-anchor-drift 999 \
  --targeted-frac 0.5 --n-target 40 --align-temp 0.45 --target-perp-brake \
  --min-modes-per-gamma 0 --min-modes-schedule \
    "$start:2" "$((start+20)):4" "$((start+40)):8" "$((start+60)):12" "$((start+80)):14" \
  --mode-hit-gate --min-target-hits 1 \
  --recovery-frac 0.3 --recovery-origin-band 0 1 -0.05 0.18 0 0.45 -0.28 0.05 \
  --recovery-goal-band 4.3 5 4.6 5.06 -0.3 0.3 -0.05 0.35 \
  --hard-quota 12 --hard-x0 oob --hard-x0-cand 64 --strip-probe-every 2 \
  --escape-replay "$replay" --escape-quota 64 --escape-eta 1.0 \
  --wall-plugs 4 --probe-cov 2 --viz-db-every 1 --ckpt-every 4 --log-comp-every 1 \
  --seed 825 --tag "$(basename "$out")" > "$log" 2>&1
