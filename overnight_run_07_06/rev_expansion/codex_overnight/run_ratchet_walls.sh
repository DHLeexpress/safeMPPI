#!/bin/bash
# Ratcheted guarded unit driver: when a generation saturates its cumulative trust-anchor budget
# (2 consecutive rollback COMP lines), stop it, take its newest full-state checkpoint, and relaunch with
# teacher/anchor re-referenced to that checkpoint (documented per-generation recipe change; the numeric
# bounds and gate mechanism never change). Exits when the trainer finishes naturally or MAX_GEN reached.
# Usage: bash run_ratchet_unit.sh <first_gen_outdir> <max_gen> <gpu>
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16
OUT0="$1"; MAX_GEN="${2:-8}"; GPU="${3:-3}"; START_GEN="${4:-1}"
ESCAPE="analysis/goal_replay/onpolicy_s766_teacher_t104_allg_seed100_199_certified.pt"
gen=$START_GEN; OUT="$OUT0"
echo "$(date '+%H:%M') driver start gen=$gen out=$OUT" >> logs/ratchet_driver.log
while true; do
  sleep 240
  LOG="logs/$(basename "$OUT").log"
  if [ -f "$OUT/final.pt" ]; then
    echo "$(date '+%H:%M') gen $gen finished naturally" >> logs/ratchet_driver.log; break
  fi
  if ! pgrep -f "grid_expand_hardtail.*$(basename "$OUT")" > /dev/null; then
    echo "$(date '+%H:%M') gen $gen process died; stopping driver" >> logs/ratchet_driver.log; break
  fi
  nroll=$(grep -c "rollback 1" "$LOG" 2>/dev/null || echo 0)
  last2=$(grep -E "rollback [01]" "$LOG" | tail -2 | grep -c "rollback 1")
  if [ "$last2" -ge 2 ]; then
    echo "$(date '+%H:%M') gen $gen saturated (rollbacks=$nroll); ratcheting" >> logs/ratchet_driver.log
    pkill -f "grid_expand_hardtail.*$(basename "$OUT")"; sleep 8
    CKPT=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then echo "no ckpt in $OUT; stop" >> logs/ratchet_driver.log; break; fi
    gen=$((gen + 1))
    if [ "$gen" -gt "$MAX_GEN" ]; then echo "max gen reached" >> logs/ratchet_driver.log; break; fi
    NEW="results/p2/walls4_gen${gen}_s$((808 + gen))"
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
      --ckpt "$CKPT" --outdir "$NEW" \
      --iters 82 --drop-train-state --legacy-prime-iters 1 --freeze --lr 2e-5 --seed $((800 + gen)) \
      --rollouts-per-iter 28 --gather-attempt-cap 300 --batch 64 \
      --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
      --quantile-schedule 0:0.50 200:0.60 400:0.70 \
      --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 \
      --early-until 100 --cooldown-from 400 \
      --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
      --demo-frac 0.125 --lwf-eta 0.05 \
      --teacher-ckpt "$CKPT" \
      --nfe-explore 8 --field-grad-clip 1.0 \
      --max-functional-step 0.025 --max-anchor-drift 0.016 \
      --targeted-frac 0.5 --n-target 40 --align-temp 0.45 --min-modes-per-gamma 1 \
      --recovery-frac 0.3 --recovery-origin-band 0.0 1.0 -0.05 0.18 0.0 0.45 -0.28 0.05 \
      --recovery-goal-band 4.3 5.0 4.6 5.06 -0.30 0.30 -0.05 0.35 \
      --hard-quota 12 --hard-x0 oob --hard-x0-cand 64 --strip-probe-every 2 \
      --wall-plugs 4 \
      --m-measure 5 --measure-every 2 --probe-cov 2 --log-comp-every 1 \
      --viz-db-every 2 --ckpt-every 2 \
      --tag "$(basename "$NEW")" > "logs/$(basename "$NEW").log" 2>&1 &
    echo "$(date '+%H:%M') gen $gen launched from $CKPT -> $NEW (PID $!)" >> logs/ratchet_driver.log
    OUT="$NEW"
  fi
done
echo "$(date '+%H:%M') driver exit at gen $gen; last out=$OUT" >> logs/ratchet_driver.log
