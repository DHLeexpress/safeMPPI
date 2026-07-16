#!/bin/bash
# AUTONOMOUS pipeline (user 2026-07-14): it10 emergent-gamma sweep (restored recipe: 28 rollouts, gp/qbuf
# 500) -> auto-pick winning beta by pooled SR -> run winner to it200 -> report at every it20 (per-gamma a-d
# vs expert + per-gamma valid2 + faithful taxonomy) -> the THREE deliverables (table, scatter, curriculum
# video) at it100 and it200. Goal: beat the demo expert per gamma on a-d. No further assumptions/prompts.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
GPU=${GPU:-3}; ITERS=${ITERS:-200}
A=analysis; R=results/p2; OUT=grand_final_reports_rev
mkdir -p logs "$OUT"
log(){ echo "$(date '+%H:%M') $*" | tee -a logs/auto_pipeline.log; }

# 1) wait for the it10 sweep + its per-gamma report (byji sentinel), to avoid GPU contention
log "waiting for it10 sweep report (RESTORED_REPORT_DONE)"
for _ in $(seq 1 240); do
  [ -f "$R/RESTORED_REPORT_DONE" ] && break
  [ -f "$R/FAITHFUL_SWEEP_IT10_DONE" ] && { pgrep -f "per_gamma_valid" >/dev/null || { sleep 30; }; }
  sleep 30
done

# 2) pick winner beta by pooled SR (tie: lower CR)
log "picking winner among fsw_b02/b03/b04"
for t in fsw_b02 fsw_b03 fsw_b04; do
  [ -f "$R/eval_pick_$t/scorecard.json" ] || \
    python $A/report_at.py --ckpt "$R/$t/final.pt" --tag "pick_$t" --M 30 \
      --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --reach 0.15 --concurrency 5 --gpu $GPU >/dev/null 2>&1
done
read WIN WBETA < <(python3 -c "
import json
B={'fsw_b02':'0.2','fsw_b03':'0.3','fsw_b04':'0.4'}
sc=[]
for t in B:
    try:
        d=json.load(open(f'$R/eval_pick_{t}/scorecard.json'))['pooled']; sc.append((d['SR'],-d['CR'],t))
    except Exception: pass
sc.sort(reverse=True); print(sc[0][2], B[sc[0][2]]) if sc else print('fsw_b03 0.3')
")
log "winner: $WIN beta=$WBETA"
TAG="final_b${WBETA/./}"; rm -rf "$R/$TAG"

# 3) launch winner -> it200 (restored recipe)
CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
  --ckpt "$PRE" --outdir "$R/$TAG" --iters $ITERS --seed 910 --lr 2e-5 --wall-plugs 8 --start-eps 0.05 --reach 0.2 \
  --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 --gp-buf 500 --qbuf-cap 500 --emergent-gamma \
  --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.30 \
  --mix-start 0.4 0.6 --mix-end 0.4 0.6 --beta "$WBETA" \
  --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
  --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" --nfe-explore 8 --field-grad-clip 1.0 \
  --max-functional-step 999 --max-anchor-drift 999 --targeted-frac 0.0 --min-modes-per-gamma 0 \
  --recovery-frac 0.0 --hard-quota 0 --m-measure 5 --measure-every 10 --probe-cov 2 --log-comp-every 1 \
  --viz-db-every 1 --ckpt-every 10 --tag "$TAG" > "logs/$TAG.log" 2>&1 &
log "launched $TAG -> it$ITERS PID $!"

deliverables(){ # ckpt tag_iter run_dir iter
  local ck=$1 ti=$2 rd=$3 it=$4 ed="$R/eval_${ti}"
  python $A/make_table.py --ours-dir "$ed" --out-prefix "$OUT/$TAG/table_${ti}" --note "(ours it$it, walled)" >/dev/null 2>&1 || true
  python $A/make_scatter.py --ours-dir "$ed" --out "$OUT/$TAG/scatter_${ti}.png" --note "it$it" >/dev/null 2>&1 || true
  python video_curriculum_fixed.py --run "$rd" --out "$OUT/$TAG/curriculum_${ti}.mp4" \
    --title "Safe Flow Expansion (emergent-gamma, beta=$WBETA, it$it)" >/dev/null 2>&1 || true
  log "$ti deliverables (table/scatter/curriculum) written"
}

# 4) milestone reports every it20; deliverables at it100/it200
mkdir -p "$OUT/$TAG"
for N in 20 40 60 80 100 120 140 160 180 200; do
  [ "$N" -gt "$ITERS" ] && break
  CK="$R/$TAG/ckpt_${N}.pt"
  for _ in $(seq 1 360); do
    [ -f "$CK" ] && break
    pgrep -f "grid_expand_hardtail.*$TAG" >/dev/null || { log "$TAG proc gone before it$N"; break; }
    sleep 20
  done
  [ -f "$CK" ] || continue
  log "it$N: reporting"
  python $A/report_at.py --ckpt "$CK" --tag "${TAG}_it${N}" --M 40 \
    --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --reach 0.15 --concurrency 5 --gpu $GPU \
    > "$OUT/$TAG/scorecard_it${N}.txt" 2>&1
  python $A/per_gamma_valid.py --ckpt "$CK" --M 25 --tag "it$N" >> "$OUT/$TAG/valid_trace.txt" 2>&1 || true
  CUDA_VISIBLE_DEVICES=$GPU python $A/faithful_taxonomy.py --ckpt "$CK" --tag "${TAG}_it${N}" \
    --M 40 --gammas 0.1 0.5 1.0 --reach 0.15 --wall-plugs 8 --start-eps 0.05 \
    --out-dir "$OUT/$TAG" > "$OUT/$TAG/taxonomy_it${N}.txt" 2>&1 || true
  touch "$R/MILE_${TAG}_it${N}"
  [ "$N" -eq 100 ] && deliverables "$CK" "${TAG}_it${N}" "$R/$TAG" "$N"
  [ "$N" -eq 200 ] && deliverables "$CK" "${TAG}_it${N}" "$R/$TAG" "$N"
done
log "$TAG PIPELINE DONE"
touch "$R/AUTO_PIPELINE_DONE"
