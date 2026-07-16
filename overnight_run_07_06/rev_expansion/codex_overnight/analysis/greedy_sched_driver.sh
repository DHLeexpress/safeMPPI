#!/bin/bash
# GREEDY knob-schedule search (user 2026-07-14, task 2): from the pretrained with the it200-winner recipe
# but WALLED demos (--demo-prefix w8d_: walled-scene expert windows teach in-scene clearance + goal-stop),
# probe the effect of beta / batch-mix / gp_buf during it20 blocks and GREEDILY adopt the best knob per
# block ("change its schedule"). Scored vs the WALLED expert (results/expert_gt_walls8) at reach 0.15:
# rank = gammas fully-winning a-d, then pooled clearance (the metric that must cross), then SR.
# Stage 1 arms (it0->20): base | beta.4 | easy-heavy mix .6/.4 | frontier-heavy .2/.8 | gp200.
# Then winner resumes it20->40 with the two runner-up knob variations re-tested around it, etc. (it20
# blocks, --resume-allow-recipe-drift covers beta/mix/gp_buf/qbuf). GPU3 only.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
GPU=${GPU:-3}
R=results/p2; OUT=grand_final_reports_rev/greedy_sched
mkdir -p logs "$OUT"
log(){ echo "$(date '+%H:%M') $*" | tee -a logs/greedy_sched.log; }

# 0) gate on walled demo windows + walled expert rows
log "waiting for w8d demo windows (7 gammas) + walled expert rows"
for _ in $(seq 1 120); do
  n=$(ls ../../dataset/w8d_windows_g*.pt 2>/dev/null | wc -l)
  m=$(ls results/expert_gt_walls8/row_g*.json 2>/dev/null | wc -l)
  [ "$n" -ge 7 ] && [ "$m" -ge 7 ] && break
  sleep 20
done
log "data ready: w8d=$(ls ../../dataset/w8d_windows_g*.pt 2>/dev/null | wc -l) expert_rows=$(ls results/expert_gt_walls8/row_g*.json 2>/dev/null | wc -l)"

launch() { # tag beta mixE mixF gpbuf resume_ckpt start_iter iters(ADDITIONAL)
  local tag=$1 beta=$2 me=$3 mf=$4 gpb=$5 rck=${6:-} sit=${7:-0} its=${8:-20}
  local src="$PRE" extra=()
  if [ -n "$rck" ]; then
    # resume: the train_state is EMBEDDED in ckpt_N.pt (--ckpt doubles as resume source; --iters = MORE its)
    src="$rck"; extra+=(--start-iter "$sit" --resume-allow-recipe-drift)
  fi
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
    --ckpt "$src" --outdir "$R/$tag" --iters "$its" --seed 910 --lr 2e-5 \
    --wall-plugs 8 --start-eps 0.05 --reach 0.2 \
    --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 --gp-buf "$gpb" --qbuf-cap "$gpb" \
    --emergent-gamma --demo-prefix w8d_ \
    --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.30 \
    --mix-start "$me" "$mf" --mix-end "$me" "$mf" --beta "$beta" \
    --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
    --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" --nfe-explore 8 --field-grad-clip 1.0 \
    --max-functional-step 999 --max-anchor-drift 999 --targeted-frac 0.0 --min-modes-per-gamma 0 \
    --recovery-frac 0.0 --hard-quota 0 --m-measure 5 --measure-every 20 --probe-cov 2 --log-comp-every 1 \
    --viz-db-every 1 --ckpt-every 10 --tag "$tag" "${extra[@]}" > "logs/${tag}.log" 2>&1 &
  log "launched $tag (beta=$beta mix=$me/$mf gp=$gpb resume='${rck}' it$sit +$its) PID $!"
}

score() { # tag ckpt -> writes eval + returns "wins clr SR" via stdout
  local tag=$1 ck=$2
  python analysis/report_at.py --ckpt "$ck" --tag "gs_${tag}" --M 30 \
    --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --reach 0.15 --concurrency 5 --gpu $GPU \
    --expert-dir results/expert_gt_walls8 > "$OUT/score_${tag}.txt" 2>&1
  python3 -c "
import json
d=json.load(open('$R/eval_gs_${tag}/scorecard.json'))['pooled']
print(d['gammas_all_ad'], round(d['clr'],4), round(d['SR'],3), d['ad_pass'])"
}

wait_tags() { # tag... : wait until all have final.pt (or procs gone)
  while true; do
    local done=0 t
    for t in "$@"; do [ -f "$R/$t/final.pt" ] && done=$((done+1)); done
    [ "$done" -ge "$#" ] && return 0
    pgrep -f "grid_expand_hardtail.*gs1_\|grid_expand_hardtail.*gs2_\|grid_expand_hardtail.*gs3_" >/dev/null || {
      sleep 6
      pgrep -f "grid_expand_hardtail.*gs" >/dev/null || { log "gs procs gone (done=$done/$#)"; return 1; }
    }
    sleep 30
  done
}

# ---- stage 1: it0 -> 20, 5 arms ----
log "STAGE 1: it0->20 (5 arms)"
launch gs1_base  0.2 0.4 0.6 500
launch gs1_b04   0.4 0.4 0.6 500
launch gs1_easy  0.2 0.6 0.4 500
launch gs1_front 0.2 0.2 0.8 500
launch gs1_gp200 0.2 0.4 0.6 200
wait_tags gs1_base gs1_b04 gs1_easy gs1_front gs1_gp200
best=""; bestkey=""
for t in gs1_base gs1_b04 gs1_easy gs1_front gs1_gp200; do
  [ -f "$R/$t/final.pt" ] || continue
  read wins clr sr adp < <(score "$t" "$R/$t/final.pt")
  log "stage1 $t: fullwin=$wins clr=$clr SR=$sr ad=$adp"
  key=$(python3 -c "print(f'{int($wins):03d}_{float($clr):.4f}_{float($sr):.3f}')")
  if [ -z "$bestkey" ] || [ "$key" \> "$bestkey" ]; then bestkey=$key; best=$t; fi
done
log "STAGE 1 WINNER: $best ($bestkey)"
echo "$best" > "$OUT/stage1_winner.txt"

# ---- stage 2: winner it20 -> 40 with 3 variations around it ----
wb=0.2; we=0.4; wf=0.6; wg=500
case "$best" in gs1_easy) we=0.6; wf=0.4;; gs1_front) we=0.2; wf=0.8;; gs1_gp200) wg=200;; gs1_b04) wb=0.4;; esac
log "STAGE 2: it20->40 around ($wb $we/$wf gp$wg)"
launch gs2_keep  "$wb" "$we" "$wf" "$wg" "$R/$best/ckpt_20.pt" 20 20
launch gs2_bswap "$([ "$wb" = "0.2" ] && echo 0.4 || echo 0.2)" "$we" "$wf" "$wg" "$R/$best/ckpt_20.pt" 20 20
launch gs2_mswap "$wb" "$wf" "$we" "$wg" "$R/$best/ckpt_20.pt" 20 20
wait_tags gs2_keep gs2_bswap gs2_mswap
best2=""; bestkey2=""
for t in gs2_keep gs2_bswap gs2_mswap; do
  [ -f "$R/$t/final.pt" ] || continue
  read wins clr sr adp < <(score "$t" "$R/$t/final.pt")
  log "stage2 $t: fullwin=$wins clr=$clr SR=$sr ad=$adp"
  key=$(python3 -c "print(f'{int($wins):03d}_{float($clr):.4f}_{float($sr):.3f}')")
  if [ -z "$bestkey2" ] || [ "$key" \> "$bestkey2" ]; then bestkey2=$key; best2=$t; fi
done
log "STAGE 2 WINNER: $best2 ($bestkey2)"
echo "$best2" > "$OUT/stage2_winner.txt"

# ---- stage 3: continue winner it40 -> 100 unchanged (schedule locked) ----
# NOTE: keep the STAGE-2 winner's knobs (bswap/mswap may have flipped beta or mix)
case "$best2" in
  gs2_bswap) wb=$([ "$wb" = "0.2" ] && echo 0.4 || echo 0.2);;
  gs2_mswap) tmp=$we; we=$wf; wf=$tmp;;
esac
log "STAGE 3: $best2 it40->100 (locked schedule: beta=$wb mix=$we/$wf gp$wg)"
launch gs3_final "$wb" "$we" "$wf" "$wg" "$R/$best2/ckpt_40.pt" 40 60
wait_tags gs3_final
read wins clr sr adp < <(score gs3_final "$R/gs3_final/final.pt")
log "STAGE 3 (it100): fullwin=$wins clr=$clr SR=$sr ad=$adp"
log "GREEDY SCHED DONE"
touch "$R/GREEDY_SCHED_DONE"
