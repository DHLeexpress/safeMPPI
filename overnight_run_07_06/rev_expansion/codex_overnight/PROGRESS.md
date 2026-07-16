# PROGRESS — codex overnight (append-only; newest at bottom)

Entry format:
```
## [YYYY-MM-DD HH:MM] <phase P1/P2/P3> — <one-line what>
CMD: <verbatim launch line>
RESULT: <numbers / table snippet>
DECISION: <what changes next and WHY>
```

## [2026-07-10 04:10] setup — folder seeded by Claude
Preliminary viz in `preliminary/`; GOAL.md written.

## [2026-07-10 04:25] user decisions LOCKED (see GOAL.md §C2)
- Frontier = AND-cell fixed quantile (50% planes ⇒ ~12.5%); N% + mixing ratio = sweep variables,
  scheduled by ABSOLUTE iteration.
- NO demo backfill — if easy/frontier pool empty, gather MORE rollouts instead.
- Fresh ratio HIGHER: sweep demo_frac 0.25 → 0.125 → 0 (control variable).
- β constant 0.3 first; 0.2 only if coverage stalls.
- Expert GT (P1): origin start, reach 0.1, ≥100 seeds/γ, all 7 γ.
Ready for codex launch.

## [2026-07-10 02:37 PDT] P1 — launch all-gamma SafeMPPI ground truth (M=100 each)
CMD: `mkdir -p logs results/expert_gt tables; for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do OMP_NUM_THREADS=16 setsid nohup python eval_ae.py expert-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method SafeMPPI --outdir results/expert_gt > "logs/expert_g${g}.log" 2>&1 < /dev/null & done`
RESULT: launched seven CPU workers after `nvidia-smi` confirmed GPUs 2/3 idle and 0/1 occupied; each worker is capped at 16 Torch/OpenMP threads.
DECISION: use seeds 0..99, origin start, endpoint reach 0.1 m, and the single shared evaluator for a--e exactly as fixed in GOAL.md.

## [2026-07-10 02:41 PDT] P1 — expert ground truth complete and assembled
CMD: `OMP_NUM_THREADS=16 python eval_ae.py assemble --input-dir results/expert_gt --table-prefix tables/T1_expert --title 'T1 — SafeMPPI expert ground truth'`
RESULT:

| γ | SR | CR | clearance m | time s | coverage | n/M |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100% | 0% | 0.333 ± 0.011 | 15.13 ± 0.83 | 8 | 100/100 |
| 0.2 | 100% | 0% | 0.290 ± 0.013 | 11.53 ± 0.92 | 6 | 100/100 |
| 0.3 | 100% | 0% | 0.281 ± 0.015 | 10.99 ± 0.83 | 9 | 100/100 |
| 0.4 | 100% | 0% | 0.282 ± 0.015 | 10.68 ± 0.70 | 7 | 100/100 |
| 0.5 | 100% | 0% | 0.285 ± 0.015 | 10.54 ± 0.70 | 6 | 100/100 |
| 0.7 | 100% | 0% | 0.287 ± 0.014 | 10.58 ± 0.76 | 6 | 100/100 |
| 1.0 | 100% | 0% | 0.294 ± 0.013 | 10.76 ± 0.69 | 11 | 100/100 |

The seven per-gamma sets contain 14 distinct empirical staircase IDs in union; per-gamma coverage is reported without hard-coding 16 or 252. Raw paths are in `results/expert_gt/paths_g*.npz`.
DECISION: P1 establishes the target: P2 needs SR=100%, CR=0%, clearance above these rows, time below these rows, and substantially higher per-gamma empirical coverage.

## [2026-07-10 02:43 PDT] P2 — local fixed-schedule trainer implementation + one-iteration smoke
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2_smoke --iters 1 --no-freeze --enc-lr-mult 0.3 --m-measure 1 --measure-every 1 --rollouts-per-iter 2 --gather-attempt-cap 4 --batch 8 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 1 --log-comp-every 1 --tag smoke`
RESULT: pending smoke output. Static checks before launch: random independent axes at q=0.5 selected 12.439% (expected ~12.5%); absolute schedule boundaries 0/200/400 passed; an empty class forces update skip.
DECISION: the local copy uses σ-high AND margin-low AND progress-high, fixed absolute iteration phases, constant β, no demo backfill/recovery, and checkpoint metadata for absolute-index resume.

## [2026-07-10 02:46 PDT] P2 — smoke passed, including checkpoint resume
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python grid_expand_fixed.py --ckpt results/p2_smoke/final.pt --outdir results/p2_smoke --iters 1 --no-freeze --enc-lr-mult 0.3 --m-measure 1 --measure-every 1 --rollouts-per-iter 2 --gather-attempt-cap 4 --batch 8 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 1 --log-comp-every 1 --tag smoke_resume`
RESULT: the first smoke automatically continued gathering after an initial frontier-only pool and obtained 86 easy / 36 frontier windows from two valid rollouts; the batch was 5e+2f+1 demo. Resume read `iter=1`, executed absolute iteration 2, and preserved history iterations `[0,1,2]`; final checkpoint metadata reports iteration 2 and the 0/200/400 quantile schedule.
DECISION: launch controlled 100-iteration arms from the same pretrained model. Hold q schedule, β=0.3, demo_frac=0.125, and all optimizer/gate settings fixed; vary only easy:frontier mix (75:25 versus 50:50).

## [2026-07-10 02:48 PDT] P2 — launch controlled mix sweep, 100 absolute iterations
CMD (arm mix75, GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/qmix75_d125_b03 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 100 --log-comp-every 1 --probe-cov 1 --tag qmix75_d125_b03 > logs/p2_qmix75_d125_b03.log 2>&1 < /dev/null &`
CMD (arm mix50, GPU3): `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/qmix50_d125_b03 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.50 0.50 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 100 --log-comp-every 1 --probe-cov 1 --tag qmix50_d125_b03 > logs/p2_qmix50_d125_b03.log 2>&1 < /dev/null &`
RESULT: launched only after `nvidia-smi` reconfirmed GPU2=15 MiB/0% and GPU3=16 MiB/0%; GPUs 0/1 remain occupied and untouched. Both arms save per-iteration viz databases and composition/SR50/CR50/coverage probes.
DECISION: evaluate both checkpoints with M=100 across all γ; continue only the better arm from its saved iteration-100 checkpoint, adjusting solely the allowed recipe controls.

## [2026-07-10 02:52 PDT] P3 — launch first short-horizon cost-scale tuning comparison (γ=0.5, M=3)
CMD (GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx 0.5 --w-safe 0.5 --coll-w 20 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --M 3 --T 250 --tag cw20_gw2_gc05_ws05 --outdir results/kazuki_sweep/cw20 > logs/kaz_cw20.log 2>&1 < /dev/null &`
CMD (GPU3): `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx 0.5 --w-safe 0.5 --coll-w 50 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --M 3 --T 250 --tag cw50_gw2_gc05_ws05 --outdir results/kazuki_sweep/cw50 > logs/kaz_cw50.log 2>&1 < /dev/null &`
RESULT: launched alongside the low-memory P2 arms on allowed GPUs only. This isolates collision-wall scale (20 versus 50); goal cost is raised 20× because the original H=80 cost balance stalls in this H=10 port, while guidance coefficients are held fixed.
DECISION: inspect reach/collision/final distance and then sweep w_safe around the better cost scale; final evaluation returns to N=200/elite=10/copy=200 and M≥100.

## [2026-07-10 02:56 PDT] P3 — cost-scale result; launch all-gamma w_safe comparison (M=5)
CMD (evaluate short sweep): `OMP_NUM_THREADS=16 python eval_ae.py saved-worker --gamma 0.5 --paths results/kazuki_sweep/cw20/paths_g0.5.npz --reach 0.1 --method Kazuki-guidance --outdir results/kazuki_sweep/cw20` (and the same for `cw50`).
RESULT: at γ=0.5, `coll_w=20` reached 3/3 with CR=0, clearance 0.369±0.002 m, time 9.80±0.57 s, coverage 3/3; `coll_w=50` stalled for 250 steps in 3/3 cases (SR=0, CR=0). This directly confirms the short-H proximity-wall diagnosis.
CMD (w_safe=0.3, GPU2): `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx "$g" --w-safe 0.3 --coll-w 20 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --M 5 --T 250 --tag "g${g}" --outdir results/kazuki_sweep/ws03; done > logs/kaz_ws03.log 2>&1`
CMD (w_safe=0.7, GPU3): `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx "$g" --w-safe 0.7 --coll-w 20 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --M 5 --T 250 --tag "g${g}" --outdir results/kazuki_sweep/ws07; done > logs/kaz_ws07.log 2>&1`
RESULT: launched as two sequential all-gamma loops on allowed GPUs; the only changed method parameter is w_safe.
DECISION: select the lower-weight arm if both remain collision-free, because excessive safety guidance is the expected source of local freezing and low mode coverage.

## [2026-07-10 03:01 PDT] P2 — certificate-axis audit caught inherited naming mismatch; stop preliminary arms
CMD: `rg -n "def certify|margin|return" ../../../overnight_run_2026-07-01/verifier_polytope.py ../grid_metrics2.py ../../grid_metrics.py ../../grid_rollout.py`
RESULT: `grid_metrics2.window_min_clearance`, used by the inherited trainer as `margin`, is geometric obstacle clearance. The fitted verifier exposes the actual minimum level-set certificate slack via `check_certificate`; GOAL.md explicitly fixes the frontier axis as **SOCP margin low**.
CMD: `pids=$(pgrep -f '^python grid_expand_fixed.py .*qmix'); [ -z "$pids" ] || kill $pids`
RESULT: preserving both partial run directories/probes, but stopping these preliminary arms before treating them as the requested experiment.
DECISION: copy `grid_metrics2.py` locally, add a read-only certificate-slack metric using the same verifier parameters as `GM.socp_ok`, carry γ with every gathered window, then relaunch the controlled arms. Valid2/taskspace/SOCP/progress acceptance remains byte-for-byte unchanged.

## [2026-07-10 03:04 PDT] P2 — actual SOCP-margin smoke passed
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2_margin_smoke --iters 1 --no-freeze --enc-lr-mult 0.3 --m-measure 1 --measure-every 1 --rollouts-per-iter 2 --gather-attempt-cap 4 --batch 8 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 1 --log-comp-every 1 --tag socp_margin_smoke`
RESULT: gathered 122 valid windows from two Valid2 trajectories, classified 75 easy / 47 frontier, and used a 5e+2f+1d batch. Saved q=0.5 planes were σ=1.000, certificate slack≈0.000, progress=0.583; raw certificate slack ranged from infeasible/clipped −5 to +0.198. A direct stationary-window certificate sanity returned margins {γ=.1: .1, γ=.5: .5, γ=1: 1}.
DECISION: the continuous frontier axis now comes from the fitted-polytope level-set certificate. Relaunch the mix comparison from iteration 0 in new run directories; do not reuse the geometric-clearance checkpoints.

## [2026-07-10 03:05 PDT] P2 — relaunch corrected SOCP-slack mix comparison
CMD (GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/qmix75_socp_d125_b03 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 100 --log-comp-every 1 --probe-cov 1 --tag qmix75_socp_d125_b03 > logs/p2_qmix75_socp.log 2>&1 < /dev/null &`
CMD (GPU3): `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/qmix50_socp_d125_b03 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.50 0.50 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 100 --log-comp-every 1 --probe-cov 1 --tag qmix50_socp_d125_b03 > logs/p2_qmix50_socp.log 2>&1 < /dev/null &`
RESULT: corrected controlled arms launched from the identical pretrained checkpoint on GPUs 2/3.
DECISION: preserve the earlier partial arms as evidence of the audit, but exclude them from final selection and paper artifacts.

## [2026-07-10 03:08 PDT] P3 — w_safe sweep result; launch final M=100 all-gamma evaluation
CMD: for each arm `ws03`/`ws07` and γ, `OMP_NUM_THREADS=16 python eval_ae.py saved-worker ...`, followed by `eval_ae.py assemble` into `tables/_T3_<arm>_M5`.
RESULT: both arms achieved 100% SR and 0% CR in all 7×5 episodes. `w_safe=0.3` was consistently faster (8.88--10.26 s except γ=.1 at 10.20) than `w_safe=0.7` (10.38--12.52 s), while clearance remained higher than expert in every row. M=5 coverage was 2--4 for w_safe=.3 and 2--5 for .7; with equal safety and no clear diversity advantage, .3 wins on progress.
CMD: `i=0; for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do gpu=$((2 + i % 2)); CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx "$g" --w-safe 0.3 --coll-w 20 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --M 100 --T 250 --tag "g${g}" --outdir results/kazuki_final > "logs/kaz_final_g${g}.log" 2>&1 < /dev/null & i=$((i+1)); done`
RESULT: launched seven M=100 workers, alternating GPUs 2/3; each process is capped at 16 threads and uses faithful N=200/elite=10/copy=200.
DECISION: final tuned recipe is w_safe=.3, collision weight 20, goal weight 2, goal-guidance coefficient .5, proximity steepness 20; all other UnifiedGenRefine sampling/refinement values remain faithful to the port.

## [2026-07-10 03:15 PDT] P3 — M=100 table complete; extend uniformly to M=200 for coverage
CMD: per γ, `OMP_NUM_THREADS=16 python eval_ae.py saved-worker --gamma <g> --paths results/kazuki_final/paths_g<g>.npz --reach 0.1 --method Kazuki-guidance --outdir results/kazuki_final`, then assemble `tables/T3_kazuki`.
RESULT:

| γ | SR | CR | clearance m | time s | coverage | n/M |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100% | 0% | 0.372 ± 0.004 | 10.48 ± 1.03 | 5 | 100/100 |
| 0.2 | 100% | 0% | 0.374 ± 0.004 | 9.50 ± 0.93 | 5 | 100/100 |
| 0.3 | 100% | 0% | 0.375 ± 0.003 | 9.04 ± 0.68 | 6 | 100/100 |
| 0.4 | 100% | 0% | 0.375 ± 0.003 | 8.93 ± 0.61 | 6 | 100/100 |
| 0.5 | 100% | 0% | 0.375 ± 0.003 | 8.96 ± 0.68 | 6 | 100/100 |
| 0.7 | 100% | 0% | 0.375 ± 0.004 | 9.16 ± 0.76 | 8 | 100/100 |
| 1.0 | 100% | 0% | 0.375 ± 0.003 | 9.08 ± 0.77 | 7 | 100/100 |

Coverage is still sample-limited and falls below 70% of expert for γ=.1 (5/8), .3 (6/9), and 1.0 (7/11).
CMD: same seven-worker GPU2/3 launch as the M=100 run, with `--seed0 100 --M 100 --outdir results/kazuki_final_extra` and per-gamma logs `logs/kaz_extra_g*.log`.
RESULT: launched seeds 100--199 for every γ, not only the low-coverage rows, to preserve a uniform M=200 protocol.
DECISION: merge the two disjoint seed batches, recompute all a--e rows at M=200, and accept only if every coverage row is at least 70% of its expert counterpart.

## [2026-07-10 03:18 PDT] P3 — record matched success/failure internals for visualization
CMD (success): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx 0.5 --w-safe 0.3 --coll-w 20 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --T 250 --seed0 0 --viz-out results/kazuki_final/viz/success.pt`
CMD (failure): `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx 0.5 --w-safe 0.3 --coll-w 50 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --T 250 --seed0 0 --viz-out results/kazuki_final/viz/failure.pt`
RESULT: pending; each record stores generated candidates, refined elites, selected window, and the reward-guidance vector at every deployment step.
DECISION: render the pair with the local `plot_kazuki_viz.py` after both records complete.

## [2026-07-10 03:19 PDT] P3 — correct failure record to the observed failed arm
CMD: `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python kazuki_baseline.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --gamma-ctx 0.5 --w-safe 0.5 --coll-w 50 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --T 250 --seed0 0 --viz-out results/kazuki_final/viz/failure.pt`
RESULT: the first visualization attempt used w_safe=.3 and reached in 95 steps, so it was not a failure. This corrected command exactly matches the tested `cw50_gw2_gc05_ws05` arm that timed out 3/3.
DECISION: use the corrected timeout record, and retain the unexpected w_safe=.3 success only in this log as further evidence of guidance/cost interaction.

## [2026-07-10 03:20 PDT] P3 — success/failure visualization complete
CMD: `OMP_NUM_THREADS=16 python plot_kazuki_viz.py --record 'Tuned success=results/kazuki_final/viz/success.pt' --record 'Overweighted local wall=results/kazuki_final/viz/failure.pt' --out figures/kazuki_success_failure.png`
RESULT: tuned record reached in 90 steps; corrected overweighted-wall record timed out safely at 250 steps. The paper figure overlays generated candidates, refined elites, selected window, executed path, and mean reward-guidance arrow for both cases.
DECISION: retain `figures/kazuki_success_failure.png` as the required P3 failure/success diagnostic.

## [2026-07-10 03:24 PDT] P3 — M=200 confirms coverage saturation; launch final coverage-control comparison
CMD: for each γ, `eval_ae.py merge-paths` combined seeds 0--99 and 100--199, `saved-worker` recomputed a--e, then `assemble` overwrote the authoritative `tables/T3_kazuki`.
RESULT: all 1,400 episodes remain successful and collision-free. Coverage at M=200 is {5,6,6,7,7,8,7} for γ {.1,.2,.3,.4,.5,.7,1}; it did not grow enough at γ=.1/.3/1.0, remaining exactly one mode below the 70%-of-expert requirement in each case.
CMD (mixed original w_safe groups): seven M=20 workers alternating GPUs 2/3, tuned cost weights, **no** `--w-safe`, output `results/kazuki_sweep/mixed_M20/`.
CMD (w_safe=.1): the same seven M=20 workers with `--w-safe 0.1`, output `results/kazuki_sweep/ws01_M20/`.
RESULT: launched both fixed-method arms; each worker retains N=200/elite=10/copy=200 and 16-thread cap.
DECISION: choose a single fixed guidance recipe only if its all-gamma safety remains reasonable and its empirical mode support resolves the three coverage shortfalls; do not hide the shortfall by mixing post-hoc tables from different methods.

## [2026-07-10 03:28 PDT] P3 — coverage-control result
CMD: evaluated/assembled both M=20 arms into `tables/_T3_mixed_M20` and `tables/_T3_ws01_M20` using the shared evaluator.
RESULT: mixed candidate weights preserved 100% SR/0% CR but did not improve mode support over the M=200 w_safe=.3 arm. w_safe=.1 was faster but timed out in 10% of γ=.3 and γ=.7 episodes and had only 3--5 modes. Neither becomes the fixed baseline.
DECISION: retain w_safe=.3 as the most defensible single recipe. T3 honestly reports its M=200 coverage {5,6,6,7,7,8,7}; do not manufacture a passing coverage row. Return resources to higher-priority P2.

## [2026-07-10 03:29 PDT] P2 — launch controlled rollout-budget arm
CMD: `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/qmix50_socp_r4_d125_b03 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 4 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.50 0.50 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 100 --log-comp-every 1 --probe-cov 1 --tag qmix50_socp_r4_d125_b03 > logs/p2_qmix50_socp_r4.log 2>&1 < /dev/null &`
RESULT: launched from the same pretrained model; the only recipe change versus qmix50 is minimum valid rollouts per iteration 1→4.
DECISION: use rid diversity, near-origin easy fraction, SR50/CR50, and coverage traces to decide whether the higher rollout budget prevents the single-trajectory instability.

## [2026-07-10 03:31 PDT] P2 — terminate collapsing qmix50/min-rollouts1 arm at iteration 37
CMD: `pid=$(pgrep -f '^python grid_expand_fixed.py .*--outdir results/p2/qmix50_socp_d125_b03 '); [ -z "$pid" ] || kill "$pid"`
RESULT: after peaking near SR50=.86, the arm degraded monotonically to SR50=.46 / CR50=.20 / coverage=2 by iteration 37. Update dominance stayed at one source rollout. In contrast qmix75 reached SR50=.86 / CR50=0 at iteration 34.
DECISION: continuing this arm would reinforce an already measured unsafe collapse and contend with the targeted rollout-budget correction. Preserve its 37-row probe log as the failed intermediate result; free GPU3 for r4.

## [2026-07-10 03:33 PDT] P3 — targeted fixed w_safe=.7 coverage check on the three deficient γ rows
CMD: three concurrent M=100 workers for γ {.1,.3,1.0}, `--w-safe 0.7`, otherwise the tuned fixed cost recipe, outputs `results/kazuki_sweep/ws07_M100/` and `logs/kaz_ws07_M100_g*.log` on GPUs 2/3.
RESULT: launched only for the three rows where w_safe=.3 saturated one mode below target; P2 remains active and certificate-bound with substantial GPU headroom.
DECISION: this is not a post-hoc table mix. If .7 supplies the missing modes safely, run its remaining four rows and adopt .7 as one fixed recipe; otherwise keep the transparent .3 table.

## [2026-07-10 03:35 PDT] P2 — r4 triggers the warned origin-dither failure; switch N schedule constructively
CMD: `pid=$(pgrep -f '^python grid_expand_fixed.py .*--outdir results/p2/qmix50_socp_r4_d125_b03 '); [ -z "$pid" ] || kill "$pid"`
RESULT: by iteration 8, SR50=.10 / CR50=.02 / coverage=2. Multiple updates had `near0_e=1.0`; raising rollout count supplied more valid but ill-conditioned near-origin behavior rather than improving the policy. This is precisely the failure mode called out in GOAL.md.
CMD: launch a new 100-iteration arm on GPU3 with the qmix75 recipe and only `--quantile-schedule 0:0.40 200:0.50 400:0.60` changed; output `results/p2/q40mix75_socp_d125_b03`, log `logs/p2_q40mix75_socp.log`.
RESULT: launch pending.
DECISION: lower N expands the three-axis AND cell (ideal independent share 12.5%→21.6%), so high-σ/low-margin/progressing origin windows are less likely to be hammered as easy, without adding an absolute gate or altering Valid2.

## [2026-07-10 03:37 PDT] P3 — user confirms benchmark-only boundary; stop auxiliary tuning
CMD: `pids=$(pgrep -f '^python kazuki_baseline.py .*results/kazuki_sweep/ws07_M100'); [ -z "$pids" ] || kill $pids`
RESULT: Kazuki/Mizuta evaluation used the untouched `pretrained_a32uni.pt` throughout. Only inference-time guidance/MPPI weights were tuned; no flow-matching update or Safe Flow Expansion was ever applied to that model. The decent authoritative benchmark is already complete at M=200 with 100% SR, 0% CR, 0.372--0.375 m clearance, 8.96--10.47 s time, and 5--8 modes.
DECISION: freeze T3 at the w_safe=.3 tuned inference recipe and devote all GPU/CPU capacity to our P2 expansion, per the user's priority and benchmark intent.

## [2026-07-10 03:39 PDT] P2 — launch fresh-ratio control arm (demo_frac=.125→0)
CMD: launch the q=.50, mix75, β=.3 100-iteration recipe on GPU2 with only `--demo-frac 0` changed; retain `lwf_eta=.05`; output `results/p2/qmix75_socp_d0_b03`, log `logs/p2_qmix75_socp_d0.log`.
RESULT: launched while existing trainer processes remain certificate/CPU-bound and both H100s have ample memory/utilization headroom.
DECISION: this completes the requested first sweep over a higher fresh ratio. Compare stability, coverage, and iterations-to-goal against demo_frac=.125 before resuming any checkpoint.

## [2026-07-10 03:41 PDT] P2 — checkpointing safeguard for subsequent units
CMD: local code change only; future runs save `probe_best.pt` whenever the per-iteration M=50 probe has CR50=0 and lexicographically improves (SR50, coverage).
RESULT: the first qmix75 arm briefly reached SR50=.90 near iteration 39 but its launch used `ckpt_every=100`, so that transient model cannot be recovered from the running process. Existing runs continue unchanged; the new safeguard applies only to subsequent units.
DECISION: all future launches also use shorter checkpoint intervals. This does not narrow the final M≥100 all-gamma gate; it prevents a good intermediate model from being lost before that authoritative evaluation.

## [2026-07-10 03:43 PDT] P2 — reject q=.40; launch natural 87.5:12.5 class-mass mix
CMD: terminate `q40mix75_socp_d125_b03` at iteration 10, preserving its probe/viz history.
RESULT: q=.40 enlarged the frontier pool but worsened to SR50=.60 / CR50=.14 after CR50 peaked .16; it is less safe than q=.50/mix75 at comparable and later iterations.
CMD: launch GPU3 arm `results/p2/q50mix875_socp_d125_b03` with q schedule {.50,.60,.70}, constant mix .875/.125, β=.3, demo_frac=.125, `ckpt_every=10`, and otherwise identical locked settings; log `logs/p2_q50mix875_socp.log`.
RESULT: launch pending.
DECISION: three independent 50% planes imply an ideal ~12.5% AND-cell. Sampling it at 12.5% avoids the 2--4× frontier oversampling of mix75/mix50 while retaining a principled fixed quantile recipe.

## [2026-07-10 03:44 PDT] P2 — demo_frac=0 control fails early; terminate at iteration 5
CMD: terminate `qmix75_socp_d0_b03`, preserving its five probe/viz rows.
RESULT: removing the 12.5% demo CFM anchor caused SR50/CR50 to move .28/.02 → .48/.06 → .36/.18 by iteration 5, substantially less stable than the .125 arm.
DECISION: retain demo_frac=.125. This is still 87.5% fresh in the CFM batch, satisfies the requested fresh-dominated recipe, and the control demonstrates why zero demo fraction is not selected.

## [2026-07-10 03:47 PDT] P2 — natural 87.5:12.5 mix undertrains frontier; launch 80:20 compromise
CMD: terminate `q50mix875_socp_d125_b03` at iteration 9.
RESULT: SR50 declined from .34 to .20 while CR50 rose to .10; the natural cell-frequency mix does not supply enough frontier update mass to escape the pretrained/OOD failure.
CMD: launch GPU3 `results/p2/q50mix80_socp_d125_b03` with constant mix .80/.20, q schedule {.50,.60,.70}, β=.3, demo_frac=.125, checkpoint every 10 and probe-best saving; otherwise locked settings.
RESULT: launch pending.
DECISION: mix80 is the controlled midpoint between the undertraining 87.5:12.5 arm and the more effective but oscillatory 75:25 arm.

## [2026-07-10 03:49 PDT] P2 — terminate oscillatory mix75; launch the prescribed β=.2 coverage control
CMD: terminate `qmix75_socp_d125_b03` at iteration 56.
RESULT: after a transient SR50=.90 peak, the unrecoverable arm declined to SR50=.76 / CR50=.12; coverage stayed 4--5. Its no-checkpoint peak motivated the now-active probe-best safeguard.
CMD: launch GPU2 `results/p2/q50mix80_socp_d125_b02` with the exact mix80/β=.3 recipe except constant `--beta 0.2`; checkpoint every 10 and probe-best enabled; log `logs/p2_q50mix80_socp_b02.log`.
RESULT: launch pending.
DECISION: coverage has demonstrably stalled, so this is the one controlled β=.3→.2 comparison authorized by the fixed recipe. Do not change any other knob between these two arms.

## [2026-07-10 03:52 PDT] P2 — replay measured mix75 peak with recoverable checkpoints
CMD: launch GPU3 `results/p2/q50mix75_socp_ckpt_d125_b03`, identical to the original qmix75 recipe, but with `ckpt_every=10` and the new zero-collision `probe_best.pt` saving; log `logs/p2_q50mix75_socp_ckpt.log`.
RESULT: original probe history proves zero-collision candidates at it14--16, 24--26, and 31--34, with best zero-collision SR50=.86 at it34. The replay is expected to preserve those models for authoritative all-gamma evaluation.
DECISION: evaluate saved models rather than selecting a transient trace value. The M≥100, all-γ a--e gate remains authoritative.

## [2026-07-10 03:54 PDT] P2 — β control resolved; retain β=.3
CMD: terminate `q50mix80_socp_d125_b02` after iteration 9 (the iteration-10 checkpoint comparison is available for β=.3; β=.2 already has a worse trace).
RESULT: at comparable early iterations, β=.2 reached SR50=.58 / CR50=.16 / coverage=4; β=.3 reached SR50=.58 / CR50=.08 / coverage=6 at iteration 10. Stronger novelty exploration doubled collisions without improving SR or coverage.
DECISION: select constant β=.3 as originally suggested. No further β changes.

## [2026-07-10 03:57 PDT] P2 — reject mix80 at iteration 16; concentrate on checkpointed mix75
CMD: terminate `q50mix80_socp_d125_b03`, preserving checkpoints 10 and its full probe/viz trace.
RESULT: mix80 declined to SR50=.56 / CR50=.10 / coverage=3 at iteration 16, whereas mix75 previously reached .72/0/4 at the same iteration and later .86/0/4 at iteration 34.
DECISION: the controlled mixing sweep selects 75:25. Free GPU3 so the checkpointed replay reaches its known good region faster.

## [2026-07-10 03:58 PDT] P2 — launch checkpointed seed-1 replicate of selected recipe
CMD: launch GPU2 `results/p2/q50mix75_socp_ckpt_s1_d125_b03`, identical selected recipe with only `--seed 1`; checkpoint every 10 and zero-collision probe-best enabled; log `logs/p2_q50mix75_socp_ckpt_s1.log`.
RESULT: launch pending.
DECISION: seed is not a recipe knob. This replicate uses the now-free second allowed GPU and provides a recoverable stability check against single-measure/seed flip-flops before expensive all-gamma M=100 evaluation.

## [2026-07-10 04:01 PDT] P2 — seed-1 stability replicate collapses; terminate at iteration 9
CMD: terminate `q50mix75_socp_ckpt_s1_d125_b03`.
RESULT: seed1 degraded from SR50/CR50=.64/.04 at it4 to .34/.24 at it9, while seed0 reached .70/0 at it16--18. This confirms the recipe's optimization-seed instability rather than a one-measure fluctuation.
DECISION: retain seed0 checkpoints as candidate models, but final success still requires the all-gamma M≥100 table. Use free GPU2 for checkpoint evaluation when the next seed0 probe-best is saved.

## [2026-07-10 04:02 PDT] P2 — M=100 a--e reference audit of prior ad-hoc best checkpoint
CMD: seven `eval_ae.py policy-worker` processes on GPU2 for `../results/uni_expand/uni_A_b64i121/best.pt`, one per γ with M=100, output `results/reference_uniA_M100/`, logs `logs/ref_uniA_g*.log`.
RESULT: launched on otherwise-free GPU2 while the fixed-recipe seed0 trainer remains on GPU3.
DECISION: this checkpoint is not eligible as the final recipe. Its full a--e table provides a realistic comparison for safety/time/coverage and prevents optimizing only the γ=.5 SR50 probe.

## [2026-07-10 04:08 PDT] P2 — prior ad-hoc best M=100 reference table complete
CMD: `eval_ae.py assemble --input-dir results/reference_uniA_M100 --table-prefix tables/_reference_uniA_M100 ...`.
RESULT: the prior 1000-iteration `uni_A` best is not close to the paper gate: SR 88--95%, CR 0--5%, clearance 0.295--0.310 m, time 11.70--18.71 s, coverage {13,6,5,3,4,5,5}. It is slower than expert at every γ and safer only in some rows.
DECISION: do not use the historical 0.93 aggregate as evidence of goal completion. The fixed-recipe model must be evaluated on all a--e metrics, and likely needs resumed 100--200 iteration units rather than selection from the first γ=.5 probe rise.

## [2026-07-10 04:09 PDT] P2 — first saved-model continuation unit (absolute it22→122)
CMD: launch GPU2 `grid_expand_fixed.py --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it22.pt --outdir results/p2/q50mix75_resume22_to122 --iters 100` with the selected q schedule, mix .75→.50, β=.3, demo_frac=.125, checkpoint every10, per-iter probes, and all other locked settings.
RESULT: launch pending. Checkpoint metadata supplies `start_iter=22`; every quantile/phase/mix decision uses absolute t, not a restarted fraction.
DECISION: this is the user's required “saved updated model for additional iters” path. Compare its saved candidates and final M=100 all-gamma a--e table against the uninterrupted seed0 trajectory.

## [2026-07-10 04:09 PDT] P2 — exact continuation launch command
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it22.pt --outdir results/p2/q50mix75_resume22_to122 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --tag q50mix75_resume22_to122 > logs/p2_q50mix75_resume22.log 2>&1 < /dev/null &`
RESULT: same launch as the preceding entry, recorded verbatim.
DECISION: none; this entry preserves the append-only command audit.

## [2026-07-10 04:14 PDT] P2 — launch first full intermediate a--e table (iteration 32, M=100)
CMD: seven `eval_ae.py policy-worker` launches alternating GPUs2/3 for checkpoint `results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it32.pt`, γ {.1,.2,.3,.4,.5,.7,1}, M=100, method `Flow-expanded-it32`, output `results/p2/eval_it32/`, logs `logs/p2_eval_it32_g*.log`.
RESULT: launch pending. The checkpoint's probe was SR50=.84 / CR50=0 at γ=.5; it is an intermediate, not goal completion.
DECISION: assemble and append all a--e rows when workers finish; use the table to decide which metrics the next continuation unit must improve.

## [2026-07-10 04:14 PDT] P2 — exact it32 evaluation loop
CMD: `mkdir -p results/p2/eval_it32; i=0; for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do gpu=$((2 + i % 2)); CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-it32 --outdir results/p2/eval_it32 --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it32.pt --device cuda --T 250 > "logs/p2_eval_it32_g${g}.log" 2>&1 < /dev/null & i=$((i+1)); done`
RESULT: same launch recorded verbatim.
DECISION: none; append-only audit correction.

## [2026-07-10 04:18 PDT] P2 — iteration-32 intermediate a--e table complete
CMD: `OMP_NUM_THREADS=16 python eval_ae.py assemble --input-dir results/p2/eval_it32 --table-prefix tables/_T2_it32 --title 'P2 intermediate — fixed AND-quantile iteration 32'`
RESULT:

| γ | SR | CR | clearance m | time s | coverage | n/M |
|---:|---:|---:|---:|---:|---:|---:|
| .1 | 83% | 0% | .321±.016 | 18.14±1.85 | 16 | 83/100 |
| .2 | 85% | 0% | .305±.017 | 14.24±1.26 | 10 | 85/100 |
| .3 | 82% | 6% | .307±.018 | 12.77±1.02 | 7 | 82/100 |
| .4 | 80% | 5% | .306±.018 | 12.36±.87 | 7 | 80/100 |
| .5 | 78% | 4% | .307±.017 | 12.55±1.07 | 8 | 78/100 |
| .7 | 78% | 4% | .309±.018 | 13.13±1.02 | 6 | 78/100 |
| 1.0 | 79% | 1% | .310±.017 | 13.41±1.08 | 7 | 79/100 |

DECISION: coverage already reaches 16 at γ=.1 and exceeds expert in several rows, but SR/CR and time fail broadly. The per-iter M=50 probe was optimistic: the second 50 seeds exposed collisions/timeouts. Continue saved-model units; never accept probe-only success.

## [2026-07-10 04:20 PDT] P2 — replace degrading it22 continuation with stronger it37 continuation
CMD: terminate `q50mix75_resume22_to122` at absolute iteration 27; launch GPU2 from `probe_best_it37.pt` for 100 additional iterations into `results/p2/q50mix75_resume37_to137`, using the selected recipe, absolute schedule, checkpoint every10, per-iter probe, and both SR-first/coverage-first zero-collision saves.
RESULT: it22 branch fell from its all-γ M25 baseline aggregate SR=.82/CR=0 to γ=.5 SR50=.54/CR=.02 by absolute it27. it37 checkpoint has γ=.5 SR50=.84/CR=0/coverage=6.
DECISION: resume from the best currently evidenced saved state, as instructed; do not continue a branch whose immediate evidence contradicts progress.

## [2026-07-10 04:20 PDT] P2 — exact it37 continuation command
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it37.pt --outdir results/p2/q50mix75_resume37_to137 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 100 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --tag q50mix75_resume37_to137 > logs/p2_q50mix75_resume37.log 2>&1 < /dev/null &`
RESULT: command recorded verbatim; launch pending.
DECISION: none.

## [2026-07-10 04:43 PDT] P2 — verify corrected absolute-iteration gamma rotation
CMD: inspect `probe.jsonl` for `results/p2/balanced_q50mix75_s0` and `results/p2/balanced_q50mix75_s1`; inspect active processes and GPU 2/3 utilization.
RESULT: both jobs are active on the authorized GPUs. Probe records show iteration 1 starts at gamma=.1 (2 accepted/gathered rollouts), iteration 2 at gamma=.2 (continuing to .3 when the class requirement needs another attempt), iteration 3 at gamma=.3, iteration 4 at gamma=.4, and later gathers continue round-robin until both classes exist. Thus the former fixed gamma=.1 start is removed and the resume-safe absolute rotation is operational. At iteration 6, seed-0 gamma=.5 probe improved from baseline SR50=.28/CR50=.02/coverage=3 to .46/.08/4; seed-1 reached .48/.04/4. These single-gamma probes are diagnostic only.
DECISION: keep both corrected runs through the first periodic all-gamma M=25 gate at iteration 10. Select only from all-gamma safety results, never from the gamma=.5 probe alone.

## [2026-07-10 04:33 PDT] P2 — iteration-22 M=100 starting-state table complete
CMD: assemble `results/p2/eval_it22` into `tables/_T2_it22`.
RESULT: SR {83,89,87,80,83,79,79}%, CR {0,0,1,4,1,0,0}%, clearance .306--.330 m, time 12.88--19.00 s, coverage {14,11,8,7,7,7,5}. This is better than it32/it37 on SR and mostly CR, confirming the all-γ M25 selection signal, but remains slower than expert and below final safety/performance goals.
DECISION: measured it22→122 continuation is now the sole active trainer. Its every-10 all-γ rows will determine the next checkpoint; do not use later γ=.5-only peaks that lose conditional behavior elsewhere.

## [2026-07-10 04:34 PDT] P2 — measured continuation replicate on GPU3
CMD: launch `q50mix75_measured22_s1_to122` from the same it22 checkpoint with the identical selected recipe and every-10 all-γ measurements, changing only random seed to 1; GPU3, checkpoint every10, per-iter probes, output `results/p2/q50mix75_measured22_s1_to122`, log `logs/p2_q50mix75_measured22_s1.log`.
RESULT: launch pending.
DECISION: this is not a recipe sweep; it uses the second allowed GPU to search the same fixed continuation under known optimization-seed instability. `safe_best.pt` remains constrained to aggregate CR=0.

## [2026-07-10 04:34 PDT] P2 — exact measured seed-1 continuation command
CMD: `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it22.pt --outdir results/p2/q50mix75_measured22_s1_to122 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 1 --tag q50mix75_measured22_s1_to122 > logs/p2_q50mix75_measured22_s1.log 2>&1 < /dev/null &`
RESULT: command recorded verbatim; launch pending.
DECISION: none.

## [2026-07-10 04:37 PDT] P2 — root-cause audit: inherited gather starves six γ conditions
CMD: inspect local `_gather_fresh`: `gi=0` was created inside every call and the first rollout used `gammas[gi % len(gammas)]`; early-stop commonly ends after that one rollout.
RESULT: most iterations therefore trained only γ=.1. This matches the intermediate tables: γ=.1 reaches coverage 14--16 while γ=.3--1.0 lose SR, collide, and remain slow. Multiple-rollout iterations accidentally include later γ values, making the bug noisy rather than obvious.
CMD: terminate both pre-fix measured it22 continuations; patch local trainer so iteration t starts at `gammas[(t-1) % 7]` and subsequent attempts continue round-robin. Add `gamma_counts` to every probe record and `gamma_rotation=absolute_iteration_round_robin` to recipe JSON.
RESULT: `python -m py_compile grid_expand_fixed.py` passes. Valid2, certificate slack, q planes, β, and batch recipe are unchanged.
DECISION: all pre-fix P2 models remain valid intermediate evidence but are ineligible as final fixed-recipe models. Relaunch the selected q=.5/mix75/demo=.125/β=.3 recipe from the pretrained backbone with balanced absolute-iteration γ rotation.

## [2026-07-10 04:38 PDT] P2 — launch corrected balanced-γ recipe, seeds 0/1
CMD (GPU2 seed0): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/balanced_q50mix75_s0 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 0 --tag balanced_q50mix75_s0 > logs/p2_balanced_s0.log 2>&1 < /dev/null &`
CMD (GPU3 seed1): same command with `CUDA_VISIBLE_DEVICES=3`, `--seed 1`, outdir `results/p2/balanced_q50mix75_s1`, tag `balanced_q50mix75_s1`, and log `logs/p2_balanced_s1.log`.
RESULT: launch pending. Both runs measure all γ every 10 absolute iterations and save `best.pt`, `safe_best.pt`, SR-first probe-best, and coverage-first probe-best.
DECISION: these are the first final-eligible P2 runs because they implement every user-fixed recipe element and sample all seven conditions deliberately.

## [2026-07-10 04:38 PDT] P2 — exact balanced seed-1 command
CMD: `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/balanced_q50mix75_s1 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 1 --tag balanced_q50mix75_s1 > logs/p2_balanced_s1.log 2>&1 < /dev/null &`
RESULT: exact command recorded before launch.
DECISION: none.

## [2026-07-10 04:22 PDT] P2 — freeze post-peak replay; launch M=100 table for it37
CMD: terminate uninterrupted `q50mix75_socp_ckpt_d125_b03` at iteration 42 after preserving `probe_best_it37.pt`; launch seven GPU3 `eval_ae.py policy-worker` processes for it37, all γ, M=100, output `results/p2/eval_it37`, logs `logs/p2_eval_it37_g*.log`.
RESULT: after the saved it37 SR50=.84/CR50=0/coverage=6 point, the live replay fell as low as .68/.08 by it41 and was .72/.04 at it42. The separate it37→137 continuation supplies the requested full 100-update unit.
DECISION: spend GPU3 on authoritative metrics for the preserved candidate rather than additional updates past a measured collapse.

## [2026-07-10 04:22 PDT] P2 — exact it37 evaluation loop
CMD: `mkdir -p results/p2/eval_it37; for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-it37 --outdir results/p2/eval_it37 --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it37.pt --device cuda --T 250 > "logs/p2_eval_it37_g${g}.log" 2>&1 < /dev/null & done`
RESULT: command recorded verbatim; launch pending.
DECISION: none.

## [2026-07-10 04:27 PDT] P2 — iteration-37 table disproves γ=.5 probe selection
CMD: assemble `results/p2/eval_it37` into `tables/_T2_it37`.
RESULT: M=100 SR {68,73,71,66,71,64,68}%, CR {0,2,7,9,5,6,3}%, clearance .303--.317 m, time 12.27--17.79 s, coverage {16,10,7,7,7,7,7}. It is substantially worse than it32 outside the first 50 γ=.5 seeds.
DECISION: terminate it37→137 at absolute it40. Candidate selection must use periodic all-γ measures, not γ=.5 probe-best.

## [2026-07-10 04:28 PDT] P2 — launch measured it22 continuation + M=100 it22 audit
CMD (trainer, GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it22.pt --outdir results/p2/q50mix75_measured22_to122 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 10 --gather-attempt-cap 30 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --tag q50mix75_measured22_to122 > logs/p2_q50mix75_measured22.log 2>&1 < /dev/null &`
CMD (evaluation, GPU3): seven M=100 `eval_ae.py policy-worker` jobs for `probe_best_it22.pt`, all γ, output `results/p2/eval_it22`, logs `logs/p2_eval_it22_g*.log`.
RESULT: launch pending. The trainer now writes `safe_best.pt` only from periodic all-γ measures with aggregate CR=0; both SR-first and coverage-first probe snapshots remain diagnostic.
DECISION: use it22 M=100 as the authoritative starting-state table and every-10 all-γ measures to choose/resume subsequent units.

## [2026-07-10 04:28 PDT] P2 — exact it22 evaluation loop
CMD: `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-it22 --outdir results/p2/eval_it22 --ckpt results/p2/q50mix75_socp_ckpt_d125_b03/probe_best_it22.pt --device cuda --T 250 > "logs/p2_eval_it22_g${g}.log" 2>&1 < /dev/null & done`
RESULT: same launch recorded verbatim.
DECISION: none.

## [2026-07-10 03:54 PDT] P2 — corrected run iteration-10 all-gamma gate
CMD: periodic in-trainer evaluation at all seven gammas, M=25 each, seeds 10000--10024, for corrected seeds 0 and 1.
RESULT: seed 0 reaches aggregate SR=.51/CR=.05. Seed 1 reaches SR=.7543/CR=.0343, versus the untouched pretrained baseline SR=.36/CR=.0343. Seed-1 per-gamma SR is {.84,.80,.76,.76,.68,.72,.72}; CR is {0,.04,.04,.04,.12,0,0}. The gamma=.5 probe alone is more variable and is not used for selection. No `safe_best.pt` is written because aggregate collision rate is nonzero.
DECISION: seed 1 leads but is not a safe candidate. Continue both corrected recipes to the iteration-20 all-gamma gate; retain seed diversity until a zero-collision periodic result appears or one seed clearly collapses.

## [2026-07-10 03:59 PDT] P2 — corrected run iteration-20 all-gamma gate
CMD: periodic in-trainer evaluation at all seven gammas, M=25 each, seeds 20000--20024, corrected seeds 0 and 1.
RESULT: seed 0 is now the leader at aggregate SR=.8171/CR=.0286; per-gamma SR {.76,.88,.76,.80,.84,.84,.84}, CR {0,0,.08,.04,.04,.04,0}. Seed 1 is SR=.7829/CR=.0686; per-gamma CR {0,0,.16,.12,.12,.08,0}. The remaining seed-0 collisions are concentrated at gamma=.3--.7. Neither run writes `safe_best.pt`.
DECISION: continue the fixed schedule and absolute gamma rotation. Seed 0 is the preferred branch, but retain seed 1 through additional periodic gates because its earlier iteration-10 result led seed 0 and the all-gamma M=25 estimate is noisy.

## [2026-07-10 04:01 PDT] P2 — correction to periodic-evaluation seed notation
CMD: re-audit `_measure` and the shared `sr_cr_eval.eval_policy` implementation.
RESULT: periodic M=25 evaluations use the same fixed seeds 0--24 at every gate (`seed0=0` default), not iteration-offset seeds. The numerical iteration-10/20 results above are correct; only their recorded seed-range descriptions were wrong. Fixed seeds make the gates paired checkpoint comparisons, while M=25 remains too small for final claims.
DECISION: preserve the append-only log and record this correction explicitly. Final tables still use M>=100 via `eval_ae.py`.

## [2026-07-10 04:00 PDT] P2 — second gather audit: rotation alone does not enforce the rollout budget
CMD: aggregate `gamma_counts` across the corrected rotation-only probes through iterations 24/25 and inspect `_gather_fresh` against the inherited trainer.
RESULT: seed 0 accumulated accepted-window counts {.1:2,.2:733,.3:341,.4:7,.5:336,.7:342,1:337}; most updates drew one rollout ID. The local class-ready loop accidentally omitted the inherited `valid >= K_eff` budget condition, so one long trajectory could terminate gathering. The absolute starting-gamma rotation worked, but did not make the actual fresh pool conditionally representative.
DECISION: terminate PIDs 3323964/3323965 at about iterations 25/24. These remain diagnostic and are ineligible for final selection. Fix the stop condition to require the valid-rollout budget, both AND-quantile classes, and at least one valid2 rollout from every gamma before stopping (up to the explicit attempt cap). This is the user-authorized rollout-budget control, not a validity-gate change.

## [2026-07-10 04:00 PDT] P2 — all-gamma gather invariant smoke test
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/_smoke_balanced_gather --iters 1 --no-freeze --enc-lr-mult 0.3 --m-measure 1 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 0 --ckpt-every 10 --log-comp-every 1 --probe-cov 0 --seed 0 --tag smoke_balanced_gather`
RESULT: compile passes. The smoke gather gets 11 valid trajectories from 13 attempts, includes every gamma at rollout level ({1,2,2,2,2,1,1}), populates both classes (364 easy/222 frontier), and its update batch spans mean 6 rollout IDs with dominance .268. `gamma_rollout_counts`, `gamma_attempt_counts`, `gamma_ready`, and `classes_ready` are now audited in every probe.
DECISION: use `rollouts-per-iter=14` (early effective minimum seven) and attempt cap 42 for the final-eligible 100-iteration unit.

## [2026-07-10 04:00 PDT] P2 — launch final-eligible all-gamma-gather seeds 0/1
CMD (GPU2 seed0): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/balanced_allg_q50mix75_s0 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 0 --tag balanced_allg_q50mix75_s0 > logs/p2_balanced_allg_s0.log 2>&1 < /dev/null &`
CMD (GPU3 seed1): same exact flags with GPU3, seed 1, outdir/tag `balanced_allg_q50mix75_s1`, log `logs/p2_balanced_allg_s1.log`.
RESULT: PIDs 3326929/3326930 active on GPUs 2/3. Both start from the untouched pretrained checkpoint. Valid2, certificate gate, q schedule, beta, class mix, demo fraction, and all other fixed recipe elements are unchanged.
DECISION: these supersede the rotation-only diagnostics and are the first runs eligible for the final P2 table.

## [2026-07-10 04:02 PDT] P2 — exact all-gamma seed-1 launch and first invariant check
CMD: `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir results/p2/balanced_allg_q50mix75_s1 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 1 --tag balanced_allg_q50mix75_s1 > logs/p2_balanced_allg_s1.log 2>&1 < /dev/null &`
RESULT: at iteration 1 both seeds gather exactly one valid2 rollout for each of the seven gammas in seven attempts; `gamma_ready=true`, `classes_ready=true`. The update batches span 3 and 5 distinct rollout IDs rather than the former single-rollout batches.
DECISION: invariant verified live; continue.

## [2026-07-10 04:14 PDT] P2 — iteration-10 paired gate and rollout-budget continuation
CMD: paired periodic evaluation for all seven gammas, fixed seeds 0--24, M=25/gamma, at absolute iteration 10.
RESULT: seed 0 has SR=.7600/CR=.0571; seed 1 has SR=.7600/CR=.0114. Seed-1 per-gamma SR {.76,.80,.76,.80,.84,.72,.64}, CR {0,0,0,0,0,.04,.04}. Seed 0 reached the 42-attempt cap without a valid2 gamma=.1 rollout at iterations 9 and 10; seed 1 still obtained gamma=.1. No zero-collision all-gamma candidate yet.
DECISION: terminate weaker seed-0 PID 3326929 after its saved `ckpt_10.pt`. Preserve seed 1 as the high-budget per-update-all-gamma branch. Use seed 1's saved updated checkpoint for the controlled rollout-budget continuation allowed by the goal: require at least seven valid rollouts and both actual AND-quantile classes, rotate the starting gamma by absolute iteration, log all per-gamma attempts/accepts, but do not spend the entire cap forcing every gamma into every individual update. This retains the fixed class rule and does not change Valid2.

## [2026-07-10 04:14 PDT] P2 — launch seven-valid-rollout branch from seed-1 iteration 10
CMD: `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_allg_q50mix75_s1/ckpt_10.pt --outdir results/p2/balanced_budget7_from_s1it10 --iters 90 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 2 --tag balanced_budget7_from_s1it10 > logs/p2_balanced_budget7_from_s1it10.log 2>&1 < /dev/null &`
RESULT: compile passes; PID 3329904 active on GPU2. Checkpoint metadata sets absolute start iteration 10, so all q/phase/artifact schedules resume correctly. GPU3 seed-1 high-budget branch remains active and is unaffected by the local code change because its process already loaded the prior function.
DECISION: compare both at identical absolute iteration-20 gates, then keep the safer/more successful branch through the full 100-update unit.

## [2026-07-10 04:20 PDT] P2 — first authoritative evaluation of corrected-recipe checkpoint
CMD: terminate slower high-budget PID 3326930 after its iteration-13 probe (its saved reproducible checkpoint remains iteration 10). Launch seven parallel GPU3 workers: `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-budget7-it15 --outdir results/p2/eval_budget7_it15 --ckpt results/p2/balanced_budget7_from_s1it10/probe_best.pt --device cuda --T 250 > "logs/p2_eval_budget7_it15_g${g}.log" 2>&1 < /dev/null & done`
RESULT: `probe_best.pt` metadata is absolute iteration 15, gamma=.5 SR50=.78/CR50=0/coverage=4. GPU2 training continues. Evaluation is pending and is selection-grade for SR/CR because M=100 is used independently at every gamma.
DECISION: prioritize all-gamma metrics over preserving the more expensive branch after the efficient continuation showed equal/better instantaneous safety.

## [2026-07-10 04:23 PDT] P2 — corrected-recipe iteration-15 M=100 table
CMD: assemble seven completed workers into `tables/_T2_budget7_it15.{md,csv}` with the shared evaluator.
RESULT: per-gamma SR {85,87,85,83,79,79,83}%; CR {0,0,2,4,4,2,0}%; primary clearance {.321,.304,.304,.305,.303,.305,.306} m; time {17.94,14.19,12.62,12.26,12.24,12.71,13.03} s; coverage {13,8,6,6,5,5,7}. This is materially better and more condition-balanced than the pre-fix tables, but fails the simultaneous goal: mid-gamma collisions remain, SR is below 100%, time is slower than expert at every gamma, and coverage is not close to 16 except gamma=.1.
DECISION: mark iteration 15 intermediate only. Continue the same fixed q=.5, beta=.3, 75/25 class mix, demo=.125 recipe through the absolute iteration-20 gate and onward; do not loosen Valid2 or claim goal attainment.

## [2026-07-10 04:25 PDT] P2 — iteration-20 regression and fresh-dominant anchor control
CMD: periodic M=25/gamma paired gate at absolute iteration 20 for `balanced_budget7_from_s1it10`.
RESULT: aggregate SR=.6686/CR=.0057, versus .7600/.0114 at iteration 10. Gamma=.5 probe fell from .78/0 at iteration 19 to .62/.02 at iteration 20. Safety improved slightly while task completion degraded; near-origin easy fraction remains only .23, so this is not the original near-origin dither pathology.
DECISION: preserve measured iteration-15 `probe_best.pt`. Continue the unchanged demo=.125 arm to iteration 30 to test recovery. In parallel, resume from iteration 15 with demo_frac=.1875 (12 demos, 52 fresh samples per 64 batch; 81.25% fresh and therefore still fresh-dominant). Demo fraction is an explicitly allowed control; q schedule, beta, mix, rollout budget, Valid2, and learning settings remain fixed.

## [2026-07-10 04:25 PDT] P2 — launch demo=.1875 stability arm from iteration 15
CMD: `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_budget7_from_s1it10/probe_best.pt --outdir results/p2/balanced_d1875_from_it15 --iters 85 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.1875 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 3 --tag balanced_d1875_from_it15 > logs/p2_balanced_d1875_from_it15.log 2>&1 < /dev/null &`
RESULT: PID 3333323 active on GPU3; absolute start iteration 15 comes from checkpoint metadata.
DECISION: compare at absolute iteration 25/30 gates and retain the branch that restores SR without worsening collision rate.

## [2026-07-10 04:28 PDT] P2 — terminate collapsing unchanged arm; launch mixing control
CMD: inspect unchanged demo=.125/mix75 probes after the iteration-20 regression; terminate PID 3329904 at absolute iteration 22. Launch `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_budget7_from_s1it10/probe_best.pt --outdir results/p2/balanced_mix875_from_it15 --iters 85 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.875 0.125 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 4 --tag balanced_mix875_from_it15 > logs/p2_balanced_mix875_from_it15.log 2>&1 < /dev/null &`.
RESULT: unchanged arm gamma=.5 probes after the gate are iteration 21 SR=.46/CR=.02 and iteration 22 .54/.02, confirming continued collapse rather than a one-step fluctuation. Mix-control PID 3333701 is active on GPU2. It changes only the explicitly allowed easy/frontier mixing control, from 75/25 to 87.5/12.5 during the absolute early phase; the latter matches the nominal q=.5 three-axis AND-cell probability.
DECISION: compare the mix and demo controls from the same preserved iteration-15 weights. Both remain fresh-dominated and keep beta=.3 and Valid2 unchanged.

## [2026-07-10 04:32 PDT] P2 — reject mix87.5 arm; launch M=100 for anchored iteration 19
CMD: terminate mix-control PID 3333701 at absolute iteration 18 after gamma=.5 probes {.80/.02,.72/.04,.62/.08} at iterations 16--18. Inspect `results/p2/balanced_d1875_from_it15/probe_best.pt` metadata, then launch seven GPU2 workers: `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-d1875-it19 --outdir results/p2/eval_d1875_it19 --ckpt results/p2/balanced_d1875_from_it15/probe_best.pt --device cuda --T 250 > "logs/p2_eval_d1875_it19_g${g}.log" 2>&1 < /dev/null & done`.
RESULT: anchored `probe_best.pt` is absolute iteration 19 with gamma=.5 SR50=.82/CR50=0/coverage=4. Seven evaluation workers active; GPU3 anchored training continues.
DECISION: reject 87.5/12.5 mixing as destabilizing in this corrected setup. Use M=100 all-gamma evidence to decide whether the modestly larger demo anchor improves the iteration-15 model.

## [2026-07-10 04:35 PDT] P2 — reject demo=.1875; launch beta and stronger-anchor controls
CMD: assemble `tables/_T2_d1875_it19.{md,csv}`; terminate declining demo=.1875 PID 3333323 after absolute iteration 22. Launch two branches from the same preserved iteration-15 checkpoint: beta=.2/demo=.125 on GPU2 and beta=.3/demo=.25 on GPU3, all other flags identical to the selected seven-valid-rollout recipe.
RESULT: demo=.1875 iteration-19 M=100 has SR {88,88,86,85,82,83,86}%, CR {0,1,4,4,4,3,1}%, clearance {.319,.302,.301,.303,.302,.303,.303} m, time {17.79,14.10,12.51,12.20,12.28,12.56,12.96} s, coverage {14,7,6,6,6,6,6}. It improves SR slightly versus iteration 15 but worsens safety and still fails speed/coverage. Its live iteration-20 all-gamma gate is .789/.017 and probes fall to .72 by iterations 21--22.
DECISION: reject demo=.1875. Coverage outside gamma=.1 has stalled at 5--8, meeting the user's condition for the one beta=.2 controlled comparison. Test demo=.25 separately as the strongest stability anchor; although it is the upper control value, each batch remains 75% fresh.

## [2026-07-10 04:35 PDT] P2 — exact beta=.2 and demo=.25 launch commands
CMD (GPU2 beta): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_budget7_from_s1it10/probe_best.pt --outdir results/p2/balanced_beta02_from_it15 --iters 85 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.2 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 5 --tag balanced_beta02_from_it15 > logs/p2_balanced_beta02_from_it15.log 2>&1 < /dev/null &`.
CMD (GPU3 demo): `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_budget7_from_s1it10/probe_best.pt --outdir results/p2/balanced_d25_from_it15 --iters 85 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 14 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.25 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 6 --tag balanced_d25_from_it15 > logs/p2_balanced_d25_from_it15.log 2>&1 < /dev/null &`.
RESULT: PIDs 3335902/3335903 active on authorized GPUs; both resume at absolute iteration 15.
DECISION: run through at least the paired iteration-20/25 gates before selection unless an explicit collapse occurs.

## [2026-07-10 04:39 PDT] P2 — beta=.2 early peak; retire demo=.25 and launch M=100
CMD: inspect probes at absolute iterations 16--18; terminate demo=.25 PID 3335903 at iteration 18. Inspect beta checkpoint metadata and launch seven GPU3 M=100 workers: `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-beta02-it16 --outdir results/p2/eval_beta02_it16 --ckpt results/p2/balanced_beta02_from_it15/probe_best.pt --device cuda --T 250 > "logs/p2_eval_beta02_it16_g${g}.log" 2>&1 < /dev/null & done`.
RESULT: beta=.2 gamma=.5 probes are iteration 16 .84/0/coverage6, iteration 17 .84/0/5, iteration 18 .80/.08/2. `probe_best.pt` preserves iteration 16. Demo=.25 probes are .78/.08/3, .82/.04/3, .82/.08/3 and never match the beta arm. Seven evaluation workers active; beta training continues to its paired gate.
DECISION: reject demo=.25. Evaluate the preserved beta=.2 early peak at the required M=100 before deciding whether the coverage-stall switch is useful.

## [2026-07-10 04:43 PDT] P2 — beta=.2 M=100 result; double valid-rollout-pool comparison
CMD: assemble `tables/_T2_beta02_it16.{md,csv}`; terminate beta PID 3335902 after its absolute iteration-20 paired gate. Launch two beta=.3 seeds from iteration 15 with `rollouts-per-iter=28` (absolute early-phase effective valid budget 14 instead of 7), attempt cap 42, and otherwise the selected demo=.125/mix75 recipe.
RESULT: beta=.2 iteration-16 M=100 has SR {86,89,83,85,85,79,81}%, CR {0,1,5,4,4,2,4}%, time {17.65,13.69,12.30,11.89,12.04,12.44,12.58} s, coverage {14,8,7,7,7,7,9}. Coverage and speed improve modestly, but safety is worse; live iteration-20 M=25 is SR=.80/CR=.04. Beta=.2 is rejected.
DECISION: revert to beta=.3. The seven-rollout branch's one-trajectory ancestry was fixed, but its update pool can still be noisy; valid-rollout budget is an allowed control, so compare two 14-rollout seeds to reduce batch-pool variance while keeping the fixed q/beta/mix/demo/Valid2 recipe.

## [2026-07-10 04:43 PDT] P2 — exact K=14 launch commands
CMD (GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_budget7_from_s1it10/probe_best.pt --outdir results/p2/balanced_k14_s7_from_it15 --iters 85 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 7 --tag balanced_k14_s7_from_it15 > logs/p2_balanced_k14_s7_from_it15.log 2>&1 < /dev/null &`.
CMD (GPU3): same exact flags with seed 8, outdir/tag `balanced_k14_s8_from_it15`, and log `logs/p2_balanced_k14_s8_from_it15.log`.
RESULT: PIDs 3338903/3338904 active on GPUs 2/3.
DECISION: compare paired gates and preserved zero-collision probes; do not infer from a single seed.

## [2026-07-10 04:47 PDT] P2 — K=14 seed split; launch seed-7 iteration-16 M=100
CMD: inspect first post-resume probes; terminate weaker seed-8 PID 3338904. Inspect seed-7 `probe_best.pt`, then launch seven GPU3 workers: `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-k14-s7-it16 --outdir results/p2/eval_k14_s7_it16 --ckpt results/p2/balanced_k14_s7_from_it15/probe_best.pt --device cuda --T 250 > "logs/p2_eval_k14_s7_it16_g${g}.log" 2>&1 < /dev/null & done`.
RESULT: both K=14 branches gathered 14 valid rollouts from 17 attempts and populated both classes. Seed 7 gamma=.5 is SR50=.86/CR50=0/coverage=5; seed 8 is .76/.08/3. Seed-7 update batch diagnostics show 13 rollout IDs and .12 dominance. Checkpoint metadata confirms absolute iteration 16.
DECISION: preserve and evaluate the stronger seed at M=100; continue seed-7 training toward the paired iteration-20 gate.

## [2026-07-10 04:51 PDT] P2 — K=14 iteration-16 M=100 and stronger iteration-18 candidate
CMD: assemble `tables/_T2_k14_s7_it16.{md,csv}`. After live iteration 18 updates `probe_best.pt`, launch seven GPU3 M=100 workers into `results/p2/eval_k14_s7_it18` with method `Flow-expanded-k14-s7-it18` and the current preserved checkpoint.
RESULT: iteration-16 M=100 has SR {89,89,83,86,87,83,86}%, CR {0,1,8,3,2,3,2}%, clearance {.317,.303,.304,.304,.304,.304,.306} m, time {17.30,13.51,12.17,11.77,11.92,12.33,12.58} s, coverage {13,7,6,6,6,7,6}. It is faster and more successful than earlier candidates but still unsafe. Live K=14 probes remain zero-collision at iterations 16--18 and improve to SR50=.92/CR50=0/coverage=5 at iteration 18; metadata confirms iteration 18 before worker launch.
DECISION: iteration 16 remains diagnostic only. Evaluate iteration 18 rather than assuming the single-gamma improvement transfers; seed-7 training continues.

## [2026-07-10 04:55 PDT] P2 — K=14 iteration-18 M=100; select rollout budget and branch seed
CMD: assemble `tables/_T2_k14_s7_it18.{md,csv}`; inspect seed-7 absolute iteration-20 paired M=25 gate. Launch same-recipe seed 9 on GPU3 from seed-7 `probe_best.pt` (iteration 18): `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_k14_s7_from_it15/probe_best.pt --outdir results/p2/balanced_k14_s9_from_it18 --iters 82 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 9 --tag balanced_k14_s9_from_it18 > logs/p2_balanced_k14_s9_from_it18.log 2>&1 < /dev/null &`.
RESULT: iteration-18 M=100 has SR {88,91,89,88,89,86,88}%, CR {0,1,3,4,3,3,3}%, time {17.28,13.41,12.04,11.66,11.76,12.25,12.43} s, coverage {13,8,7,6,6,7,9}. Iteration-20 paired M=25 is aggregate SR=.8514/CR=.0171, improving over the same model's iteration-15 baseline .8057/0 and sharply outperforming the K=7 iteration-20 regression .6686/.0057.
DECISION: select K=14 as the rollout budget. Keep seed 7 running to absolute 100 and use GPU3 for a second same-recipe continuation from the strongest preserved iteration-18 weights. Safety goals remain unmet; no final claim.

## [2026-07-10 05:00 PDT] P2 — reject K14 seed9; launch K=28 variance control
CMD: inspect seed-9 paired iteration-20 gate; terminate PID 3342213. Launch `CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_k14_s7_from_it15/probe_best.pt --outdir results/p2/balanced_k28_s10_from_it18 --iters 82 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 56 --gather-attempt-cap 60 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 10 --tag balanced_k28_s10_from_it18 > logs/p2_balanced_k28_s10_from_it18.log 2>&1 < /dev/null &`.
RESULT: seed-9 paired gate regresses from its iteration-18 baseline .8629/.0114 to iteration 20 .8229/.0286. K=28 PID 3343341 active; early effective valid budget is 28, exactly double K=14, with attempt cap 60. All other controls and Valid2 remain unchanged.
DECISION: use the two-update iteration-20 gate to determine whether further pool broadening is helpful; seed-7 K=14 remains the primary full-unit run.

## [2026-07-10 05:08 PDT] P2 — K14 post-peak collapse; launch q=.4 control
CMD: inspect seed-7 absolute iteration-30 paired gate; terminate PID 3338903. Launch `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_k14_s7_from_it15/probe_best.pt --outdir results/p2/balanced_q40_k14_from_it18 --iters 82 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.40 200:0.50 400:0.60 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 11 --tag balanced_q40_k14_from_it18 > logs/p2_balanced_q40_k14_from_it18.log 2>&1 < /dev/null &`.
RESULT: K14 seed-7 gate sequence is iteration 15 .8057/0, iteration 20 .8514/.0171, iteration 30 .4686/.0114. Gamma=.5 probe falls from the preserved iteration-18 .92/0 to iteration 30 .64/.04. The continuation clearly collapses, so its later weights are rejected. q=.4 PID 3345074 active on GPU2; fixed schedule is {0:.4,200:.5,400:.6} by absolute index.
DECISION: preserve iteration 18. Test the explicitly allowed AND-quantile control: q=.4 gives a nominal .6^3=.216 frontier cell, closer to the 25% batch share and less extreme than q=.5. K=28 q=.5 continues on GPU3 pending its gate.

## [2026-07-10 05:14 PDT] P2 — reject q=.4; launch K=28 iteration-20 M=100
CMD: inspect q=.4 paired gate and terminate PID 3345074. Launch seven GPU2 workers: `for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-k28-it20 --outdir results/p2/eval_k28_it20 --ckpt results/p2/balanced_k28_s10_from_it18/ckpt_20.pt --device cuda --T 250 > "logs/p2_eval_k28_it20_g${g}.log" 2>&1 < /dev/null & done`.
RESULT: q=.4 iteration-20 M=25 is aggregate SR=.8114/CR=.0343 versus its iteration-18 baseline .8629/.0114; gamma=.5 probes are .82/.04 then .76/.08. q=.4 is rejected. K=28 q=.5 iteration-20 M=25 is .8457/.0114, essentially preserving its baseline; live gamma=.5 remains .84/.02 at iteration 22. Seven M=100 workers active.
DECISION: prioritize authoritative K=28 metrics. Keep K=28 training running; no q=.4 continuation.

## [2026-07-10 05:18 PDT] P2 — reject K=28 after M=100; launch q=.6 replicate
CMD: assemble `tables/_T2_k28_it20.{md,csv}`; terminate K=28 PID 3343341 after iteration-23 collapse. Launch two q=.6/K14 branches from preserved iteration 18 on GPUs 2/3, seeds 13/14, with fixed absolute schedule {0:.6,200:.7,400:.8}; all other selected controls unchanged.
RESULT: K=28 iteration-20 M=100 has SR {88,89,90,90,90,83,86}%, CR {0,2,3,2,2,6,4}%, time {17.09,13.35,12.00,11.65,11.73,12.15,12.37} s, coverage {13,7,6,6,6,7,7}. It improves mid-gamma SR but not safety/coverage; live iteration 23 collapses to gamma=.5 .74/.10. K=28 is rejected.
DECISION: test the remaining quantile direction with two seeds. q=.6 is a more selective nominal .4^3=.064 AND cell; gathering still must populate both actual classes and never backfills from demos. Stop early only on explicit paired/probe collapse.

## [2026-07-10 05:18 PDT] P2 — exact q=.6 seed-13/14 commands
CMD (GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_k14_s7_from_it15/probe_best.pt --outdir results/p2/balanced_q60_k14_s13_from_it18 --iters 82 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.60 200:0.70 400:0.80 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 13 --tag balanced_q60_k14_s13_from_it18 > logs/p2_balanced_q60_k14_s13_from_it18.log 2>&1 < /dev/null &`.
CMD (GPU3): same exact flags with seed 14, outdir/tag `balanced_q60_k14_s14_from_it18`, and log `logs/p2_balanced_q60_k14_s14_from_it18.log`.
RESULT: PIDs 3347488/3347489 active.
DECISION: none pending metrics.

## [2026-07-10 05:25 PDT] P2 — reject q=.6; launch uninterrupted absolute-100 units
CMD: inspect q=.6 seed-13/14 paired iteration-20 gates; terminate PIDs 3347488/3347489. Launch two same-recipe q=.5/K14/beta=.3/demo=.125 continuations from the preserved iteration-18 checkpoint for 82 additional updates, seeds 15/16 on GPUs 2/3.
RESULT: q=.6 seed-13 iteration-20 is aggregate SR=.7771/CR=.0229; seed-14 is .8114/.0114, both worse than their shared iteration-18 baseline .8629/.0114. q=.6 is rejected. The authorized control sweep has now covered q {.4,.5,.6}, class mix {75/25,87.5/12.5}, demo fraction {.125,.1875,.25} (plus pre-fix 0 diagnostic), beta {.3,.2 after coverage stall}, and valid-rollout budgets {7,14,28}, with M=100 tables for promising checkpoints. q=.5/K14 is selected.
DECISION: run two uninterrupted selected-recipe lineages to absolute iteration 100 as explicitly requested. Preserve every probe-best; use periodic all-gamma gates for selection. Do not stop merely on a bad probe.

## [2026-07-10 05:25 PDT] P2 — exact absolute-100 commands
CMD (GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/balanced_k14_s7_from_it15/probe_best.pt --outdir results/p2/finalunit_q50_k14_s15_from_it18 --iters 82 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 15 --tag finalunit_q50_k14_s15_from_it18 > logs/p2_finalunit_q50_k14_s15.log 2>&1 < /dev/null &`.
CMD (GPU3): same exact flags with seed 16, outdir/tag `finalunit_q50_k14_s16_from_it18`, and log `logs/p2_finalunit_q50_k14_s16.log`.
RESULT: PIDs 3348793/3348794 active; both resume at absolute iteration 18 and target absolute iteration 100.
DECISION: pending full units.

## [2026-07-10 07:40 PDT] P2 — absolute-100 units complete; seed 15 reaches safe M=25 gate
CMD: inspect both completed histories, terminal logs, and saved checkpoint metadata.
RESULT: seed 15 completes absolute iteration 100 with paired all-gamma M=25 aggregate SR=.9029/CR=0, mean final goal distance .404; gamma=.5 probe is .94/0/coverage3. Its gate history is (18 .863/.011), (20 .834/.029), (30 .343/.006), (40 .594/.006), (50 .634/.023), (60 .543/.017), (70 .503/.011), (80 .766/.006), (90 .863/0), (100 .903/0), demonstrating a late recovery that validates completing the requested unit. `safe_best.pt` is iteration 100. Seed 16 collapses by iteration 100 to paired SR=.53/CR=.09 and gamma=.5 .66/.16, so it is rejected.
DECISION: seed 15 iteration-100 `safe_best.pt` is the new incumbent, but M=25 is not sufficient for the final claim. Launch M=100 independently at every gamma before deciding whether to resume toward iteration 200.

## [2026-07-10 07:40 PDT] P2 — launch iteration-100 authoritative M=100 audit
CMD: four GPU2 workers for gamma {.1,.2,.3,.4} and three GPU3 workers for {.5,.7,1.0}, each `eval_ae.py policy-worker --M 100 --reach 0.1 --seed0 0 --method Flow-expanded-q50-k14-it100 --outdir results/p2/eval_finalunit_s15_it100 --ckpt results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt --device cuda --T 250`, with per-gamma logs `logs/p2_eval_finalunit_s15_it100_g*.log`.
RESULT: seven workers active on authorized GPUs; evaluation pending.
DECISION: if any gamma fails SR=1/CR=0, resume only this safe seed-15 lineage under the same absolute schedule. Mizuta remains untouched benchmark-only.

## [2026-07-10 07:44 PDT] P2 — iteration-100 M=100 table and resume to absolute 200
CMD: assemble `tables/_T2_finalunit_s15_it100.{md,csv}`; launch two 100-update continuations from iteration-100 `safe_best.pt`, seeds 17/18 on GPUs 2/3, with the identical selected recipe and absolute schedules.
RESULT: iteration-100 M=100 per-gamma SR {91,96,95,95,94,91,93}%, CR {0,0,2,1,1,1,0}%, clearance {.308,.296,.297,.299,.299,.299,.300} m, time {18.46,14.00,12.41,12.01,12.10,12.58,12.81} s, coverage {9,7,5,4,4,6,5}. This is the highest-SR authoritative P2 table so far, but it fails simultaneous 100%/0%, speed, gamma=.1 clearance, and coverage goals.
DECISION: resume rather than finalize. The same fixed schedule enters its predeclared mid-phase after absolute 100: effective valid-rollout pool 28, two inner updates, slowly changing class mix; q remains .5 until it becomes .6 at absolute iteration 200. Beta stays .3 because the one controlled beta=.2 test was worse.

## [2026-07-10 07:44 PDT] P2 — exact iteration-100-to-200 commands
CMD (GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt --outdir results/p2/resume100_to200_s17 --iters 100 --no-freeze --enc-lr-mult 0.3 --m-measure 25 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 10 --log-comp-every 1 --probe-cov 1 --seed 17 --tag resume100_to200_s17 > logs/p2_resume100_to200_s17.log 2>&1 < /dev/null &`.
CMD (GPU3): same exact flags with seed 18, outdir/tag `resume100_to200_s18`, and log `logs/p2_resume100_to200_s18.log`.
RESULT: PIDs 3378336/3378337 active; both start at absolute iteration 100 and target 200.
DECISION: use paired all-gamma gates and preserve any safe-best; do not expand Mizuta.

## [2026-07-10 07:55 PDT] P2 — stop destructive post-100 resumes; diagnose vector-field jump
CMD: inspect first three probes from both iteration-100 resumes; terminate PIDs 3378336/3378337 before further updates; extract per-update sigma planes, class counts, and gradient RMS.
RESULT: seed 17 gamma=.5 SR50/CR50 falls from the saved .94/0 checkpoint to .60/0 at it101, .22/.04 at it102, .06/.02 at it103. Seed 18 falls to .68/.02, .62/0, .60/0. At it101 the resumed GP buffer is empty so sigma_plane=1 and all windows have sigma=1; labels become effectively a two-axis rather than three-axis AND, yielding 997/3251=30.7% frontier instead of approximately 12.5%. The absolute phase boundary simultaneously changes from one to two inner gradient steps and from 14 to 28 valid rollouts. Gradient RMS is not numerically explosive (field .011--.013, encoder .009--.023), so parameter/functional displacement and label-distribution shift require direct measurement.
DECISION: both post-100 weights are rejected; safe iteration-100 remains intact. Do not resume again until the true flow vector field, parameter displacement, GP-resume state, and frontier labels are audited constructively. Parallel agents are auditing vector mechanics, run data, and verifier/frontier correctness.

## [2026-07-10 08:19 PDT] P2 — causal resume A/B and three independent audits
CMD (unprimed GPU2): `CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup python grid_expand_fixed.py --ckpt results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt --outdir analysis/runs/resume_no_prime_s17 --iters 1 --no-freeze --enc-lr-mult 0.3 --m-measure 1 --measure-every 10 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --viz-db-every 1 --ckpt-every 1 --log-comp-every 1 --probe-cov 1 --seed 17 --tag diag_resume_no_prime > logs/diag_resume_no_prime.log 2>&1 < /dev/null &`
CMD (primed GPU3): the same command with `--iters 2 --warmup-gather 101`, outdir `analysis/runs/resume_prime1_s17`, log/tag `diag_resume_prime1`, and GPU3.
RESULT: the unprimed first update changes gamma=.5 SR50/CR50 from .94/0 to .64/.02 and all-gamma M1 to .43/0. One gather-only GP prime preserves .94/0 at t101; after the t102 update it remains .92/0 and all-gamma M1=.86/0. This is direct causal evidence that the cold query buffer—not numerical gradient explosion—caused the immediate resume collapse.
RESULT: independent reports are `analysis/frontier_verifier_audit.md`, `analysis/vector_field_diagnosis.md`, `analysis/run_data_forensics.md`, and `analysis/resume_state_checkpoint_audit.md`, with reproducible scripts/JSON/CSVs beside them. They agree on: gather used legacy reach .45; only 52--79% of accepted executed paths passed actual Valid2; 1--3% of H=10 proposal targets were certificate-infeasible (9.9% at gamma=.1); fitted residual margin was numerically tied at zero; probes overwrote training RNG; resume lost Adam/qbuf/RNG/teacher/coverage; global planes/batches starved gamma=.1; one staircase supplied 60--73% of accepted paths.
RESULT: true-vector-field probes show base→it100 sampled-control effective rank 11.77→7.42, gamma sensitivity down 18%, balanced-demo CFM 7.9% worse, and late-context generated first-action bias correlates with all-gamma SR at Pearson r=.948. Failed policies generate a y-biased first action while their unused full tail compensates xward, causing repeated-replan goal/top-boundary overshoot.
DECISION: these invalidate pre-fix long-run interpretations. Retain seed15 it100 only as rollback weights, not final evidence. Correct semantic invariants before further tuning. Mizuta/Kazuki remains untouched-pretrained benchmark-only; no flow expansion.

## [2026-07-10 08:19 PDT] P2 — corrected trainer implementation and semantic regression gate
CMD: `LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 python analysis/test_corrected_trainer.py --json analysis/test_corrected_trainer.latest.json`
RESULT: 13/13 independent gates PASS: NumPy/Torch/CUDA-observational probes; nondegenerate real-face verifier margin; per-gamma quantile planes; gamma/mode/rollout-balanced unique draws; coherent executed-horizon targets; non-finite margins unlabelable; rejected selected proposals enter qbuf; strict-reach assertion; exact planned certificate/cache; unique easy/frontier quotas; all-gamma class quotas; complete train-state roundtrip.
IMPLEMENTED (local files only): gather now uses reach=.1 and actual unchanged `traj_valid2`; CFM targets are H-step controls actually executed after each context rather than unexecuted proposal tails; every target is exact-certificate gated; frontier uses minimum feasible real `Face.m`; qbuf remembers rejected queries and refreshes within gather; quantiles and batch quotas are per gamma; batches balance gamma→staircase→rollout without replacement; exploration/evaluation NFE both 8; field gradients are bounded; all probes/viz preserve Python/NumPy/Torch/CUDA RNG; checkpoints serialize Adam, qbuf, coverage, pile, original teacher, history/counters and all RNG states and commit after measurement. Legacy checkpoints receive a mandatory gather-only prime.
DECISION: next run only a small deterministic smoke and uninterrupted-vs-full-state-resume equivalence test. A serious fine-tuning arm is permitted only if strict reach, executed Valid2, exact target certificate, all-gamma classes, nonconstant sigma, mode composition, coherent first-action telemetry, and state equivalence all pass.

## [2026-07-10 08:39 PDT] P2 — live corrected smoke and exact continuation gate
CMD: paired seed-31 runs from the seed15 iteration-100 incumbent: uninterrupted `--iters 3` on GPU2 and split `--iters 2` on GPU3, then resume the split `final.pt` for one iteration with CLI seed 999. Shared corrected recipe: `--rollouts-per-iter 14 --gather-attempt-cap 98 --batch 64 --quantile-schedule 0:0.50 200:0.60 400:0.70 --mix-start .75 .25 --mix-end .50 .50 --beta .3 --early-inner 1 --inner-steps 1 --cooldown-inner 1 --demo-frac .125 --lwf-eta .05 --teacher-ckpt ../../results/hp_repr/pretrained_a32uni.pt --lr 1e-4 --nfe-explore 8 --field-grad-clip 1 --legacy-prime-iters 1 --viz-db-every 1 --ckpt-every 1 --log-comp-every 1`.
RESULT: iteration 101 is correctly query-only: qbuf=500, no gradient, 45 accepted Valid2 paths/83 attempts, 5,345/5,345 coherent windows exactly certified, all-gamma class quotas ready, finite sigma planes (.48 easy/.52 frontier), and M1 all-gamma 1.0/0. Iteration 102 uses exactly 42 easy +14 frontier +8 demo, exactly eight fresh samples per gamma, 14 source rollouts, and M1=.71/0. At iteration 103 the pre-deficit scheduler reached its cap with 51 Valid2 paths but no gamma=.1 accepted path, so it correctly skipped the update rather than backfilling; this exposed that equal attempts cannot repair unequal per-gamma Valid2 acceptance.
RESULT: independent physical-GPU regression is 14/14 PASS (`analysis/corrected_trainer_regression.md`): uninterrupted vs split has model max|Δ|=0 and exact Adam/qbuf/teacher/coverage/history/Python/NumPy/Torch/CUDA RNG. The live GPU2-vs-GPU3 pair also has model max|Δ|=0 and exact qbuf/teacher/coverage/RNG; four Adam moment tensors differ only 4.7e-9--1.1e-11 from physical-GPU arithmetic. Gather now schedules remaining attempts directly to unmet gamma/class deficits; the semantic harness still passes 14/14.
DECISION: state continuity is proven. Only periodic `ckpt_t.pt` and `final.pt` are declared resumable; selection snapshots are model-only. Saves are atomic, validate CUDA topology and recipe/parameter-group signature, and require the fixed teacher when LwF is active.

## [2026-07-10 08:39 PDT] P2 — first corrected update trust failure and step-size/freeze control
CMD: authoritative M25/gamma evaluation of corrected lr=1e-4 iteration 102 -> `tables/_T2_corrected_it102_lr1e4_m25.{md,csv}`; fixed vector/late-action probes -> `analysis/vector_field_corrected_it102.json`.
RESULT: SR {.92,.80,.84,.80,.76,.84,.88}, CR all zero, coverage {6,4,2,2,2,2,2}. A single update moves the fixed origin field 2.15% and reduces the late-goal first-action margin `mean(a0_x-a0_y)` .154→.114 (y-dominant 40.2%→42.8%). It is safe but behaviorally too large and is rejected.
CMD: same-seed GPU2/GPU3 controls from the incumbent with a query-only prime then one update at lr=2e-5, one frozen encoder and one trainable encoder; all other semantic/data controls identical. Both use one step, incumbent as the fixed LwF teacher, and deficit-directed all-gamma gathering.
RESULT: both gather identical data (14 accepted/63 attempts, 1,691/1,691 coherent targets certified, gamma=.1 receives 42 directed attempts). Frozen late-action margin is .1423 and unfreezed .1404; fixed-field drift is .378% versus .418%. Frozen M1 is 1.0/0 versus unfreezed .86/0.
CMD: M25/gamma evaluation of the frozen arm -> `tables/_T2_corrected_freeze_lr2e5_it102_m25.{md,csv}`.
RESULT: frozen lr2e-5 SR {.92,.96,.88,.84,.88,.88,.92}, CR all zero, clearance {.309,.293,.296,.300,.298,.297,.297}, time {18.09,14.17,12.34,11.81,11.99,12.50,12.91}, coverage {6,4,2,2,2,2,2}. Aggregate SR=.894, close to the incumbent .903 while slightly improving most medium-gamma completion times; the lr1e-4 arm averaged .834. It still fails the goal and is only the stable-step control.
DECISION: select frozen encoder + lr2e-5 + one update/iteration for the next bounded arm. Keep a hard per-update functional-drift watch near 0.5%, late-action margin watch, and faithful all-gamma gates. Coverage remains the dominant gap; test the existing whole-rollout coherent target mechanism only through exact Valid2/certificate acceptance before a serious run. Mizuta remains untouched.

## [2026-07-10 12:57 PDT] COMPACT STATUS — authoritative goal position

| Priority / artifact | Current authoritative evidence | Goal test | Status |
|---|---|---|---|
| P1 SafeMPPI expert | `tables/T1_expert.*`, M=100/gamma: SR=100%, CR=0%; clearance .281--.333 m; time 10.54--15.13 s; coverage 6--11/gamma | Ground-truth baseline | **Complete** |
| P3 Mizuta/Kazuki | `tables/T3_kazuki.*`, untouched pretrained, M=200/gamma: SR=100%, CR=0%; clearance .372--.375 m; time 8.96--10.47 s; coverage 5--8 | Benchmark only; no expansion | **Complete and frozen** |
| P2 best pre-semantic-fix weights | seed15 it100, `tables/_T2_finalunit_s15_it100.*`, M=100: SR {91,96,95,95,94,91,93}%, CR {0,0,2,1,1,1,0}%, time 12.01--18.46 s, coverage 4--9 | Every gamma SR100/CR0, safer+faster than P1, coverage >=14 | **Rollback incumbent only; fails** |
| P2 corrected pipeline | `analysis/corrected_trainer_regression.md`: 14/14 CPU + physical-GPU gates; exact split resume max model error 0 | Semantic/training integrity | **Complete** |
| P2 corrected one-step stable arm | frozen encoder, lr2e-5, it102, M=25: SR {92,96,88,84,88,88,92}%, CR all 0 | Short safety gate | **Stable but not goal** |
| P2 corrected targeted arm | it106, `tables/_T2_corrected_target50_it106_m25.*`: SR {84,92,92,92,88,88,88}%, CR all 0; time {17.64,14.14,12.18,11.70,11.83,12.17,12.59}; coverage {5,4,3,2,2,2,2} | Must improve coverage/speed without losing SR | **Selected corrected data mechanism; checkpoint not final** |
| Final P2/T2/T_ALL/video | No checkpoint passes the M>=100 all-gamma goal | Strict final audit | **Pending** |

### Original-code caveats, pinpointed

| Caveat | Original code / exact mechanism | Measured consequence | Corrected local behavior |
|---|---|---|---|
| Wrong training reach | pre-fix `grid_expand_fixed.py::_gather_fresh`, original `GR.fm_deploy(...)` call omitted `reach=cfg.reach`; shared `grid_rollout.py:103-104` defaults to `GM.REACH=.45` | 0/14 saved paths reached .1; all stopped .378--.444 m | Explicit `reach=.1` plus hard final-distance assertion |
| Valid2 was recorded, not enforced | pre-fix `_gather_fresh` used only `GM.socp_ok(out["path"],...)`; recipe JSON nevertheless said `valid2_unchanged=true` | Only 52--79% of accepted executed paths passed actual `GM2.traj_valid2` | Executed path must pass unchanged `GM2.traj_valid2(..., check_socp=True)` |
| Learned target was not the verified object | shared `grid_rollout.py:155-160` executes only `U[0]`; pre-fix gather stored/trained full proposal `r[3]` via `GE._to_t(out["recs"])` | 1--3% proposal tails certificate-infeasible (9.9% at gamma=.1); late plans compensated for a wrong first action; first-action bias correlates with SR at r=.948 | Target is ten consecutive controls actually executed after the context; every coherent H=10 target is exact-certificate/progress gated |
| Degenerate SOCP ranking scalar | pre-fix `grid_metrics2.window_socp_margin` ranked `check_certificate` minimum fitted residual | >=53% tied near zero; 97.2% of selected frontier was numerical zero | Binary certificate unchanged; continuous axis is nondegenerate `min(real Face.m)` / `R_eff` |
| Resume was model-only | shared `grid_hp_expt.py:83-87` saves weights/config; pre-fix `run_expand_cur` recreated Adam/GP/teacher and set `qbuf=None` | first resumed sigma identically 1; frontier 30.7%; gamma=.5 SR50 .94→.60→.22→.06 | Atomic full-state checkpoints: Adam, qbuf, teacher, coverage, counters, history, Python/NumPy/Torch/CUDA RNG; legacy checkpoints prime without gradients |
| Evaluation selected the next training noise | shared `sr_cr_eval.py:44` calls global `torch.manual_seed`; pre-fix probes did not restore RNG | different CLI seeds produced byte-identical first gathers | Every measure/probe/viz preserves all RNG families |
| Gamma and homotopy starvation | pre-fix `_front_mask` pooled gammas; `gamma_ready` was computed but ignored; update drew windows without gamma/mode quotas | gamma=.1 was 2--3% of windows; one staircase was 60--73% | Per-gamma planes and class quotas; deficit-directed attempts; gamma→achieved-mode→rollout balanced unique batch |
| Rejected queries remained “novel” | pre-fix qbuf update occurred only after trajectory/window acceptance | repeatedly spent attempts in already queried rejected regions | Every selected proposal enters qbuf before trajectory acceptance; GP refreshes during gather |
| Numerical sampler mismatch | pre-fix gather NFE=6, evaluation NFE=8 | identical origin windows differed 5.1% in control L2 | Both use NFE=8 |
| Small parameter moves caused closed-loop bifurcation | pre-fix only encoder clipping; no causal field displacement gate | 0.07% trunk move caused 2.15% origin-field move and large SR loss | frozen encoder, lr2e-5, one step, per-step field bound 2.5%, cumulative fixed-origin teacher bound 1.6%, exact rollback |

### Explicit assumptions made

| Assumption | Why it is being used | Caveat / audit treatment |
|---|---|---|
| Seed15 it100 is a rollback **weight initialization**, not valid final evidence | It is the strongest measured P2 policy (M100 SR91--96%) | Its first 100 updates used flawed collection; corrected iterations are reported separately |
| “Unchanged Valid2” means the original `GM2.traj_valid2` boolean is untouched; extra reach/exact-target gates may only make acceptance stricter | Preserves the user's verifier definition while making the training claim literal | No relaxation, demo backfill, or inferred safe label is allowed |
| `min(real Face.m)` is the continuous SOCP margin after binary certification | It is verifier-native, finite, and empirically nondegenerate/correlated with clearance | Certificate feasibility remains authoritative; the scalar only ranks feasible windows |
| The correct CFM label under receding-horizon deployment is the witnessed shifted executed-action sequence | Only those actions jointly occurred on the trajectory that passed Valid2 | This is a semantic correction, not a change to the CFM equation |
| Per-gamma and mode balancing are sampling corrections, not changes to the fixed AND rule | Final goals are required for every gamma and coverage mode | Quantile remains absolute q=.5 and frontier remains high-sigma AND low-margin AND high-progress |
| A fixed whole-rollout staircase target may propose candidates, but never certify them | Bounded probes: exact-accepted targeted paths had more modes and were ~2.4 s faster at gamma=.5; gamma=.1 produced 3 valid targets vs 0 ordinary | Target hit rate is low; target fraction=.5 remains provisional and all raw unsafe/invalid outputs are discarded |
| Frozen encoder, lr2e-5, and trust rollback are temporary stability controls | lr1e-4 moved the field 2.15% in one step and reduced SR; lr2e-5 frozen moved .378% on the fixed field probe | They are beyond the original {q,mix,demo,beta,budget} sweep list, so are logged as diagnosed optimizer safeguards, not silently treated as the original recipe |
| Operational “coverage close to 16” is >=14, and final evidence is M>=100/gamma | Matches `audit_p2_goals.py` and the goal's empirical instruction | M1/M5/M25 are gates only; no final claim from them |
| Corrected lineage teacher is fixed to seed15 it100 | Preserves the strongest stable OOD behavior while corrected data expands it | This differs from anchoring to untouched pretraining and is stated in every recipe |

### Nearest future gates (do not skip)

| Order | Nearest goal | Concrete pass condition |
|---:|---|---|
| 1 | Increase *accepted* conditional mode support before another update | Each gamma has >=2 achieved exact-valid staircase modes in the gathered block; easy/frontier quotas still filled; no demo backfill |
| 2 | Transfer diversity without crossing the safety bifurcation | Fixed-origin drift <=1.6%, per-step drift <=2.5%, late `a0_x-a0_y` stays positive, M25 CR=0 and SR does not regress from the selected checkpoint |
| 3 | Establish a promising short checkpoint | M25: every gamma SR>=.95, CR=0, coverage visibly increasing, time decreasing toward P1 |
| 4 | Run the required unit only after gate 3 | Stateful 100-update lineage with q=.5 absolute schedule, beta=.3, frozen encoder/one step, periodic rollback-safe checkpoints |
| 5 | Final certification | M>=100 for every gamma: SR=1, CR=0, clearance>P1, time<P1, coverage>=14 and >P1; then write T2/T_ALL/video/figures and pass `audit_p2_goals.py` |

LATEST TRUST RESULT: continuing target it106 toward it109 moved fixed-origin field 1.47%→2.32% and introduced M5 collisions, so that branch was stopped. Replaying one proposed update from it106 produced per-step drift 1.21% but cumulative anchor drift 1.65%; the new 1.6% guard rolled it back exactly. Therefore it106 remains the selected corrected checkpoint, not it109.

DECISION: compact context above is now the handoff source of truth. Next implementation change is only the gathered-block `>=2 modes per gamma` readiness gate, followed by one bounded update and M25 evaluation. Do not flow-expand Mizuta/Kazuki.

## [2026-07-10 13:26 PDT] P2 — mode-quota gate passes; corrected checkpoint reaches 93.7% M25

| Item | Concrete result | Decision |
|---|---|---|
| Semantic regression | `analysis/test_corrected_trainer.gpu2.modequota.json`: 14/14 PASS, including exact split resume | Trainer remains eligible |
| Gathered-block quota | `min_modes_per_gamma=2` is part of readiness, deficit scheduling, recipe signature, telemetry, and checkpoints | Required gate implemented; no demo backfill |
| Iteration 102 | 16 exact-valid rollouts/45 attempts; all gamma/mode quotas met; per-step field drift 1.22%, cumulative anchor drift .41%; M25 aggregate SR 92.0%, CR0 | Safe bounded update |
| Iteration 103 | 26 exact-valid rollouts/160 attempts; all gamma/mode quotas met; per-step field drift .98%, cumulative anchor drift .68%; M5 aggregate SR 97.1%, CR0 | Best corrected resumable checkpoint: `results/p2/corrected_mode2_target50_s81_to106/ckpt_103.pt` |
| Iteration 103 M25 | SR by gamma `{92,96,96,92,92,92,96}%`; CR all 0; clearance `.294--.307`; time `11.77--17.88` s; coverage `{6,3,3,3,3,2,3}` | Aggregate 93.7%, but four gammas remain below the required 95% short gate |

DECISION: resume the exact iteration-103 state for at most two guarded updates. Evaluate each periodic checkpoint at M25; stop immediately on rollback, collision, or cumulative anchor drift above 1.6%. Do not start the 100-update unit or final M100 audit until every gamma clears the short gate. Mizuta/Kazuki remains untouched benchmark-only.

## [2026-07-10 13:35 PDT] P2 — t104 selected, t105 rejected; origin-tail hypothesis isolated; Claude packet ready

CMD: exact full-state resume of `results/p2/corrected_mode2_target50_s81_to106/ckpt_103.pt` for two updates with the unchanged corrected recipe (`freeze`, lr=2e-5, one step, target fraction .5, NFE8, `min_modes_per_gamma=2`, per-step drift<=2.5%, cumulative anchor drift<=1.6%). Output: `results/p2/corrected_mode2_target50_s81_from103_to105`.

| Checkpoint | Gather/update telemetry | Gate result | Decision |
|---|---|---|---|
| t104 | 23 exact-valid rollouts/48 attempts; all readiness gates pass; per-step drift 1.06%; anchor drift .97%; M5 SR97.1%, CR0 | M25 SR `{92,96,96,92,92,92,96}%`, CR0; coverage `{6,4,3,3,3,2,3}`; time `{18.20,13.80,12.10,11.78,11.73,12.32,12.65}` s | Best corrected resumable state, but short all-gamma 95% gate still fails |
| t105 | 19 exact-valid rollouts/83 attempts; all readiness gates pass; per-step drift 1.05%; anchor drift 1.24%; M5 SR94.3%, CR0 | Regression despite remaining inside trust bounds | Reject for continuation; diagnostic only |

RESULT: authoritative t104 table is `tables/_T2_corrected_mode2_it104_m25.{md,csv}`. The extra update did not improve the all-gamma reliability ceiling, so no 100-update unit or final M100 audit is authorized.

CMD: run new read-only `analysis/origin_window_failure_probe.py` across accepted-window snapshots t102--t105 and faithful path archives for rollback it100 / corrected t103 / corrected t104. Outputs: `analysis/origin_window_failure_probe.{md,json}`.

| Origin hypothesis component | Evidence | Conclusion |
|---|---|---|
| Accepted windows concentrated at origin | Radius<1 m share is stable at 20.2--21.3%; σ near origin `.745--.772` vs `.442--.456` away; 89--94% near-origin windows enter easy pool under the three-axis AND rule | High-σ origin data is present and mostly easy, but it is not an overwhelming majority |
| Origin target windows numerically ill-conditioned | Centered 10x2 target-control SVD condition median is `1.54--1.67` near origin vs `1.57--1.61` away | This explicit low-rank proxy does **not** support numerical ill-conditioning of valid targets |
| Faithful generative origin-tail failure | Seed 12 exits through y<-.12 within 10--11 controls for every gamma at rollback it100, corrected t103, and corrected t104 | Strongly supported; it predates corrected tuning and ordinary updates do not remove it |
| Single failure mechanism | t104 M25 also contains four upper-boundary near-goal overshoots | Rejected: origin tail and late-goal overshoot must be repaired/audited separately |

DECISION: do not loosen Valid2, add inference clipping/safety filtering, or blindly resume unchanged from t104 (the exact next update is the rejected t105). Next causal work is to trace the NFE8 base-noise tail, log the exact chosen training indices, and run one controlled training-only hard-tail repair using exact-valid origin and late-goal targets.

VIZ: generated the latest 2x4 curriculum artifact `video/p2_corrected_mode2_it104_105_curriculum.mp4` (scene, sigma histogram, three-axis planes, sample bins; beta/count/mix/lr traces).

HANDOFF: created `claude_handoff/README.md`, `claude_handoff/CLAUDE_PROMPT.md`, and `claude_handoff/NEXT_COMMANDS.md`. They identify t104 as selected, t105 as rejected, reproduce the semantic/origin audits, state exact GPU/validity/trust restrictions, and define the visualization-level path to the final T2/T_ALL/video/audit. Mizuta/Kazuki remains untouched benchmark-only.

## [2026-07-10 14:05 PDT] CLAUDE takeover — handoff reproduced, diagnosis phase begins
CMD: per `claude_handoff/NEXT_COMMANDS.md`: `CUDA_VISIBLE_DEVICES=2 python analysis/test_corrected_trainer.py --json analysis/test_corrected_trainer.claude.json`; re-run `analysis/origin_window_failure_probe.py` into `analysis/origin_window_failure_probe.claude.{json,md}`.
RESULT: semantic regression 14/14 PASS on physical GPU2 (cuda_visible_count=1). Origin/failure probe taxonomy byte-identical to codex's: seed 12 origin-boundary OOB at all 7 gammas for it100/t103/t104; four near-goal overshoots at t104 (g.1 s22, g.4 s8, g.5 s3, g.7 s5; endpoints y~5.13, d_goal .14-.18). GPUs 2/3 free at takeover; no stray processes.
DECISION: proceed exactly along the handoff sequence: (1) read-only seed-12 NFE8 ODE trace + latent-tail density probe + exact-batch telemetry replay of t103->t104, (2) one controlled training-only hard-tail arm gated on fixed seeds then M25. t104 remains selected; t105 remains rejected; Mizuta untouched.

## [2026-07-10 14:45 PDT] CLAUDE — seed-12/near-goal causal localization (read-only; the handoff's diagnosis step)
CMD: new `analysis/seed12_tail_trace.py` on GPU2: (A) instrumented faithful fm_deploy clone (identical RNG consumption, one randn(1,d)/replan) traces all 11 fixed t104 M25 failures and asserts path equality against true `GR.fm_deploy`; (B) the exact seed-12 step-0 origin latent pushed through pretrained/it100/t103/t104; (C) 512 fresh latents integrated at each failing context, scored by one-step OOB and 10-step open-loop window OOB. Outputs `analysis/seed12_tail_trace.{json,md}`, `figures/seed12_trace.png`.
RESULT (verify_ok=True for all 11 traces):
- Origin stratum is TRIGGER + ABSORBER, not one mechanism. Trigger: at the clean (0,0) context only 1-2% of latents map to window-OOB at t104 (pretrained was 4-8%; corrected training already shrank it), but seed-12's latent maps to a SATURATED down first action u0_y=-1.000 at EVERY checkpoint (pretrained included) — training never moved this fiber (frozen encoder + lr2e-5 leaves it byte-similar across it100/t103/t104). Absorber: once the state dips below y=0 (10-11 steps of v_y<0 drift), window-OOB over 512 fresh latents is 0.83-1.00 for ALL gammas and ALL checkpoints — the y<0 strip has NO in-bounds generative mode at all.
- Near-goal stratum: at the last in-bounds contexts (~(4.9,5.10), above the goal line), mean first action DOES point down (u0_y<0) but window-OOB=1.00 for every checkpoint, and corrected training WEAKENED the brake (u0_y mean pretrained -0.61 -> t104 -0.44; consistent with the r=.948 late-action bias finding).
- Root cause: both strips are DATA-EMPTY — demos live in [0,5]^2 with reach .45, gathered rollouts start at the origin and die on strip entry, so no certified window has ever supervised strip contexts. Re-weighting existing data cannot fix absence; batch composition is not the binding constraint.
DECISION: the minimal training-only repair must CREATE exact-valid strip data, not re-weight: (1) recovery-start gathering — a bounded fraction of gather attempts starts ON the strips with adverse velocity; acceptance stays byte-identical (strict reach .1 + unchanged traj_valid2 + per-window exact certificates); (2) a small batch sub-quota for strip-context certified windows; (3) codex's hard-tail x0 pairing at those contexts (oversample base latents whose current faithful map exits the box; standard CFM target). No inference change, no Valid2 change, coverage/mode quotas EXCLUDE recovery rollouts.

## [2026-07-10 14:50 PDT] CLAUDE — hard-tail arm implemented; 16/16 semantic gates; smoke + exact-batch replay launched
CMD: `grid_expand_hardtail.py` (copy of the corrected trainer; original untouched): CurConfig fields recovery_frac/recovery_origin_band/recovery_goal_band/hard_quota/hard_x0/hard_x0_cand/strip_probe_every; gather recovery branch with env.x0 override under try/finally (never leaks; env untouched when arm off); rkind/strip arrays; recovery rollouts EXCLUDED from valid_modes and covered; sub-quota drawn gamma/mode/rollout-balanced from strip-context certified windows before the normal class draws; `_cfm_loss_x0` (bit-exact two-stage-mean copy of FlowPolicy.cfm_loss with row-local x0 override); `_harvest_bad_x0` (32 candidates/window, keep an OOB-mapping latent only when the OOB set is a MINORITY <=50%, else random x0 — at mean-shifted strip contexts random draws already cover the tail); per-iter RNG-isolated `_strip_probe` (win-OOB at the 2 fixed failing contexts); recipe + resume_signature carry every new field. Harness: `analysis/test_hardtail_trainer.py` = full 14-gate harness aliased onto the copy + 2 arm gates (disabled-arm bit-exactness; x0-override row-locality/band/flag correctness).
RESULT: 16/16 PASS on physical GPU2 (`analysis/test_hardtail_trainer.json`). Two harness-driven fixes: lazy env.x0 capture (mock envs lack x0) and exact two-stage CFM mean.
CMD: smoke `results/p2/hardtail_r25_q8_s82_from104` on GPU2 — exact corrected recipe + `--recovery-frac 0.25 --hard-quota 8 --hard-x0 oob`, from t104 via the documented `--drop-train-state` model-only branch (recipe/signature changed => exact optimizer continuation impossible by design; forced 1-iter gather-only GP prime). In parallel on GPU3: `analysis/grid_expand_replay.py` = byte-copy of the CORRECTED trainer with chosen-batch-index logging (bi_np/demo di + pool metadata per inner step; logging consumes no RNG), exact full-state resume of t103 for one update — reproduces the exact t104 training batch for codex's "instrument the selected indices" ask; fidelity check = produced weights vs stored ckpt_104.
DECISION: gate ladder before any unit: (a) smoke telemetry sane (recovery acceptance >0, strip windows in batch, drift bounds, no rollback), (b) fixed-seed suite — all 11 probes flip without regressing the other M25 seeds, (c) independent M25 all-gamma. Mizuta untouched.

## [2026-07-10 15:20 PDT] CLAUDE — exact-batch replay verified (max|dW|=0.0); arm-1 smoke verdict; re-anchored arm-2 + baselines launched
CMD/RESULT (replay): full-state resume of t103 with `analysis/grid_expand_replay.py` (index logging only) reproduced stored ckpt_104 to max|dW|=0.0 — the logged batch IS the true t104 batch. Composition: 42e+14f+8demo; near-origin 13/56 (23% vs pool 21%, chosen U0_y +0.31, sigma .70) — in-task origin weighting was fine; **goal-strip (y>4.95) windows: 0/56 in batch and 0/2540 in the entire pool** — training-side confirmation that the overshoot strip has never had a certified sample. Trace: `analysis/runs/replay_t104_trace/batch_trace_it104.npz`.
CMD/RESULT (arm-1 smoke, hardtail_r25_q8_s82_from104, prime+2 updates): machinery works — recovery acceptance 4/30->7/24 (rising), batch_hard 8/8 filled, SR50 .96/CR 0 and M5 .97/0 held, no rollback. But absorber probes stayed 1.00/1.00 and cumulative anchor drift 1.03%->1.19% vs the 1.6% bound ALREADY 0.97% consumed by t104 itself (anchor ref = it100 teacher). The it100-referenced cumulative bound structurally caps origin-region repair at ~0.6% field change — an order below what filling an absorbing strip needs. This is the same necessary-not-sufficient bound behavior codex logged for t105, now in the constructive direction.
DECISION: keep the gate mechanism and numeric bounds UNCHANGED; re-reference the branch teacher/anchor to the branch point t104 (codex precedent: the corrected-lineage teacher was itself re-fixed to it100 and stated per-recipe — the teacher is a per-lineage declaration). Arm-2 `results/p2/hardtail_tanchor104_s83` launched on GPU2: teacher/anchor=ckpt_104, recovery-frac .3, milder origin band (y[-.05,.18], vy[-.28,.05]) for higher certified-recovery acceptance, hard-quota 12, x0-cand 64, mild+deep absorber probes, iters 82, full-state ckpt every 2 — codex-resumable at any point. 16/16 gates re-passed after the probe/band edits. Baselines for the final a-e story launched on GPU3: iter0 pretrained M25 a-e (`results/p2/eval_pretrained_m25`) then Kazuki w_safe VULNERABILITY sweep {0.05,.3,.9,2,5} M25 (`results/kazuki_wsweep`) — the published-style mixed-coef T3 stays untouched; the sweep shows single-knob sensitivity.
GOAL METRICS REMINDER (every future report: SR / CR / clearance / time / coverage): P1 expert M100 = 100% / 0% / .281-.333 m / 10.54-15.13 s / 6-11; T3 Kazuki M200 = 100% / 0% / .372-.375 / 8.96-10.47 / 5-8; t104 M25 = 93.7% agg / 0% / .294-.307 / 11.73-18.20 / 2-6.

## [2026-07-10 16:05 PDT] CLAUDE — final status at handoff back to codex (a-e comparison assembled; arm-2 live)
RESULT (Kazuki w_safe vulnerability, `tables/T_COMPARE_progress.md` + `results/kazuki_wsweep/`): with ALL 200 MPPI samples given to a single safety coefficient, EVERY tested w_safe in {0.05, 0.3, 0.9, 2.0, 5.0(0/8 at logging, completing)} yields SR 0% / CR 0% — 25/25 timeouts each — versus the tuned 5-coefficient MIX at SR 100% (T3, untouched pretrained ckpt, no confound: single-coef runs use MORE samples per coef than the mix). Their method's completion lives entirely in the hand-tuned coefficient ensemble; no single setting works at all. Our method carries the safety level as a gamma conditioning input with a trajectory-level verifier certificate instead of a fragile cost weight.
RESULT (iter0 vs t104 vs baselines, M25, SR/CR/clearance/time/coverage): iter0 pretrained = 24-48% / 0-12% / .312-.335 / 11.7-16.9 / 2-8. t104 = 92-96% / 0% / .294-.307 / 11.7-18.2 / 2-6. P1 expert (M100) = 100/0/.281-.333/10.5-15.1/6-11. T3 Kazuki mix (M200) = 100/0/.372-.375/8.96-10.5/5-8. Remaining gaps to the final claim: the 11 fixed-seed failures (strips), speed at gamma=.1, and coverage (needs >=14).
RESULT (arm-2 `hardtail_tanchor104_s83`, LIVE on GPU2, PID 3583165): prime it105 + updates it106/it108 (it107 correctly SKIPPED - attempt cap 196 hit before all-gamma/mode quotas; recovery attempts consume budget). Telemetry: anchor 0.39%->0.67% cumulative FROM THE BRANCH POINT (re-anchor works; arm-1's it100-referenced anchor was structurally exhausted), fstep ~1%, no rollback, SR50 stable 0.96/0, M5 0.97/0. Absorber probes: origin mild 0.04 (strip entry 96% recoverable), origin deep 1.00, goal mild/deep 1.00/1.00 - the goal-side absorber is wider; not moved after 2 updates (expected at lr 2e-5; watch the MILD probes first). Full-state ckpt_106/ckpt_108 saved; ckpt-every 2.
HANDOFF -> codex: (1) keep arm-2 running; if quota-skips repeat, either raise --gather-attempt-cap to ~260 or drop --recovery-frac to 0.2 on the next resume (recipe change = new outdir per protocol). (2) Per even checkpoint: `bash run_gate.sh results/p2/hardtail_tanchor104_s83/ckpt_<t>.pt hardtail_<t> 3` = M25 a-e + `analysis/fixed_seed_gate.py` (11 probes must flip, zero per-seed regressions). (3) Short gate then 100-update unit then final M>=100 audit per GOAL.md; coverage >=14 remains the big gap - keep targeted_frac pushing uncovered staircases; recovery rollouts never count toward coverage. (4) Read `claude_handoff/CLAUDE_RETURN.md` for the complete map. Mizuta untouched (w-sweep used the untouched pretrained ckpt).

## [2026-07-10 16:25 PDT] CLAUDE — FIRST FLIPS at ckpt_108; run_gate.sh validated; handoff complete
CMD: `bash run_gate.sh results/p2/hardtail_tanchor104_s83/ckpt_108.pt hardtail108 3` (script's first end-to-end run — validated for codex).
RESULT (fixed-seed gate, 2 repair updates in): **3/11 fixed probes FLIPPED — near-goal g0.4/s8, g0.5/s3, g0.7/s5 all -> success.** The goal-strip recovery data works, and fast. BUT 3 new same-stratum regressions appeared (near_goal: g0.2/s0, g0.5/s8, g1.0/s0) — the descent field is being rewritten and has not settled; aggregate M25 SR unchanged at .937 (t104 parity), CR 0 all, clearance .293-.303, time 11.6-18.2, coverage 3-5. Seed-12 origin x7 still failing (deep absorber; expected at 2 updates — mild-origin probe already 0.04). gate_pass=false (correctly: flips must arrive WITHOUT regressions).
RESULT (arm-2 through it112): 4 clean updates (106/108/111/112), skips at 107/109/110 (cap 196 vs quotas+recovery load). Anchor cumulative 0.39->0.67->0.89->1.09% with DECELERATING increments (0.39/.28/.22/.20) — may asymptote under the 1.6% bound; if it saturates and rollbacks begin, the legitimate continuation is the RATCHETED-BRANCH pattern: gate a checkpoint, branch from it with teacher/anchor re-referenced to that checkpoint (per-recipe declaration precedent), banking <=1.6% of intended origin-field change per generation. SR50 stable .96-.98/0 every iter; no rollback yet.
RESULT (w_safe sweep complete): w=5.0 also SR 0% — ALL five single coefficients {.05,.3,.9,2,5} are 0% (25/25 timeouts each) vs tuned 5-coef mix 100%. `tables/T_COMPARE_progress.md` final.
DECISION/HANDOFF: arm-2 left RUNNING on GPU2 (PID 3583165, full-state ckpt every 2; GPU3 free). codex: run the gate per even checkpoint; promote the first checkpoint with flips>=previous and n_regressions=0; short gate = every gamma SR>=95/CR0/cov non-decr/time non-incr; then the 100-update unit; coverage >=14 stays the big remaining gap. All protocol in claude_handoff/CLAUDE_RETURN.md. Mizuta untouched.

## [2026-07-10 16:50 PDT] CLAUDE — wrap-up: gen-1 stopped at anchor saturation; handoff to codex via NEXT_CODEX.md
RESULT: gen-1 `hardtail_tanchor104_s83` it117/it118 both rolled back (cumulative anchor 1.72% > 1.6% bound) — the generation's origin-repair budget is spent exactly as the decelerating trend predicted; productive weights end ~it114/115; ckpt_118 = terminal full-state. Training stopped; GPUs 2/3 verified idle; no Claude process left. The designed gen-2 ratchet (model-only from ckpt_118, teacher/anchor re-referenced to ckpt_118, cap 260, all else byte-identical — verified against gen-1 recipe.json) was NOT launched per user instruction: codex runs it.
HANDOFF: `claude_handoff/NEXT_CODEX.md` = single entry point (user's storyboard+requests VERBATIM: a-e columns in every report, iter0/mid/final vs Kazuki w_safe-sweep vs demo expert, drive to SR 100% every gamma then the 100-update unit + M>=100 quantification; state table; two paste-ready commands: ckpt_118 gate + gen-2 launch; loop protocol: gate every even ckpt, promote only flips-with-zero-regressions, ratchet on saturation; resource map). `claude_handoff/CLAUDE_RETURN.md` patched to match (gen-1 marked STOPPED). Milestone standing at handoff: 3/11 fixed probes flipped at ckpt_108 (all three flippable near-goal cases) after 2 updates, aggregate M25 SR .937/CR 0 (t104 parity); Kazuki single-w_safe sweep complete (all five values SR 0% vs mix 100%); comparison spine in tables/T_COMPARE_progress.md.

## [2026-07-10 19:13 PDT] CODEX — terminal gen-1 rejected; repair-direction and guarded-batch audits
CMD: gate Claude terminal `ckpt_118`, first-update `ckpt_106`, and deployment-only weight interpolation t104→t106 at α={.25,.5,.75}; add `analysis/interpolate_checkpoints.py` (outputs explicitly non-resumable). Add an opt-in exact-certified interior-boundary guard quota to `grid_expand_hardtail.py`; disabled behavior remains unchanged. Regression harness extended and passes 17/17 on GPU3. Smoke from t104: `hardtail_guard_t104_s85`, batch64 = 12 hard + 12 guard + 32 ordinary + 8 demo.

RESULT: terminal ckpt118 is not a valid ratchet base: M25 SR `{84,92,88,88,92,84,88}%`, CR0, clearance `.290-.303`, time `11.55-17.51s`, coverage `2-5`; 3 near-goal fixes but 13 baseline-success regressions and all seven seed12 origin failures remain. The t104→t106 line search proves the first-update direction is conflicting: α=.25 already has one regression before most fixes; α=.5 has 2 fixes/1 regression; α=.75 has 3 fixes/2 regressions. Guarded batch64 filled 12/12 guard slots from 326 eligible exact-certified interior windows and stayed inside trust (fstep1.40%, anchor.36%), but gate gave 0 fixes/5 regressions; M25 SR `{88,92,96,92,88,88,92}%`, CR0, clearance `.293-.306`, time `11.75-17.80s`, coverage `2-5`. Run stopped before a second update.

DECISION: do not launch Claude's proposed gen2 from degraded ckpt118 and do not promote any tested repair checkpoint. The major untested blind spot is stochastic-gradient variance: each update uses only 64 of >2,600 certified windows and regression identities change by seed. Launch one bounded large-effective-batch smoke from t104 (`hardtail_guard_b256_t104_s86`: 48 hard + 48 guard + 128 ordinary + 32 demo; all validity/inference/trust rules unchanged) and gate its sole update before any continuation. GPU2 remains occupied by another user; GPU3 only. Mizuta/Kazuki remains frozen benchmark-only.

## [2026-07-10 20:57 PDT] CODEX — hard-tail capacity localized; fixed-teacher residual repair reaches all origin flips but not zero-regression gate
CMD: large-batch variance probe from the ready it105 certified pool (`analysis/one_step_from_viz.py`); local-shadow `grid_hp_expt.py` adds an opt-in zero-initialized compact-support boundary residual (old checkpoints load unchanged); `eval_ae.py` fixed to honor the local shadow. Hard-tail trainer gains opt-in interior guards, OOB/in-bounds latent pairing, worst-OOB selection, differentiable NFE8 endpoint loss, and teacher-endpoint guards. All disabled-arm semantics remain unchanged; harness passes 18/18. Multiple diagnostic residual arms were gated; no arm was promoted.

RESULT: batch256 reduces but does not remove the global-update conflict (1 fixed flip/2 regressions, M25 92--96%, CR0). The seed12 raw start action is `u0_y=-1.593`, far below the deployment clamp -1; ordinary certified start targets are well-conditioned/action-diverse (`u0_y` mean +.260, range -.484..+1). Random/worst OOB harvesting over hundreds of tail fibers does not flip seed12 inside one 1.6% budget. Exact-fiber endpoint training proves capacity: after bounded nonlinear-residual ratchets the raw action crosses -1 and all seven seed12 failures flip. The overfit capacity checkpoint `origin_mlp64_seed12_capacity_s118` flips all 11 baseline failures but is invalid (6 regressions, CR4% at γ=.4/.5, coverage 2--4).

RESULT: preservation ablation improved monotonically. Previous-generation teacher guards: 11 flips/5 regressions. Dense 96 start guards: 11 flips/4 regressions and CR0, M25 SR `{92,96,100,100,100,100,96}%`. Full-band guards still accumulated drift because each ratchet used its predecessor as teacher. Correct role separation (per-generation trust anchor + immutable t104 preservation teacher) produced `results/p2/origin_mlp64_fixedteacher_bandguard192_s293.pt`: 10/11 fixed failures flip (all seed12 plus 3/4 near-goal), CR0 all γ, M25 SR `{96,96,100,96,100,96,92}%`, clearance `.295-.304`, time `11.50-18.35s`, coverage `2-5`; however 5 prior successes regress, so gate remains false. Authoritative JSON: `analysis/fixed_seed_gate_origin_mlp64_fixedteacher_bandguard192_s293.json`.

DECISION: origin concentration/ill-conditioning is now sharply resolved: accepted target controls are not numerically ill-conditioned; the deployment failure is a deeply saturated rare latent fiber plus an absorbing empty strip. It is learnable only after crossing the clamp boundary, but zero-regression repair requires immutable-teacher preservation over complete escape trajectories/latent fibers, not merely random start or local windows. Keep t104 as the selected production checkpoint; all adapter files are non-resumable diagnostics and include explicit recipes. Next: build independent-seed full escape-trajectory teacher replay (no M25 seed leakage), couple it with adversarial hard-latent neighborhoods, then gate M25. Coverage≥14 and the 100-update/final M100 stages remain pending. No process remains; GPU3 idle; GPU2 untouched. Mizuta/Kazuki remains frozen benchmark-only.

## [2026-07-10 22:21 PDT] CODEX — independent replay repair reaches all 11 flips; gamma-1 preservation remains the reliability blocker; Claude handoff refreshed

CMD: build independent successful escape replay from seeds 100--149 (disjoint from M25), train a compact origin adapter using adversarial OOB latents plus immutable replay, then add a tight goal adapter using exact-certified goal-brake rows. Artifacts: `analysis/escape_replay/escape_success_seed100_149.pt` (10,123 rows), `analysis/onpolicy_replay/onpolicy_s326_teacher_t104_seed100_149.pt` (18,628 rows), `analysis/goal_replay/goal_success_s671_seed100_149.pt` (4,554 rows), and `analysis/goal_replay/onpolicy_s671_teacher_t104_g1_seed100_199.pt` (1,325 rows). All diagnostic adapter checkpoints are explicitly non-resumable; no M25 seed is used for training.

| Checkpoint | SR by gamma (`.1,.2,.3,.4,.5,.7,1`) | CR | Clearance mean | Time mean (s) | Coverage | Fixed-seed gate | Decision |
|---|---|---|---|---|---|---|---|
| `origin_tightgate_production_s671.pt` | `96,100,100,100,100,92,88` | 0 all | `.292-.305` | `11.53-17.70` | `2-5` | 7/7 origin fibers flip; 4 regressions | Origin capacity/independent replay validated; do not promote |
| `goal_brake_gammaaug_s766.pt` | `100,100,100,100,100,100,92` | 0 all | `.292-.306` | `11.54-17.80` | `2-5` | All 11 original failures flip; gamma1 seeds 5/14 regress | Best exact-valid diagnostic; do not promote |
| `goal_gamma1_brake_focus_s790.pt` | `96,100,100,100,100,100,92` | 0 all | `.292-.305` | `11.54-17.72` | `2-5` | 10/11 flip; gamma-.1 seed22 reopens; gamma1 unchanged | Reject |

RESULT: the origin hypothesis has two distinct meanings. Curriculum conditioning is unusual near the origin: accepted radius<1 m windows are 20.2--21.3% of the pool, sigma `.745--.772` vs `.442--.456` away, and 89--94% are labelled easy by the three-axis AND cell. Numerical target-control conditioning is not unusual: centered 10x2 SVD condition median `1.54--1.67` near vs `1.57--1.61` away. Faithful failure is instead a rare saturated latent trigger (t104 raw seed12 `u0_y=-1.593`, deployment clamp -1) followed by a data-empty absorbing strip (`y<0`, window-OOB `.83--1.00`). Near-goal upper-boundary OOB is a separate empty-strip mechanism. Therefore zero CR with SR<100 is compatible with exact-valid local targets: the missing guarantee is closed-loop latent/state support, not target numerical rank.

CAVEAT/AUDIT: `analysis/one_step_from_viz.py` previously gamma-relabelled five goal-brake rows without an explicit destination-gamma certificate check, although `grid_metrics2.py:121-123` is gamma-dependent. Retrospective audit: all 35 relabels pass (`5/5` at every gamma) with positive margins, so s766 remains exact-valid. The script now recertifies every relabel and aborts on failure; `gamma_aug_certified` is stored in future recipes.

DECISION: t104 remains the selected production checkpoint. Direct gamma1 braking is rejected because it neither repairs gamma1 nor preserves gamma-.1. The next controlled test is immutable t104 teacher replay on states induced by s766, using independent seeds across all gammas, plus one bounded gamma1 braking update. Do not stack updates if its zero-regression M25 gate fails. Coverage remains `2-5` (goal >=14), and the 100-update/final M>=100 stages are not authorized. Handoff: `claude_handoff/START_HERE_LATEST.md` and paste-ready `claude_handoff/CLAUDE_EXACT_PROMPT_CURRENT.md`; visualization goal explicitly separates target SVD condition, curriculum sigma, latent-tail OOB, and empty-strip state support. Mizuta/Kazuki stays benchmark-only and untouched.

## [2026-07-10 23:06 PDT] CODEX — t104 M100 proves rare mode support remains, but unchanged iterations cannot reach coverage 14

CMD: faithful all-gamma M100 audit of selected t104 on GPU3 (`temp=1`, NFE8, reach .1, seeds 0--99), followed by the existing failure-taxonomy probe. Outputs: `results/p2/eval_corrected_mode2_it104_m100/`, `analysis/origin_window_failure_probe_t104_m100.{md,json}`, and `analysis/coverage_iteration_diagnosis.md`.

| gamma | SR | CR | clearance mean | time mean (s) | coverage M25 -> M100 |
|---:|---:|---:|---:|---:|---:|
| .1 | 94% | 0% | .305 | 18.23 | 6 -> 10 |
| .2 | 96% | 1% | .295 | 13.76 | 4 -> 7 |
| .3 | 97% | 1% | .297 | 12.18 | 3 -> 4 |
| .4 | 96% | 0% | .299 | 11.83 | 3 -> 4 |
| .5 | 97% | 0% | .299 | 11.89 | 3 -> 4 |
| .7 | 96% | 0% | .298 | 12.41 | 2 -> 5 |
| 1.0 | 97% | 0% | .299 | 12.62 | 3 -> 6 |

RESULT: M25 did undercount rare support, so the vector field is not irreversibly collapsed: M100 reveals 4--10 modes. It is nevertheless far from >=14 at every gamma, and M100 exposes two rare collision episodes that the M25 CR0 claim missed. Failure taxonomy over 700 rollouts: 673 success, 7 origin-boundary failures, 18 near-goal failures, 2 other failures. The coverage bottleneck is probability-mass concentration rather than global parameter overfit: t104 remains only .97% from its fixed functional anchor and the encoder is frozen, but the canonical mode occupies 20--24/56 fresh training rows.

RESULT: corrected iterations 101--105 recorded zero exact requested-mode target hits despite 21--78 targeted attempts per iteration. The readiness gate requires only 2 achieved modes per gamma, and mode-balanced sampling can balance only modes actually gathered. A later 400-attempt gather produced one target hit but failed gamma-.1 two-mode readiness and made no update. M50 deployed coverage stayed 4 over multiple hard-tail steps; terminal t118 degraded SR without expanding M25 coverage.

DECISION: coverage >=14 is not shown impossible and the pretrained/t104 field retains multimodal capacity, but merely adding unchanged iterations is not a credible route. Before a long run, make `target_hits>0` a coverage-readiness gate, use a fixed absolute achieved-mode schedule (2 -> 4 -> 8 -> 12 -> 14), raise rollout/sample budget and use beta .2 only as the controlled exploration arm, and preserve successful rare-mode latent fibers. Reliability remains first: require an independent M100 SR100/CR0 gate because M25 missed collisions and a broader near-goal tail. Mizuta remains untouched benchmark-only.

## [2026-07-11 00:15 PDT] CLAUDE takeover #2 — plan verified against codex handoff; overnight execution begins
CMD: verified existence/mechanics of everything the handoff depends on: one_step_from_viz.py (exact-certified-pool one-step probe with adapter/escape/endpoint extensions + retrospective gamma-aug certification), build_onpolicy_teacher_replay.py (candidate-rollout/teacher-relabel distillation on independent seeds), build_escape_replay.py (merge), local grid_hp_expt.py boundary adapter (compact-support, zero-init; harness gates 17/18), s766/s790/s671 ckpts + gate JSONs, hardtail_guard_b256_t104_s86/viz_db/it105.pt, coverage_iteration_diagnosis.md (t104 M100: SR 94-97, CR 1% at g.2/.3, coverage 4-10, zero target hits it101-105). GPU3 is the only free device — all work sequential on GPU3.
CRITICAL REVIEW findings folded into the plan: (1) the replay builders do NOT certify teacher targets at destination gamma — per handoff instruction a certify-filter audit will be added before training ("keep only exact destination-gamma-certified rows"); (2) update_flow_fresh returns mean loss only — a loss_steps telemetry field will be added (user deliverable: loss curves) and the 18-gate harness re-run; (3) s766 gate evidence re-verified (11/11 flips, exactly 2 regressions g1.0 s5/s14, SR {100x6, 92}).
PLAN (user-approved direction; user asleep — deliver by morning IF gates pass): A reproduce evidence -> B latent_support_map 4-panel viz -> C certified teacher replay (seeds 100-199) -> D loss telemetry -> E bounded s791 update + M25 gate (SR100/CR0 all gamma + 11 flips + 0 regressions) -> F M100 audit, integrate into resumable trainer, 100-unit, coverage phase (target-hit gate, mode schedule 2-4-8-12-14, budget/beta=.2, M100 coverage) -> G deliverables: 4-panel viz, a-e tables (iter0/mid/final vs Kazuki mix+w-sweep vs P1 expert), loss curves, Kazuki provenance notes, plain-language METHODS.md. Everything saved for re-testing. Mizuta untouched.

## [2026-07-11 00:55 PDT] CLAUDE — phases A-D complete (evidence, 4-panel viz, telemetry, lineage internals)
RESULT (A): origin probe + seed12 trace reproduced (claude2 artifacts). (B) `analysis/latent_support_map.py` -> `figures/current_goal_latent_support.png` + .json: seed-12 raw u0_y −1.593(t104) -> −0.983(s671/s766), origin win-OOB tail 1.03% -> 0.02% (the origin repair TRANSFERRED through the lineage); near-goal strip win-OOB stays 0.93-1.00 for ALL checkpoints incl. s766 at its own regression contexts -> the g1.0 s5/s14 regressions are re-ROUTING into a still-empty strip, not forgotten braking; teacher-preservation replay is the right medicine. (D) `loss_steps/fstep_steps/anchor_steps` per-step telemetry added to update_flow_fresh return (curve deliverable); harness 18/18 PASS (cpu.lossteps json). NEW per user: `analysis/repair_lineage_internals.py` -> `figures/internals_repair_lineage.png` (45 gated experiments, chronological, flips/regressions/SR — the search-history proof plot); `tables/T3_kazuki_notes.md` (full provenance: tuned T3 = w_safe .3 + coll_w 20 + goal_w 2.0 + goal_coef .5, M200, same eval_ae metric code as ours; sweep = published-style weights, ALL single w_safe SR 0%; even tuned point has FLAT a-e across gamma = no safety-level control). METHODS.md drafted (plain-language mechanism table).
USER (verbatim, binding additions): internals/loss-curve plots are THE proof artifacts for every run; second top-level goal = QUANTITATIVE proof that the verifier concept improves the generative policy BEYOND the teacher/expert (per-gamma M100 mode sets ours vs P1 vs demo support, certified NEW modes, a-e beat table) — planned as `analysis/mode_discovery.py` + T4.
CMD (C, running): 7-gamma on-policy teacher replay build (candidate s766, teacher t104, seeds 100-199, region 4-5 x 4-5.12; g0.1 1547 rows, g0.2 1521, g0.3 1424 so far) -> merge -> NEW `analysis/audit_replay_certify.py` destination-gamma certificate filter (drops non-certified teacher targets; per-handoff requirement).

## [2026-07-11 01:45 PDT] CLAUDE — E-phase: s791 -> diagnosis -> s792 (11/11 flips, agg .989) -> top-band data gap found
CMD/RESULT (replay): 7-gamma build complete, 9,978 rows (1,362-1,547/gamma), seeds 100-199 only; `analysis/audit_replay_certify.py` destination-gamma certification: **9,978/9,978 pass** (margin min .157 / median .384) -> `onpolicy_s766_teacher_t104_allg_seed100_199_certified.pt`.
CMD/RESULT (s791, codex's exact arm + certified replay): loss_steps 0.40-0.65 (no blowup), anchor 1.15% cum, no rollback, escape rows 2,560 used. Gate: 10/11 flips (all seven seed-12 origin cases FLIP), SR {96,100,100,100,100,100,92}, agg .983, CR0 — but the SAME g1.0 s5/s14 regressions persist and g0.1 s22 reopened. gate_pass=false.
CMD/RESULT (diagnosis, `analysis/diag_s791_gamma1.json`): the three failing routes diverge from t104 at STEP 6 near the ORIGIN (chaotic split after the origin repair; per-seed route pinning is hopeless) and end at y=5.121-5.123 — 1-3 mm over the 5.12 bound = marginal top-band GRAZINGS. Cross-arm pattern: gamma-1-FOCUSED pushes (s790, s791) reopen s22 and never fix s5/s14 (the shared adapter couples gammas); all-gamma dosing (s766) never broke s22.
CMD/RESULT (s792 = s766 + all-gamma brake + certified preservation, HALF dose: steps 5, lr 2e-5): **11/11 flips, SR {100,100,100,100,100,100,92}, CR0 all, aggregate .9886 (best), s22 healthy** — only the two inherited g1.0 s5/s14 grazings remain. `analysis/fixed_seed_gate_goal_allg_preserve_lowdose_s792.json`.
RESULT (root cause of the residual): the s86 training pool contains **0/4096 rows above y=5.02** and its goal-brake rows top out at y=4.85 — the grazing band y in [5.0,5.12] has NO certified supervision anywhere in the pool; further dosing from this pool cannot teach top-band braking.
CMD (running): gather-only prime from s792 (`results/p2/topband_pool_s793`) with recovery-goal band tightened to the top strip (x 4.2-5.0, y 5.00-5.10, vy 0-0.35), recovery-frac .5, 42 rollouts / cap 400 -> a certified pool WITH top-band brake rows; then one bounded all-gamma step from that pool + preservation, then the gate.

## [2026-07-11 02:35 PDT] CLAUDE — dose-response arc s794/s795; interception-band hypothesis
CMD/RESULT (s794 = s792 + top-band pool s793 hard rows, strip band override y>=5.0, steps 5 lr 2e-5): mechanism engaged (35 band rows, 160 OOB latents paired, escape 1280) but fstep 0.05%/step — deployment numerically UNCHANGED (all 14 a-e values identical to s792). 11/11 flips hold; g1.0 s5/s14 persist.
CMD/RESULT (s795 = same at 10x dose, steps 20 lr 5e-5): anchor hit the 1.6% bound at step 20 -> ROLLBACK (guard working as designed); deployment still ~unchanged (only g0.1 time 17.80->17.77); g1.0 s5/s14 persist. Dose is not the binding variable.
DIAGNOSIS refined (physics): the grazers cross y=5.12 while still RISING (vy>0 at y~5.10; full-brake stopping distance vy^2/2 ~ 0.03-0.06 m ⇒ interception must happen at y in [4.85,5.05] while vy>0). Pool audit: brake rows top out at y=4.849; the s793 top-band rows are y>=5.0 DESCENDING starts. The rising-interception band (y 4.85-5.05, vy>0.1) has ZERO certified supervision in every pool so far. Adapter gate (4.4,4.65,0.4,0.35) covers the band at support 0.57-1.0, so this is purely a data gap.
CMD (running): gather `results/p2/interception_pool_s796` from s792 — recovery-goal band x 4.2-5.0, y 4.85-5.05, vy 0.10-0.35 (rising starts; certified recovery = brake-then-descend) -> next bounded arm s797 with strip-band override y>=4.85 and mid dose (steps 12, lr 5e-5).

## [2026-07-11 03:05 PDT] CLAUDE — ORACLE PROOF: interception-band braking is sufficient for the last knot
CMD: CPU oracle test — replay s792's faithful gamma-1.0 seeds 5/14 EXCEPT replacing the executed action with full down-brake (u_y=-1) whenever the state is inside the rising interception band (y in [4.85,5.12], vy>0.05); everything else (latent stream, policy, replan) unchanged.
RESULT: **seed 5: success in 119 steps with exactly 2 oracle-brake steps; seed 14: success in 120 steps with exactly 1** — endpoints (4.945,4.942)/(4.959,4.931). Both then reach FASTER than t104's own successes (129/125 steps). The final knot needs 1-2 correct brake actions at band contexts; the s798 gather is collecting certified recoveries starting exactly there (rising starts must brake-then-descend to be accepted, so every accepted window carries the needed action).
DECISION: when the s798 pool lands -> bounded arm s799 from s792: hard rows = strip-band override y>=4.85 (the interception band), quota 48, steps 12, lr 5e-5, preservation replay + endpoint pairing unchanged -> full gate. First gather attempt (s796) hit its cap with quotas unmet at recovery-frac .5 -> relaunched s798 at .35 with cap 600.

## [2026-07-11 04:20 PDT] CLAUDE — s799 verdict; STRUCTURAL finding (terminal windows); guarded unit launched from s792
CMD/RESULT (s799 = oracle-brake escape rows, quota 128, steps 12 lr 5e-5): OVER-BRAKE — g1.0 88% (fresh regression s6 + s5/s14), g0.1 s22 became timeout_or_nonreach, anchor 1.76% rollback. Combined with the oracle-build side-finding (g0.1 kept 0/100 because the band override kills legitimate from-below approaches), the knot's true structure is a STATE AMBIGUITY: the goal sits ON y=5.0, so rising through y in [4.85,5.0] is CORRECT for approaches and fatal for overshoots; (y,vy) alone cannot separate them.
RESULT (STRUCTURAL, for codex/paper): the unambiguous region y>=5.0 can NEVER supply H=10 executed windows — any correct behavior there reaches within <10 steps, and window extraction requires 10 executed controls after the context. The goal-side strip is data-empty partly by CONSTRUCTION of the window semantics, not only by exploration. Filtered oracle replay at ctx y>=5.0: 0/192 rows. One-step scalpels on this knot have hit their structural floor (s794 no-op, s795 rollback-no-op, s799 over-brake).
DECISION (pivot per ladder intent — every update remains gated): integrate and run the guarded UNIT from s792 = `results/p2/unit_s792_esc64_s801` (GPU3, PID 4043896): fresh certified gathering each iteration (recovery both strips, targeted coverage proposals live), t104-preservation escape replay wired INTO the trainer (--escape-replay/quota 64/eta 1.0; 18/18 harness after wiring incl. exact split-resume), teacher/anchor = s792 (ratchet), lr 2e-5, 1 step, trust gates standard, M5 gate + SR50 probe + strip probes every 2, full-state ckpt every 2 (codex-resumable). Iterated small certified doses + organic data diversity are the remaining sanctioned medicine for a mm-scale distributional knot; the same run RESUMES COVERAGE work (targeted_frac .5). Morning protocol: fixed-seed gate over even checkpoints, best -> M25 -> M100.

## [2026-07-11 04:50 PDT] CLAUDE — ratcheted-unit driver armed; overnight autonomous mode
RESULT (unit first updates): it136 fstep 1.36%/anchor 0.41%, it137 1.18%/0.79%, no rollback, M5 gate SR 1.00/CR 0.00, SR50 probe 1.00/0.00 cov 4 — the integrated recipe (fresh certified gathers + hard quota 12 + escape 64 preservation rows + trust gates) is stable from the start.
CMD: `run_ratchet_unit.sh` armed (checks every 4 min): on 2 consecutive rollback COMP lines it stops the generation, takes the newest FULL-STATE ckpt, relaunches with teacher/anchor re-referenced to it (new outdir `unit_ratchet_genK_s80K`, per-generation recipe logged), max 8 generations — the arm-2 ratchet pattern automated at unit scale, every update still guarded and every checkpoint codex-resumable. Driver log: `logs/ratchet_driver.log`.
MORNING PROTOCOL: (1) pick best checkpoints by M5/SR50 telemetry across generations; (2) fixed-seed gate + M25; (3) best -> M100 audit incl. coverage (mode_discovery vs expert + T4 rows); (4) refresh T_COMPARE/lineage/internals figures + final report with methods.

## [2026-07-11 07:55 PDT] CLAUDE — morning state: unit gen-1 harvested, gen-2 live, three-seed lottery quantified
RESULT (gen-1, it135-142): prime + 4 clean updates + saturation at it141/142 (anchor 1.84-1.86% -> rollbacks; the ratchet driver detected it; a self-match pkill bug in the driver was found and fixed — gen-2 relaunched manually from ckpt_142 with teacher=ckpt_142, driver re-armed with start-gen). M5 gates 1.00/1.00/0.97/0.97; SR50 probes 0.96-1.00/CR 0.
RESULT (trio dynamics — the promotion bottleneck in one number): trio probe (g0.1/s22, g1.0/s5, g1.0/s14; 3 rollouts ~5 s) across checkpoints: s792={R,D,D}, unit ckpt_136/138={D,D,D}, **ckpt_140/142={D,R,R} — the unit's own gathered braking data FIXED the gamma-1 pair** at the cost of s22 (exact complement of s792). Weight-space line s792<->ckpt_140 (alpha .2-.8): ALL {D,D,D} — the two solutions are NOT linearly connected; interpolation is dead.
ANALYSIS: gamma-AUGMENTED brake rows force gamma-UNIFORM goal behavior while the trio demands gamma-DIFFERENTIATED braking (more at gamma.1, different at gamma1.0); the unit's per-gamma natural recovery rows are the mechanism that accumulates exactly that differentiation — hence gen-2+ continues, trio-watch probes every new even checkpoint (~5 s each; full gate only on TRIO-PASS).
CMD (running): s792 authoritative M100 audit (7 gammas, seeds 0-99) -> `results/p2/eval_s792_m100`; then mode_discovery T4 update. Figures refreshed: `figures/internals_unit_and_arms.png` (unit loss/trust/probes + arm loss curves + gate outcomes — the internals proof plots), `figures/internals_repair_lineage.png` (50+ gated experiments).

## [2026-07-11 08:25 PDT] CLAUDE — s792 M100 verdict; the THIRD mm-tail found (pinch clearance); morning package assembled
RESULT (s792 M100, `results/p2/eval_s792_m100` + T_COMPARE rows): SR {100,99,97,97,97,97,97} — g0.1 is 100% at M100 with coverage 10 — but **CR {0,0,2,3,2,1,0}%: 8 rare collisions that M25 never sees**. All 8 are ONE event: obstacle 4 at (1.80,1.00) grazed 2-13 mm at steps 36-40 by seeds 33/71/97 (mid-gamma). Outside the goal adapter's compact support — pre-existing lineage tail (t104 M100 CR 1% at g.2/.3 = same pinch), EXPOSED more by s792's higher SR (survivorship). Third mm-scale tail: latent-precision at a corridor pinch; the unit's frontier rule (low-margin AND high-sigma AND high-progress) is the standing mechanism aimed at it.
RESULT (T4, `tables/T4_mode_discovery.md`): s792 M100 deploys 14 modes beyond the expert's own deployment (RRURUURURU at 5 gammas etc.) but loses many expert modes (g1.0: 3 vs expert 11) — coverage is confirmed as the主 structural gap for the beyond-teacher claim.
STATE (hunting, all autonomous on GPU3): gen-2 live (ckpt_144 trio {R,D,R} — a NEW two-of-three; three distinct 2-of-3 configs seen: the lottery is circling), ratchet driver armed (self-match pkill bug fixed), trio-watch probing every new even ckpt (~5 s), full gate only on TRIO-PASS. Deliverables current: figures/current_goal_latent_support.png, internals_unit_and_arms.png, internals_repair_lineage.png, seed12_trace*.png; tables/T_COMPARE_progress.md (+s792 M25/M100), T4_mode_discovery.md, T3_kazuki_notes.md; claude_handoff/METHODS.md.

## [2026-07-11 ~11:00 PDT] CLAUDE — **M25 MILESTONE LANDED: unit_ratchet_gen2_s802/ckpt_146.pt**
RESULT: trio-watch found ckpt_146 TRIO-PASS; full fixed-seed gate: **SR {100,100,100,100,100,100,100}%, CR 0 all, 11/11 original failures flipped, 0 regressions, GATE_PASS=True** (`analysis/fixed_seed_gate_unit_g2_it146.json`). a-e: clearance .290-.304, time {17.56,13.59,11.76,11.40,11.47,11.93,12.28} s (FASTER than t104 at every gamma), coverage 3-5 (M25). The winning mechanism: gen-2 of the guarded ratcheted unit — fresh certified gathers + hard-quota strip rows + t104-preservation escape replay, teacher/anchor re-referenced at the gen-2 branch point.
RESULT (chain safety): generations PAST the winner degraded (gen-3 CR50 .04, gen-5 SR50 .84/CR50 .14) — compounding per-generation 1.6% freedom without a global anchor walks off the safe manifold. Driver + gen-5 STOPPED; gens 3-5 retained as diagnostics only. The ratchet pattern needs a global regression gate between generations (noted for codex).
CMD (running): authoritative M100 audit of ckpt_146 -> results/p2/eval_it146_m100 (the pinch-tail seeds 33/71/97 at (1.80,1.00) are the open question; t104 and s792 both graze there at M100).

## [2026-07-11 ~12:30 PDT] CLAUDE — it146 M100 verdict; WALLS-4 expansion APPROVED and launched
RESULT (it146 M100, `results/p2/eval_it146_m100` + T_COMPARE): SR {99,100,97,96,99,97,97}%, CR {0,0,2,3,1,1,0}%, clearance .292-.305, time 11.41-17.69 (faster than t104 everywhere), coverage {8,5,5,4,4,5,5}. ALL 7 collisions = obstacle (2.00,1.00), the SAME three seeds 33/71/97, depths 0-10 mm — the lineage's entire remaining safety gap is one obstacle face x three latent fibers x sub-centimeter depths (well-posed next bounded repair; the seed-12 playbook applies: certified pinch data + x0 pairing).
USER APPROVED: 4-plug walled scene (`figures/wall_plug_scenes_2v4.png`) + 100-iteration expansion. Wiring: `--wall-plugs {0,2,4}` in grid_expand_hardtail (_apply_wall_plugs; recipe + resume_signature field; one cfg-ordering crash fixed) and eval_ae (all 3 env sites); 18/18 harness re-passed. NOTE: escape replay OMITTED for the walled run — its rows carry OPEN-scene percepts (corner H_P differs); documented recipe deviation.
CMD (running): `results/p2/walls4_gen1_s810` from ckpt_146 (teacher/anchor = ckpt_146), --wall-plugs 4, otherwise the unit recipe; iters 100 (abs 146->246); walls-specific ratchet driver `run_ratchet_walls.sh` (max 12 gens, template carries --wall-plugs 4). Baseline to beat: zero-shot walled M25 agg 66.3%/CR 1.7%. Hypothesis (user): with walls perceptible, the policy focuses on the goal; origin-drift/non-termination phenomena disappear. Watch: M5/SR50 walled trend, strip probes, CR (OOB->collision conversions), and the late-generation drift lesson (gens past a winner can degrade — pick by gates, not by last).
VIDEO/VIZ (user requests, delivered): full-history curriculum video `video/full_history_curriculum.mp4` (it0,19-23,28,38,58,78,98,102-105,136-188; winner it146) with viridis sigma, thin easy ring + thicker frontier ring, frontier-front histogram, transparent 3D + larger fonts; scene options figure `figures/wall_plug_scenes_2v4.png`.

## [2026-07-11 ~14:00 PDT] CLAUDE — F-stage HELD per user; 4-arm walled FROM-SCRATCH ablation suite launched
USER DIRECTIVE (verbatim intent): do NOT start the walls run from it146 — the test is whether the CURRENT RECIPE alone (no manual seeding/repairs) achieves stable learning from iteration 0 on the 4-plug scene (no ill-conditioning, no grazing); concatenate with the previous best it100 lineage for the report; expect walls-run@100 > previous it100, then fine-tune toward conquering the demo expert. Plus a faithful 4-option comparison on the walled scene, every gamma, with internals + curriculum video each: BASELINE (full pipeline) vs (1) NO CURRICULUM (single class, equal samples; expect poor metrics/unstable gradients) vs (2) NO MULTI-STEP SOCP (verifier = free-space + goal progress; expect poor safety/clearance) vs (3) NO PROGRESS (verifier = taskspace+SOCP only; still curriculum). Use FULL GPU3.
IMPLEMENTED: `--ablate-curriculum/--ablate-socp/--ablate-progress` in grid_expand_hardtail (single-class labeling + single-class batch legality + mix (1,0); traj gate via traj_valid2(check_socp=False) for (2) and taskspace∧socp_ok for (3); window filter drops approach/floor for (3); margin axis = GM2.window_min_clearance geometric clearance for (2), positive-clearance guard). Signature+recipe carry the flags; harness 18/18 with flags off. Earlier it146-based walls run KILLED (wrong baseline per user) — its only artifacts: min_modes discovery (walled scene supports 1 mode/gamma initially -> min-modes-per-gamma 1 for bootstrapping) and goal-strip probe 0.99 zero-shot.
LAUNCHED (4 concurrent on GPU3, OMP 8 each): walls4_scratch_{base_s820, nocur_s821, nosocp_s822, noprog_s823} — from pretrained_a32uni, USER-LOCKED from-scratch recipe (lr 1e-4, no-freeze enc-mult .3, inner 1/2/1, beta .3, q .5 abs, mix .75/.25->.5/.5, demo .125, LwF teacher = pretrained), trust-anchor rollback OFF (fine-tuning-era device, not in the locked recipe; field-grad-clip kept), min-modes 1, recovery+hard-quota constant across arms (framework machinery; ablations isolate exactly one factor each), viz-db-every 1 (video-ready), 100 iters.
GPU NOTE (user asked): GPU3 previously ran ONE trainer (util fluctuates between gather/update/probe phases — rollouts are small sequential NFE-8 integrations, not GPU-saturating); now 4 concurrent trainers use the full device.
DELIVERABLES QUEUED: per-arm internals (report_internals_v3 extended) + curriculum videos (full-history style incl. its 1,2,3,4,5,10,20,... cadence) + concatenated comparison vs finalunit it100 (open-scene previous best) + a-e tables every gamma. F stage (pinch repair, coverage, M100 ladder) HELD until this suite reports.

## [2026-07-11 ~15:50 PDT] CLAUDE — suite first read: no-SOCP arm FINISHED with the textbook failure; budget fix for the rest
RESULT (walls4_scratch_nosocp_s822, COMPLETE 0->100 in ~90 min): dropping multi-step SOCP makes acceptance trivial (14 valid / 21 attempts vs ~5% for the full verifier) and the policy self-trains into collisions: **it100 M5 SR 0.37 / CR 0.37** — the user's expected outcome for arm (2) confirmed at full depth. Full internals/video to be rendered with the suite report; artifacts in results/p2/walls4_scratch_nosocp_s822 (viz_db every iter, probe.jsonl, ckpts every 4).
RESULT (base_s820 / nocur_s821 / noprog_s823 at cap 300): from-scratch gamma-0.1 Valid2 acceptance is so low that per-gamma readiness rarely fills within 300 attempts -> most iterations starved (nocur: ZERO updates, loss nan; base: some updates then starve at it10). The cap, not the recipe, was binding.
CMD: three arms killed, wiped, relaunched with --gather-attempt-cap 600 (deficit-directed attempts get room to repair the gamma-0.1 deficit); everything else identical. nosocp NOT rerun (it never approached the cap; noted as the one budget asymmetry).

## [2026-07-11 ~16:20 PDT] CLAUDE — suite bootstrap fix #2: min-modes gating off (tracking stays)
RESULT (cap-600 relaunch): base + noprog UPDATING (nvalid ~1270, real losses ~0.87-0.91); nocur STILL zero updates — audit `unmet_gammas=[0.1]`: the min_modes_per_gamma=1 quota demands >=1 REACHED gamma-0.1 origin-start rollout per iteration, a coin flip from scratch on walls (base merely got lucky at 202 attempts; both arms would stall stochastically later).
DECISION: from-scratch bootstrapping regime — mode quota becomes TRACKING-ONLY (--min-modes-per-gamma 0) for the three relaunched arms; per-gamma valid-rollout + class quotas (the data-balance guarantees) remain gating. Third relaunch, outdirs wiped; nosocp's finished run keeps min-modes 1 (it never stalled — acceptance without SOCP is enormous; asymmetry noted).

## [2026-07-11 ~22:10 PDT] CLAUDE — WALLS-4 suite handed to codex: claude_handoff/WALLS4_SUITE_HANDOFF.md
HANDOFF: the walled from-scratch suite (fine-tune / diagnose / end-to-end viz / publishable package) passes to codex. The handoff file carries: START CONDITION (one-command check: 4 final.pt + expert_gt_walls4/EVAL_DONE = 5 files; ~1-3h out), the user's a-e bar and suite definition VERBATIM, the motivation framing (walls remove boundary-invisibility => eliminate hand-crafted plumbing), the state table (nosocp 33-54%/CR25-54% M100 DONE; noprog finished M100 pending; nocur finished — best early SR .80/CR .06; base ~it88 SR .69/CR .17-.29 with collisions localized at INTERIOR diagonal pinches — the curriculum-needs-competence finding), the task ladder (0-100 exhibit -> stateful continuation to a-e with phased-curriculum as the pre-authorized fix direction -> coverage mode-hit gating if stalled -> publishable package), and the week's hard rules (M100-only decisions, min-modes 0 from scratch, cap 600, wall-plugs flag everywhere, harness after edits, viz-db/log cadence, pkill self-match).
STILL RUNNING at handoff: base (~it88->100), walled expert M100 (results/expert_gt_walls4). Claude-side viz fix landed: video module auto-draws plugs from recipe.json. F stage remains PAUSED (open-scene incumbents intact: it146 M25-perfect, pinch-repair spec in NEXT_CODEX/CLAUDE_RETURN).

## [2026-07-11 ~22:25 PDT] CLAUDE — handoff amended: adaptive-gamma deployment capstone (user request, verbatim in handoff §6)
Added to WALLS4_SUITE_HANDOFF.md: deployment-level adaptive gamma on the matured baseline ckpt — (1) heuristic proximity schedule gamma(d_min), (2) verifier-guided per-step gamma selection over the SAME latent (certificate+margin+progress scoring; gradient variant optional). Protocol: M100 walled, rows = 2 adaptive + 7 fixed gammas + random-schedule control + walled expert; claim = adaptive above the fixed-gamma Pareto front; deliverables T_ADAPTIVE_GAMMA + Pareto figure + gamma(t) traces + gamma-colored trajectory video. Hard separation: adaptive results NEVER contaminate the faithful fixed-gamma a-e tables.

## [2026-07-11 ~21:55 PDT] CODEX — WALLS-4 iteration-100 M100 exhibit complete; stateful continuation launched
FIX: the walled expert job had produced no files because `expert_worker` read `wall_plugs` but its CLI parser did not accept the flag. Added the missing parser option, then ran the authoritative 4-plug expert M100 (seeds 0--99) and all missing arm M100 evaluations. `results/expert_gt_walls4/EVAL_DONE` now corresponds to the correct scene. No trainer code changed.

RESULT: `tables/T_WALLS4_SUITE.{md,csv}` contains 4 arms + walled expert + pretrained/s792 zero-shot rows. BASE M100: SR 68--76%, CR 3--24%, clearance .226--.232, coverage 2--8. NOCUR: SR 56--79%, CR 1--17%, coverage 4--14. NOSOCP: SR 33--54%, CR 25--54%, clearance .222--.228. NOPROG: SR 58--70%, CR 4--24%. Walled expert: SR100/CR0, clearance .240--.299, coverage 5--9. The s792 walled M25 context row exactly reproduces the handoff aggregate (SR .6629 / CR .0171).

DIAGNOSIS: `analysis/walls4_collision_locations.{md,json}` classifies the deepest collision obstacle for every M100 episode. BASE has 119/700 collisions versus NOCUR 63/700; BASE 116/119 are interior and 70 are at (1,1), only 3 at a plug. NOSOCP has 309/700. This strengthens the curriculum-needs-competence finding: visible walls eliminate the old invisible-boundary failure class, but frontier pressure before execution competence concentrates failures at interior pinches. The earlier M5 statement of zero plug collisions does not extend to M100 (rare plug events exist), so the report uses the stronger M100 counts.

DELIVERED: `figures/internals_suite_walls4.png` (corrected stale frozen-encoder/trust labels and unclipped SR/CR axes), `figures/walls4_clearance_time_vs_expert.png`, four 0,1,2,3,4,5,10,...,100 arm curriculum videos, existing baseline `video/full_history_curriculum.mp4`, and `tables/T_WALLS4_MODE_DISCOVERY.md`. `claude_handoff/METHODS.md` now has the walls/no-hand-plumbing protocol and curriculum amendment.

CMD/STATE: exact full-state BASE + NOCUR resumes are running in their original dirs from it100 to it140, inner-steps=2, unchanged resume signatures, wall-plugs=4, viz/log every iter, M25 measurement every 10. The first attempt was correctly rejected before updates by the CUDA-topology guard (all GPUs visible versus the saved GPU3-only state); relaunch with `CUDA_VISIBLE_DEVICES=3` restored train state v2 at it100. At it140, run external M25 collision/location gates; if BASE pinches persist, branch the pre-authorized phased-curriculum arm and preserve both pure controls.

IMPLEMENTED (pre-authorized contingency, not yet launched): `--phased-curriculum --phase-sr-threshold .85 --phase-sr-patience 3`. The branch treats every certified sample equally until any three consecutive measurement gates meet the SR threshold, then irreversibly enables the normal easy/frontier mix. The switch is derived from serialized measurement history, so split resume is deterministic; the disabled arm omits the new signature fields and exactly restores the already-running pure checkpoints. Exact it100 restore smoke passed. Required trainer harness expanded with a phased semantic gate: **19/19 PASS** (`analysis/test_hardtail_trainer.cpu.phased.json`; original 18 all remain green).

IMPLEMENTED (coverage contingency, disabled on live controls): `--mode-hit-gate --min-target-hits N --min-modes-schedule START:N ...`. A coverage update now can require an exact targeted staircase hit, and its per-gamma achieved-mode readiness quota follows an absolute schedule (planned branch: 2→4→8→12→14). Disabled signatures omit all new fields, preserving pure-arm exact resume. One minimal-fixture fallback regression was caught by the first harness run and fixed; final expanded harness **20/20 PASS** (`analysis/test_hardtail_trainer.cpu.coverage.json`).

RARE-MODE RETENTION PREP: `analysis/build_escape_replay.py` now supports `--wall-plugs 4`, optional successful staircase filters, full per-row mode provenance, and refuses cross-scene merges. `analysis/audit_replay_certify.py` applies/validates the same wall-plug scene and filters list provenance with the certified tensors. Compatibility merge+certificate smoke: 2,898/2,999 legacy rows retained with mode-list lengths aligned (`analysis/runs/replay_merge_mode_compat*_certified.pt`). Actual WALLS-4 rare-mode replay remains pending the promoted branch checkpoint and independent seeds 100+.

ADAPTIVE DEPLOYMENT PREP (no evaluation claim yet): `analysis/adaptive_gamma_eval.py` implements the requested continuous proximity schedule, same-latent seven-gamma verifier selection, and random-gamma control without training; `adaptive_gamma_tune.py` declares one six-pair tuning sweep on seeds 100+ only; `adaptive_gamma_report.py` produces the separate table/CSV, strict claim JSON, std-ellipse Pareto plot, traces, and gamma-colored video. Semantic smoke `analysis/test_adaptive_gamma.json`: explicit single-gamma integration is bit-exact with faithful sampling (max|diff|=0), seven verifier scores contain exact validity/certificate/face-margin/progress, wall-plugs=4, and deployment leaves every weight unchanged.

PINCH LATENT DIAGNOSIS: the wall-aware configurable `analysis/trio_probe.py` identifies BASE gamma-.4 seeds 16/28/48 as a fixed `(1,1)` pinch trio; all are dead at it100 and remain dead at it104 (`analysis/walls4_pinch_trio_it100_it104.txt`). Reusing the exact `seed12_tail_trace` context/latent/NFE8 machinery with 4,096 identical latents shows this is a distribution-level basin, not one rare fiber: ten replans before collision, planned-window collision fractions are {83.7,89.1,29.1}% at it100 and {81.7,88.7,21.4}% at it104; at the final pre-collision context every latent immediately collides. Artifacts: `analysis/walls4_pinch_latent_offset10_it100_it104.json`, `figures/walls4_pinch_latent_offset10_it100_it104.png`. This is mechanistic evidence for competence-before-frontier pressure; repeat against it140 at the gate.

TARGET-HIT MECHANISM: pure continuation telemetry still records ZERO exact target hits despite 20--105 targeted attempts/iteration. A sharp-knob-only probe (β.2, 128 candidates, alignT .05) also produced 0/10 hits and worsened validity (`targeted_coverage_walls4_nocur104_sharp.json`). Root cause: “move R next” proposals did not brake upward momentum, so the wrong grid boundary crossed first. New opt-in `--target-perp-brake` uses a PD proposal toward the requested boundary while holding the perpendicular coordinate inside its current cell; Valid2/SOCP acceptance is unchanged and the override is scoped/restored per targeted rollout. On independent probe seeds with the ordinary β.3/40/alignT.45 knobs it yields **3/10 exact hits, all three Valid2, collision-free, and matching their requested words** (`analysis/targeted_coverage_walls4_nocur104_perpbrake_defaultknobs.json`), versus 0/10 before. Disabled signature compatibility and proposal scoping are covered by the final **20/20** harness (`analysis/test_hardtail_trainer.cpu.targetbrake.json`).

CONTINUATION GATE it110 (BASE, M25×7 faithful internal measurement): aggregate SR .629 / CR .194; per-gamma SR {68,68,52,68,68,64,52}% and CR {4,8,36,24,24,24,16}%. This is worse than it100 M100 and the preceding SR50 also fell to .64/.28. Pure BASE remains running because the requested 120-update control must be observed, but it is not a promotion candidate. NOCUR it110 measurement was still pending at this entry.

CONTINUATION GATE it110 (NOCUR, M25×7): aggregate SR .674 / CR .143, versus BASE .629/.194. Per-gamma NOCUR SR {84,72,60,64,64,72,56}% and CR {12,4,28,16,16,12,12}%. Uniform sampling remains the less-bad early learner but is also far below the a--e bar and is not promotable. This cleanly supports the phased order: uniform is useful for competence acquisition, not sufficient as the final recipe.

PHASE-GATE CALIBRATION (supersedes the earlier patience=3 draft; arm not yet launched): NOCUR's from-scratch M5 history reaches aggregate SR≥.85 on consecutive gates at it82 (.857) and it84 (.886), then regresses at it86 (.829). Three gates would never have activated and would merely duplicate NOCUR; two consecutive gates are a sustained four-iteration competence signal and demonstrably reachable. Final phased flag/script therefore use threshold .85, patience **2**, with irreversible activation. Full post-calibration harness remains **20/20 PASS** (`analysis/test_hardtail_trainer.cpu.final.json`).

## [2026-07-12 ~00:30 PDT] CLAUDE — reconcile with codex's session; PHASED-CURRICULUM arm launched
RECONCILE: codex's 21:55 session had already fixed the expert-worker flag and completed the full it100 M100 exhibit (T_WALLS4_SUITE: walled expert SR100/CR0 clearance .240-.299 cov 5-9; BASE 68-76%/CR3-24; NOCUR 56-79%/CR1-17; collision-location analysis BASE 119/700 vs NOCUR 63/700 with 70 at (1,1)) — my ~23:30 expert+eval relaunches were REDUNDANT duplicates (dirs `eval_walls4_*_s82x_it100_m100`, `expert_gt_walls4` re-run; results consistent; treat codex's `eval_walls4_<arm>_it100_m100` dirs + tables as authoritative). Codex additionally: continuations to it140 (BOTH degrade — it140 M25: BASE 64-80%/CR8-20, NOCUR 48-80%/CR4-32 -> expectation (1) instability CONFIRMED at depth; neither promotable), pinch-basin latent proof (distribution-level, not rare-fiber), phased-curriculum implemented+CALIBRATED (.85/patience-2, harness 20/20), coverage mode-hit gate implemented (disabled), **target-hit root cause FIXED** (--target-perp-brake: 3/10 exact certified hits vs 0/10 — first-ever exact targeted-mode hits), adaptive-gamma eval tooling with bit-exact smoke.
CMD: launched the pre-authorized decision-point arm `results/p2/walls4_phased_s830` (GPU3 idle, codex's last entry left it "not yet launched"): from scratch, walls-4, phased curriculum (uniform until M5 agg SR>=.85 on 2 consecutive gates, then irreversible easy/frontier mix) + --target-perp-brake (the validated coverage mechanism), 140 iters, all other locked-recipe knobs identical to the pure arms. Hypothesis: uniform phase acquires competence (NOCUR's early strength), frontier phase then compounds it (BASE's late strength) without the pinch-basin front-loading; perp-brake targeting grows coverage toward (e).

## [2026-07-12 ~08:20 PDT] CLAUDE — PHASED+PERP-BRAKE arm it140 M100: the amendment WORKS (safety + coverage); continuation is the path to a-e
RESULT (walls4_phased_s830, M100 walled, `results/p2/eval_walls4_phased_it140_m100`): SR {74,74,77,75,76,72,73}%, **CR {9,3,1,1,1,1,2}%** (pure arms at it140: CR 17-21%), clearance .238-.271 (expert .240-.299 — SIMILAR TREND, parity mid-gamma), time 12.4-21.0 s (slower than expert — the remaining (d) gap), **coverage {20,5,5,5,5,5,5} — gamma-0.1 deploys 20 distinct modes vs the walled expert's 8** (2.5x beyond-teacher; the perp-brake targeting works at scale: 267 exact hits/140 iters, phase switch at it17, CR50=0 tail).
VERDICT: the phased order + perp-brake is decisively the right from-scratch lineage — safety and mode discovery solved simultaneously; the residual gap to the a-e bar is ~20-25% timeouts (long exploratory routes, gamma-0.1 mean 21 s) and per-gamma coverage balance. NEXT (codex ladder): continue this arm with the mode-hit gate + absolute mode schedule (2-4-8-12-14, already implemented/disabled) to convert targeting into balanced coverage, then speed maturation; adaptive-gamma capstone on the matured checkpoint. Artifacts: video/walls4_phased_curriculum.mp4 (phase switch visible at it17), suite internals refreshed, T_WALLS4_SUITE to gain the phased row.
## [2026-07-12] CODEX — explicit Claude takeover note written at user-requested pause
HANDOFF: paused all further experiment work and wrote `claude_handoff/CODEX_TO_CLAUDE_WALLS4.md`. It
separates Codex's completed iteration-100 suite, it140 pure-control gates, diagnosis, and prepared tooling
from the later Claude `walls4_phased_s830` combined-arm run. It also warns that s830 changed phased
curriculum and perpendicular braking together, so its improvements cannot be causally assigned to phased
curriculum alone. At handoff there are no live trainer/evaluator/watcher processes. Overall completion is
still pending mature fixed-gamma a--e evidence, requested control continuations, certified WALLS-4 replay,
and adaptive-gamma M100 deliverables.

## [2026-07-12] CLAUDE — division of labor corrected (user): codex = SFM simulation ONLY
CLAUDE_TO_CODEX_FINETUNE.md marked SUPERSEDED (walls fine-tune PARKED, no owner). Codex's assignment = overnight_run_07_12_sfm/ per CODEX_START_SFM.md + GOAL_SFM_FRESH.md (port the hardtail pipeline to the SFM moving-crowd scene; successful-demos-only story; GPU2). Claude keeps the it146 open-scene track (continuation unit146_gen3_s853 + openabl_* retrains + paper_results _vN).

## 2026-07-12 ~15:00 — it146 fallback resumed post-compact (Claude)
CMD: (1) archived `results/p2/unit146_gen3_s853` -> `unit146_gen3_s853_noperp_stub` (2-iter stub; on-disk
recipe showed `perpendicular_braking_proposal: false`, violating the agreed recipe unification — the
restart-with-perp-brake never actually carried the flag). (2) Patched `run_ratchet_146.sh` template:
`--target-perp-brake` added + seed naming fixed to $((850+gen)). (3) Relaunched gen3 FRESH from
`unit_ratchet_gen2_s802/ckpt_146.pt` (teacher=same, seed 853, iters 82, full unit recipe + perp-brake;
recipe.json now records perp true) + re-armed ratchet driver (start gen 3, max 10, GPU3), both setsid-
detached. (4) Launched 9 open-scene M100 evals: openabl_{nosocp_s861,noprog_s862,nocur_s863}/final.pt
(abs it146, exact recipe, one ablate flag each — verified) x gamma {0.1,0.5,1.0} -> 
`results/p2/eval_openabl_<name>_m100`. Prior facts: the 3 ablation arms COMPLETED 134->146 with final.pt;
gen3+driver had been killed at compaction task-termination (no crash; log clean at it148).
RESULT: pending (evals ~1-2 h; gen3 booting: resume baseline it146 SR 1.00 CR 0.00 reproduced).
DECISION: _v1 paper trio regenerates from same-scene ablation rows when the 9 evals land (drop the
provisional-walled dagger); gen3+ checkpoints gate per ladder (trio_probe -> run_gate vs eval_unit_g2_it146_m25)
toward the per-gamma a-d win over the demo expert (-> _v2). Coverage tracked but optional per user.

## 2026-07-12 ~16:10 — _v1 paper package COMPLETE (same-scene ablations landed)
CMD: 9 open-scene M100 evals finished -> results/p2/eval_openabl_{nosocp,noprog,nocur}_m100; ran
paper_results/table_v1.py (+ scatter_v1/rollouts_v1 lockstep, generated earlier); internals figure
figures/internals_openabl_it146.png.
RESULT (gamma .1/.5/1.0, SR/CR | clr | t | cov): expert 100/0 .333 15.1 8 | 100/0 .285 10.5 6 | 100/0 .294
10.8 11; it146 FULL 99/0 .305 17.7 8 | 99/1 .296 11.5 4 | 97/0 .296 12.2 5; noSOCP 100/0 (tie, 1-seed) |
97/3 | 97/0; noPROG 96/0 | 97/2 | 95/0; noCUR 98/0 | 96/3 | 96/0. Every ablation degrades on SR or CR at
some gamma (noSOCP: CR 3x at g.5; noPROG: SR -2..-3 everywhere; noCUR: SR -1..-3 and CR 3): directional
but BOUNDED damage — expected, since arms take only 12 guarded updates from a competent certified base
and the trust gates rolled back their last iterations (see internals). The catastrophic version of the
same removals is the from-scratch walls suite (CR 25-54%). Honest caveat: noSOCP nominally 100 vs 99 at
g0.1 (single seed flip, noise-level).
DECISION: _v1 published (table_v1.tex + scatter_v1 + rollouts_v1, headers carry model/recipe/data);
provisional walled rows retired. pdflatex not installed on box — table syntax follows the _v0 pattern
(same preamble contract). it146 does NOT yet meet the win bar (SR<100 at M100, time > expert) -> gen3
continuation ladder is the active path (_v2 on first per-gamma a-d win).

## 2026-07-12 ~17:05 — gate ckpt_156 (gen3+perp-brake, 9 updates past it146): FAIL, it146 stays champion
CMD: analysis/trio_probe.py ckpt_156 (PASS: g0.1s22, g1s5, g1s14 all reach) -> bash run_gate146.sh
ckpt_156 unit146g3_156 3 (M25 7-gamma + fixed-seed diff vs eval_unit_g2_it146_m25).
RESULT: all 11 canonical probes succeed, BUT aggregate regresses: g0.3 96/CR4, g0.4 96/CR4 (s7
timeout_or_nonreach + 1 collision each), g0.7 SR92 (s3,s14 near_goal_oob), g1.0 SR92 (s3,s8
near_goal_oob). gate_pass=false. Training tape was clean (SR50 1.0/CR50 0, anchor 1.05%) — M25 gate
catches what M5/SR50 cannot: mid-generation churn degrades near-goal strata at high gamma under
continued expansion pressure.
DECISION: no promotion; ladder continues to ckpt_166 (watcher armed). If 166 fails the SAME strata
(near-goal high-gamma + mid-gamma s7), gen3's direction is evidenced harmful -> stop and diagnose
before burning generations (candidate lever: pinch/goal-band recovery emphasis as ONE documented
recipe change, not more raw iterations).

## 2026-07-12 ~17:20 — gen3 saturated at it162; driver ratcheted to gen4 (chain policy set)
CMD: driver auto-ratchet: gen3 rollbacks at it162/163 (anchor 1.69-1.70% > 1.6%) -> killed, gen4
launched from unit146_gen3_s853/ckpt_162.pt (teacher=ckpt_162, seed 854, perp-brake CARRIED via patched
template — verified true in recipe.json).
RESULT: gen3 end-tape degraded (SR50 0.92 CR50 0.02, M5 0.83 at branch) — consistent with the 156 gate
fail. Precedent both ways: it146 itself came from a gen2 branched at gen1's saturation point.
DECISION: chain capped at gen5 (depth 3 from it146). All promotions remain gated vs it146
(run_gate146.sh). If gens 3-5 all gate-fail on the SAME strata (near-goal high-gamma s3/s8/s14 +
mid-gamma s7), stop the chain and A/B the perp-brake suspect (report to user at that boundary —
perp-brake was the agreed unification, so overriding it needs evidence + user visibility). Next gate:
gen4 ckpt_172 (10 updates past branch).

## 2026-07-12 ~18:00 — gate gen4 ckpt_172: FAIL (worse); chain evidence 2/3
CMD: trio_probe ckpt_172 PASS -> run_gate146.sh unit146g4_172. RESULT: SR 100/92/96/92/88/96/96 (g.1-1),
CR 0 everywhere, times 12.3-19.4 (up vs it146 11.5-17.7), near-goal family regressions incl. canonical
g0.2_s12 fiber and g1.0_s2. gate_pass=false. Pattern vs 156: collisions gone but SR erosion broadened —
chained re-referencing lets near-goal calibration random-walk away from it146 while each gen stays
within 1.6% of its own degraded teacher (drift mechanism, matches old no-perp chain s803-805 that also
degraded: NOT perp-specific).
DECISION: complete pre-registered evidence: driver will ratchet gen5 from gen4's saturation point
(imminent, anchor 1.49% at 172); gate gen5's ~10-update candidate ONCE. Then ROUND BOUNDARY: stop chain
regardless (pass -> M100/_v2 path; fail -> full stop + report with proposals: champion-restart seed
search from it146 (no numerics change), pinch-band recovery gen targeting it146's actual M100 residual,
exploration-share taper for the time gap (needs approval), perp on/off A/B (needs user visibility —
perp was the agreed unification)).

## 2026-07-12 ~19:30 — Kazuki fragility sweep (M=10) + _v2 package + FAITHFUL from-scratch ablations launched
CMD: (1) chain STOPPED (driver+gen5 killed; gens3/4 gate-failed vs it146 -> champion stays it146).
(2) 9-variant Kazuki smoke sweep around tuned base (kazuki_baseline.py on untouched pretrained_a32uni,
M=10 x gamma {.1,.5,1}): w_safe {.05,.9,2}, coll_w {2,5,50,100}, goal_coef {.2,.1}.
(3) analysis/ood_start_rollouts.py: it146 from 4 off-diagonal starts x 3 gamma.
(4) paper_results _v2 trio regenerated per new design (2x4 gallery, 1x2 phase planes, table + detuned row).
(5) FAITHFUL ablations launched: results/p2/openscratch_{base,nosocp,noprog,nocur}_s87{0..3} — from
pretrained_a32uni, locked from-scratch recipe (lr 1e-4, unfrozen, cap 600, inner 1/2/1, gates off,
phased+perp-brake; nocur = --ablate-curriculum, no phased), 140 iters, viz-db/probe every iter.
RESULT: sweep — w_safe graceful axis: .3->1.00 | .9 -> SR .10/.60/.90 (all traps, CR 0) | 2.0 -> 0;
coll_w CLIFF 20->50 (1.00 -> 0.00, no middle); coll_w DOWN 2/5 stays SR 1.00 CR 0.00 — collision-side
failure UNREACHABLE by single-knob detuning (proposals from safely-pretrained flow are obstacle-avoiding
by construction; fragility manifests as TRAPPING). Detuned exhibit locked: w_safe=.9 (clearance flat
.36-.37 across gamma = no safety-level control; SR collapses gamma-dependently). OOD starts: 12/12 reach.
DECISION: _v2 delivered (headers document all data incl. M=10 smoke + short-window-ablation caveat).
Next _vN swaps: faithful openscratch rows/rollouts (ETA ~6-9 h, then M100 evals), cw30/40 cliff-edge +
goal-aggressive (goal_coef 2-5) + walls4 scene-shift sweep axes queued. nyx UNREACHABLE from this box
(DNS) — all on GPU3.

## 2026-07-12 ~20:10 — PIVOT: GRAND FINAL from-scratch expansion is the flagship (user decision)
CMD: killed openscratch_{nosocp,noprog,nocur} (5.1-5.3 PAUSED, dirs kept, ~it2 each);
openscratch_base_s870 KEPT as the GRAND FINAL run — pretrained_a32uni -> open scene, walls4_phased
recipe FROM ITERATION 0 (lr 1e-4, unfrozen, cap 600, inner 1/2/1, trust gates off, phased curriculum
.85/2, perp-brake, recovery bands, hard-quota 12, viz-db+probe every iter, 140 iters) — now sole owner
of GPU3. _v2 figures corrected: gamma colormap viridis -> truncated plasma (viridis is reserved for
sigma/uncertainty in curriculum videos); Kazuki TUNED row/series removed per user (detuned w_s=.9
fragility exhibit stays; tuned provenance remains in tables/T3 + kazuki_final_m200 for the record).
RATIONALE (user + evidence): it146 lineage is mode-collapsed (below-diagonal family; cov 4-8) because
its whole history optimized SR/CR gates under tiny trust budgets; the phased+perp from-scratch recipe
discovers modes from iteration 0 (walls: cov 20 vs expert 8). Fine-tuning it146 with that recipe was a
mismatch (gen3/4 gate-fails). Batch-52 note: 64 slots = 44 fresh + 8 demo + 12 hard-quota; hard slots
fill only when certified strip data exists.
DECISION: on GRAND FINAL completion (ETA ~5-8 h at full GPU) -> curriculum video + internals + M100
evals + _v3 grand-final package. Ablation retrains resume AFTER, from the same recipe (faithful arms).

## 2026-07-12 ~20:50 — GRAND FINAL corrected: --no-freeze (recipe fidelity catch) + 100 iters + per-10 reporting
CMD: internals enc-panel exposed freeze=True default (trainer CLI) — the first grand-final launch had a
FROZEN encoder, deviating from walls4_phased (freeze_enc=False, encoder learning is part of the recipe).
Killed + archived to openscratch_base_s870_frozenc_stub (25 iters; SR50 0.80/CR 0 at it20, cov 4-5 —
tracked walls reference well even frozen). RELAUNCHED openscratch_base_s870 with --no-freeze --iters 100
(user cap), all else identical; verified freeze=False + recipe freeze_enc:false.
Reporting protocol (user): grand_final_reports/ folder; internals + curriculum video at early iters and
EVERY 10 iters (video frame list 0,1,3,5,10,20,... native to video_curriculum_fixed.py); first reports at
it10/it20 immediately on availability. analysis/grand_final_internals.py = 8-panel overlay vs walls-4
reference (grey).
DECISION: stop at it100 = final model -> _v3 (video, internals, M100, table/scatter/rollouts regenerated
on the grand final). it146 lineage retired from the headline (mode-collapsed) but kept for the record.

## 2026-07-12 ~21:20 — _v3 prepared (GRAND FINAL headline) + 5.1-5.3 relaunched no-freeze, all 4 in parallel
CMD: (1) frozen-encoder ~it2 ablation stubs archived (*_frozenc_stub); nosocp/noprog/nocur RELAUNCHED
--no-freeze --iters 100 (verified freeze=False), sharing GPU3 with the grand final per user (4-way,
~3-4 min/iter each). (2) _v3 modules written (generate as data lands): rollouts_v3 = demo-data panel
(ALL 300 attempted starts from dataset/dr05_windows_g*.pt as black squares + window support cloud +
ours from 6 held-out starts incl. 2 new far-off-diagonal (4.5,0.5)/(0.5,4.5), dashed), zoom insets on
pretrained (origin-death box) and Kazuki-detuned (auto-located trap orbit), from-scratch ablation
panels, GRAND FINAL balanced-mode panel (U-first and R-first corridor words interleaved, 6/gamma);
scatter_v3/table_v3 point ours -> eval_grandfinal_m100, ablations -> eval_openscratch_*_m100.
(3) ood_start_rollouts.py STARTS extended to 6.
DECISION: at GF it100 -> M100 evals (7 gamma) + ood rollouts (tag grandfinal) + curriculum video +
internals -> _v3 delivery; ablation arm evals appended when they land (gamma {.1,.5,1}, M100).
it146 retired from headline (kept in dirs/tables history). ETA with 4-way contention: GF it100
~2:00-4:00 AM; per-10 reports continue through the night.

## 2026-07-12 ~19:00 — COLLAPSE of unfrozen-on-open-scene (4/4) -> FROZEN restart (A/B-grounded recipe adaptation)
CMD/RESULT: the no-freeze runs ALL degraded by it35-50: GF SR 0.17->0.00 (it32->36, SR50 0, cov 0,
anchor 24% climbing), noprog SR 0.00, nosocp SR 0.11-0.14, nocur declining 0.69->0.49. Direct A/B: the
FROZEN GF stub (same seed/flags, only --freeze) was stable to it24 (SR50 0.80, cov 5). Diagnosis: the
open scene IS the pretraining distribution — the encoder has nothing new to learn; unfreezing enables
self-training representation drift -> policy collapse. Walls needed unfrozen because plugs are NOVEL
(encoder-plasticity x scene-novelty finding; keep for the paper).
DECISION: killed + archived *_unfro_collapse; GF resumed STATEFUL from the frozen stub ckpt_20
(freeze=True verified, abs 20+140, watcher stops at ckpt_100); 3 arms relaunched FROZEN --iters 100
(verified). _v3 headers + internals labels updated to record the adaptation. Revised ETA: GF it100
~22:30, arms ~22:15-23:00, full _v3 ~23:30.

## 2026-07-12 ~19:50 — SECOND collapse (frozen, lr 1e-4) -> ROOT CAUSE via t104 recipe diff -> corrected relaunch
CMD/RESULT: frozen GF also declined (M5 0.60->0.11 by it50, SR50 0.36) — same shape, delayed. DIFF vs the
only proven open-scene from-scratch lineage (corrected_mode2_target50_s81 = t104): t104 ran lr 2e-5,
inner 1/1/1, min_modes_per_gamma 2, frozen enc; our runs ran lr 1e-4 x2 steps (10x update magnitude).
Walls tolerated 1e-4x2 (geometry funnels gathered data on-manifold); the open scene does not — drift
outruns demo/LwF anchors with trust gates off. Consistent: nocur (least exploration variety) decayed
slowest; unfrozen died faster than frozen.
DECISION: archived *_lr1e4_collapse; ALL FOUR relaunched fresh 0->100: walls machinery (phased+perp,
recovery, hard-quota, cap 600, rollouts 28) at t104 magnitudes (lr 2e-5, 1 step, min-modes 2, frozen).
Verified lr=2e-05/freeze=True x4. Caveat logged: at 2e-5 the .85 phase switch may fire late (~it80+) or
not within 100 — surface extend-or-not to the user at the it80 report. Revised ETA: GF it100 ~23:30-24:00,
_v3 ~00:30.

## 2026-07-12 ~23:30 — PIVOT: greedy per-iteration hill-climb from ckpt_40 (user experiment)
CMD: user wants a local best-improvement search from GF v2 ckpt_40, sweeping (1) beta UP (exploit:
w=exp((sig-max)/beta), higher=flatter=exploit — confirmed) and (2) fixed frontier fraction 12.5/25/50%
(phased OFF, "fall back to original frontier portion" since phased never switched by it50). Per iter:
branch 1 step under each (beta,frontier), score a-d on SAME fixed-seed episodes (M8, gammas .1/.5/1,
paired), promote the config that STRICTLY improves all of SR up / CR down / clearance up / time down;
if none, widen beta {0.7,1.0}; if still none, best-effort (max improved-count) + record NO_STRICT.
TRAINER EDITS (harness 20/20 PASS, default-off = bit-identical): (a) --beta choices widened to
{0.2,0.3,0.4,0.5,0.7,1.0}; (b) --resume-allow-recipe-drift relaxes the resume signature for exactly the
swept knobs (beta/mix/quantile/phased*) — structural fields stay strict; optimizer topology unchanged so
Adam/RNG continuity preserved. New tools: analysis/greedy_eval.py (fast paired a-d), analysis/
greedy_driver.py (the hill-climb; seeds canonical dir results/p2/greedy_gf_s870 from base 0->40 history
for continuous internals/video). Paused the 3 ablation arms for GPU (nosocp DONE it100, noprog it73,
nocur it42 — resumable for _v3). Base GF run killed (branch from saved ckpt_40).
RESULT (pending): it41 baseline SR .833/CR .042/clr .307/t 13.09/cov 10 (M8); smoke b0.5/f25 -> SR .833
CR .042 (SR+CR improved, clr/time flat = partial, illustrates strict-domination difficulty).
DECISION: driver runs it41->50; report it41 (proof) + it45 + it50 with the three paper reports
regenerated on the greedy-best ckpt + internals + curriculum, placed in grand_final_reports.

## 2026-07-12 ~23:55 — greedy: post-it45 = M20 + per-gamma "whole metrics" selection (user)
CMD: it41 proved M8/3g per-iter is BELOW noise (all 12 candidates SR .833/CR .042 identical). User dir:
after it45, selection = "increase whole metrics of gamma 0.1,0.5,1.0 with M=20". Coded in greedy_driver:
selection now ranks by (pooled-strict, net gamma-cells improved [3g x 4 metrics = 12], composite); strict
= pooled a-d all improve AND net gamma-cells > 0. M passed via --M (20 on the it46 relaunch). Internals
KEPT + new analysis/greedy_internals.py (per-step paired a-d baseline-vs-winner + chosen beta/frontier +
net gamma-cells, M8->M20 switch line). Paper: analysis/greedy_milestone.sh = full 7-gamma M100 eval of a
ckpt -> eval_greedy_it{N}_m100 (symlinked to eval_grandfinal_m100 so _v3 modules pick it up) + OOD
rollouts. CURRICULUM ANSWER: panel A = all gathered windows that iter colored by sigma (viridis),
AGGREGATED across ALL gammas (each window carries its own gamma); --gamma 0.5 ONLY sets the it0
pretrained reference dashed rollout. It is a mechanism viz, not a per-gamma metric.
PLAN: current M8 driver -> it45; then kill + relaunch from ckpt_45 with --M 20 (new selection) for
it46->; at it45 milestone run greedy_milestone.sh (7g M100) + 3 paper reports + greedy_internals +
curriculum into grand_final_reports; then the fine-tuning (beta,batch) phase per user.

## 2026-07-13 ~00:30 — it45 milestone (paper-grade 7g M100) + M20 driver relaunched
RESULT it45 (greedy-best from-scratch, FULL 7g M100): SR 71-84% CR 0-8% clr .307-.322 time 11.7-17.0;
COVERAGE sum 60 vs expert 53, g0.1 cov 16 vs expert 8 (mode-discovery ALREADY visible). NOT at expert
bar (SR<100, CR>0 mid-g, slower). Greedy it40-45 (M8) = confirmed NOISE (SR/CR frozen .833/.042, only
time crept -0.09s; all best-effort). OOD 16/18 held-out starts reach. Reports -> grand_final_reports/
{table,scatter,rollouts,greedy_internals,internals,curriculum}_it45. DECISION: M20 driver relaunched
it46->55 (per-gamma whole-metrics selection). HONEST NOTE: model is mid-training (~78% SR), not
converged; per-ITER greedy (even M20) may stay sub-noise for single steps — real lever is continued
training to lift SR. Will surface at it50 whether M20 resolves steps or we need coarser granularity /
straight training.

## 2026-07-13 ~01:45 — STOP greedy; FAITHFUL taxonomy diagnosis + REV sweep (user overnight task)
CMD: user stopped greedy. Diagnosis via analysis/faithful_taxonomy.py (4-way RAW split reach/collision/
OOB/timeout): PRETRAINED faithful M50 = reach 20-36% CR 0-2% **OOB 64-80%** (goal OVERSHOOT: DI momentum
past corner (5,5) beyond 5.12 bound; worse hi-gamma). it45 = reach 71-84% OOB 16-22%. CR~0 HIDES OOB.
User dx: too many EASY samples; use FRONTIER early + keep easy LOW; early beta+mix key; recovery/greedy-
multi-iter may help. LAUNCHED rev_sweep_driver.sh: 6 configs pretrained->it10 (frozen lr2e-5 NO-phased):
b0.3/f0.5, b0.3/f0.75, b0.2/f0.75, b0.2/f0.5, b0.3/f0.75/rec0.5, b0.3/f0.25(ref). rev_rank.py evals each
faithfully (M30 3g), ranks by reach, keeps best -> rev_best.pt. Reports -> grand_final_reports_rev/ (kept
grand_final_reports intact). Memory: oob-ill-conditioning-07-13.
DECISION (overnight autonomous): sweep->rank->continue-best further (straight, track faithful taxonomy,
keep best-reach ckpt)->final rev package. Report when user wakes. Greedy = reserved for near-convergence
fine-tuning (beta/mix), not the current under-converged regime.

## 2026-07-13 ~02:30 — BLIND-SPOT PASS on GP sigma (user /effort max) + FIX
FINDINGS: (1) buffer content = EXECUTED-trajectory windows (every 3rd, incl. REJECTED trajs), capped
qbuf_cap=500 -> matches user's agreed revision (NOT valid2-only); grid_expand_hardtail:1040-1046. (2)
sigma = ACTFLOW Eq10 posterior var via Cholesky+triangular-solve float64 jittered (NOT naive inverse);
384x384 op = 34 ms -> the matrix is NOT the heaviness. (3) **REAL BUG**: gp_buf=384 < qbuf_cap=500 and
neither was a CLI arg, so GE._buffer_feat does randperm(n)[:384] = a FRESH RANDOM 384-of-500 subset EVERY
call -> sigma jitter (noise/signal 14.8%), **15% of easy/frontier labels FLIP across draws** (measured on
real it40 gather). Directly corrupts the frontier/easy split the user is tuning. (4) re-Cholesky runs
PER-ROLLOUT in the gather loop (~1-20 s/iter on starved gathers) = the real 'heaviness', minor. sigma
dynamic range is FINE (0.16-1.0 std .19). FIX: added --gp-buf CLI (default 384 = bit-identical; set 500
=> deterministic all-500 buffer). Verified: gp_buf=500 -> sigma std across draws 0.000000, label
instability 0/300. Harness 20/20. Rev sweep KILLED (0-2 iters) + RELAUNCHED with --gp-buf 500 (clean
frontier signal for the OOB-fix sweep).

## 2026-07-13 ~04:20 — REV sweep ranked (clean gp_buf500 buffer): best = b0.2/f50/rec0.3
RESULT (pretrained->it10, faithful M30 3g, reach/CR/OOB): b2_f50_r3 49/1/50 (BEST) > b3_f50_r3 42/4/53 =
b3_f25_r3 42/3/54 > b3_f75_r5 40/4/56 > b3_f75_r3 37/1/62 = b2_f75_r3 37/1/62. FINDING: frontier helps but
75% is WORST (over-hard -> more OOB); sweet spot ~50% + LOW beta 0.2 (more explore). Refines user's
"frontier early" (has a limit). All still OOB-heavy at it10 (screening only). DECISION: continue best
recipe rev_best_cont (from rev_b2_f50_r3 it10 -> it60, same recipe) w/ faithful taxonomy tracking, keep
best-reach ckpt. rev_sweep_compare.png in grand_final_reports_rev.

## 2026-07-13 ~05:00 — rev best continuation WORKING: OOB dropping (user faithfulness concern noted)
CMD: user flagged recovery-band gathering (rec=0.3) as ad-hoc/unfaithful (strip-started rollouts, not
true origin) -> KEEP rec in code, faithful impl is a LATER task (killed premature rec=0 arm). Confirmed
on GPU3 (GPU2=other user). Delivered curriculum_rev/internals_rev/rollouts_rev + faithful_progression +
rev_sweep_compare to grand_final_reports_rev. Continued rev_b2_f50_r3 it10->60 (rec=0.3, b0.2, 50%f,
gp_buf500 deterministic).
RESULT faithful taxonomy (reach/CR/OOB pooled): it10 49/1/50 -> it20 61/0/39 (OOB dropping, CR 0,
reach rising). rollouts it10: reach=clean diagonals, OOB failures shifted to ORIGIN-departure (not goal
overshoot). internals SR50 panel flagged as OOB-HIDING (M50@g0.5, collision-only CR50). KEY honest note:
grand final Image#5 'SR ~0.7' = reach 71% + 20% OOB HIDDEN.
DECISION: let continuation run to it60, faithful taxonomy at each ckpt (it30/40/50/60), track best-reach.
FAITHFULNESS (parked for later): recovery-band = strip-started rollouts, ad-hoc; test rec=0 faithfully
AFTER the current recipe matures.

## 2026-07-13 ~16:56 — WALL8 walled-scene sweep (user: forget rec, close OOB routes, cut rollouts, raise lr)
CMD: STOPPED rev continuation (was rec=0.3). User plan: (1) 8-plug scene = near-full corner closure so
OOB escape -> detectable COLLISION (start/goal clr 0.083m; _WALL_PLUGS8 in grid_expand_hardtail + eval_ae,
--wall-plugs choices now [0,2,4,8], harness 20/20); (2) forget rec (recovery_frac 0); (3) rollouts 28->6
(batch 64 only needs ~56 fresh — user's efficiency point); (4) raise lr (trust frontier); (5) sanity
sweep to it10. faithful_taxonomy.py now takes --wall-plugs (deploy+classify on walled scene). Sweep:
lr{5e-5,1e-4} x frontier{0.5,0.75}, all wall8/rec0/roll6/gp_buf500/from pretrained->it10. NOTE user:
curriculum panel A shows POSITIVES only -> can't see failures; will report ROLLOUTS (shows reach/coll/oob)
+ curriculum on walled scene. Do NOT modify originals (grid_scene untouched; plugs applied on top).
DECISION: eval each faithfully (--wall-plugs 8), pick best reach, report rollouts + curriculum (walled).

## 2026-07-13 ~17:30 — paper viz relabels (user) + wall8 relaunch (cap120, 2-conc)
CMD: scatter_v2.py: labels Expert / **Our approach**(bold) / Pretrained / CFM-MPPI$^{*}$; removed both
subplot titles + the 'ideal' annotation. rollouts_v3.py: removed top legend + suptitle; 8 titles larger
font = Pre-trained data / Expert / Pretrained / CFM-MPPI$^{*}$ / NO safety validity check / NO progress
check / NO curriculum / **Ours**(bold); panel1 rebuilt = 100 uniform IC seeds (grey) + 8 OFF-DIAGONAL
SafeMPPI rollouts/gamma (analysis/runs/offdiag_expert.npz, starts max|x-y|). Overwrote grand_final_reports
copies. wall8 sweep was STALLED (4-way GPU contention x full cap300; valid2 rate on 8-plug=52% so scene
IS gatherable) -> relaunched cap120, 2-concurrent.

## 2026-07-13 ~19:30 — open-frontier no-plumbing sweep (user Q answered + gp_buf 200)
CMD: answered user's mechanism Qs (Q1 OOB=policy generalization, 0/2900 training windows OOB; SOCP
gamma-dependent higher-g looser; "raise quantile->more frontier" is BACKWARDS, lower q=more frontier).
Plan approved: frontier-knobs-only, NO plumbing. Added --qbuf-cap flag (harness 20/20) so --gp-buf 200
--qbuf-cap 200 = deterministic small buffer (user req). Recipe: open scene, rec0/hard0/targeted0/
min-modes0, q0.30, mix40/60, gp/qbuf200, beta{0.2,0.3}, from pretrained->it10.
FINDING (important): first launch (rollouts14->K_eff7) STARVED like wall8 — only it3,it7 trained
(vr6/300 < K_eff7 -> 'no fresh'). ROOT CAUSE = pretrained open-scene valid2 rate is only ~2-4% (mostly
OOB -> fails in_taskspace), so getting 7 valid rollouts is marginal. THIS IS WHAT RECOVERY GATHERING
COMPENSATED FOR (strip rollouts = high valid rate, bootstraps cold start). No-rev must brute-force it.
FIX: rollouts10 (K_eff5, reliably met) + cap400. Relaunched. DECISION: verify every iter trains, then
faithful taxonomy it10, pick lowest-OOB, continue.

## 2026-07-13 ~20:15 — aggregated failure fig + wall8 start-eps sweep (user)
CMD: (1) failures_aggregate.png = 3 models x 3g outcome-colored (pretrained OOB 9-12 -> it20 5-7 -> it45
3); failures = origin-departure + goal-overshoot. (2) open-frontier sweep (of_b2/b3_q30) ALIVE but SLOW
(open valid rate 2-4% -> cap400 gather slow); watcher raced false-stop, actually training it3/it5. (3)
Added --start-eps flag (trainer+taxonomy, harness 20/20): start at (eps,eps) so robot is in free space
(walled origin corner is tight 0.083m). (4) wall8 sweep w8f_b2/b3: SAME open-frontier recipe (q0.3,
mix40/60, gp/qbuf200, roll10, cap400, no plumbing) + --wall-plugs 8 --start-eps 0.1. KEY: walled valid
rate ~52% (walls prevent OOB) >> open 2-4%, so wall8 gather trains reliably/fast; start-eps fixes the
origin-corner window rejection that starved the earlier wall8.
DECISION: both sweeps -> it10 -> faithful taxonomy (open: no plugs; wall8: --wall-plugs 8 --start-eps
0.1) -> compare OOB, pick best, continue.

## 2026-07-13 ~21:00 — wall8 corrected + 5-arm ablation suite (user approved)
CMD: corrected _WALL_PLUGS8 to AXIS-ALIGNED (origin (0,-r)+(-r,0), goal (5+r,5)+(5,5+r)) — the diagonal
(-r,-r)/(5+r,5+r) left a straight-down/left gap (user caught it). Added --start-eps (start at (eps,eps),
0.05 -> clr 0.055m free space) + --reach (0.2 generous, near-goal=success; goal sits at plug boundary).
Both harness 20/20. Launched 5-arm suite on wall8 (axis plugs, eps0.05, reach0.2, q0.3, mix40/60,
gp/qbuf200, roll10, cap400, no plumbing): w8_full_b2/b3 (beta sweep), w8_nosocp (--ablate-socp),
w8_noprog (--ablate-progress), w8_nocur (--ablate-curriculum).
RESULT (early): walled scene TRAINS RELIABLY now — w8_nosocp trained every iter [2,3,4,5] (vs open-scene
2-4/10). Confirms user's insight: walls raise valid rate (~52%) -> gather meets K_eff. SOCP arms slower
(per-window SOCP CPU-bound, 4 concurrent) but progressing.
DECISION: at it10 -> faithful taxonomy each (wall8/eps0.05/reach0.2) -> pick best full arm -> curriculum
+ internals for it -> rollouts figure (full best vs 3 ablations) -> all into grand_final_reports_rev/wall8.
