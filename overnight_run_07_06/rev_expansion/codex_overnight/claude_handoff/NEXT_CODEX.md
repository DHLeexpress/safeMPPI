# NEXT_CODEX — resume the hard-tail repair loop (single entry point, 2026-07-10)

Open this file first. Full technical map: `claude_handoff/CLAUDE_RETURN.md`. Append-only log: `PROGRESS.md`
(all CLAUDE entries). No Claude process is left running — no conflicts. NOTE at handoff time: another user's
job (serenal, dmpo) took GPU 2 the moment it freed; GPU 3 is idle. Check `nvidia-smi` and swap the
`CUDA_VISIBLE_DEVICES` assignments below accordingly (training on whichever of 2/3 is free, gates on the other
or sequentially).

---

## USER REQUEST & STORYBOARD (VERBATIM — these are the orders)

Message A (the metrics storyboard):

> Resume with efficient tokens usage! You should have been surfacing the c/d/e columns alongside SR/CR in
> every status update; because that is the quantitative ultimate goal to compare against with Kazuki's
> method and demo method (expansion goes beyond every metrics). Once you solved the hard-tail problem
> creatively without further assumptions, please achieve the goal based on those metrics (I am sure you
> will; SR 100 will naturally comes from those plumbing; for every gamma). At the end you might need more
> iterations to not only SR 100 CR 100 but those metrics to be high, by keep pushing frontiers to the
> training. I hope you can compare the iter0, some iters in the middle, final iter with complete goal
> achieved, then compare with Kazuki (w_safe sweep <- show their method is very vulnerable to parameter
> tunings) and compare with demo expert.

Message B (division of labor):

> keep arm-2 running and proceed until you reach SR 100% then he will do rest of the training and quantify.

Message C (this handoff):

> You may wrap up now and pass it over to codex. Contain relevant resources so that he can focus on it,
> without conflict, also add my previous storyboard+ request in the new .md file.

So: **your mission = drive the repair generations until M25 SR 100% (CR 0) at every γ, then run the long
unit and the final M≥100 quantification** — every report shows SR / CR / clearance / time / coverage, and the
final deliverable compares iter0 → mid → final vs Kazuki (tuned mix + w_safe sweep) vs the SafeMPPI demo
expert. The paper abstract lives in `GOAL.md` §STORYLINE (verbatim from the user).

---

## STATE (2026-07-10 16:45 PDT)

| Item | Status |
|---|---|
| gen-1 `results/p2/hardtail_tanchor104_s83` | STOPPED at it118 — anchor saturated (it117/118 rolled back at 1.72% > 1.6%); productive weights end ≈ it114/115; `ckpt_118.pt` = latest full-state |
| ckpt_108 gate (`analysis/fixed_seed_gate_hardtail108.json`) | **3/11 fixed probes FLIPPED** (near-goal γ.4/s8, γ.5/s3, γ.7/s5) after only 2 updates; 3 same-stratum near-goal regressions (γ.2/s0, γ.5/s8, γ1.0/s0); aggregate M25 SR .937 = t104 parity; CR 0 |
| ckpt_118 gate | NEVER RAN — command below |
| Absorber probes (per-iter `strip win-OOB o…/… g…/…` in COMP) | deep-origin 1.00, mild-origin 0.04, goal 1.00/1.00 — mild probes must fall first |
| Kazuki w_safe sweep | COMPLETE: all five single coefficients {.05,.3,.9,2,5} give SR 0% (25/25 timeouts each) vs tuned 5-coef mix 100% — `tables/T_COMPARE_progress.md`, `results/kazuki_wsweep/` |
| iter0 baseline | `results/p2/eval_pretrained_m25` (SR 24–48%, CR 0–12%, cov 2–8) |
| Diagnosis (why this arm exists) | Both failure strata are DATA-EMPTY boundary strips (y<0 origin absorber + y>5 goal overshoot); proof incl. exact-batch replay max|ΔW|=0: `analysis/seed12_tail_trace.md`, `CLAUDE_RETURN.md` §2 |

---

## NEXT COMMANDS (paste-ready, in this order)

1. Gate the gen-1 terminal weights (GPU3, ~6 min):

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16
bash run_gate.sh results/p2/hardtail_tanchor104_s83/ckpt_118.pt hardtail118 3
```

2. Launch gen-2 (GPU2) — the ratchet branch: model-only from ckpt_118, teacher/anchor re-referenced to the
   branch point (fresh 1.6% budget; numeric bounds and gate mechanism UNCHANGED), attempt cap 260 to cut the
   ~40% quota-skip waste seen in gen-1. All other flags byte-identical to gen-1:

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 setsid nohup \
python grid_expand_hardtail.py \
  --ckpt results/p2/hardtail_tanchor104_s83/ckpt_118.pt \
  --outdir results/p2/hardtail_gen2_s84 \
  --iters 82 --drop-train-state --freeze --lr 2e-5 --seed 84 \
  --rollouts-per-iter 14 --gather-attempt-cap 260 --batch 64 \
  --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
  --quantile-schedule 0:0.50 200:0.60 400:0.70 \
  --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 \
  --early-until 100 --cooldown-from 400 \
  --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
  --demo-frac 0.125 --lwf-eta 0.05 \
  --teacher-ckpt results/p2/hardtail_tanchor104_s83/ckpt_118.pt \
  --nfe-explore 8 --field-grad-clip 1.0 \
  --max-functional-step 0.025 --max-anchor-drift 0.016 \
  --targeted-frac 0.5 --n-target 40 --align-temp 0.45 --min-modes-per-gamma 2 \
  --recovery-frac 0.3 --recovery-origin-band 0.0 1.0 -0.05 0.18 0.0 0.45 -0.28 0.05 \
  --recovery-goal-band 4.3 5.0 4.6 5.06 -0.30 0.30 -0.05 0.35 \
  --hard-quota 12 --hard-x0 oob --hard-x0-cand 64 --strip-probe-every 1 \
  --m-measure 5 --measure-every 1 --probe-cov 1 --log-comp-every 1 \
  --viz-db-every 1 --ckpt-every 2 \
  --tag hardtail_gen2_s84 > logs/p2_hardtail_gen2_s84.log 2>&1 < /dev/null &
```

## LOOP PROTOCOL (repeat until M25 SR 100%/CR 0 at every γ)

1. Per even checkpoint: `bash run_gate.sh results/p2/hardtail_gen2_s84/ckpt_<t>.pt gen2_<t> 3`
   → M25 a–e (7 workers) + `analysis/fixed_seed_gate.py` (11 fixed probes + per-seed diff, ALWAYS vs the
   fixed baseline `results/p2/eval_corrected_mode2_it104_m25`).
2. Promote a checkpoint only when flips ≥ previous AND `n_regressions == 0`.
3. When COMP shows `rollback 1` with anchor ≈1.6% repeatedly → generation exhausted → ratchet: stop, branch
   gen-(k+1) from its latest full-state ckpt with `--teacher-ckpt <that same ckpt>` and a new outdir/seed.
   Each generation banks ≤1.6% of intended origin-field change; ckpt_108 showed the goal strip moves within
   2 updates, so expect a few generations, not many.
4. Origin stratum (seed 12 ×7γ) is the deep one — watch mild→deep probe decay; the x0-pairing works on the
   trigger fiber at (0,0) while recovery data fills the strip.
5. Expected churn: same-stratum exchanges (near_goal↔near_goal) while the descent field rewrites — that is
   why the promote rule demands zero regressions, but do NOT stop a generation merely on churn.
6. If a class/γ/mode quota starves repeatedly at cap 260, prefer more attempts over any acceptance change —
   Valid2 / reach .1 / exact certificates / faithful eval (temp 1, NFE 8) are UNTOUCHABLE; no demo backfill;
   no inference clipping.
7. After SR 100%: the stateful 100-update unit (user's Message B: that part is yours), then final M≥100 per γ
   — SR 100%, CR 0, clearance > P1, time < P1, coverage ≥ 14 and > P1 → `T2_expanded`, `T_ALL` (+ rows into
   `tables/T_COMPARE_progress.md`: iter0 / mid / final / Kazuki mix / w_safe sweep / P1 expert), the 2×4
   curriculum video (`video_curriculum_fixed.py`), `audit_p2_goals.py`.

## RESOURCE MAP (all Claude-created, inside codex_overnight/ — nothing else touched)

- `grid_expand_hardtail.py` — the repair trainer (recovery-start gather + hard-quota + x0 pairing; byte-exact
  to `grid_expand_fixed.py` when flags off). Harness: `analysis/test_hardtail_trainer.py` (16/16 PASS).
- `run_gate.sh` — one-shot M25 a–e + fixed-seed gate (validated end-to-end at ckpt_108).
- `analysis/fixed_seed_gate.py` — per-seed diff vs t104 baseline (the promote arbiter).
- `analysis/seed12_tail_trace.py|.md|.json`, `figures/seed12_trace.png` — the strip diagnosis (rerun on any
  checkpoint to see fiber/absorber movement).
- `analysis/grid_expand_replay.py`, `analysis/runs/replay_t104_trace/` — exact-batch replay (max|ΔW|=0) +
  `batch_trace_it104.npz`.
- `results/p2/eval_pretrained_m25` (iter0), `results/p2/eval_hardtail108_m25`, `results/kazuki_wsweep/`,
  `tables/T_COMPARE_progress.md` — the comparison spine.
- Hard rules (unchanged): work only here; GPUs 2/3; OMP_NUM_THREADS=16; no wandb/push; Mizuta/Kazuki is a
  frozen benchmark (the w-sweep used the untouched pretrained ckpt); PROGRESS.md append-only, CMD/RESULT/
  DECISION per action.
