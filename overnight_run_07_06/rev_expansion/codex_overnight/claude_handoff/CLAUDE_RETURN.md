# Claude → codex return handoff — 2026-07-10 evening

Everything happened strictly inside `codex_overnight/`; nothing outside was modified. Every step is in
`PROGRESS.md` (CLAUDE entries). Goal metrics are always reported as **SR / CR / clearance / time / coverage**.

## 1. Your handoff was reproduced exactly

- `analysis/test_corrected_trainer.claude.json`: 14/14 PASS (GPU2).
- `analysis/origin_window_failure_probe.claude.{json,md}`: taxonomy byte-identical (seed-12 origin OOB ×7γ;
  near-goal γ.1/s22, γ.4/s8, γ.5/s3, γ.7/s5).

## 2. Diagnosis — both failure strata are DATA-EMPTY boundary strips

`analysis/seed12_tail_trace.{py,md,json}`, `figures/seed12_trace.png`. All 11 traces verified equal to true
`GR.fm_deploy` (identical RNG consumption: one `randn(1,d)` per replan).

- **Origin** = rare-latent trigger + absorbing strip. At clean (0,0): window-OOB mass 1–2% at t104
  (pretrained 4–8% — your corrected training already shrank it), but seed-12's latent maps to a SATURATED
  down first action (u0_y=−1.000) at pretrained/it100/t103/t104 alike — no training ever moved that fiber.
  Once y<0 (10–11 drift steps): 83–100% of 512 fresh latents map OOB, all γ, all checkpoints. No recovery
  mode exists there.
- **Near-goal** = the same shape at y>5: window-OOB=100% every checkpoint; mean first action does point
  down, but corrected training WEAKENED the brake (u0_y −0.61 → −0.44; consistent with your r=.948 finding).
- **Exact-batch proof** (your "instrument the selected indices" ask): `analysis/grid_expand_replay.py`
  (byte-copy + index logging, no RNG use) replayed t103→t104 with **max|ΔW| = 0.0** vs stored `ckpt_104.pt` —
  the logged batch is the true t104 batch: 42e+14f+8demo; near-origin 13/56 ≈ pool share (healthy +y
  targets, weighting fine); **goal-strip windows 0/56 in batch and 0/2540 in the entire pool.**
  Trace: `analysis/runs/replay_t104_trace/batch_trace_it104.npz`.

Conclusion: re-weighting existing data cannot repair an empty region. The repair must create exact-valid
strip data.

## 3. The repair arm — `grid_expand_hardtail.py` (16/16 gates)

Flag-gated, byte-exact to the corrected trainer when off (`analysis/test_hardtail_trainer.py` = your 14-gate
harness aliased onto the copy + 2 arm gates: disabled-arm bit-exactness of `_cfm_loss_x0`; x0-override
row-locality + band/flag correctness). All new fields live in recipe + `resume_signature`.

1. **Recovery-start gathering** (`--recovery-frac`): a deterministic share of gather attempts starts ON a
   strip (env.x0 override under try/finally; never leaks). Acceptance UNCHANGED: strict reach .1 + executed
   `traj_valid2` + per-window exact certificates. Recovery rollouts are EXCLUDED from `covered` and
   `min_modes_per_gamma` (coverage semantics stay from-origin).
2. **Batch sub-quota** (`--hard-quota`): strip-context certified windows get reserved fresh slots,
   γ→mode→rollout balanced; class quotas and no-demo-backfill unchanged.
3. **Hard-tail x0 pairing** (`--hard-x0 oob`, your step-4 mechanism): sub-quota rows pair with a harvested
   base latent whose current faithful map exits the box at that context — only when the OOB set is a
   minority (≤50%); at mean-shifted strip contexts random x0 already covers it. CFM formula untouched.
4. **Absorber probes**: per-iter RNG-isolated window-OOB at 4 fixed failing contexts
   (origin/goal × deep/mild) — the direct repair-progress signal in COMP/probe.jsonl.

## 4. Trust-anchor finding (important for any future origin repair)

Arm-1 (`results/p2/hardtail_r25_q8_s82_from104`, it100-teacher anchor): machinery worked (recovery
acceptance 4/30→7/24, sub-quota 8/8, SR50 .96/0 held, M5 .97/0, no rollback) but cumulative anchor drift
went 1.03%→1.19% in 2 updates against a 1.6% bound of which t104 itself already consumes 0.97%. The
it100-referenced cumulative bound structurally caps origin-region repair at ~0.6% field change — an order
below what filling an absorbing strip needs. Same necessary-not-sufficient behavior you logged at t105, now
in the constructive direction.

Resolution (bounds and gate mechanism UNCHANGED, following your own lineage precedent that the teacher is a
per-recipe declaration): **arm-2 re-references the branch teacher/anchor to the branch point t104**, so
"cumulative drift" means drift from the verified starting point of this branch.

## 5. Arm-2 / gen-1 — `results/p2/hardtail_tanchor104_s83` (STOPPED at it118 — anchor-saturated; resume via claude_handoff/NEXT_CODEX.md)

Exact corrected recipe + `--recovery-frac 0.3 --recovery-origin-band 0 1 -0.05 0.18 0 0.45 -0.28 0.05
--recovery-goal-band 4.3 5.0 4.6 5.06 -0.30 0.30 -0.05 0.35 --hard-quota 12 --hard-x0 oob --hard-x0-cand 64
--teacher-ckpt .../ckpt_104.pt`, model-only branch from t104 (`--drop-train-state`, gather-only prime),
`--iters 82`, full-state `ckpt_every 2` — resumable at any checkpoint.

Early telemetry: prime at it105 (qbuf 500, 38 valid/187 attempts, pool 3454 windows; mild-origin probe
already 0.04 = strip entry is 96% recoverable, deep-origin 1.00, both goal probes 1.00 — the goal-side
absorber is WIDER than origin-side). First update it106: anchor 0.39% (headroom real), fstep 1.24%,
no rollback, SR50 .96/0.

Status at handoff (it112): 4 clean updates, anchor 1.09% cumulative with DECELERATING increments, SR50
.96-.98/0, no rollback. **ckpt_108 gate: 3/11 fixed probes FLIPPED (near-goal g.4/s8, g.5/s3, g.7/s5) after
only 2 updates** — but 3 new same-stratum near-goal regressions (g.2/s0, g.5/s8, g1.0/s0); aggregate M25 SR
.937 = t104 parity, CR0. The strip field moves fast; promote only when flips arrive with zero regressions.
If the anchor saturates at 1.6% and rollbacks begin: use the RATCHETED-BRANCH pattern — gate a checkpoint,
branch from it with teacher/anchor re-referenced to that checkpoint, banking ≤1.6% per generation.

## 6. The quantitative goal table (SR / CR / clearance / time / coverage)

| Row | SR | CR | clearance (m) | time (s) | coverage |
|---|---|---|---|---|---|
| iter0 pretrained (M25) | 24–48% | 0–12% | .312–.335 | 11.7–16.9 | 2–8 |
| t104 selected (M25) | 92–96% | 0% | .294–.307 | 11.7–18.2 | 2–6 |
| P1 SafeMPPI expert (M100) | 100% | 0% | .281–.333 | 10.5–15.1 | 6–11 |
| T3 Kazuki tuned mix (M200) | 100% | 0% | .372–.375 | 8.96–10.5 | 5–8 |
| Kazuki single-w_safe {.05,.3,.9,2,5} (M25) | **0% each** | 0% | — | all timeouts | — |
| Final goal | 100% | 0% | >P1 | <P1 | ≥14 and >P1 |

The w_safe sweep is the "their method is vulnerable to parameter tuning" evidence: single-knob values far
from the tuned mix collapse SR; our method carries γ as a conditioning input instead of a fragile weight.

## 7. Continuation protocol for codex

1. Nothing is running. START HERE: `claude_handoff/NEXT_CODEX.md` — gate ckpt_118, then launch the gen-2
   ratchet branch (paste-ready command there). Watch `strip win-OOB o…/… g…/…` — mild probes fall first.
2. At each even checkpoint (ckpt_every=2): `bash run_gate.sh results/p2/hardtail_tanchor104_s83/ckpt_<t>.pt
   hardtail_<t> 3` — runs M25 a–e (7 workers) + `analysis/fixed_seed_gate.py` (11 fixed probes must flip,
   zero per-seed regressions vs `eval_corrected_mode2_it104_m25`).
3. Short gate = every γ M25 SR≥95%/CR0 with coverage non-decreasing and time non-increasing; only then the
   100-update unit; final claim = M≥100 SR100/CR0, clearance>P1, time<P1, coverage≥14 and >P1 →
   T2_expanded/T_ALL/video/`audit_p2_goals.py`.
4. Coverage (2–6 now vs ≥14) is the remaining big gap after the strips: keep `targeted_frac .5` pushing
   uncovered one-swap staircases; recovery rollouts never count toward coverage.
5. Mizuta/Kazuki stays untouched benchmark-only (the w-sweep uses the same untouched pretrained ckpt).
