# GOAL_SFM_TRANSFER — port the HP hold-while-explore winner to SFM, re-benchmark vs Kazuki

**Intent (task #54, 2026-07-06):** the 0702 chessboard study converged on a hold-while-explore recipe (frozen
encoder + demo replay + LwF anchor + grad-clip) that holds validity while coverage climbs, on an OD-augmented,
γ-balanced base. Transfer that recipe to the SFM (moving-pedestrian) expansion and see if a stronger base policy
lets the certified-deployment stage (already 3%/97% @6ms vs Kazuki 10%/95ms) go further.

## Honest starting point (don't re-litigate)
- SFM **expansion-only** plateaued ~9.7% collision (lost to Kazuki 4.0%). But **certified DEPLOYMENT**
  (`stage_e_policy.py`: N=64 FM + brake windows → SOCP cert ∧ anticipated clearance p+i·dt·v → yield fallback)
  already reached **3% collision / 97% success at 6 ms** — a WIN on the 300-scenario DI benchmark. So the gap is
  closed at deployment; this transfer tests whether a better-explored BASE improves it (less pad/yield needed,
  higher raw success, or lower collision pre-certification).
- The winning HP recipe's biggest lever was **OD-in-pretraining** (balanced γ). The SFM analogue = ensure the SFM
  pretrain data covers diverse crowd encounters; if γ0.1 is starved on SFM too, that is the first thing to fix.

## THE PORT — add 3 knobs to `grid_expand_sfm.py` (mirror `overnight_run_2026-07-02/grid_expand2.py`)
`grid_expand_sfm.update_flow()` is positive-only + α-neg today. Add, copying grid_expand2 verbatim:
1. **Config fields** (the Config dataclass): `demo_frac: float = 0.0`, `lwf_eta: float = 0.0`,
   `grad_clip: float = 0.0`. CLI args `--demo-frac --lwf-eta --grad-clip` (argparse ~line 229-249) + wire into cfg.
2. **`update_flow(policy, opt, demo, pos, neg, cfg, device)`** — add a `demo` param (the SFM training windows,
   `sfm_windows_g*.pt`, loaded once in run_expand like grid_expand2's `load_demo_all`). Then:
   - **demo_frac**: `ndf = round(cfg.demo_frac * cfg.batch)`; draw ndf rows from demo + (batch−ndf) from positives
     for the cfm batch (grid_expand2 update_flow2, the demo_frac branch).
   - **lwf_eta**: at expansion start `cfg._teacher = copy.deepcopy(policy).eval()` (params `requires_grad_(False)`)
     when lwf_eta>0; in the loop add `cfg.lwf_eta * ((v_student − v_teacher)**2).mean()` on demo contexts via the
     cfm interpolation (x_τ=(1−τ)x0+τx1) — grid_expand2 lines ~230-239.
   - **grad_clip**: after `loss.backward()`, before `opt.step()`:
     `if cfg.grad_clip>0: torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], cfg.grad_clip)`.
3. **Hard freeze** (run_expand, ~line 273): when `cfg.enc_lr_mult <= 0`, `for p in enc_params: p.requires_grad_(False)`
   and DON'T append the enc group (mirror grid_expand2) — freezes the SFM DeepSets scene-encoder.
Smoke-test: `python grid_expand_sfm.py --iters 3 --demo-frac 0.4 --lwf-eta 0.05 --grad-clip 10 --enc-lr-mult 0 …`
→ expect drift≈0 (frozen enc) and no crash in the demo/lwf branches.

## RUN — expansion with the locked recipe, from the SFM base
Base = `pretrained_sfm.pt` (grid_expand_sfm `--policy` default). Locked recipe (ov_mine):
```bash
python grid_expand_sfm.py --policy pretrained_sfm.pt \
  --iters 5000 --ckpt-every 1000 --enc-lr-mult 0 --lr 1e-4 --grad-clip 10 \
  --demo-frac 0.4 --lwf-eta 0.05 --beta 0.1 --alpha 0.02 --s 0.9 --temp 1.5 --ell <sfm ell*> \
  --outdir results/expand_sfm_holdexplore
```
(Re-calibrate ell on the SFM φ_s first if not done — the σ-kernel is scene-dependent.) Report per-round:
collision / success / time-to-completion per γ on held-out SFM scenes + coverage proxy + the drift/demoCFM watch.
Max-coverage variant: `--s 0.99`.

## RE-BENCHMARK vs Kazuki (300 DI scenarios)
Deploy the expanded policy through the certified stage, then the paired benchmark:
```bash
python stage_e_benchmark.py --side ours --ep-start 0 --ep-end 300   # certified-FM deploy on the new base
python stage_e_benchmark.py --side merge --ep-start 0 --ep-end 300  # vs cached kazuki: Δcollision, Δsuccess, CIs
```
WIN metric = our γ-Pareto dominates Kazuki on collision AND success (compute already 15× better). Compare to the
existing 3%/97% — did the better base lower collision further or cut the pad/yield rate?

## Files (SFM side, `overnight_run_2026-07-04/`)
`grid_expand_sfm.py` (port target) · `stage_e_policy.py` (certified deploy) · `stage_e_benchmark.py` (paired vs
Kazuki, `--side ours|kazuki|merge`) · `grid_policy_sfm.py` · `grid_metrics_sfm.py` · `sfm_scene.py` ·
`dataset/sfm_windows_g*.pt` (demo source) · `pretrained_sfm.pt` (base). Recipe reference:
`overnight_run_2026-07-02/grid_expand2.py` (the 3 knobs, verbatim) + `goal_07_02_Hp.md` §FINAL JUDGMENT.

## Open decision
Transfer the **balanced** recipe (s.9, better γ0.1 → likely lower collision) or the **max-coverage** one
(s.99, more modes → maybe higher success but weaker safety)? For a COLLISION-gap goal, start with **s.9**.
