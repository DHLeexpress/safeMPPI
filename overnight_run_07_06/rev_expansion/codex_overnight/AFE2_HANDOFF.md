# AFE2 handoff — what happened, where it lives in code, and which recipes to try next

> **Historical result only.** Do not execute the recipe matrix below for the
> radius-1 replication. The locked, non-sweep continuation contract is
> `../codex_challenging/afe_restart/AFE2_RADIUS1_HANDOFF.md`; it first corrects
> absorbing-goal termination and experimental provenance while leaving both
> update arms unchanged.

Audience: another agent (codex) continuing this line, possibly on a different task. Read
`README.md` §9-§10 first for the full spec; this document is the RESULT story + the exact code
pinpoints + a prioritized recipe matrix. Repo: `DHLeexpress/safeMPPI`, branch
`codex/safe-mppi-publish` (mirrored to `master`); dir
`overnight_run_07_06/rev_expansion/codex_overnight/`.

## 1. The study in one paragraph

Two matched arms use the same protocol, initial checkpoint, and indexed random streams (their
learned representations, candidates, archives, and A matrices then diverge): evolving
representation φ_s⁽ⁿ⁾, per-round rebuilt uncertainty matrix A, σ-tilted B=8-of-K=64 verified acquisition, argmax-progress execution among
SOCP-positive plans, EXPERT-FREE: an episode terminates `NO_VERIFIED_POSITIVE` when nothing
certifies — no SafeMPPI, no fallback, ever) except the update: **prox** (batch 128, lr 2e-5,
η=0.01, ≤40 steps, fstep-bound 0.03) vs **afe** (batch 128, lr 1e-4, exactly 250 steps, no prox).
10 rounds × all seven γ each round; β=0.02 fixed by ESS calibration; fixed-index M=8/γ controller
evaluation each round (same rollout seeds every round → paired).

## 2. What happened (final numbers, `results/afe2/*/probe.jsonl`)

| | round 0 | prox final | afe final (r10) | afe peak (r9) |
|---|---|---|---|---|
| controller SR (pooled) | **0.00** | 0.00 (never >0.05) | 0.16 | **0.34** |
| NVP rate | 1.00 | 1.00 | 0.84 | 0.66 |
| CR | 0.00 | 0.00 | 0.00 | 0.00 |
| audit V (raw untilted) | 0.716 | 0.716 (flat) | **0.671** (eroding) | 0.679 |
| audit V_adverse | 0.438 | 0.435 | **0.363** (−7.5 pts) | — |
| rep cosine vs φ⁰ | 1.000 | 1.000 | 0.982 | 0.985 |
| per-round fstep | — | 0.013–0.030 (bound) | 0.12–0.39 | — |

1. **Baseline reality**: the pretrained's expert-free verified controller is SR 0.00 / NVP 1.00 at
   every γ. In Study 1 the certified SafeMPPI fallback was silently completing EVERY episode.
   CR = 0.00 everywhere always — verify-or-terminate never executes an uncertified action.
2. **Where episodes die (the two walls)**: 65–78 % of NVP terminations are within 0.8 m of the
   goal (median 0.3–0.4 m) — the goal-corner certification hole (goal (4.7,4.7) against the wall
   plugs is OOD vs the (5,5)-goal pretraining demos). Exception: **γ0.1 dies at the START**
   (median distance-to-goal 5.35) — strict-γ certification fails in the first corridor. Ten rounds
   moved neither wall for prox; afe punched partial holes (r9: γ0.1 0.38, γ0.2 0.50, γ1.0 0.50).
3. **prox = stable but frozen**: the η=0.01/lr 2e-5 leash (sized for the 100-round Study 1) allows
   ~1–3 % field change per round — nothing measurable in 10 rounds. Audit perfectly flat.
4. **afe = learns, oscillates, erodes**: SR 0.09 → **0.34** → 0.16 over rounds 8–10 (round-to-round
   whiplash, the known unbounded-update signature), while the audit erodes monotonically
   (V −4.5 pts, V_adverse −7.5 pts) and the representation drifts (cos 0.982). It buys task
   success by spending held-out validity — the −Prox lesson from Study 1, reproduced under the
   corrected protocol.
5. **The acquisition signal is nearly blind**: effective rank of A ≈ 1.1–1.3 (the 32-d φ_s
   features of all proposed plans lie along ~one direction), σ-uplift of selected-vs-pool → 0.003,
   ESS/K drifts OUT of the calibration band (0.18 → 0.83) as σ flattens — the lottery converges to
   uniform. β ∈ {0.01, 0.02, 0.05} never hit the [0.25, 0.5] band at calibration (0.036 / 0.125 /
   0.626; 0.02 chosen by band-midpoint rule, margin 0.001). **Plan-level σ selection is doing
   almost no work**; feature rank, not β, is the bottleneck.

Artifacts: `paper_results/afe2_report_v1.png` (8-panel diagnostics),
`paper_results/afe2_{afe,prox}.mp4` (per-round 7-γ videos: gray K-pool / green positive / red
rejected / orange socp_error / blue selected+executed / X = NVP), `results/afe2/*/viz_db/round*.pt`,
`dstore.pt`, per-round ckpts.

## 3. Exact code pinpoints (`grid_expand_afe2.py` unless noted)

| mechanism | where | current value |
|---|---|---|
| candidate pool / verifier budget | `AFE2Config` `:67-68` (`--K/--B`) | K=64, B=8 |
| acquisition temperature β | `:69` (`--beta`); ESS calibration `calibrate_beta` `:361` (`--calibrate`) | 0.02 |
| BLR ridge λ | `:71` (`--lam`) | 10 |
| σ features z | `AC.frozen_feat` call `:182` (embedder = CURRENT policy); `afe_core.py:frozen_feat` (φ_s at s=0.9, normalized) | 32-d, effR≈1.1 (**bottleneck**) |
| A rebuild per round | `rebuild_A` `:117` (re-embed archive `embed_queries` `:102`); called `:480` (gather) and `:504` (eval) | full rebuild, float64 |
| σ-lottery draw | `torch.multinomial` `:188` (without replacement) | B of K ∝ e^((σ−σmax)/β) |
| socp_error handling | `:198-201` (skip store, skip A) | updates nothing |
| execution rule J_exec | `:207` (`best` = argmax `v["prog"]`) and `_pick`-free — argmax only | max progress |
| NVP termination | `run_episode` `:245` (`best is None` → `status="nvp"`) | terminate, no expert |
| episode start / context process ρ(c) | `env.x0` set in `main` `:589`; used `run_episode` `:232` | fixed (0.3, 0.3); goal (4.7,4.7) |
| γ sweep | `run_afe2` `:485` (`for ep, g in enumerate(cfg.gammas)`) | all 7, fixed order, 1 episode each |
| prox arm knobs | `:79-82` (`--prox-lr/--prox-eta`); prox term `update_round` `:301`; stop `:316` | lr 2e-5, η 0.01, fstep 0.03/40 |
| afe arm knobs | `:83-84` (`--afe-lr/--afe-steps`); step count `:279` | lr 1e-4, 250 steps |
| replay draw | `store.sample_pos` `:286` (uniform WITH replacement over cumulative D⁺; `afe_core.py:DStore.sample_pos`) | batch 128 |
| fixed-index controller eval | `controller_eval` `:331`; rollout seeds `:340` | M=8/γ, seeds shared across rounds |
| audit (never buffered) | `afe_core.py:build_audit_contexts/run_audit`; called `:441,:502` | 12 pos × {rest, adverse} × 7γ × 4 plans |
| rep-drift probe | `rep_probe_build/rep_cos_drift` `:138,:161` | 24 ctx × 8 plans, cosine vs φ⁰ |

## 4. Recipe matrix for the next arms (priority order, one change per arm)

Every arm below is launched with the SAME command skeleton (README §9) and decided by the SAME
paired instruments: fixed-index controller SR/NVP (learning), audit V per γ (erosion guard),
NVP-location split (which wall moved), ESS/uplift (acquisition alive), rep cosine (drift).

1. **Middle update (highest value, zero code)** — the gap between frozen and whiplash:
   `--arm afe --afe-steps 80 --afe-lr 3e-5` (and a second point `--afe-steps 120 --afe-lr 5e-5`).
   Hypothesis: monotone SR rise without the r8-r10 oscillation and with V flat. Success = SR ≥ afe
   peak with V within 2 pts of 0.716.
2. **Looser prox (zero code)** — same target from the other side:
   `--arm prox --prox-eta 0.05 --prox-lr 5e-5` (the 0.01/2e-5 leash was sized for 100 rounds).
3. **Fix the blind σ (small code, big principle)** — raise feature rank: at `:182` (and
   `embed_queries:113`, `calibrate_beta:395` — keep all three consistent) concatenate to φ_s the
   normalized raw window `U.flatten()/u_max` (20-d) and the plan displacement `seg[-1]−x_t` (2-d)
   → z ∈ R^54, then L2-normalize; A becomes 54×54 (BLRSigma `dim=` picks it up automatically if
   you pass the new dim at `:437/:481`). MUST re-run `--calibrate` afterwards (β is
   feature-scale-dependent). Success = uplift stays > 0.02 and effR > 5 at round 10.
4. **Move the walls via ρ(c) (one line, but flag it as a context-process change)** — at `:232`,
   start a deterministic fraction of episodes near the walls: e.g. episodes with
   `ep % 3 == 2` start at `(4.2, 4.2, 0, 0)` (goal-corner approach) — and/or give γ0.1 a second
   episode per round. This is the certified-recovery analogue; it changes ρ(c), which both studies
   identified as the binding constraint (validity only expands along visited contexts). Keep the
   audit unchanged so the comparison stays honest.
5. **Execution rule π-sample (one line)** — at `:207`, replace argmax with sampling ∝ the lottery
   weight among positives (Study 1 measured this as the within-γ diversity lever: covΣ 34 vs
   26–30). Watch coverage + SR together.
6. **Longer horizon** — `--rounds 30` for the winner of (1)/(2); 10 rounds was the study size, not
   a convergence claim (afe was mid-breakthrough at r9).

**Do NOT change** (these are the paradigm, not knobs): pre-execution full verification
(`afe_core.verify_plan`), NVP termination semantics (no fallback execution), audit isolation
(never buffer audit samples), fixed-index paired evaluation, cumulative raw archive D, per-round
A rebuild under the current representation, socp_error-updates-nothing.

**Operational warnings**: (a) module shadowing — `grid_metrics2` etc. exist in multiple dirs;
the local bootstrap (`:44-48`) must keep `_HERE` first; verify with `module.__file__` before
editing anything imported; (b) `verifier_polytope` resolves from `overnight_run_2026-07-01/`;
(c) run from inside `codex_overnight/` (relative `--ckpt`); export
`LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib`, `CUDA_VISIBLE_DEVICES=3`; one arm ≈ 5 min on this
box (gather ~20 s + eval ~3 min per round); (d) rollback tag `pre-afe-2026-07-16`.

## 5. Launch skeleton

```bash
cd overnight_run_07_06/rev_expansion/codex_overnight
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib CUDA_VISIBLE_DEVICES=3
C="--ckpt ../../results/hp_repr/pretrained_a32uni.pt --rounds 10 --K 64 --B 8 --beta 0.02 \
   --lam 10 --T 300 --reach 0.15 --M-eval 8 --batch 128 --wall-plugs 8 --start-eps 0.3 \
   --goal-xy 4.7 4.7 --seed 910"
python grid_expand_afe2.py $C --arm afe --afe-steps 80 --afe-lr 3e-5 --outdir results/afe2/mid80_s910
python analysis/afe2_report.py --arms results/afe2/prox_s910 results/afe2/afe_s910 results/afe2/mid80_s910
python video_afe2.py --run results/afe2/mid80_s910 --out paper_results/afe2_mid80.mp4
```
(β recalibration when features/λ change: `python grid_expand_afe2.py $C --calibrate --outdir results/afe2/calib2`.)
