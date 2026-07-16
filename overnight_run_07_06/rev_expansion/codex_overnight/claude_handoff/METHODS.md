# METHODS — what we actually did, in plain language

One page. Every mechanism below is gate-checked code in this folder; nothing here changes the safety
definition (Valid2), the certificate, the strict reach=0.1, or faithful evaluation (temp=1, NFE=8).

## The base system (unchanged throughout)

A SafeMPPI expert produces demonstrations at seven safety levels γ. A conditional flow-matching policy
learns them: given the scene features and γ, integrating a learned velocity field for 8 Euler steps turns a
random "latent" vector into a 10-step control window; the robot executes the first control and replans.
"Expansion" = fine-tuning this policy on ITS OWN deployed rollouts, but only windows that pass the
verifier-polytope certificate + net-progress + task-space checks (exact-valid data, never relabeled).

## Why SR was stuck at ~93–96% with zero collisions

Three separate phenomena (the four-panel figure `figures/current_goal_latent_support.png` shows them apart):

1. **Certified targets were fine** — near-origin windows are high-uncertainty but NOT ill-conditioned
   (SVD condition ≈ same near/away). Batch re-weighting was never the bottleneck.
2. **One rare latent fiber was bad**: ~1% of latent draws at the start state map to a saturated downward
   first action (the "seed-12" failure — reproducible because evaluation seeds the noise).
3. **Two boundary strips had NO data at all**: just below y=0 near the origin, and just above y=5 next to
   the goal. Demos never go there (they stop at 0.45 m in the old convention) and self-rollouts die on
   entry. Once the robot drifts in, essentially EVERY latent maps out of bounds — an absorbing strip.
   Proof: the exact training batch that produced the incumbent t104 was replayed bit-identically
   (max|ΔW|=0) — the pool of 2,540 certified windows contained ZERO goal-strip rows.

## The repair toolbox (in the order it was built)

| Mechanism | What it does | Why it is legitimate |
|---|---|---|
| Recovery-start gathering | A bounded share of data-gathering rollouts STARTS on a strip (with adverse velocity) instead of the origin | The rollout must still reach the goal and pass the unchanged certificate/Valid2 — it only creates certified data where none existed. Strip rollouts never count toward coverage/mode quotas. |
| Hard-quota + x0 pairing | Reserves a few batch slots for strip windows and pairs them with the specific latents that currently map out of bounds | Importance sampling on the CFM base noise only; the loss formula is untouched (bit-exact when off — gate 15) |
| Trust-anchor ratcheting | Each repair branch may move the origin-region field ≤1.6% from its own verified starting checkpoint; saturation ⇒ gate a checkpoint and re-anchor | The numeric bounds and rollback mechanism are the original ones; only the reference point ratchets, and only through gated checkpoints |
| Boundary adapters (codex) | Tiny zero-init residual networks added to the velocity field that are EXACTLY zero outside the strips (compact support — gate 18) | The base policy is frozen; interior behavior provably unchanged; used as capacity probes, promoted only through the full gates |
| Goal-brake + γ-augmentation (codex) | Certified braking windows near the goal, duplicated across γ AFTER re-certifying each copy at its destination γ (the certificate is γ-dependent) | Retrospective audit: all 35 relabels certified; the script now enforces this for every future run |
| Teacher-preservation replay (this run) | Roll the CANDIDATE at 100 independent seeds (100–199, disjoint from the 25 evaluation seeds), record every near-goal state it visits, and record what the INCUMBENT t104 would generate there from the same latent; train the candidate to keep that mapping | Anti-churn: earlier repairs fixed 3 failures but broke 3 other seeds; this pins known-good behavior while the adapter adds braking. Every replay row is re-certified at its own γ before use (`analysis/audit_replay_certify.py`) |
| One-step probe harness | Every candidate = ONE bounded, non-resumable update from a frozen certified pool, then judged | No gathering, no acceptance change, deterministic; keeps the search honest and cheap |

## The judgment ladder (what "success" means)

1. `analysis/fixed_seed_gate.py`: the 11 original failures (seed 12 at all 7 γ + 4 near-goal seeds) must
   flip to success AND no previously-passing seed may regress — per-seed diff vs the frozen t104 M25
   archive.
2. Independent M25: every γ SR 100%, CR 0 (a–e recorded).
3. Independent M100: SR 100%, CR 0 (M25 provably misses rare collisions: t104's M100 exposed CR 1% at
   γ.2/.3).
4. Only then: integrate into the resumable trainer, run the 100-update unit, push coverage (≥14 of the 16
   staircase modes; aspiration: all 16), final M≥100 audit with clearance>expert, time<expert.

## Current results ladder (SR / CR / clearance / time / coverage)

| Checkpoint | SR by γ (.1–1) | CR | Coverage | Status |
|---|---|---|---|---|
| iter0 pretrained | 24–48% | 0–12% | 2–8 | start |
| t104 (selected incumbent) | 92,96,96,92,92,92,96 | 0 | 2–6 (M25) / 4–10 (M100) | production |
| s671 origin repair | 96,100,100,100,100,92,88 | 0 | 2–5 | diagnostic |
| s766 goal-brake + γ-aug | 100,100,100,100,100,100,92 | 0 | 2–5 | 11/11 flips; 2 new γ1 regressions |
| s791 preservation (γ1-focus) | 96,100,100,100,100,100,92 | 0 | 2-5 | 10/11 flips; γ-focus reopens s22 |
| **s792 all-γ brake + preservation** | **100,100,100,100,100,100,92** | **0** | 2-5 | **11/11 flips, agg .989 — unit base** |
| s794/s795/s799 band scalpels | — | — | — | no-op / rollback / over-brake (see PROGRESS) |

## Two structural findings from the last knot (γ1.0 s5/s14, mm-scale grazings)

1. **State ambiguity**: the goal sits ON the y=5.0 line, so "rising through y 4.85-5.0" is correct for
   approaches and fatal for overshoots — no (y, vy) rule separates them (oracle build: γ0.1 lost 100/100
   rollouts under a band brake; s799 over-brake broke a fresh seed).
2. **Terminal-window blindness**: contexts within 10 steps of reaching can never produce an H=10 executed
   window, so the unambiguous y≥5.0 region is data-empty BY CONSTRUCTION of the window semantics.

Consequence: one-step scalpels plateaued; the sanctioned path is the guarded unit (fresh certified
gathering + preservation replay + trust gates, every update gated) — running as
`results/p2/unit_s792_esc64_s801`. An oracle intervention proves sufficiency: 1-2 brake steps in the band
flip both seeds (and beat t104's own completion time).

## WALLS-4 from-scratch amendment (2026-07-11)

The original task boundary was not represented in the policy input: the H_P scene feature sensed circular
obstacles but not the open outer boundary.  We therefore added four ordinary circular "plugs" just outside
the corner openings.  This changes only the sensed scene; the flow architecture, seven-gamma conditioning,
exact certificate, Valid2 acceptance, and faithful temp=1/NFE8 deployment remain unchanged.  The scientific
test is deliberately from the pretrained iteration-0 checkpoint, with no repaired checkpoint, boundary
adapter, preservation replay, oracle intervention, evaluation-seed replay, or per-seed surgery.

The iteration-100 M100 audit exposes an important ordering effect.  The full curriculum arm has 119
collision episodes in 700 fixed-gamma rollouts, versus 63 for the no-curriculum arm.  In the full arm,
116/119 collisions occur at interior obstacles and 70 occur at the first diagonal pinch (1,1); only three
are assigned to an added wall plug.  Thus the visible boundary removes the old boundary ambiguity, but an
unskilled from-scratch policy is harmed when low-margin frontier examples are emphasized immediately.
The recipe amendment being tested is phased curriculum: uniform certified learning until sustained
competence, then enable frontier pressure.  The pure full/no-curriculum arms remain immutable controls.

Dropping the multi-step SOCP condition gives the expected mechanistic ablation: 309/700 collision episodes
at iteration 100, with successful-episode clearance around 0.22 m.  Completion-time comparisons always use
successful episodes only, so early collision deaths are never credited as speed.

The pinch failure is not merely another seed-12-style rare latent.  At gamma 0.4, seeds 16/28/48 form a
fixed marginal trio and all remain failed from iteration 100 through 104.  Holding each trajectory context
and 4,096 base latents fixed, the planned-window collision rate ten replans before impact is 84%, 89%, and
29% at iteration 100 (82%, 89%, and 21% at iteration 104).  At the final pre-collision context, every latent
causes an immediate collision.  The curriculum has driven the robot into a distribution-level pinch basin
that becomes absorbing; emphasizing additional low-margin samples before the policy can execute the pinch
does not target a single bad fiber.  Evidence: `analysis/walls4_pinch_latent_offset10_it100_it104.json` and
`figures/walls4_pinch_latent_offset10_it100_it104.png`.

<FINAL_STATUS>
