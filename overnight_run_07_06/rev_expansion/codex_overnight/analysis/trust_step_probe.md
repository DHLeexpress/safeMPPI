# Corrected P2 trust-step probe

Date: 2026-07-10 PDT  
Scope: the seed-15 iteration-100 P2 incumbent and corrected-semantic fine tuning only. Mizuta/Kazuki was not modified or expanded.

## Decision

Use **`lr=2e-5`, one optimizer step per fresh gather, with the encoder frozen** for the next bounded P2 continuation. Treat every optimizer step as a transaction: snapshot the full training state, apply one step, run the deterministic trust panel, and restore the model **and** optimizer/train state if any hard bound fails.

`lr=1e-4` is outside the measured behavioral trust radius. A single corrected update moved the origin field by 2.15%, reduced the causal late-goal first-action bias by 0.041, and reduced paired all-gamma M25 SR by 6.9 points on average (16 points at the worst gamma). `lr=2e-5` frozen moved the field by 0.378%, reduced the bias by only 0.012, and changed paired M25 SR by -0.6 point on average (4 points at the worst gamma), with CR=0 throughout.

Freezing is the conservative choice, not a large-effect claim: at the same `lr=2e-5`, unfreezing moved the origin field 0.418% versus 0.378%, gave mean M25 SR .886 versus .897, and did not improve any diagnostic enough to justify encoder drift. Reconsider unfreezing only after a coverage-stall checkpoint, as a separately gated branch.

## Controlled comparison

All corrected arms used the same semantic fixes and the same structure: iteration 101 was a gather-only legacy-state prime and iteration 102 applied exactly one corrected update. The field panel uses identical balanced demo contexts, origin inputs, latent draws, and integration settings.

| Checkpoint | Trunk / head / encoder relative drift | Origin-field relative L2 | Fixed-demo CFM MSE | Late-goal `a0_x-a0_y` | Y-dominant first actions | First-action sample drift | M25 mean / min SR | Worst paired SR loss | max CR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 0 / 0 / 0 | 0 | 1.0846 | +.1543 | 40.16% | 0 | .903 / .88 | 0 | 0 |
| `lr=1e-4`, encoder live | .0704% / .0445% / .0346% | **2.150%** | 1.0771 | +.1135 | 42.81% | **5.90%** | .834 / .76 | **.16** | 0 |
| `lr=2e-5`, encoder live | .0141% / .0089% / .0069% | .418% | 1.0828 | +.1404 | 40.93% | 1.72% | .886 / .84 | .04 | 0 |
| `lr=2e-5`, encoder frozen | .0141% / .0089% / 0 | **.378%** | 1.0830 | **+.1423** | **40.87%** | **1.50%** | **.897 / .84** | .04 | 0 |

The lower fixed-demo CFM loss of the rejected `lr=1e-4` arm is important: batch/offline loss improved while closed-loop success worsened. CFM loss is telemetry, not an acceptance criterion.

Paired M25 SR by gamma (all CR values are zero):

| Arm | .1 | .2 | .3 | .4 | .5 | .7 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | .88 | .96 | .92 | .88 | .88 | .88 | .92 |
| `lr=1e-4`, live encoder | .92 | .80 | .84 | .80 | .76 | .84 | .88 |
| `lr=2e-5`, live encoder | .88 | .92 | .88 | .84 | .88 | .88 | .92 |
| `lr=2e-5`, frozen encoder | .92 | .96 | .88 | .84 | .88 | .88 | .92 |

The missing unfreezed M25 panel was run as the one bounded GPU2 probe. It evaluated the existing checkpoint only; no additional model was trained and production code was untouched.

## Temporal and late-goal bias

The fixed panel reproduces the previously validated late-goal causal diagnostic on 64 contexts and 128 identical noise draws per context. The incumbent has `mean(a0_x-a0_y)=+.1543`; historically this metric correlates strongly with all-gamma SR because only `a0` is executed before replanning.

The rejected `lr=1e-4` step changes the generated action sequence much more than its small parameter drift suggests:

| Arm | Early-half action drift | Late-half action drift | Late/early ratio | Full-plan `sum(ax)-sum(ay)` |
|---|---:|---:|---:|---:|
| `lr=1e-4`, live encoder | 10.38% | 8.55% | .824 | 2.788 |
| `lr=2e-5`, live encoder | 1.50% | 1.19% | .789 | 3.476 |
| `lr=2e-5`, frozen encoder | 1.45% | 1.19% | .821 | 3.475 |

Thus the corrected coherent target removed the old unexecuted-tail supervision mismatch, but a too-large optimizer step still perturbs the causally executed early controls most. There is no evidence here for increasing late-action weight; the appropriate control is a functional field/action trust gate.

## Hard per-update trust gate

Measure changes against the immediately preceding accepted checkpoint on fixed, versioned inputs. An update is accepted only if every bound holds:

- Origin-field mean relative L2 across all seven gammas: **<= 0.005**.
- Relative parameter L2: trunk **<= 0.00020**, head **<= 0.00013**. The encoder remains exactly unchanged in the recommended phase.
- Fixed late-goal panel: `mean(a0_x-a0_y) >= +0.10`, its one-step drop **<= 0.03**, and Y-dominant fraction **<= .45**.
- Fixed late-goal sampled controls: first-action relative L2 **<= .03** and early-half mean relative L2 **<= .03**.
- Balanced-demo fixed CFM MSE may not increase by more than **0.5%**. An improvement does not override another failed bound.
- On a paired all-gamma M25 gate: every gamma has **CR=0** and no gamma loses more than **.08 SR** versus the preceding accepted checkpoint.
- Until a new all-gamma M100-safe checkpoint is promoted, cumulative origin-field drift from the current M100 anchor stays **<= .01**.

Rollback means restoring model weights, Adam moments, q-buffer/GP state, teacher, pile, coverage/history counters, and all RNG states. Restoring weights alone changes the next update and is not a valid rollback.

## Stop/measure and promotion thresholds

Stop training before another update and run the paired all-gamma M25 panel when any soft boundary is reached:

- one-step origin-field drift **>= .004**;
- late-goal first-action bias drops by **>= .02**;
- early-half sampled-control drift **>= .02**;
- any gamma loses **>= .04 SR** on the paired M25 panel;
- cumulative origin-field drift from the M100 anchor reaches **.008**.

These are warning thresholds, not promotion criteria. Promotion still requires the goal's all-gamma M>=100 evidence, CR=0, and the required safety/speed/coverage comparisons. After such a checkpoint is accepted, make it the new cumulative trust anchor.

## Why no `lr=1e-5` training branch

The existing arms already bracket the upper stable step: `1e-4` fails three independent functional/behavioral gates while `2e-5` frozen lies inside all hard bounds. Another half-step arm would test conservatism, not locate the unsafe boundary, and would consume a gather without evidence that smaller movement improves expansion. If `2e-5` repeatedly touches a soft boundary, the prescribed response is rollback and retry that update at `1e-5`; that is the useful adaptive role for the smaller rate.

Reproduce the deterministic JSON with:

```bash
LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 \
  python analysis/trust_step_probe.py
```

Primary artifacts: `analysis/trust_step_probe.json`, `analysis/trust_step_vector_field.json`, and the read-only M25 outputs under `analysis/runs/eval_corrected_*_it102_m25/`.
