# Giant-obstacle local-minimum OOD benchmark

## Scientific question

Can Safe Flow Expansion adapt a gamma-conditioned generative controller to a **local scene-geometry
shift** without changing the start, goal, dynamics, or policy structure?

The in-distribution (ID) task uses the ordinary 8-plug stadium and one fixed southwest-to-northeast
diagonal pair: start `(0.5,0.5)` and goal `(4.5,4.5)`. Both endpoints have 0.507 m obstacle clearance.
SafeMPPI supplies demonstrations for every gamma and the pretrained policy learns the diagonal prior.

The out-of-distribution (OOD) task keeps that exact pair and replaces only the four central obstacles
at `(2,2), (2,3), (3,2), (3,3)` with one circle centered at `(2.5,2.5)`. Its radius is selected by an
approval-gated sweep. The circle should nearly meet the surrounding obstacle ring while preserving a
SafeMPPI-feasible detour. This isolates visual/local-geometry OOD from endpoint, direction, corner, and
path-length shifts.

## Locked comparison

- Same start, goal, dynamics, horizon, reach radius, gamma values, network, and random seeds.
- ID scene differs from OOD scene only in the four-center-to-one-giant replacement.
- Gamma uses truncated plasma in figures; uncertainty sigma uses viridis.
- Report success, collision, task-space validity, time/path length, minimum clearance, detour side,
  giant-boundary clearance, and mode coverage per gamma.
- The active `codex_challenging` checkpoints and results are never overwritten.

## Approval-gated stages

### Stage 1A — geometry and expert-feasibility sweep

1. Select symmetric diagonal endpoints with clearance >=0.30 m in the ID scene.
2. Sweep giant radii `0.90, 1.00, 1.10, 1.20, 1.28` m.
3. Record the physical gap to the nearest surrounding obstacle.
4. Run one deterministic SafeMPPI rollout for all seven gamma values in the ID scene and each OOD
   candidate.
5. Render all trajectories and recommend, but do not silently choose, a radius.

The sweep recommended radius 1.20 m as the largest tested candidate with 7/7 expert feasibility. The
original automatically selected endpoints are retained only in the Stage 1A artifacts and are superseded
by the exact Stage 1B endpoints below.

### Stage 1B — smooth long-horizon expert at the provisional radius

1. Tentatively fix giant radius 1.20 m.
2. Fix start `(0.5,0.5)` and goal `(4.5,4.5)` exactly.
3. Tune only SafeMPPI `smooth_weight`, leaving the remaining mode-1 configuration unchanged.
4. Use an 800-control (80 s) ceiling so conservative gamma trajectories can finish.
5. Evaluate two matched-seed trajectories per gamma and report a--e plus independent progress/SOCP audits.

The selected pilot value is `smooth_weight=8.0`, the highest tested weight with 7/7 success. Weights 16
and 32 lose expert feasibility even with the long horizon.

**Approval gate:** user approves the exact pair, provisional radius, expert style, and disclosed validity
caveat before any training-data generation.

### Stage 1C — window-level validity and moving certificates

1. Reuse the exact 14 physically successful Stage 1B trajectories.
2. Score every emitted H=10 training-style sample on task space, goal progress, fitted-verifier SOCP, and
   their valid2 conjunction. Report executed-full and terminal-padded subsets separately.
3. Test the distinction between geometric nominal-polytope existence and an executed window satisfying
   that nominal polytope's gamma level-set schedule.
4. Animate one matched seed across all seven gammas: moving nominal polytope and H-step level set in blue,
   fitted verifier polytope and H-step level set in green.

Result: 61.6% of all training-style samples and 60.3% of executed-full windows pass joint valid2. Every
nominal-schedule-certified window also passes the fitted verifier (0 counterexamples), while the fitted
verifier additionally certifies 860 windows beyond the nominal schedule.

**Approval gate:** user approves the window-level interpretation and animation before Stage 2.

### Stage 2A — soft anti-retreat expert recipe

Add an optional exponential penalty only for predicted increases in goal distance. Keep its default at
zero, verify byte-exact compatibility with the locked Stage-1B expert, and tune its weight with matched
all-gamma rollouts. Validate the selected value with M=2 per gamma before generating any dataset.

Result: `goal_retreat_exp_weight=1`, scale `0.05 m`, cap `6` preserves 14/14 physical success and both
global detour homotopies while reducing mean executed retreat 28.2%, radial direction switches 18.5%,
time 7.6%, and path length 2.7%. The response is non-monotone above weight 1, so stronger is not better.

**Approval gate:** approve the soft expert recipe and disclosed whole-path-valid2 caveat.

### Stage 2B — geometrically balanced fixed-pair ID demonstrations

Generate stochastic fixed-pair demonstrations on the ordinary 4x4 stadium only, for all seven gamma
values, using the approved Stage-2A recipe. Classify successful paths by monotone right/up crossing word
(four R and four U crossings, at most `C(8,4)=70` signatures) and detour side. Oversample seeds and accept
an equal quota over the geometric signatures observed for each gamma. Give every accepted trajectory
equal loss mass before H=10 slicing so long or stalled episodes cannot bias the pretrained model.

Store the complete unbalanced candidate pool, selected balanced paths, exact H=10 windows, per-window
validity masks, nominal-polytope diagnostics, quota audit, and per-gamma clearance/time statistics.

Result: 168 real successful SafeMPPI paths (24/gamma) produce 10,752 complete executed windows. R/U
reflection counts are exact at both trajectory (84/84) and window (5,376/5,376) level; padding and
physical window failures are both zero. An independent audit traces every target back to its source
controls with zero error.

**Approval gate:** approve the balanced ID expert distribution and dataset visualization.

### Stage 3 — ID pretraining

Train the original endpoint-free model structure on Stage 2B. The stopping criterion is strong ID behavior
for every gamma, with no OOD samples. Save training curves and an all-gamma ID rollout panel.

**Approval gate:** approve the diagonal pretrained controller before OOD evaluation.

### Stage 4 — frozen OOD baselines and Mizuta tuning

On the approved giant scene, evaluate:

- SafeMPPI expert (feasibility ceiling),
- the frozen pretrained policy (expected diagonal collision), and
- tuned low-guidance Mizuta/CFM-MPPI (expected pocket/local-minimum trapping, conservatism, or mode
  collapse while retaining visible pretrained-policy behavior).

Produce failure-taxonomy and local-minimum zooms. No learning occurs in this stage.

**Approval gate:** confirm that the selected scene is difficult for learned baselines but feasible for the
expert. If not, return only to the Stage 1 radius sweep.

### Stage 5 — bounded Safe Flow Expansion and the three No controls

Run short, matched full, `-SOCP`, `-Progress`, and `-Curriculum` expansions using the approved OOD scene.
Track exact accepted/rejected windows, demo fraction, gamma composition, SOCP margin, update magnitude,
boundary-following arc, detour side, and faithful evaluation. `-SOCP` removes only SOCP from valid2;
`-Progress` removes only progress; `-Curriculum` retains the same accepted-sample count as Full but does
not separate easy/frontier pools. The desired qualitative signal is successful boundary following whose
clearance and route shape change with gamma.

Result (bounded it0--20): the corrected LR `5e-6` suite is mechanically clean and independently audited.
Full gathered 5,511 coherent valid2 windows with zero whole-valid2 parent trajectories; `-SOCP` made zero
SOCP calls, `-Progress` made zero progress calls, and `-Curriculum` matched Full's positive count exactly.
All arms updated every iteration with zero rollback. The behavioral gate did not pass: Full is 0/42 with
42 collisions at temperature 0.5 (also 0/42 at 0.1 and 1.0), so no successful route-mode diversity exists.

**Approval gate:** review the curriculum video, internals, rollouts, scatter, and table before any long run.

### Stage 6 — exact-style reports and rollout visualization

Only after Stage 5 approval, produce the exact requested curriculum, internals, rollout, scatter, and table
styles. The rollout panel includes pretraining data, Expert, Pretrained, tuned Mizuta/CFM-MPPI, all three
No controls, and Ours. The second Expert panel uses the approved anti-retreat recipe. A claim requires high
success, low collision, meaningful gamma-conditioned behavior, ID retention, and a clean improvement over
Pretrained and CFM-MPPI.

Result: exact-style rollout, internals, scatter, 21-second curriculum MP4, and matched temperature diagnostic
were generated under `stage_results/06_exact_reports/`. The mechanics/artifact audit passes; its scientific
outcome is explicitly `ALERT`, not a positive claim.

## Stop conditions

- Never launch the next stage without explicit user approval.
- Never weaken the expert/validity definition to manufacture success.
- Never report physical expert success as `valid2`: Stage 1B paths are collision-free and reach the goal,
  but boundary-following windows violate the current net-progress criterion and three tight high-gamma
  paths fail external fitted-SOCP re-certification.
- Keep trajectory-level and sample-level claims separate: a trajectory fails whole-path valid2 if any one
  window fails, even though a majority of its individual training samples can remain valid2-positive.
- If SafeMPPI is not reliably feasible, change only the giant radius at Stage 1.
- If Pretrained already solves the OOD scene, increase radius within the feasible sweep rather than
  changing endpoints or adding another distribution shift.
- If every baseline fails because the scene is geometrically sealed, decrease radius.
