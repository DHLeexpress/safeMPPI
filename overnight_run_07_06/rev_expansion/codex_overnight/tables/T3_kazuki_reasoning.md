# Kazuki / UnifiedGenRefine baseline tuning rationale

The local port retains the original generate--refine structure: reward-gradient flow guidance, top-10 mode
selection, 200 perturbations per elite, and MPPI softmax temperature 0.1. The source method used 80-step
pedestrian windows; this grid policy predicts 10 controls. With the original stage weights, the obstacle
term `100 * (1 + .99^t) * exp(-20(d-r))` dominates the `0.1 * goal_distance` term throughout that short
horizon. The observed failure is therefore a safe timeout roughly 0.45 m from the goal, not collision.

The controlled γ=0.5 scale test held guidance fixed at `w_safe=.5`, goal guidance `.5`, and goal cost `2.0`:

| Collision weight | M | SR | CR | Outcome |
|---:|---:|---:|---:|---|
| 50 | 3 | 0% | 0% | all three episodes timed out at 250 steps |
| 20 | 3 | 100% | 0% | 93--106 steps; 0.369±0.002 m clearance |

The next controlled test held collision weight 20 and compared only `w_safe` over all γ (M=5 per row).
Both `.3` and `.7` produced 100% SR / 0% CR, but `.3` completed in 8.88--10.26 s versus 10.38--12.52 s
for `.7`, without a consistent coverage disadvantage. The final settings are therefore:

| Parameter | Original port | Tuned | Reason |
|---|---:|---:|---|
| collision cost weight | 100 | 20 | remove the H=80→H=10 proximity-wall scale mismatch |
| goal stage/terminal weight | .1 | 2.0 | make progress resolvable inside a one-second prediction window |
| flow goal-guidance coefficient | .1 | .5 | keep generated endpoints moving rather than relying only on refinement |
| `w_safe` | mixed {.1,.3,.5,.7,.9} | .3 | lowest tested safe setting; faster and less locally conservative |
| proximity steepness | 20 | 20 | retained |
| candidates / elite / copies | 200 / 10 / 200 | 200 / 10 / 200 | retained |
| MPPI temperature / noise | .1 / .2 | .1 / .2 | retained |

This tuning makes the baseline credible on the new scene while retaining its defining single-reward local
guidance and generate--refine mechanism. It does not add the Safe Flow Expansion verifier or train the policy.
The authoritative M=200 results are in `T3_kazuki.md`. Coverage saturated at 5--8 modes; the retained fixed
recipe stays safe and fast but remains one mode below 70% of expert coverage at γ=.1/.3/1.0. Mixed candidate
weights did not improve that support, while w_safe=.1 introduced 10% timeouts at γ=.3/.7.
