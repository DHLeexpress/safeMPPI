# Double-shift SFM fresh restart

## Frozen scientific timepoint

The quoted double-shift OOD statistics were produced from the clean source
commit `ca7f0d718f8d70cf74833b1c75157caf7f1b13f2` on July 20, 2026.  The
authenticated environment-and-visualization child commit is
`b27df76fe461fd7f7e86ecf87e80cbf52e7f01d5`; this is the restart base.

The benchmark contract is:

- scene: `double_density_velocity_ood`;
- pedestrians: 40;
- pedestrian desired-speed range: 1.0--2.0 m/s;
- episode bank: 250000--250099;
- evaluation: raw temperature 1, NFE 8, M=100 per gamma;
- pretrained checkpoint SHA-256:
  `1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215`;
- historical A-r10 checkpoint SHA-256:
  `bf6f521dd2dd6de4cffcce672a8ce4adbf00bb14e71dd9fd27704d205f65744c`.

The authenticated pooled OOD results were:

| Method | SR | CR | Successful clearance | Successful time |
|---|---:|---:|---:|---:|
| Hp10 r0 raw | 70.00% | 30.00% | 0.131 m | 8.69 s |
| historical A-r10 raw | 69.43% | 30.29% | 0.128 m | 8.79 s |
| Kazuki default | 81.29% | 18.71% | 0.181 m | 4.39 s |
| Kazuki goal-stress | 75.29% | 24.71% | 0.163 m | 3.81 s |

The original metrics artifact is
`/home/dohyun/projects/sfm_hp10_b1_runs/ca7f0d7_preexp_double_shift/double_shift_ood/metrics.json`
on Helios.  Its local Mac copy is
`/Users/dhl/Documents/SFM_HP10_DOUBLE_SHIFT_PREEXP_b27df76/double_shift_ood/metrics.json`
with SHA-256
`566ac3fc87b727ad0957b837aca68a1fdd24777040584791bd372eb25e3b8977`.

## Repository relationship

`DHLeexpress/safe_flow_expansion_SFM` has a standalone packaging history, so
Git cannot express it as being a number of commits ahead of this historical
safeMPPI branch.  Its current source snapshot records safeMPPI commit
`e5ab47b`, which is eight safeMPPI source commits after `b27df76` and changes
34 SFM files.  Therefore this restart is published on isolated archive/restart
branches; `master` is not reset or force-pushed.

## One deliberate correction after the timepoint

The historical model, scene, exact K=16 moving-obstacle SOCP, and checkpoint are
preserved.  One user-approved semantic correction is applied before new
experiments: every queried action window is certified over all H=10
transitions.  Crossing the goal does not truncate a queried window.  Goal reach
only terminates the closed-loop episode after the selected first action.

Thus new source should be described as **b27df76 plus the full-H10 correction**,
not as bitwise historical b27df76.

## Pre-expansion full-episode diagnostic

`sfm_b1_full_episode_audit.py` is an isolated diagnostic.  It does not modify
the historical fail-closed trainer or enter any sample into D, D+, the GP, or
gradient replay.

For three fixed episodes and all seven gamma values it starts with the
pretrained generator and the round-1 B1 mechanism:

1. generate K=16 windows;
2. select B=4 using the empty-buffer RBF acquisition with pending-point
   conditioning;
3. run the exact full-H10 verifier on B;
4. if an admissible B query exists, execute the max-one-step-Hp-margin action;
5. on finite-B NVP, independently sample and verify one raw temperature-1
   window, execute its first action, and continue the offline simulator;
6. stop only at realized collision, goal reach, or T=180 timeout.

Step 5 is deliberately **not** a certified controller.  It exists only to
observe post-NVP states that fail-closed gathering would hide.

Labels remain separate:

- `verifier_positive` / `verifier_negative`: exact safety label of the
  actually executed H=10 window;
- `finite_B_NVP`: B=4 failed to contain an admissible candidate; it is not
  itself a verifier-negative label;
- `trap`: displacement over the last ten executed transitions is below 0.2 m;
- `collision`: realized simulator outcome; it does not retroactively relabel
  earlier safe windows.

Consequently, NVP, trap, and collision must not be pooled blindly into the
negative verifier loss.  They can support a later, separately defined
continuation/viability label.

The companion `sfm_b1_full_episode_viz.py` renders:

- K=16 generated paths in gray;
- B=4 queried paths in orange;
- verifier-positive/rejected B paths in green/red;
- the complete executed trail in blue/red according to the exact H=10 label;
- the executed positive candidate's exact K=16 verifier polytope and H=1..10
  level sets in green;
- finite-B NVP, first trap entry, and collision as separate markers.

The resulting movie is a **pretrained-generator round-1 gathering diagnostic**,
not a pure raw-policy rollout and not a safety-certified deployment.
