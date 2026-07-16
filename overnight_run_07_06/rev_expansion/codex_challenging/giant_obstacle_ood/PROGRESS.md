# Progress

| Stage | State | Decision |
|---|---|---|
| 1A. Geometry and expert-feasibility sweep | Complete | Radius 1.20 m is the provisional choice |
| 1B. Smooth long-horizon expert, exact endpoints, M=2/gamma | Complete | Smooth weight 8; physical SR 100%/CR 0% |
| 1C. Window valid2 + moving nominal/verifier certificates | Complete | 61.6% sample valid2; nominal-cert implies verifier on all observed windows |
| 2A. Soft exponential anti-retreat expert | Complete and approved | Weight 1, scale 0.05 m, cap 6; 14/14 physical success; retreat -28.2% |
| 2B. Geometrically balanced fixed-pair ID demonstrations | Complete and approved | 168 real successful paths; 10,752 full H=10 windows; exact R/U balance; independent audit PASS |
| 3. ID pretraining | Complete and approved | Selected A32: ID SR 93.8%, CR 4.5%, R/U 57/48; gamma=1 caveat accepted |
| 4. Frozen OOD baselines + Mizuta tuning | Complete, awaiting approval | Expert 42/42; pretrained 0/112 with 112 collisions; low-guidance Mizuta 0/42 with 42 local-minimum timeouts |
| 5. Full expansion + three No controls | Blocked by approval gate | Not started |
| 6. Exact-style reports and rollout visualization | Blocked by approval gate | Not started |

The benchmark folder remains isolated from prior checkpoints and results. Stage 3 used the approved
Stage-2B files only and trained the original endpoint-free A32 policy from scratch. The selected
trajectory-held-out model uses exact reflection augmentation with an equivariance weight of 1.0. Its
plain M=16 ID rollout gate records SR 93.8%, CR 4.5%, and successful R/U 57/48. Gamma 1.0 is the only
strict per-gamma failure (SR 75%, CR 25%, R/U 6/6). The independent checkpoint/data/evaluator audit is
`PASS_WITH_GAMMA1_CAVEAT`. The user approved this caveat and explicitly launched Stage 4. No expansion
learning has started; Stage 4 remains frozen and learning-free.

## Stage 3 command / result / decision

**CMD (selected training):** `CUDA_VISIBLE_DEVICES=2 python giant_obstacle_ood/stage3_pretrain_balanced.py
--tag sanity_a32_sym500_eq1 --epochs 500 --batch 512 --lr 3e-4 --repr 32 --trunk-hidden 160 96
--symmetry-augment --equivariance-weight 1.0 --device cuda:0`

**CMD (selected gate):** `CUDA_VISIBLE_DEVICES=2 python giant_obstacle_ood/stage3_eval_id.py --checkpoint
.../sanity_a32_sym500_eq1/checkpoint_best.pt --history .../history.csv --outdir .../eval_m16_t01 --M 16
--T 300 --temperature 0.1 --nfe 12 --h-exec 1 --device cuda:0`

**RESULT:** best held-out objective 0.96553 at epoch 478; visual-token effective rank 6.28. The ID gate is
105/112 successes, 5/112 collisions, R/U 57/48. Gamma 0.1--0.7 pass; gamma 1.0 is 12/16 success and 4/16
collision. The exact panel and curves are under `stage_results/03_pretrain/viz/`.

**DECISION (approved):** retain the trajectory-held-out A32 checkpoint. Reject A20, A48, global OT coupling,
route-bit coupling, symmetry weight 5, and the U-biased all-data refit. Publish
`data/pretrained_id_balanced_a32.pt`; the gamma=1 caveat was accepted before Stage 4 launch.

A separate legacy cleared-stadium window-native expansion sanity lives under
`../stage_results/05_window_native`. It validates the new per-window gather semantics and all three No
controls, but is **not** a giant-obstacle Stage 5 result and must not be mixed into the current benchmark.
The giant-obstacle Stage 4 is now complete; Stage 5 still requires its own explicit approval.

## Stage 4 command / result / decision

**CMD:** `CUDA_VISIBLE_DEVICES=2 giant_obstacle_ood/run_stage4.sh`

**RESULT:** the approved Stage-2A M=2 paths plus four fresh matched replicates give 42/42 expert
successes and zero collisions. The frozen approved-temperature policy has 0/112 success and 112/112
collisions (109 on the giant obstacle). The faithful temperature-1.0 diagnostic also has 0/28 success
and 28/28 collisions. A bounded low-guidance Mizuta sweep selected `w_safe=.04`, `coll_w=4`,
`goal_w=2`, and `goal_coef=.2`; its final M=6×7 evaluation has zero collisions but all 42 trajectories
stall in the lower-left entry pocket. The final-30-step displacement is at most 0.0248 m. Runtime was
254.1 s on physical GPU 2.

**DECISION:** Stage 4 is complete and learning-free. The independent audit is `PASS`; all controls
re-integrate to the saved paths, labels recompute against the true per-obstacle radii, and the frozen
checkpoint hash is unchanged. Await explicit approval of `stage_results/04_frozen_ood/` before Stage 5.
