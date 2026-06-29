# CFM-MPPI Dataset And Dynamics Notes

This note is based on the local checkout of `m-kazuki/cfm_mppi`, the `upstream/master`
remote, and direct inspection of the downloaded tensors under `dataset/`.

## Short Answer

Yes: in `eval80_ego_{ucy,sdd}.pt`, each of the 300 scenarios appears to choose one
pedestrian trajectory and reinterpret that pedestrian as the ego navigation task.
The planner does not replay that pedestrian trajectory. In evaluation code it uses
only the selected ego pedestrian's final relative position as the robot goal:

- `batch_ego = torch.load('./dataset/eval80_ego_{dataset}.pt')`
- `goal = batch_ego[idx, :2, -1]`
- robot state is initialized independently as zero:
  - double integrator: `state = torch.zeros(1, 4)`
  - unicycle: `state = torch.zeros(1, 3)`

The other pedestrians in the same scenario are stored in `eval80_obs_{ucy,sdd}.pkl`
and used as moving obstacles.

For training, the core model is also simpler than the artifact shape suggests.
`train.py` loads only `dataset/train80_ego.pt`, and `train_loop.py` uses only:

- channels `0:2`: position
- channels `2:4`: velocity, interpreted as single-integrator control

The extra channels in `train80_ego.pt` are not used by the original CFM loss.
So the training target is basically pedestrian velocity conditioned on pedestrian
start and final position.

## What The Tensors Contain

Local artifact shapes:

- `dataset/train80_ego.pt`: `[273989, 9, 80]`, `float32`
- `dataset/eval80_ego_ucy.pt`: `[300, 6, 80]`, `float32`
- `dataset/eval80_ego_sdd.pt`: `[300, 6, 80]`, `float32`
- `dataset/eval80_obs_ucy.pkl`: list of 300 tensors, each roughly `[1, N, 6, 80]`
- `dataset/eval80_obs_sdd.pkl`: list of 300 tensors, each roughly `[1, N, 6, 80]`

Downloaded ETH/BIWI raw-data audit:

- Downloaded official full archive to
  `external_data/eth_biwi/ewap_dataset_full.tgz`.
- The full archive contains `seq_eth` and `seq_hotel`, including videos and the
  same `obsmat.txt` annotations as the light archive.
- Raw annotation counts:
  - `seq_eth`: 8,908 observation rows, 360 unique pedestrian IDs.
  - `seq_hotel`: 6,544 observation rows, 390 unique pedestrian IDs.
  - combined: 15,452 observation rows, 750 sequence-local pedestrian IDs.
- Therefore `dataset/train80_ego.pt` with 273,989 samples is not a one-raw-
  pedestrian-to-one-training-sample dataset. Each training sample is an ego-only
  trajectory snippet/window derived from a raw pedestrian track.

The `80` means 80 future samples at `dt = 0.1`, i.e. an 8 second horizon.
The stored positions are ego-relative future displacements. The current ego
position is implicit at the origin; the first stored position is already one
time step into the future. This is why direct inspection gives:

```text
pos[:, :, 0] == vel[:, :, 0] * 0.1
pos[:, :, t] - pos[:, :, t-1] == vel[:, :, t] * 0.1
```

up to floating point error.

Important start-token detail:

- In training, the transformer receives `start=pos[:, :, 0]`, so the start token
  is the first stored future relative position, not the implicit current origin.
- In evaluation, `synthesize_control` calls `run_CFM` with `start_pos =
  torch.zeros(1, 2)`, so the CFM start token is exactly the current robot-relative
  origin.
- Therefore the learned model is effectively trained on near-origin first-step
  starts and evaluated with an exact zero start token. The current position is
  not stored as an explicit `t=0` state in the tensor.

The 6-channel eval ego tensors are effectively:

- `0:2`: selected ego pedestrian future position, relative to the planning origin
- `2:4`: selected ego pedestrian velocity
- `4:6`: heading-like helper channels, approximately `sin(theta), cos(theta)`

The original evaluation scripts only need the final `0:2` position as the robot
goal. The full ego trajectory is mostly a source artifact and useful for
visualization/diagnosis, not a demonstrated robot rollout.

The 9-channel training tensor includes the same position and velocity channels,
plus helper channels such as heading/speed/angular-rate style metadata. Original
CFM training ignores those helper channels.

## What The Code Actually Uses

Training:

- `cfm_mppi/train.py`
  - `LightDataset("dataset/train80_ego.pt")`
  - random crop length `L in [10, 80]`
- `cfm_mppi/training/train_loop.py`
  - `pos = samples[:, :2, :]`
  - `control = samples[:, 2:4, :]`
  - CFM target is the velocity/control sequence.
  - model conditioning is `start=pos[:, :, 0]`, `goal=pos[:, :, -1]`.

Evaluation:

- `cfm_mppi/evaluation/eval_cfm_mppi_doubleintegrator.py`
  - loads `eval80_ego_{ucy,sdd}.pt` and `eval80_obs_{ucy,sdd}.pkl`
  - filters NaN-padded obstacle agents
  - sets `goal = batch_ego[idx, :2, -1]`
  - initializes robot state at zero
- `cfm_mppi/evaluation/eval_utils.py`
  - at each planning step, takes current obstacle position/velocity
  - predicts future obstacle positions with a constant-velocity rollout:
    `pos_obs_seq = obs_positions + cumsum(obs_velocities * dt)`

So they do not train the CFM with known future pedestrian states. The CFM is
conditioned only on start/goal. Obstacle information is injected later through
CFM reward-gradient markup and MPPI costs.

The "known other-agent position/velocity" assumption comes from the evaluation
wrapper, not from CFM training. For UCY/SDD, the evaluation script loads obstacle
tensors, slices current obstacle positions and velocities at time `t`, and passes
those exact values into `synthesize_control`. `synthesize_control` then predicts
future obstacle positions with a constant-velocity model. The planner is not
given the ground-truth future obstacle trajectory for planning, but the benchmark
does assume exact current obstacle positions and velocities are observable.

## Why They Convert Raw Pedestrian Data

The raw ETH/UCY/SDD data is not already in the form needed by this planner.
The conversion is mostly packaging, not a change of physical model:

- make every example a fixed horizon or random crop up to 80 steps
- resample/represent trajectories at `dt = 0.1`
- translate each selected pedestrian to an ego-centered relative frame
- turn finite-difference pedestrian velocity into the single-integrator control
  target
- store the final relative position as the goal
- store neighboring agents as variable-size obstacle tensors for evaluation

This is why `train80` does not need separate double-integrator and unicycle
versions. The learned object is a single-integrator velocity sequence. Robot
dynamics are handled downstream.

## Dynamics Mapping

The original CFM generator outputs single-integrator planar velocities
`[vx, vy]`. `FlowMPPI._convert_si_to_dynamics` maps those velocities into the
actual robot control space:

- unicycle:
  - projects `[vx, vy]` onto the current heading for linear velocity
  - uses a look-ahead/control-point offset `d = 0.1` for angular velocity
- double integrator:
  - computes feedforward acceleration from the velocity sequence
  - adds proportional velocity tracking feedback
  - the evaluation script sets `DI_KP = 3`

Therefore, `train80_ego.pt` is one SI-style training set, not two robot-specific
training sets.

## Benchmark Implications

The double-integrator notebook is a legitimate baseline target because it uses
`doubleintegrator_dynamics` and `FlowMPPI(..., dynamics_type="doubleintegrator")`.
But it is not a separately trained double-integrator imitation model. It is the
same CFM velocity prior, retargeted into double-integrator acceleration controls
inside MPPI.

Good comparison axes:

- Out-of-distribution generalization: train on ETH-like pedestrian SI velocities,
  evaluate on UCY, SDD, and synthetic SFM crowds.
- Runtime: compare CFM-MPPI refinement cost against geometric/safe-MPPI overhead.
- Model scope: original benchmark is planar 2D; it does not represent 3D.
- Information restriction: original CFM has no obstacle context in the learned
  model, while the planner observes all current obstacle positions/velocities
  and uses constant-velocity prediction. A useful tweak is to restrict the
  planner/model to only the nearest obstacle and compare against our geometric
  method under that information bottleneck.

## Caveat

The public repo does not include the raw ETH/UCY/SDD preprocessing script that
created the Google Drive `.pt`/`.pkl` files. The raw-to-artifact step above is
therefore inferred from tensor consistency checks and from how the original
training/evaluation code consumes the artifacts.
