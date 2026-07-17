# Five-round adaptive-acquisition and replay screen

This is an additive diagnostic protocol. Existing fixed-beta runners retain
their defaults.

The acquisition screen compares three arms at replay window `W=5`:

1. uniform B-without-replacement control;
2. previous-round RBF-GP with round-local median ESS target `0.5`;
3. cumulative five-member deep ensemble with round-local median ESS target
   `0.5`.

Set `AFE_SCREEN_PARALLEL=1` to schedule the three independent screen arms
concurrently on the caller's single visible GPU. Each arm writes a separate
`launcher.log`; estimator, seeds, and rollout semantics are unchanged. Leave it
at the default `0` when shared-GPU scheduling is not appropriate.
Both screen launchers cap OpenMP at 4 threads and BLAS pools at 1 thread by
default, separately from `--verifier-workers`; use the `AFE_*_NUM_THREADS`
environment overrides only after checking host load.

For adaptive arms, beta-neutral K-pools are generated after round `n` at every
context visited during that round, using the updated policy and estimator.
No extra verifier call or verifier label is used. The solved `beta_(n+1)` is
frozen during round `n+1`. `probe.jsonl` records `beta_used`, `beta_next`, the
solver witness, realized ESS, and replay exposure. Panel I of each report shows
the beta schedule and achieved calibration ESS.

The optional RBF counterfactual diagnostic evaluates length-scale multipliers
`{0.5,1,2}` and buffer caps `{128,512}` once, on the pretrained-policy
calibration feature pools. It is stored in `rbf_calibration.json` and is not
repeated during expansion. It does not select, verify, execute, or train on a
plan.

The replay screen accepts finite `W` in `{1, 3, 5, 10}`. The append-only D+ archive
is identical in semantics for every option. Only the positive IDs eligible for
each CFM minibatch change:

- `W=1`: current round;
- `W=3`: current and previous two rounds;
- `W=5`: current and previous four rounds;
- `W=10`: current and previous nine rounds.

The cumulative-replay control is not repeated: the completed 100-round runs
already exhibit its fresh-sample dilution. Run `W={1,3,5}` first; use `W=10`
only if the five-round learning/forgetting comparison is inconclusive.

Both screens are fixed to five rounds and preserve `K=64`, `B=8`, batch 128,
250 CFM steps, learning rate `1e-4`, the deterministic verifier, expert-free
NVP termination, and maximum-progress verified execution.

After the screen, `run_afe50_final.sh` runs the fixed single-arm confirmation:
adaptive deep ensemble, median ESS target `0.5`, replay `W=5`, and 50 rounds.
It emits authenticated PNG/PDF reports and a 14-frame video for rounds
`1..10,20,30,40,50`. Final held-out M=20 evaluation remains a separate step so
its seeds cannot select the training recipe or checkpoint.
