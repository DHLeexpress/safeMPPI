# Five-round adaptive-acquisition and replay screen

This is an additive diagnostic protocol. Existing fixed-beta runners retain
their defaults.

The acquisition screen compares three arms at replay window `W=5`:

1. uniform B-without-replacement control;
2. previous-round RBF-GP with round-local median ESS target `0.5`;
3. cumulative five-member deep ensemble with round-local median ESS target
   `0.5`.

For adaptive arms, beta-neutral K-pools are generated after round `n` at every
context visited during that round, using the updated policy and estimator.
No extra verifier call or verifier label is used. The solved `beta_(n+1)` is
frozen during round `n+1`. `probe.jsonl` records `beta_used`, `beta_next`, the
solver witness, realized ESS, and replay exposure. Panel I of each report shows
the beta schedule and achieved calibration ESS.

The optional RBF counterfactual diagnostic evaluates length-scale multipliers
`{0.5,1,2}` and buffer caps `{128,512}` on the same unlabeled feature pools. It
does not select, verify, execute, or train on a plan.

The replay screen accepts `W=1`, `W=5`, or `W=all`. The append-only D+ archive
is identical in semantics for every option. Only the positive IDs eligible for
each CFM minibatch change:

- `W=1`: current round;
- `W=5`: current and previous four rounds;
- `W=all`: full cumulative D+.

Both screens are fixed to five rounds and preserve `K=64`, `B=8`, batch 128,
250 CFM steps, learning rate `1e-4`, the deterministic verifier, expert-free
NVP termination, and maximum-progress verified execution.
