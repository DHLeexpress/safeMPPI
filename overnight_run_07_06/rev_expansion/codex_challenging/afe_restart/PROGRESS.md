# AFE restart progress

## Stage 0 — immutable memory and conceptual reset

**CMD**

- Archived the complete 21 GB legacy workspace with ZIP compression outside
  the source folder.
- Ran SHA-256 and `unzip -tq` integrity validation.
- Audited the legacy acquisition, buffer, verifier, labeling, replay, and
  optimizer paths line by line.

**RESULT**

- Archive gate PASS; see `stage_results/00_archive/manifest.json`.
- All nine reported faults are supported by code evidence.
- Old Stage-2B tensor targets are closed-loop composites from ten replans, not
  complete planned SafeMPPI windows, and are invalid for the new contract.
- Old Stage 3 onward is invalidated. Old paths and signature census remain
  reference-only.

**DECISION**

Build a separate `afe_restart` implementation. Do not import the legacy
expansion trainer or promote any legacy checkpoint as a result of this method.

## Stage 1 — planned-window contract and mechanics

**CMD**

- Implemented immutable query/replay identities, cumulative 32-D linear
  uncertainty, atomic verifier batches, exact H=10 dynamics/full verifier,
  sigma-only acquisition, same-verifier SafeMPPI backup, fail-closed control,
  full-positive-ledger proximal updates, and isolated temperature-1 audits.
- Resolved and SHA-256 hashed every reused dependency in
  `stage_results/01_contract/logs/dependencies.json`.
- Ran `PYTHONPATH=$PWD pytest -q afe_restart/tests`.
- Ran a one-step production-SOCP controller smoke on physical GPU 1 using a
  legacy checkpoint only as a mechanics fixture, never as a new result.

**RESULT**

- 23/23 tests pass.
- The production smoke made exactly eight verifier calls, appended exactly
  eight design-matrix observations, found eight safe plans, and executed the
  exact first action of the highest-progress certified planned window.
- A verifier acquisition batch retains one shared pre-batch `sigma_n`; its
  records are committed atomically, so within-batch matrix updates cannot
  rewrite the score that caused selection.
- Every proximal optimizer step covers 100% of the positive ledger; batch size
  is memory-only gradient accumulation.

**DECISION**

The mechanics gate passes. Legacy Stage-2B executed-composite tensors remain
invalid. Proceed by regenerating real, balanced SafeMPPI full-plan targets on
physical GPU 1.
