# Corrected P2 trainer regression audit

Date: 2026-07-10. Scope: independent read/test harness only; this audit did not edit the production
trainer or metrics and did not touch Mizuta/Kazuki.

## Result

All **14/14** semantic gates pass on CPU and on physical GPU 2. The GPU run saw exactly one visible CUDA
device and verified that its RNG bytes are restored. Machine-readable results are in
`test_corrected_trainer.cpu.json` and `test_corrected_trainer.gpu2.json`.

```bash
OMP_NUM_THREADS=16 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib \
  CUDA_VISIBLE_DEVICES='' python analysis/test_corrected_trainer.py \
  --json analysis/test_corrected_trainer.cpu.json

OMP_NUM_THREADS=16 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib \
  CUDA_VISIBLE_DEVICES=2 python analysis/test_corrected_trainer.py \
  --json analysis/test_corrected_trainer.gpu2.json
```

## Gates covered

- NumPy, Python, Torch CPU, and visible-CUDA RNG isolation, including `_measure`, `_cov_probe`, and
  `_escape_probe` integration.
- Literal verifier face margin: 28/28 feasible realistic windows matched independent
  `min(real Face.m)`/`R_eff` recomputation exactly; 28/28 were distinct at 1e-8 and ranged 0.0916--0.5944.
- Per-gamma AND planes and nonempty classes for all seven gammas; NaN/nonpositive margins are unlabelable.
- Gamma -> staircase mode -> rollout-balanced draws, without repeats while unique data is available.
- Closed-loop CFM targets contain ten consecutive actions that were actually executed, with no proposal
  tail leakage or context misalignment.
- Strict `reach=0.1`, exact executed Valid2, rejected-proposal query memory, exact full-target certificate
  rejection, and exact face-margin caching.
- Gather stop semantics: unique easy/frontier quotas and easy+frontier coverage for every gamma are both
  required before readiness.
- Full state capture/restore for Adam, qbuf, coverage, pile, fixed LwF teacher, history, rolling counters,
  selection/collapse counters, and all RNG families.
- Deterministic two-step continuation: uninterrupted `2` iterations and split `1 + resume + 1` iterations
  matched exactly on model tensors (**max absolute error 0.0**), optimizer, 16-row qbuf, pile, teacher,
  coverage, histories/counters, and CPU/CUDA/NumPy/Python RNG state, despite seed 999 at resume.

The harness is `analysis/test_corrected_trainer.py`; it uses synthetic mocks for gather semantics, a tiny
CFM policy for exact split-continuation testing, and a historical P2 `viz_db` only as realistic verifier
window inputs.
