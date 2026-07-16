"""Regression checks for the separate window-native expansion module."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REF = HERE.parent
sys.path.insert(0, str(REF))

import window_expand_hardtail as WT  # noqa: E402


def _cfg(**kwargs):
    base = dict(valid_prog_floor=0.15, ablate_progress=False, ablate_socp=False)
    base.update(kwargs)
    return SimpleNamespace(**base)


def _decision(cfg, *, taskspace=True, approach=True, progress=0.2,
              socp=True, clearance=0.3):
    distances = np.array([2.0, 2.0 - progress], dtype=float)
    with (
        mock.patch.object(WT, "_window_progress",
                          return_value=(progress, np.zeros((11, 2)), distances)),
        mock.patch.object(WT.GM, "in_taskspace", return_value=taskspace),
        mock.patch.object(WT.GX2, "state_from_low5", return_value=np.zeros(4)),
        mock.patch.object(WT.GM2, "approach_ok", return_value=approach) as approach_fn,
        mock.patch.object(WT.GM2, "window_socp_stats",
                          return_value=(socp, clearance, 0.0)) as socp_fn,
        mock.patch.object(WT.GM2, "window_min_clearance", return_value=clearance) as clear_fn,
    ):
        result = WT._window_acceptance(np.zeros(5), np.zeros((10, 2)), object(), 0.3, cfg)
    return result, approach_fn.call_count, socp_fn.call_count, clear_fn.call_count


def run():
    checks = {}

    full, nap, nsocp, nclear = _decision(_cfg())
    assert full["accepted"] and (nap, nsocp, nclear) == (1, 1, 0)
    checks["full_mask"] = "taskspace+progress+SOCP"

    failed, *_ = _decision(_cfg(), socp=False)
    assert not failed["accepted"] and failed["failure"] == "socp"
    checks["sibling_failure_is_local"] = True

    no_socp, nap, nsocp, nclear = _decision(_cfg(ablate_socp=True), clearance=0.2)
    assert no_socp["accepted"] and no_socp["socp_ok"] is None
    assert (nap, nsocp, nclear) == (1, 0, 1)
    checks["minus_socp_mask"] = "taskspace+positive-clearance+progress"

    unsafe, *_ = _decision(_cfg(ablate_socp=True), clearance=-0.01)
    assert not unsafe["accepted"] and unsafe["failure"] == "safe_space"
    checks["minus_socp_still_safe_space"] = True

    no_progress, nap, nsocp, nclear = _decision(
        _cfg(ablate_progress=True), approach=False, progress=-1.0, socp=True)
    assert no_progress["accepted"] and no_progress["progress_ok"] is None
    assert (nap, nsocp, nclear) == (0, 1, 0)
    checks["minus_progress_mask"] = "taskspace+SOCP"

    assert WT._fresh_batch_plan(106, 0, (1.0, 0.0), 16) == (16, 0)
    assert WT._fresh_batch_plan(106, 0, (0.4, 0.6), 16) == (0, 0)
    checks["single_class_batch"] = "16+0 (not skipped)"

    gammas = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    idx = WT._balanced_window_prefix(gammas, 5)
    assert idx.tolist() == [0, 2, 4, 1, 3]
    checks["controlled_budget"] = {"requested": 5, "selected": idx.tolist()}

    sliced = WT._slice_window_fields({
        "U": torch.zeros(6, 10, 2), "gamma": torch.as_tensor(gammas),
        "whole_traj_valid": np.array([0, 1, 0, 1, 0, 1], dtype=bool),
        "rollout_reached": np.array([0, 0, 1, 1, 0, 1], dtype=bool),
        "paths": ["rollout metadata"],
    }, idx)
    assert sliced["whole_traj_valid"].tolist() == [False, False, False, True, True]
    assert sliced["rollout_reached"].tolist() == [False, True, False, False, True]
    assert sliced["paths"] == ["rollout metadata"]
    checks["controlled_budget_audit_fields"] = True

    source = Path(WT.__file__).read_text()
    assert "if not traj_ok and" not in source
    assert 'aggregation_semantics="window_native_v1"' in source
    checks["no_whole_trajectory_veto"] = True

    return checks


def main():
    checks = run()
    output = Path(os.environ.get(
        "WINDOW_EXPAND_TEST_JSON",
        HERE / "test_window_expand.json",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "PASS", "checks": checks}, indent=2) + "\n")
    print(f"PASS ({len(checks)}/{len(checks)}) -> {output}")


if __name__ == "__main__":
    main()
