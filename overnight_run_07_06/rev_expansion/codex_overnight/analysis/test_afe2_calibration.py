from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "afe2_calibration_test_local", _ROOT / "afe2_calibration.py"
)
assert _SPEC is not None and _SPEC.loader is not None
BC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BC)


def _table(medians):
    return {
        str(beta): {
            "ess_p10": max(0.0, median - 0.05),
            "ess_med": median,
            "ess_p90": min(1.0, median + 0.05),
        }
        for beta, median in zip(BC.CANDIDATES, medians)
    }


def test_calibration_selects_only_an_in_band_candidate() -> None:
    table = _table((0.10, 0.36, 0.70))
    assert BC.select_beta(table) == 0.02


def test_calibration_refuses_nearest_fallback_when_none_is_in_band() -> None:
    with pytest.raises(ValueError, match="no beta candidate"):
        BC.select_beta(_table((0.05, 0.12, 0.63)))


def test_success_artifact_recomputes_choice_and_provenance() -> None:
    expected = {
        "checkpoint_sha256": "a" * 64,
        "scene_sha256": "b" * 64,
        "K": 64,
    }
    payload = {
        **expected,
        "status": BC.SUCCESS_STATUS,
        "candidates": list(BC.CANDIDATES),
        "target_ess_band": list(BC.ESS_BAND),
        "acquisition": BC.ACQUISITION,
        "pool_weighting": BC.POOL_WEIGHTING,
        "selection": BC.SELECTION,
        "n_pools": 20,
        "table": _table((0.12, 0.37, 0.61)),
        "chosen": 0.02,
    }
    assert BC.validate_success(payload, expected) == 0.02
    payload["chosen"] = 0.05
    with pytest.raises(ValueError, match="chosen value"):
        BC.validate_success(payload, expected)
