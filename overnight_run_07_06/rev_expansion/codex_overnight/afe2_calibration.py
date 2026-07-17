"""Shared fail-closed contract for the one-shot AFE2 beta calibration."""
from __future__ import annotations

from collections.abc import Mapping
import math


CANDIDATES = (0.01, 0.02, 0.05)
ESS_BAND = (0.25, 0.5)
ESS_TARGET = 0.375
SUCCESS_STATUS = "CALIBRATED_AFE2_RADIUS1_BETA"
FAILURE_STATUS = "CALIBRATION_FAILED_NO_BETA_IN_BAND"
ACQUISITION = "uniform B-without-replacement; beta-neutral"
POOL_WEIGHTING = "one equal vote per visited control-step K-pool across gamma sweep"
SELECTION = "closest median ESS/K to 0.375 among in-band candidates; fail if none"


def select_beta(table: Mapping[str, Mapping[str, float]]) -> float:
    """Return the predeclared in-band candidate nearest the ESS target."""

    expected_keys = {str(value) for value in CANDIDATES}
    if set(table) != expected_keys:
        raise ValueError("beta calibration table is incomplete")
    medians: dict[float, float] = {}
    for beta in CANDIDATES:
        row = table[str(beta)]
        values = [float(row[name]) for name in ("ess_p10", "ess_med", "ess_p90")]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"beta calibration ESS statistics are invalid for beta={beta}")
        if not values[0] <= values[1] <= values[2]:
            raise ValueError(f"beta calibration ESS quantiles are unordered for beta={beta}")
        medians[beta] = values[1]
    in_band = [
        beta for beta in CANDIDATES
        if ESS_BAND[0] <= medians[beta] <= ESS_BAND[1]
    ]
    if not in_band:
        raise ValueError("no beta candidate lies in the declared ESS/K band")
    return min(in_band, key=lambda beta: (abs(medians[beta] - ESS_TARGET), beta))


def validate_success(payload: Mapping[str, object], expected: Mapping[str, object]) -> float:
    """Validate a persisted successful calibration before expensive arm 1."""

    if payload.get("status") != SUCCESS_STATUS:
        raise ValueError("beta calibration is not a successful locked artifact")
    if payload.get("candidates") != list(CANDIDATES):
        raise ValueError("beta calibration candidate set changed")
    if payload.get("target_ess_band") != list(ESS_BAND):
        raise ValueError("beta calibration ESS band changed")
    if payload.get("acquisition") != ACQUISITION:
        raise ValueError("beta calibration acquisition rule changed")
    if payload.get("pool_weighting") != POOL_WEIGHTING:
        raise ValueError("beta calibration pool weighting changed")
    if payload.get("selection") != SELECTION:
        raise ValueError("beta calibration selection rule changed")
    if int(payload.get("n_pools", 0)) <= 0:
        raise ValueError("beta calibration contains no candidate pools")
    mismatched = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatched:
        raise ValueError(f"beta calibration provenance mismatch: {mismatched}")
    chosen = select_beta(payload.get("table") or {})
    if float(payload.get("chosen")) != chosen:
        raise ValueError("beta calibration chosen value disagrees with its ESS table")
    return chosen
