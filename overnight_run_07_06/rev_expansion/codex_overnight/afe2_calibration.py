"""Shared fail-closed contract for one-shot AFE2 acquisition-temperature calibration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math

import numpy as np


ESS_TARGET = 0.375
ESS_TOLERANCE = 1e-4
MAX_BRACKET_STEPS = 80
MAX_BISECTION_STEPS = 100
SUCCESS_STATUS = "CALIBRATED_AFE2_CONTINUOUS_ESS"
FAILURE_STATUS = "CALIBRATION_FAILED_NO_ESS_ROOT"
ACQUISITION = "uniform B-without-replacement; beta-neutral"
POOL_WEIGHTING = "one equal vote per visited control-step K-pool across gamma sweep"
SOLVER = (
    "adaptive log-bisection for median ESS/K=0.375; fail on flat pools or an "
    "unbracketed root"
)


def _pool_array(pools: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(pools, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("calibration requires a nonempty rectangular collection of K-pools")
    if not np.isfinite(array).all():
        raise ValueError("calibration sigma pools contain non-finite values")
    return array


def sigma_pool_sha256(pools: Sequence[Sequence[float]]) -> str:
    """Content digest for the exact float32 sigma pools consumed by the solver."""

    array = np.ascontiguousarray(_pool_array(pools), dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def ess_summary(pools: Sequence[Sequence[float]], beta: float) -> dict[str, float]:
    """Return normalized acquisition-ESS statistics at one positive temperature."""

    array = _pool_array(pools)
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    logits = (array - array.max(axis=1, keepdims=True)) / beta
    weights = np.exp(np.clip(logits, -745.0, 0.0))
    probs = weights / weights.sum(axis=1, keepdims=True)
    values = 1.0 / (np.square(probs).sum(axis=1) * array.shape[1])
    return {
        "ess_p10": float(np.quantile(values, 0.1)),
        "ess_med": float(np.median(values)),
        "ess_p90": float(np.quantile(values, 0.9)),
    }


def solve_beta(pools: Sequence[Sequence[float]]) -> dict[str, object]:
    """Solve the predeclared median-ESS equation without a post-hoc candidate grid.

    ESS is monotone in beta.  The bracket is expanded from the median within-pool
    sigma span, then solved on log(beta).  A flat representation is reported as a
    calibration failure rather than hidden by an arbitrarily small temperature.
    """

    array = _pool_array(pools)
    spans = np.ptp(array, axis=1)
    positive = spans[spans > 0.0]
    if positive.size == 0:
        raise ValueError("all calibration sigma pools are flat")
    scale = float(np.median(positive))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibration sigma scale is not positive")

    def evaluate(beta: float) -> dict[str, float]:
        return ess_summary(array, beta)

    initial = evaluate(scale)
    low = high = scale
    low_stats = high_stats = initial
    bracket_steps = 0
    if initial["ess_med"] > ESS_TARGET:
        for bracket_steps in range(1, MAX_BRACKET_STEPS + 1):
            low *= 0.5
            low_stats = evaluate(low)
            if low_stats["ess_med"] <= ESS_TARGET:
                break
        else:
            raise ValueError("median ESS target is below the attainable tied-mode floor")
    elif initial["ess_med"] < ESS_TARGET:
        for bracket_steps in range(1, MAX_BRACKET_STEPS + 1):
            high *= 2.0
            high_stats = evaluate(high)
            if high_stats["ess_med"] >= ESS_TARGET:
                break
        else:
            raise ValueError("median ESS target could not be bracketed")

    candidates = [(abs(initial["ess_med"] - ESS_TARGET), scale, initial)]
    if low != scale:
        candidates.append((abs(low_stats["ess_med"] - ESS_TARGET), low, low_stats))
    if high != scale:
        candidates.append((abs(high_stats["ess_med"] - ESS_TARGET), high, high_stats))
    iterations = 0
    while low < high and iterations < MAX_BISECTION_STEPS:
        iterations += 1
        middle = math.sqrt(low * high)
        stats = evaluate(middle)
        candidates.append((abs(stats["ess_med"] - ESS_TARGET), middle, stats))
        if abs(stats["ess_med"] - ESS_TARGET) <= ESS_TOLERANCE:
            break
        if stats["ess_med"] < ESS_TARGET:
            low, low_stats = middle, stats
        else:
            high, high_stats = middle, stats

    error, chosen, achieved = min(candidates, key=lambda item: (item[0], item[1]))
    if error > ESS_TOLERANCE:
        raise ValueError(
            f"continuous beta solver missed ESS target by {error:.6g}"
        )
    return {
        "beta": float(chosen),
        "achieved": achieved,
        "target": ESS_TARGET,
        "tolerance": ESS_TOLERANCE,
        "bracket": [float(low), float(high)],
        "bracket_steps": int(bracket_steps),
        "bisection_steps": int(iterations),
        "sigma_span_p10": float(np.quantile(spans, 0.1)),
        "sigma_span_med": float(np.median(spans)),
        "sigma_span_p90": float(np.quantile(spans, 0.9)),
        "flat_pool_fraction": float(np.mean(spans == 0.0)),
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def validate_success(payload: Mapping[str, object], expected: Mapping[str, object]) -> float:
    """Validate a persisted successful calibration before expensive arm 1."""

    if payload.get("status") != SUCCESS_STATUS:
        raise ValueError("beta calibration is not a successful locked artifact")
    if float(payload.get("ess_target", float("nan"))) != ESS_TARGET:
        raise ValueError("beta calibration ESS target changed")
    if float(payload.get("ess_tolerance", float("nan"))) != ESS_TOLERANCE:
        raise ValueError("beta calibration ESS tolerance changed")
    if payload.get("solver") != SOLVER:
        raise ValueError("beta calibration solver changed")
    if payload.get("acquisition") != ACQUISITION:
        raise ValueError("beta calibration acquisition rule changed")
    if payload.get("pool_weighting") != POOL_WEIGHTING:
        raise ValueError("beta calibration pool weighting changed")
    if int(payload.get("n_pools", 0)) <= 0:
        raise ValueError("beta calibration contains no candidate pools")
    if not _is_sha256(payload.get("sigma_pool_sha256")):
        raise ValueError("beta calibration has no valid sigma-pool digest")
    mismatched = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatched:
        raise ValueError(f"beta calibration provenance mismatch: {mismatched}")

    solution = payload.get("solution")
    if not isinstance(solution, Mapping):
        raise ValueError("beta calibration has no solver witness")
    chosen = float(payload.get("chosen", float("nan")))
    if not math.isfinite(chosen) or chosen <= 0.0 or float(solution.get("beta", 0.0)) != chosen:
        raise ValueError("beta calibration chosen value disagrees with its solver witness")
    achieved = solution.get("achieved")
    if not isinstance(achieved, Mapping):
        raise ValueError("beta calibration has no achieved ESS statistics")
    values = [float(achieved.get(name, float("nan"))) for name in ("ess_p10", "ess_med", "ess_p90")]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("beta calibration ESS statistics are invalid")
    if not values[0] <= values[1] <= values[2]:
        raise ValueError("beta calibration ESS quantiles are unordered")
    if abs(values[1] - ESS_TARGET) > ESS_TOLERANCE:
        raise ValueError("beta calibration did not attain its ESS target")
    return chosen
