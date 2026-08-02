import numpy as np
import pytest

import sfm_hp100_eval as E


def test_clipped_verifier_rollout_uses_shared_velocity_cap():
    state = np.array([0.0, 0.0, 1.95, 0.0], np.float32)
    controls = np.tile(np.array([2.0, 0.0], np.float32), (10, 1))
    positions = E.clipped_rollout_positions(state, controls)
    assert positions.shape == (11, 2)
    # First step uses v=1.95; every later step uses the capped v=2.0.
    np.testing.assert_allclose(positions[1, 0], 0.205, atol=1.0e-6)
    np.testing.assert_allclose(np.diff(positions[1:, 0]), 0.21, atol=1.0e-6)


def test_summary_uses_only_successes_for_clearance_and_time():
    rows = []
    for gamma in E.SS.GAMMAS:
        rows.extend([
            dict(
                gamma=gamma, success=True, collision=False, timeout=False,
                validity=0.75, successful_clearance=0.2, time_to_goal=9.0,
            ),
            dict(
                gamma=gamma, success=False, collision=True, timeout=False,
                validity=0.25, successful_clearance=None, time_to_goal=None,
            ),
        ])
    summary = E.summarize(rows)
    assert summary["pooled"]["SR"] == 0.5
    assert summary["pooled"]["CR"] == 0.5
    assert summary["pooled"]["Validity"] == 0.5
    assert summary["pooled"]["successful_clearance"] == pytest.approx(0.2)
    assert summary["pooled"]["successful_time_to_goal"] == 9.0


def test_id_gate_is_raw_unit_temperature_and_matched_id(monkeypatch):
    called = {}

    def fake_evaluate(policy, **kwargs):
        called.update(kwargs)
        cells = {
            str(gamma): {
                "SR": 0.8, "CR": 0.2, "timeout": 0.0, "Validity": 0.7,
                "successful_clearance": 0.1, "successful_time_to_goal": 8.0,
            }
            for gamma in E.SS.GAMMAS
        }
        return [], {"pooled": next(iter(cells.values())), "per_gamma": cells}

    monkeypatch.setattr(E, "evaluate", fake_evaluate)
    result = E.id_raw_gate(object(), M=3, ep0=12000, device="cpu", seed=99)
    assert called["scene_profile"] == "matched_id"
    assert called["with_validity"] is True
    assert called["seed"] == result["noise_seed"] == 99
    assert result["temperature"] == 1.0
    assert result["NFE"] == 8
    assert result["pooled"]["Validity"] == 0.7
    assert set(result["per_gamma"]) == {str(gamma) for gamma in E.SS.GAMMAS}
