from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sfm_b1_offline_eval as E
import sfm_protocol as SP


def _trajectory(n_steps=3, *, collision=False):
    controls = np.arange(n_steps * 2, dtype=np.float32).reshape(n_steps, 2)
    return {
        "episode": 1,
        "gamma": 0.5,
        "status": "collision" if collision else "success",
        "success": not collision,
        "collision": collision,
        "timeout": False,
        "steps": n_steps,
        "time_to_goal": 1.0 if not collision else None,
        "successful_clearance": 0.2 if not collision else None,
        "states": np.zeros((n_steps + 1, 4), np.float32),
        "controls": controls,
        "ped_xy": np.zeros((n_steps, 0, 2), np.float32),
        "ped_vel": np.zeros((n_steps, 0, 2), np.float32),
    }


def _compact_row(episode, gamma, validity):
    evaluated = 10
    valid = int(round(float(validity) * evaluated))
    return {
        "episode": int(episode),
        "gamma": float(gamma),
        "status": "success",
        "success": True,
        "collision": False,
        "timeout": False,
        "time_to_goal": 9.0,
        "successful_clearance": 0.2,
        "validity": float(validity),
        "valid_windows": valid,
        "evaluated_windows": evaluated,
        "verifier_errors": 0,
    }


def test_terminal_windows_use_actual_executed_controls_and_all_starts(monkeypatch):
    row = _trajectory(12)
    calls = []

    def fake(state, controls, ped_xy, ped_vel, gamma):
        calls.append(np.asarray(controls).copy())
        return {
            "resolved": True,
            "y": 1,
            "window_horizon": len(controls),
        }

    monkeypatch.setattr(E.SM, "verify_executed_window", fake)
    result = E._verify_executed_episode(row)

    assert [len(controls) for controls in calls] == [
        10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    ]
    assert np.array_equal(calls[0], row["controls"][:10])
    assert np.array_equal(calls[-1], row["controls"][-1:])
    assert result == {
        "validity": 1.0,
        "valid_windows": 12,
        "evaluated_windows": 12,
        "verifier_errors": 0,
    }


def test_validity_is_fractional_and_does_not_stop_at_first_negative(monkeypatch):
    row = _trajectory(3, collision=True)
    outcomes = iter((1, 0, 1))

    def fake(state, controls, ped_xy, ped_vel, gamma):
        return {
            "resolved": True,
            "y": next(outcomes),
            "window_horizon": len(controls),
        }

    monkeypatch.setattr(E.SM, "verify_executed_window", fake)
    result = E._verify_executed_episode(row)
    assert result["validity"] == pytest.approx(2 / 3)
    assert result["valid_windows"] == 2
    assert result["evaluated_windows"] == 3
    assert result["verifier_errors"] == 0


def test_verifier_error_is_not_silently_counted_as_negative(monkeypatch):
    row = _trajectory(3)
    calls = 0

    def fake(state, controls, ped_xy, ped_vel, gamma):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"resolved": False, "error": "solver failed"}
        return {"resolved": True, "y": 1, "window_horizon": len(controls)}

    monkeypatch.setattr(E.SM, "verify_executed_window", fake)
    result = E._verify_executed_episode(row)
    assert result["verifier_errors"] == 1
    assert result["evaluated_windows"] == 1


def test_zero_transition_trajectory_has_defined_zero_validity():
    result = E._verify_executed_episode(_trajectory(0))
    assert result == {
        "validity": 0.0,
        "valid_windows": 0,
        "evaluated_windows": 0,
        "verifier_errors": 0,
    }


def test_summary_uses_mean_of_per_trajectory_fractions():
    rows = [
        _compact_row(1, .5, 1.0),
        _compact_row(2, .5, .5),
    ]
    summary = E._summarize_one(rows, seed=7)
    assert summary["Validity"]["mean"] == pytest.approx(.75)
    assert summary["Validity"]["valid_windows"] == 15
    assert summary["Validity"]["evaluated_windows"] == 20
    assert summary["Validity"]["window_weighted_fraction"] == pytest.approx(.75)
    assert "V_safe" not in summary


def test_temperature_defaults_to_one_and_is_validated():
    parser = E.build_parser()
    args = parser.parse_args([
        "--checkpoints", "a.pt",
        "--labels", "r0",
        "--output-dir", "out",
    ])
    assert args.temperature == 1.0

    args.temperature = 0.0
    args.m_per_gamma = 1
    with pytest.raises(ValueError, match="temperature"):
        E.run(args)


def test_temperature_is_part_of_noise_and_cache_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "M_PER_GAMMA", 2)
    monkeypatch.setattr(E, "TEMPERATURE", 0.7)
    _, metadata = E._noise_bank(ep0=10, d=4, seed=12)
    assert metadata["temperature"] == pytest.approx(0.7)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(E, "_sha256_file", lambda path: "evaluator")
    key_07 = E._cell_key(
        checkpoint_sha256="checkpoint",
        scene_profile="double_density_velocity_ood",
        ep0=10,
        noise_meta=metadata,
    )
    monkeypatch.setattr(E, "TEMPERATURE", 1.0)
    key_10 = E._cell_key(
        checkpoint_sha256="checkpoint",
        scene_profile="double_density_velocity_ood",
        ep0=10,
        noise_meta={**metadata, "temperature": 1.0},
    )
    assert key_07 != key_10


def test_render_uses_ball_style_validity_name_and_writes_manifest(tmp_path):
    records = []
    for round_i in (0, 1, 2):
        rows = [
            _compact_row(
                episode,
                gamma,
                validity=min(1.0, .4 + .1 * round_i),
            )
            for gamma in SP.GAMMAS
            for episode in (1, 2)
        ]
        summary = E.summarize(rows, seed=round_i + 10)
        records.append({
            "label": f"r{round_i}",
            "round": round_i,
            "cell": {"summary": summary},
        })

    outputs = E.render(records, str(tmp_path))
    assert {Path(path).suffix for path in outputs} == {".png", ".pdf", ".json"}
    assert all(Path(path).stat().st_size > 0 for path in outputs)
    manifest_path = next(Path(path) for path in outputs if path.endswith(".json"))
    manifest = json.loads(manifest_path.read_text())
    assert "Validity" in manifest["claim"]
    assert "V_safe" not in manifest["claim"]
    assert [title for _, title, _ in E.PLOT_SPECS] == [
        "Collision rate",
        "Validity",
        "Min. clearance [m]",
        "Time-to-goal [s]",
    ]
