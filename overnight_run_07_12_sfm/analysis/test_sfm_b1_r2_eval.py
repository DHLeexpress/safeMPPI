from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sfm_b1_r2_eval as E
import sfm_protocol as SP


def _row(episode, gamma, *, status, clearance=None, time=None, v_safe=False):
    return {
        "episode": int(episode),
        "gamma": float(gamma),
        "status": status,
        "success": status == "success",
        "collision": status == "collision",
        "timeout": status == "timeout",
        "successful_clearance": clearance,
        "time_to_goal": time,
        "v_safe": bool(v_safe),
        "verifier_errors": 0,
        "certified_windows": 10,
    }


def test_archive_reference_is_separate_and_m50_bank_is_disjoint():
    E._assert_disjoint_from_archive(260_000)
    with pytest.raises(ValueError, match="disjoint"):
        E._assert_disjoint_from_archive(250_050)
    assert E.ARCHIVED_M100_REFERENCE["M_per_gamma"] == 100
    assert E.M_PER_GAMMA == 50
    assert "not_a_curve_point" in E.ARCHIVED_M100_REFERENCE["role"]


def test_checkpoint_specs_require_unique_increasing_round_labels(tmp_path):
    checkpoints = []
    for name in ("a.pt", "b.pt"):
        path = tmp_path / name
        path.write_bytes(b"x")
        checkpoints.append(str(path))
    specs = E._checkpoint_specs(checkpoints, ["r0", "r1"])
    assert [spec["round"] for spec in specs] == [0, 1]
    with pytest.raises(ValueError, match="form"):
        E._checkpoint_specs(checkpoints, ["pretrained", "r1"])
    with pytest.raises(ValueError, match="increasing"):
        E._checkpoint_specs(checkpoints, ["r1", "r0"])


def test_summarize_uses_actual_collision_and_success_only_continuous_metrics():
    rows = []
    for gamma in SP.GAMMAS:
        rows.extend([
            _row(
                260_000, gamma, status="success", clearance=0.2,
                time=9.0, v_safe=True,
            ),
            _row(
                260_001, gamma, status="timeout", clearance=None,
                time=None, v_safe=True,
            ),
            _row(
                260_002, gamma, status="collision", clearance=None,
                time=None, v_safe=False,
            ),
        ])
    summary = E.summarize(rows, seed=7)
    pooled = summary["pooled"]
    assert pooled["SR"] == pytest.approx(1 / 3)
    assert pooled["CR"] == pytest.approx(1 / 3)
    assert pooled["timeout"] == pytest.approx(1 / 3)
    assert pooled["V_safe"] == pytest.approx(2 / 3)
    assert pooled["successful_clearance"]["mean"] == pytest.approx(0.2)
    assert pooled["successful_clearance"]["n"] == len(SP.GAMMAS)
    assert pooled["successful_time_to_goal"]["mean"] == pytest.approx(9.0)
    assert pooled["successful_time_to_goal"]["n"] == len(SP.GAMMAS)


def test_noise_bank_is_deterministic_and_checkpoint_common():
    first, first_meta = E._noise_bank(ep0=260_000, d=20, seed=123)
    second, second_meta = E._noise_bank(ep0=260_000, d=20, seed=123)
    assert first.shape == (len(SP.GAMMAS), 50, E.T, 20)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert first_meta == second_meta
    assert first_meta["temperature"] == 1.0
    assert first_meta["NFE"] == 8


def test_render_writes_paper_style_png_and_pdf(tmp_path):
    records = []
    for round_i in (0, 1, 2):
        per_gamma = {}
        rows = []
        for gamma in SP.GAMMAS:
            cell_rows = [
                _row(
                    260_000, gamma, status="success",
                    clearance=0.1 + round_i * 0.01,
                    time=9.0 + round_i, v_safe=True,
                ),
                _row(
                    260_001, gamma, status="collision",
                    clearance=None, time=None, v_safe=False,
                ),
            ]
            rows.extend(cell_rows)
            per_gamma[str(gamma)] = E._summarize_one(cell_rows, round_i + 1)
        pooled = E._summarize_one(rows, round_i + 100)
        for metric, key in (
            ("SR", "success"), ("CR", "collision"),
            ("timeout", "timeout"), ("V_safe", "v_safe"),
        ):
            pooled[f"{metric}_cluster_bootstrap95"] = (
                E._cluster_bootstrap_interval(rows, key, seed=round_i + 200)
            )
        pooled["successful_clearance"]["cluster_bootstrap95"] = (
            E._cluster_bootstrap_interval(
                rows, "successful_clearance", seed=round_i + 300
            )
        )
        pooled["successful_time_to_goal"]["cluster_bootstrap95"] = (
            E._cluster_bootstrap_interval(
                rows, "time_to_goal", seed=round_i + 301
            )
        )
        records.append({
            "label": f"r{round_i}",
            "round": round_i,
            "cell": {"summary": {"pooled": pooled, "per_gamma": per_gamma}},
        })
    outputs = E.render(records, str(tmp_path))
    assert {Path(path).suffix for path in outputs} == {".png", ".pdf"}
    assert all(Path(path).stat().st_size > 0 for path in outputs)
