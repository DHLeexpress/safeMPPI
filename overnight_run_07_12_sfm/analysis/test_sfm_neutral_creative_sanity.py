import json
from pathlib import Path

import numpy as np
import pytest
import torch

import run_sfm_neutral_autonomous_followup as AUTO
import run_sfm_neutral_creative_sanity as C
import run_sfm_neutral_gamma_temperature as GAMMA
import sfm_b1_cost as COST
import sfm_b1_neutral_multiround as TRAIN
import sfm_protocol as SP


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_trigger_requires_authenticated_failed_autonomous_delivery(tmp_path):
    expected_source = "a" * 40
    training_path = _write(tmp_path / "training.json", {"status": "training"})
    global_path = _write(tmp_path / "global.json", {"status": "global"})
    gamma_path = _write(tmp_path / "gamma.json", {
        "status": GAMMA.STATUS,
        "ci_clean_four_metric_win": True,
        "objective_achieved": False,
    })
    delivery = {
        "status": AUTO.STATUS,
        "action": "CREATIVE_SANITY_REQUIRED",
        "ci_clean_four_metric_win": True,
        "objective_achieved": False,
        "source_commit": expected_source,
        "selected_arm": "lr1em5_s04",
        "r100_training_delivery": str(training_path),
        "r100_training_delivery_sha256": C._sha256(training_path),
        "r100_global_delivery": str(global_path),
        "r100_global_delivery_sha256": C._sha256(global_path),
        "r100_gamma_delivery": str(gamma_path),
        "r100_gamma_delivery_sha256": C._sha256(gamma_path),
    }
    delivery_path = _write(tmp_path / "delivery.json", delivery)
    trigger = {
        "status": AUTO.CREATIVE_TRIGGER_STATUS,
        "action": "CREATIVE_SANITY_REQUIRED",
        "source_commit": expected_source,
        "autonomous_delivery": str(delivery_path),
        "autonomous_delivery_sha256": C._sha256(delivery_path),
        "selected_arm": delivery["selected_arm"],
        "r100_training_delivery": delivery["r100_training_delivery"],
        "r100_training_delivery_sha256": delivery[
            "r100_training_delivery_sha256"
        ],
        "r100_global_delivery": delivery["r100_global_delivery"],
        "r100_global_delivery_sha256": delivery[
            "r100_global_delivery_sha256"
        ],
        "r100_gamma_delivery": delivery["r100_gamma_delivery"],
        "r100_gamma_delivery_sha256": delivery[
            "r100_gamma_delivery_sha256"
        ],
    }
    result = C._validate_trigger(
        tmp_path / "trigger.json", trigger,
        expected_source=expected_source,
    )
    assert result["delivery"]["selected_arm"] == "lr1em5_s04"

    trigger["autonomous_delivery_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest mismatch"):
        C._validate_trigger(
            tmp_path / "trigger.json", trigger,
            expected_source=expected_source,
        )


def test_new_evaluation_banks_are_disjoint_from_training_and_prior_banks():
    new = [C._bank_from(530000, 10, "B"), C._bank_from(540000, 10, "A")]
    C._validate_new_banks(
        new, [C._bank_from(520000, 50, "prior")], {260000, 260001},
    )
    with pytest.raises(RuntimeError, match="overlaps prior"):
        C._validate_new_banks(
            [C._bank_from(520049, 10, "B")],
            [C._bank_from(520000, 50, "prior")], set(),
        )
    with pytest.raises(RuntimeError, match="training scenarios"):
        C._validate_new_banks(
            [C._bank_from(260000, 10, "B")], [], {260003},
        )


def _phase(round_i, phase, *, SR, CR, timeout, validity):
    return {
        "round": round_i,
        "phase": phase,
        "pooled": {
            "SR": SR, "CR": CR, "timeout": timeout,
            "Validity": validity,
        },
    }


def test_stage_b_retains_d0_unless_disjoint_overwrite_is_clear():
    beneficial = []
    harmful = []
    for round_i in (10, 20, 30):
        beneficial.extend([
            _phase(round_i, "post_Dplus", SR=.70, CR=.25, timeout=.05, validity=.70),
            _phase(round_i, "post_D0", SR=.71, CR=.23, timeout=.06, validity=.72),
        ])
        harmful.extend([
            _phase(round_i, "post_Dplus", SR=.75, CR=.22, timeout=.03, validity=.72),
            _phase(round_i, "post_D0", SR=.65, CR=.21, timeout=.13, validity=.73),
        ])
    assert not C._d0_gate(beneficial)["clear_D0_overwrite"]
    assert C._d0_gate(harmful)["clear_D0_overwrite"]


def _metric_row(round_i, *, SR, CR, timeout, validity, clearance=.1, time=9.0):
    cell = {
        "SR": SR, "CR": CR, "timeout": timeout, "Validity": validity,
        "clearance": clearance, "time_to_goal": time,
    }
    return {
        "round": round_i,
        "checkpoint": f"/checkpoint/r{round_i}.pt",
        "checkpoint_sha256": f"sha{round_i}",
        "pooled": dict(cell),
        "per_gamma": {str(gamma): dict(cell) for gamma in SP.GAMMAS},
    }


def test_stage_a_gate_requires_liveness_gain_safety_and_gamma_trend():
    candidates = [_metric_row(0, SR=.6, CR=.35, timeout=.05, validity=.6)]
    controls = []
    for round_i in range(1, 6):
        controls.append(_metric_row(
            round_i, SR=.60, CR=.35, timeout=.05, validity=.60,
        ))
        candidates.append(_metric_row(
            round_i, SR=.66, CR=.36, timeout=.04, validity=.59,
        ))
    gate = C._selector_gate(candidates, controls, controls[-1])
    assert gate["passed"]

    candidates[-1]["pooled"]["CR"] = .50
    for gamma in candidates[-1]["per_gamma"]:
        candidates[-1]["per_gamma"][gamma]["CR"] = .50
    assert any(
        not row["eligible"]
        for row in C._selector_gate(candidates, controls, controls[-1])[
            "decisions"
        ]
    )


def test_encoder_stage_stops_on_drift_regression_or_cr(tmp_path):
    marker = {
        "round": 1,
        "encoder_diagnostics": {
            "token_cosine": .97,
            "relative_parameter_drift": .001,
            "cumulative_from_reference": {
                "token_cosine": .97,
                "token_rms_change": .01,
                "relative_parameter_drift": .001,
            },
        },
        "updates": {
            "Dplus": {"encoder_gradient_norms": [.1]},
            "D0": {"encoder_gradient_norms": [.2]},
        },
        "paired_trigger_probe": {
            "Dplus_increment": {"Dplus_regressed": 0},
        },
        "gather": {
            "gp_diagnostics": {"kernel_effective_rank": 12.0},
            "acquisition": {"uplift": .01},
        },
    }
    marker_path = _write(tmp_path / "round.json", marker)
    delivery = {"round_records": [str(marker_path)]}
    candidate = [_metric_row(1, SR=.7, CR=.2, timeout=.1, validity=.7)]
    control = [_metric_row(1, SR=.7, CR=.2, timeout=.1, validity=.68)]
    control_markers = {1: {
        "gather": {
            "gp_diagnostics": {"kernel_effective_rank": 11.5},
            "acquisition": {"uplift": .009},
        },
    }}
    gate = C._encoder_stage_gate(
        delivery, candidate, control, control_markers,
    )
    assert gate["unsafe"]
    assert not gate["continue_to_round5"]


def test_legacy_control_gp_uplift_is_read_from_authenticated_trace(tmp_path):
    trace_path = tmp_path / "trace.pt"
    torch.save({
        "status": TRAIN.RA.STATUS,
        "protocol": {"acquisition": {"uplift": 0.0125}},
    }, trace_path)
    marker = {
        "gather": {
            "trace_path": str(trace_path),
            "trace_sha256": C._sha256(trace_path),
        },
    }
    fields = C._control_gp_fields(marker)
    assert fields == {
        "uplift": 0.0125,
        "effective_rank": None,
        "source": "authenticated_legacy_trace",
    }


def test_progress_gated_margin_prefers_moving_goal_progress_then_falls_back():
    state = np.zeros(4, np.float32)
    stalled = {
        "candidate_id": 0, "hp_margin": 10.0,
        "controls": np.zeros((10, 2), np.float32),
    }
    moving = {
        "candidate_id": 1, "hp_margin": 1.0,
        "controls": np.full((10, 2), 2.0, np.float32),
    }
    assert COST.select_progress_gated_margin(
        [stalled, moving], state=state,
    )["candidate_id"] == 1
    assert COST.select_progress_gated_margin(
        [stalled], state=state,
    )["candidate_id"] == 0


def test_creative_trainability_modes_are_individual_not_combined():
    assert TRAIN.StudyConfig(
        name="A", selector="progress_gated_margin",
        encoder_lr_ratio=0.0,
    ).validate()
    assert TRAIN.StudyConfig(
        name="C", selector="margin", encoder_lr_ratio=0.1,
    ).validate()
    with pytest.raises(ValueError, match="encoder_lr_ratio"):
        TRAIN.StudyConfig(name="bad", encoder_lr_ratio=0.2).validate()
    assert TRAIN.StudyConfig(name="B", neutral_replay=False).validate()
