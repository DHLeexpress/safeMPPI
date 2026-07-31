from __future__ import annotations

import json
import math
import pytest

import run_sfm_neutral_temperature_m50 as S
import sfm_protocol as SP


def _record(method, *, cr, validity, clearance, time, temperature=1.0):
    pooled = {
        "SR": 1.0 - cr,
        "CR": cr,
        "timeout": 0.0,
        "Validity": validity,
        "clearance": clearance,
        "time_to_goal": time,
    }
    return {
        "method": method,
        "round": 1,
        "temperature": temperature,
        "pooled": pooled,
        "per_gamma": {str(gamma): dict(pooled) for gamma in SP.GAMMAS},
    }


def test_reference_envelope_uses_hardest_metricwise_baseline():
    pretrained = [
        _record("pretrained", cr=.3, validity=.6, clearance=.1, time=9),
        _record("pretrained", cr=.2, validity=.5, clearance=.12, time=10),
    ]
    kazuki = _record(
        "kazuki_locked", cr=.1, validity=.4, clearance=.2, time=4,
        temperature=None,
    )
    assert S._envelope(pretrained, kazuki) == {
        "CR": .1,
        "Validity": .6,
        "clearance": .2,
        "time_to_goal": 4,
    }


def test_four_metric_gate_has_zero_shortfall_only_for_strict_envelope_win():
    target = {
        "CR": .2,
        "Validity": .6,
        "clearance": .1,
        "time_to_goal": 9,
    }
    winner = _record(
        "expanded", cr=.1, validity=.7, clearance=.2, time=8
    )
    loser = _record(
        "expanded", cr=.1, validity=.7, clearance=.08, time=8
    )
    assert S._shortfalls(winner, target) == {
        metric: 0.0 for metric in S.METRICS
    }
    assert S._shortfalls(loser, target)["clearance"] == pytest.approx(.2)


def test_no_success_metrics_can_never_win_selection():
    target = {
        "CR": .2,
        "Validity": .6,
        "clearance": .1,
        "time_to_goal": 9,
    }
    collapsed = _record(
        "expanded", cr=0.0, validity=.9,
        clearance=float("nan"), time=float("nan"),
    )
    shortfall = S._shortfalls(collapsed, target)
    assert math.isinf(shortfall["clearance"])
    assert math.isinf(shortfall["time_to_goal"])
    liveness = {
        "minimum_SR": .5,
        "maximum_timeout": .1,
        "every_gamma_has_success": True,
    }
    assert not S._liveness_eligible(collapsed, liveness)


def test_gamma_trend_is_diagnostic_not_per_gamma_temperature_tuning():
    record = _record(
        "expanded", cr=.2, validity=.5, clearance=.1, time=9,
        temperature=.7,
    )
    for index, gamma in enumerate(SP.GAMMAS):
        cell = record["per_gamma"][str(gamma)]
        cell["CR"] = .1 + .02 * index
        cell["Validity"] = .3 + .05 * index
        cell["clearance"] = .2 - .01 * index
        cell["time_to_goal"] = 12 - .5 * index
    trend = S._trend(record)
    assert trend["mean_fraction"] == 1.0
    assert record["temperature"] == .7


def test_authenticated_round_records_follow_nested_resume_chain(tmp_path):
    lineage = []
    prior_path = None
    prior_delivery = None
    for round_i in (1, 2, 3):
        checkpoint = tmp_path / f"round_{round_i:02d}.pt"
        post_positive = tmp_path / f"round_{round_i:02d}_post_positive.pt"
        checkpoint.write_bytes(f"checkpoint-{round_i}".encode())
        post_positive.write_bytes(f"positive-{round_i}".encode())
        marker = tmp_path / f"round_{round_i}.json"
        marker.write_text(json.dumps({
            "status": "SFM_B1_NEUTRAL_MULTIROUND_ROUND_COMPLETE",
            "round": round_i,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": S.FUNNEL.sha256_file(checkpoint),
        }))
        ref = S._round_ref_from_marker(marker)
        delivery = {
            "status": "SFM_B1_NEUTRAL_MULTIROUND_COMPLETE",
            "round_records": [str(marker)],
            "round_record_refs": [ref],
        }
        if prior_delivery is not None:
            delivery["resume"] = {
                "delivery": str(prior_path),
                "delivery_sha256": S.FUNNEL.sha256_file(prior_path),
                "round_record_refs": list(lineage),
            }
        path = tmp_path / f"delivery_{round_i}.json"
        path.write_text(json.dumps(delivery))
        lineage.append(ref)
        prior_path, prior_delivery = path, delivery
    records = S._authenticated_round_records(prior_delivery)
    assert [row["round"] for row in records] == [1, 2, 3]

    prior_delivery["round_record_refs"][0]["path"] = str(tmp_path / "wrong")
    with pytest.raises(RuntimeError, match="current round-record path"):
        S._authenticated_round_records(prior_delivery)


def test_legacy_delivery_refs_gain_round_and_checkpoint_identity(tmp_path):
    checkpoint = tmp_path / "round_01.pt"
    post_positive = tmp_path / "round_01_post_positive.pt"
    checkpoint.write_bytes(b"checkpoint")
    post_positive.write_bytes(b"positive")
    marker = tmp_path / "round.json"
    marker.write_text(json.dumps({
        "status": "SFM_B1_NEUTRAL_MULTIROUND_ROUND_COMPLETE",
        "round": 1, "checkpoint": str(checkpoint),
        "checkpoint_sha256": S.FUNNEL.sha256_file(checkpoint),
    }))
    delivery = {"round_records": [str(marker)]}
    refs = S._delivery_round_refs(delivery)
    assert refs[0]["round"] == 1
    assert refs[0]["post_D0_sha256"] == S.FUNNEL.sha256_file(checkpoint)
    assert refs[0]["post_Dplus_sha256"] == S.FUNNEL.sha256_file(post_positive)

    checkpoint.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="checkpoint digest"):
        S._authenticated_round_records(delivery)


def test_screen_checkpoint_must_belong_to_authenticated_round(tmp_path):
    pretrained = tmp_path / "pretrained.pt"
    checkpoint = tmp_path / "round_01.pt"
    post_positive = tmp_path / "round_01_post_positive.pt"
    pretrained.write_bytes(b"pretrained")
    checkpoint.write_bytes(b"round-one")
    post_positive.write_bytes(b"positive")
    marker = tmp_path / "round.json"
    marker.write_text(json.dumps({
        "status": "SFM_B1_NEUTRAL_MULTIROUND_ROUND_COMPLETE",
        "round": 1, "checkpoint": str(checkpoint),
        "checkpoint_sha256": S.FUNNEL.sha256_file(checkpoint),
    }))
    delivery = {
        "checkpoint": str(pretrained),
        "checkpoint_sha256": S.FUNNEL.sha256_file(pretrained),
        "round_records": [str(marker)],
        "round_record_refs": [S._round_ref_from_marker(marker)],
        "disjoint_raw_evaluation": {"records": [
            {
                "round": 0, "checkpoint": str(pretrained),
                "checkpoint_sha256": S.FUNNEL.sha256_file(pretrained),
                "pooled": {
                    "SR": .5, "CR": .5, "timeout": 0,
                    "Validity": {"mean": .5},
                    "successful_clearance": {"mean": .1},
                    "successful_time_to_goal": {"mean": 8.0},
                },
            }, {
                "round": 1, "checkpoint": str(checkpoint),
                "checkpoint_sha256": S.FUNNEL.sha256_file(checkpoint),
                "pooled": {
                "SR": .5, "CR": .5, "timeout": 0,
                "Validity": {"mean": .5},
                "successful_clearance": {"mean": .1},
                "successful_time_to_goal": {"mean": 8.0},
            },
        }]},
    }
    rows, _ = S._screen_rows(tmp_path, ["arm"], [delivery])
    assert rows[0]["checkpoint_sha256"] == S.FUNNEL.sha256_file(checkpoint)
    legacy = delivery["disjoint_raw_evaluation"]["records"][1]
    legacy.pop("checkpoint")
    legacy.pop("checkpoint_sha256")
    rows, _ = S._screen_rows(tmp_path, ["arm"], [delivery])
    assert rows[0]["checkpoint"] == str(checkpoint.resolve())
    delivery["disjoint_raw_evaluation"]["records"][1][
        "checkpoint_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="screening checkpoint"):
        S._screen_rows(tmp_path, ["arm"], [delivery])
