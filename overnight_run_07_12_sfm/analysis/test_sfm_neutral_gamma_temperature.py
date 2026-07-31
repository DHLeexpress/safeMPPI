from __future__ import annotations

import json

import pytest

import run_sfm_neutral_gamma_temperature as G


def _row(episode, gamma, *, success=True, collision=False, validity=.5,
         clearance=.1, time=8.0):
    return {
        "episode": int(episode),
        "gamma": float(gamma),
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(not success and not collision),
        "validity": float(validity),
        "successful_clearance": float(clearance) if success else None,
        "time_to_goal": float(time) if success else None,
    }


def test_point_metrics_exclude_failures_only_from_conditional_metrics():
    rows = [
        _row(1, .1, clearance=.2, time=9),
        _row(2, .1, success=False, collision=True),
    ]
    point = G._point(rows)
    assert point["SR"] == .5
    assert point["CR"] == .5
    assert point["clearance"] == .2
    assert point["time_to_goal"] == 9


def test_paired_ci_requires_complete_scenario_clusters():
    rows = [_row(episode, gamma) for episode in range(50) for gamma in G.SP.GAMMAS]
    result = G._paired_cluster_ci(rows, rows, seed=1, draws=20)
    assert all(value["paired_cluster_95"] == [0.0, 0.0] for value in result.values())
    assert not G._ci_win(result)


def test_global_temperature_reference_reuse_is_fail_closed(tmp_path):
    initial = tmp_path / "DELIVERY_COMPLETE.json"
    initial.write_text("{}")
    reference_root = tmp_path / "disjoint_m50"
    selection = {}
    frozen_records = []
    for method in ("pretrained", "expanded"):
        scenario_ids = list(range(470000, 470050))
        rows = [
            _row(episode, gamma)
            for gamma in G.SP.GAMMAS for episode in scenario_ids
        ]
        payload = {
            "scene_profile": "double_density_velocity_ood",
            "bank": {
                "ep0": 470000, "M_per_gamma": 50,
                "scenario_ids": scenario_ids,
            },
            "noise_bank": {
                "seed": 7, "gammas": list(map(float, G.SP.GAMMAS)),
                "NFE": 8, "dtype": "float32",
                "shape": [7, 50, 180, 20], "sha256": "a" * 64,
                "temperature_by_gamma": [.55] * 7,
            },
            "temperature": .55,
            "records": [{
                "round": 0,
                "cell": {
                    "rows": rows, "checkpoint_sha256": method,
                    "checkpoint": f"/{method}.pt",
                    "scene_profile": "double_density_velocity_ood",
                    "cell_key": f"cell-{method}",
                    "summary": {
                        "pooled": {
                            "SR": 1.0, "CR": 0.0, "timeout": 0.0,
                            "Validity": {"mean": .5},
                            "successful_clearance": {"mean": .1},
                            "successful_time_to_goal": {"mean": 8.0},
                        },
                        "per_gamma": {
                            str(gamma): {
                                "SR": 1.0, "CR": 0.0, "timeout": 0.0,
                                "Validity": {"mean": .5},
                                "successful_clearance": {"mean": .1},
                                "successful_time_to_goal": {"mean": 8.0},
                            }
                            for gamma in G.SP.GAMMAS
                        },
                    },
                },
            }],
        }
        destination = reference_root / method / "raw_m50_offline_metrics.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(payload))
        selection[method] = {
            "temperature": .55,
            "round": 0,
            "checkpoint": f"/{method}.pt",
            "checkpoint_sha256": method,
        }
        payload["records"][0]["cell"]["checkpoint"] = f"/{method}.pt"
        destination.write_text(json.dumps(payload))
        frozen_records.append(G.BASE._record_from_cell(
            payload, method=method, temperature=.55,
        ))
    state = {
        "banks": {"disjoint_confirmation": {
            "ep0": 470000, "M_per_gamma": 50, "noise_seed": 7,
        }},
        "final_records": frozen_records,
    }
    cells, reuse = G._reuse_global_temperature_cells(
        initial, state, selection
    )
    assert reuse["status"] == (
        "GLOBAL_TEMPERATURE_CALIBRATION_CELLS_REUSED"
    )
    assert cells["pretrained"][.55]["temperature"] == .55

    expanded_path = (
        reference_root / "expanded" / "raw_m50_offline_metrics.json"
    )
    expanded = json.loads(expanded_path.read_text())
    expanded["noise_bank"]["sha256"] = "b" * 64
    expanded_path.write_text(json.dumps(expanded))
    with pytest.raises(RuntimeError, match="do not share CRN bytes"):
        G._reuse_global_temperature_cells(initial, state, selection)
    expanded["noise_bank"]["sha256"] = "a" * 64
    expanded_path.write_text(json.dumps(expanded))

    selection["expanded"]["checkpoint_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="reference contract failed for expanded"):
        G._reuse_global_temperature_cells(initial, state, selection)


def test_prior_calibration_cells_are_content_authenticated(tmp_path):
    bank = {"ep0": 470000, "M_per_gamma": 50, "noise_seed": 17}
    methods = {
        "pretrained": {"round": 0, "checkpoint_sha256": "pre"},
        "expanded": {"round": 2, "checkpoint_sha256": "exp"},
    }
    root = tmp_path / "prior"
    for method, selected in methods.items():
        scenario_ids = list(range(470000, 470050))
        rows = [
            _row(episode, gamma)
            for gamma in G.SP.GAMMAS for episode in scenario_ids
        ]
        payload = {
            "scene_profile": "double_density_velocity_ood",
            "bank": {
                "ep0": 470000, "M_per_gamma": 50,
                "scenario_ids": scenario_ids,
            },
            "noise_bank": {
                "seed": 17, "gammas": list(map(float, G.SP.GAMMAS)),
                "NFE": 8, "dtype": "float32",
                "shape": [7, 50, 180, 20], "sha256": "c" * 64,
            },
            "temperature": .7,
            "temperature_by_gamma": [.7] * 7,
            "records": [{
                "round": selected["round"],
                "cell": {
                    "rows": rows,
                    "checkpoint_sha256": selected["checkpoint_sha256"],
                    "scene_profile": "double_density_velocity_ood",
                    "cell_key": f"{method}-cell",
                },
            }],
        }
        path = (
            root / "calibration_m50" / f"{method}_temp0p7"
            / "raw_m50_offline_metrics.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))
    cells = {"pretrained": {}, "expanded": {}}
    reuse = G._reuse_prior_calibration_cells(
        root, methods, bank, [.7], cells,
    )
    assert len(reuse["records"]) == 2
    assert not reuse["missing_cells_will_be_run"]
    assert set(cells) == {"pretrained", "expanded"}
    assert cells["expanded"][.7]["records"][0]["round"] == 2

    path = (
        root / "calibration_m50" / "expanded_temp0p7"
        / "raw_m50_offline_metrics.json"
    )
    broken = json.loads(path.read_text())
    broken["records"][0]["round"] = 3
    path.write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="prior calibration contract"):
        G._reuse_prior_calibration_cells(
            root, methods, bank, [.7],
            {"pretrained": {}, "expanded": {}},
        )


def test_locked_checkpoint_bytes_and_fresh_cell_are_fail_closed(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen")
    locked = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": G.BASE.FUNNEL.sha256_file(checkpoint),
        "round": 2,
        "temperature_by_gamma": [.7] * 7,
    }
    G._authenticate_method_checkpoints({"expanded": locked})
    rows = [
        _row(episode, gamma)
        for gamma in G.SP.GAMMAS for episode in range(480000, 480050)
    ]
    payload = {
        "scene_profile": "double_density_velocity_ood",
        "bank": {
            "ep0": 480000, "M_per_gamma": 50,
            "scenario_ids": list(range(480000, 480050)),
        },
        "noise_bank": {
            "seed": 9, "gammas": list(map(float, G.SP.GAMMAS)),
            "NFE": 8, "dtype": "float32",
            "shape": [7, 50, 180, 20], "sha256": "d" * 64,
        },
        "temperature_by_gamma": [.7] * 7,
        "records": [{
            "round": 2,
            "cell": {
                "rows": rows,
                "checkpoint_sha256": locked["checkpoint_sha256"],
            },
        }],
    }
    assert G._validate_fresh_raw_cell(
        payload, locked=locked, ep0=480000, noise_seed=9,
    ) == "d" * 64

    checkpoint.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="locked expanded checkpoint"):
        G._authenticate_method_checkpoints({"expanded": locked})
    payload["records"][0]["cell"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fresh raw confirmation"):
        G._validate_fresh_raw_cell(
            payload, locked=locked, ep0=480000, noise_seed=9,
        )


def test_ci_win_requires_all_four_intervals_strictly_favorable():
    result = {
        "CR": {"paired_cluster_95": [-.2, -.01]},
        "Validity": {"paired_cluster_95": [.01, .2]},
        "clearance": {"paired_cluster_95": [.001, .02]},
        "time_to_goal": {"paired_cluster_95": [-2.0, -.1]},
    }
    assert G._ci_win(result)
    result["CR"] = {"paired_cluster_95": [-.2, .01]}
    assert not G._ci_win(result)


def test_schedule_selection_requires_gamma_trend_before_shortfall():
    def record(name, trend_ok, cr):
        pooled = {
            "SR": .8, "CR": cr, "timeout": 0.0,
            "Validity": .7, "clearance": .2, "time_to_goal": 8.0,
        }
        rows = {}
        for index, gamma in enumerate(G.SP.GAMMAS):
            cell = dict(pooled)
            cell["clearance"] = (.3 - .01 * index) if trend_ok else (.1 + .05 * index)
            cell["time_to_goal"] = (11 - .2 * index) if trend_ok else (7 + 2.0 * index)
            cell["Validity"] = .5 + .02 * index
            cell["CR"] = .1 + .01 * index
            rows[str(gamma)] = cell
        return {
            "method": name, "round": 1, "pooled": pooled,
            "per_gamma": rows, "temperature_by_gamma": [1.0] * 7,
        }

    target = {"CR": .1, "Validity": .7, "clearance": .2, "time_to_goal": 8.0}
    liveness = {"minimum_SR": .5, "maximum_timeout": .1, "every_gamma_has_success": True}
    selected = G._pick(
        [record("bad_trend", False, .05), record("good_trend", True, .15)],
        target=target,
        liveness=liveness,
    )
    assert selected["method"] == "good_trend"


def test_final_objective_requires_ci_liveness_and_gamma_trend():
    def record(method, *, sr=.8, timeout=0.0, trend_ok=True):
        pooled = {
            "SR": sr, "CR": .1, "timeout": timeout,
            "Validity": .7, "clearance": .2, "time_to_goal": 8.0,
        }
        per_gamma = {}
        for index, gamma in enumerate(G.SP.GAMMAS):
            cell = dict(pooled)
            cell["clearance"] = (.3 - .01 * index) if trend_ok else (.1 + .05 * index)
            cell["time_to_goal"] = (11 - .2 * index) if trend_ok else (7 + 2 * index)
            cell["Validity"] = .5 + .02 * index
            cell["CR"] = .1 + .01 * index
            per_gamma[str(gamma)] = cell
        return {"method": method, "pooled": pooled, "per_gamma": per_gamma}

    comparisons = {
        name: {
            "CR": {"paired_cluster_95": [-.2, -.01]},
            "Validity": {"paired_cluster_95": [.01, .2]},
            "clearance": {"paired_cluster_95": [.001, .02]},
            "time_to_goal": {"paired_cluster_95": [-2.0, -.1]},
        }
        for name in ("expanded_minus_pretrained", "expanded_minus_kazuki")
    }
    good = [record("pretrained"), record("expanded"), record("kazuki_locked")]
    assert G._final_objective_gates(good, comparisons)["objective_achieved"]

    bad_trend = [record("pretrained"), record("expanded", trend_ok=False), record("kazuki_locked")]
    gates = G._final_objective_gates(bad_trend, comparisons)
    assert gates["paired_ci_clean_four_metric_win"]
    assert not gates["final_gamma_trend_eligible"]
    assert not gates["objective_achieved"]

    bad_liveness = [record("pretrained"), record("expanded", sr=.1, timeout=.8), record("kazuki_locked")]
    gates = G._final_objective_gates(bad_liveness, comparisons)
    assert not gates["final_liveness_eligible"]
    assert not gates["objective_achieved"]


def test_one_fully_reversed_trend_family_cannot_hide_in_mean():
    pooled = {
        "SR": .8, "CR": .1, "timeout": 0.0,
        "Validity": .7, "clearance": .2, "time_to_goal": 8.0,
    }
    per_gamma = {}
    for index, gamma in enumerate(G.SP.GAMMAS):
        cell = dict(pooled)
        cell["CR"] = .1 + .01 * index
        cell["Validity"] = .5 + .02 * index
        cell["clearance"] = .3 - .01 * index
        cell["time_to_goal"] = 7.0 + 2.0 * index
        per_gamma[str(gamma)] = cell
    record = {"pooled": pooled, "per_gamma": per_gamma}
    assert G.BASE._trend(record)["mean_fraction"] == .75
    assert not G.BASE._trend_eligible(record)
