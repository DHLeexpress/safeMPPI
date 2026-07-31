from __future__ import annotations

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
