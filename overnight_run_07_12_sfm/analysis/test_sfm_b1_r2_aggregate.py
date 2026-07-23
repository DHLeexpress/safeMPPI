import pytest

import sfm_b1_r2_aggregate as A


def test_arm_grid_names_are_unique():
    names = {
        A.arm_name(alpha, epochs)
        for alpha in A.ALPHAS
        for epochs in A.REPLAY_EPOCHS
    }
    assert len(names) == 9
    assert "margin_alpha0p1_epochs100" in names


def test_selection_is_safety_first():
    safe = {
        "CR": 0.2, "SR": 0.7, "clearance": 0.1, "time": 10.0,
        "round": 2, "alpha": 0.1, "replay_epochs": 100,
    }
    fast = {
        "CR": 0.3, "SR": 0.9, "clearance": 0.2, "time": 5.0,
        "round": 1, "alpha": 0.0, "replay_epochs": 1,
    }
    assert min((safe, fast), key=A._post_expansion_key) is safe


def test_paired_cluster_delta_respects_episode_pairing():
    baseline, candidate = [], []
    for episode in (1, 2):
        for gamma in (0.1, 1.0):
            baseline.append({
                "episode": episode, "gamma": gamma, "collision": episode == 1,
            })
            candidate.append({
                "episode": episode, "gamma": gamma, "collision": False,
            })
    value = A.paired_cluster_delta(
        baseline, candidate, "collision", seed=1, draws=1000,
    )
    assert value["estimate"] == pytest.approx(-0.5)
    assert value["paired_scenarios"] == 2
    assert value["paired_gamma_cells"] == 4
