import numpy as np

import sfm_metrics2 as M


def test_label_is_only_task_collision_and_moving_certificate():
    # A stationary, non-progressing plan is still y=1 when the three label clauses pass.
    result = M.verify_query(
        np.zeros(4), np.zeros((10, 2)), np.zeros((0, 2)), np.zeros((0, 2)), .5, n_theta=36
    )
    assert result["resolved"] and result["y"] == 1
    assert result["taskspace"] and result["collision_free"] and result["certificate"]
    assert "progress" not in result and "cost" not in result


def test_early_goal_is_prefix_only_and_not_replay_eligible():
    state = np.array([5.7, 6.0, 2.0, 0.0], np.float32)
    result = M.verify_query(
        state, np.zeros((10, 2)), np.zeros((0, 2)), np.zeros((0, 2)), .5, n_theta=36
    )
    assert result["resolved"] and result["terminal_step"] < 10
    assert not result["full_h"] and not result["train_eligible"]
