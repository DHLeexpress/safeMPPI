import sfm_b1_branch_compare_viz as V
import sfm_b1_d_branch_viz as D
import sfm_b1_selector_compare_viz as S
import numpy as np


def test_summary_separates_window_labels_from_episode_outcomes():
    bundle = {
        "traces": [
            {"executed_label": "verifier_positive"},
            {"executed_label": "verifier_negative"},
            {"executed_label": "verifier_positive"},
        ],
        "outcomes": [
            {"success": True, "collision": False, "timeout": False},
            {"success": False, "collision": True, "timeout": False},
        ],
    }
    report = V.summarize(bundle)
    assert report["contexts"] == 3
    assert report["executed_positive"] == 2
    assert report["executed_negative"] == 1
    assert report["executed_positive_fraction"] == 2 / 3
    assert (report["success"], report["collision"], report["timeout"]) == (
        1, 1, 0,
    )


def test_selector_pair_requires_same_checkpoint_and_bank():
    common = {
        "scenarios": [1, 2, 3],
        "gammas": [.1, .2, .3, .4, .5, .7, 1.],
        "environment": {"name": "test"},
        "sample_seed": 8,
        "audit_seed": 9,
        "checkpoint_sha256": "a" * 64,
    }
    S._validate_bundles((("margin", common), ("cost", dict(common))))
    different = dict(common, checkpoint_sha256="b" * 64)
    try:
        S._validate_bundles((("margin", common), ("cost", different)))
    except ValueError as error:
        assert "one pretrained checkpoint" in str(error)
    else:
        raise AssertionError("checkpoint mismatch must be rejected")


def test_robot_frame_uses_velocity_direction():
    trace = {"state": np.array([2., 3., 0., 2.])}
    path = np.array([[2., 3.], [2., 4.], [3., 4.]])
    local = D._robot_frame(path, trace)
    np.testing.assert_allclose(local, [[0., 0.], [1., 0.], [1., -1.]])
