import sfm_b1_branch_compare_viz as V


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
