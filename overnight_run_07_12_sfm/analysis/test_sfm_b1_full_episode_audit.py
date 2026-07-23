import matplotlib.pyplot as plt
import numpy as np
import pytest

import sfm_b1_full_episode_audit as A
import sfm_b1_full_episode_viz as V
import sfm_b1_viz as BV


def test_result_label_requires_a_resolved_full_h_positive():
    assert A._result_label(dict(resolved=False)) == "verifier_error"
    assert A._result_label(dict(resolved=True, y=0)) == "verifier_negative"
    assert A._result_label(
        dict(resolved=True, y=1, full_h=True, terminal_step=10)
    ) == "verifier_positive"
    with pytest.raises(RuntimeError, match="full-H=10"):
        A._result_label(
            dict(resolved=True, y=1, full_h=False, terminal_step=3)
        )


def test_trap_is_a_separate_exact_ten_transition_event():
    short = [np.array([0.0, 0.0, 0.0, 0.0])] * 10
    assert not A._trap(short)
    stationary = [np.array([0.0, 0.0, 0.0, 0.0])] * 11
    assert A._trap(stationary)
    moving = [
        np.array([0.03 * step, 0.0, 0.0, 0.0]) for step in range(11)
    ]
    assert not A._trap(moving)


def test_nvp_does_not_remove_later_steps_from_the_trace_index():
    traces = [
        dict(scenario_id=7, gamma=.5, step=0, nvp_context=True),
        dict(scenario_id=7, gamma=.5, step=1, nvp_context=False),
    ]
    index = V._index(traces)
    assert sorted(index[(7, .5)]) == [0, 1]
    with pytest.raises(ValueError, match="duplicate trace key"):
        V._index(traces + [dict(traces[1])])


def test_executed_color_is_the_verifier_label_not_the_nvp_event():
    assert V._executed_color(
        dict(executed_label="verifier_positive", nvp_context=True)
    ) == BV.BLUE
    assert V._executed_color(
        dict(executed_label="verifier_negative", nvp_context=False)
    ) == BV.RED


def test_verifier_geometry_is_drawn_only_for_positive_executed_window(monkeypatch):
    calls = []
    monkeypatch.setattr(
        V.DV, "checked_verifier_levels",
        lambda trace, query, H: calls.append(H) or dict(
            polygons=[], outer_polygon=None
        ),
    )
    monkeypatch.setattr(V.DV, "_draw_verifier_geometry", lambda axis, audit: None)
    base = dict(
        state=np.zeros(4), next_state=np.zeros(4),
        executed_controls=np.zeros((10, 2)),
        executed_result=dict(segment=np.zeros((11, 2))),
    )
    figure, axis = plt.subplots()
    V._draw_executed(
        axis, dict(base, executed_label="verifier_negative")
    )
    assert calls == []
    V._draw_executed(
        axis, dict(base, executed_label="verifier_positive")
    )
    assert calls == [10]
    plt.close(figure)
