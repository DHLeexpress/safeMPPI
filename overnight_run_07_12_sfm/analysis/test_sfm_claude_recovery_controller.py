import numpy as np

import sfm_claude_recovery_controller as C


class _Executor:
    def map(self, function, payloads):
        return map(function, payloads)


def _resolved(y=1, *, full_h=True, terminal_step=10):
    return dict(
        resolved=True,
        y=int(y),
        full_h=bool(full_h),
        terminal_step=int(terminal_step),
        taskspace=True,
        collision_free=True,
        certificate=bool(y),
        diagnostics={"slack": 0.1},
    )


def test_exact_candidate_counts_shapes_bounds_and_determinism():
    state = np.array([0.2, 0.4, 0.5, -0.3], np.float32)
    v1_first = C.recovery_candidates_v1(state)
    v1_second = C.recovery_candidates_v1(state)
    v2_first = C.recovery_candidates_v2(state)
    v2_second = C.recovery_candidates_v2(state)
    assert len(v1_first) == 145
    assert len(v2_first) == 130
    for first, second in ((v1_first, v1_second), (v2_first, v2_second)):
        for (controls_a, provenance_a), (controls_b, provenance_b) in zip(
            first, second,
        ):
            assert controls_a.shape == (10, 2)
            assert float(np.abs(controls_a).max()) <= C.SS.U_MAX
            np.testing.assert_array_equal(controls_a, controls_b)
            assert provenance_a == provenance_b


def test_v2_objective_rewards_terminal_toward_goal_velocity(monkeypatch):
    state = np.zeros(4, np.float32)
    controls = np.zeros((10, 2), np.float32)
    monkeypatch.setattr(
        C.SM,
        "rollout_positions",
        lambda _state, _controls: np.array(
            [[0.0, 0.0], [5.0, 5.0]], np.float32,
        ),
    )
    toward = controls.copy()
    toward[:, :] = 0.5
    away = -toward
    assert C._v2_objective(state, toward) < C._v2_objective(state, away)


def test_pool_uses_v2_objective_and_strict_full_h_gate(monkeypatch):
    state = np.zeros(4, np.float32)
    candidates = []
    for index in range(4):
        controls = np.zeros((10, 2), np.float32)
        controls[0, 0] = index
        candidates.append((controls, {"index": index}))
    monkeypatch.setattr(C, "recovery_candidates_v2", lambda _state: candidates)
    monkeypatch.setattr(C, "_prefilter", lambda *args: 10.0)
    monkeypatch.setattr(
        C,
        "_v2_objective",
        lambda _state, controls: float(3 - controls[0, 0]),
    )

    def verify(payload):
        _, rank, *_ = payload
        results = {
            0: _resolved(1, full_h=False),
            1: _resolved(1, terminal_step=9),
            2: _resolved(0),
            3: _resolved(1),
        }
        return 0, rank, results[rank]

    monkeypatch.setattr(C.SM, "verify_in_worker", verify)
    pool = C.evaluate_recovery_pool(
        state,
        np.array([[5.0, 5.0]], np.float32),
        np.zeros((1, 2), np.float32),
        0.5,
        family="v2",
        executor=_Executor(),
    )
    assert [row["provenance"]["index"] for row in pool["queried"]] == [
        3, 2, 1, 0,
    ]
    assert [row["certified"] for row in pool["queried"]] == [
        False, False, False, True,
    ]
    assert pool["best"]["provenance"]["index"] == 0


def test_pool_submits_at_most_24_and_keeps_at_most_two(monkeypatch):
    candidates = []
    for index in range(30):
        controls = np.zeros((10, 2), np.float32)
        controls[0, 0] = index / 100.0
        candidates.append((controls, {"index": index}))
    monkeypatch.setattr(C, "recovery_candidates_v1", lambda _state: candidates)
    monkeypatch.setattr(
        C, "_prefilter", lambda _state, controls, *_: controls[0, 0],
    )
    monkeypatch.setattr(
        C.SM,
        "verify_in_worker",
        lambda payload: (payload[0], payload[1], _resolved(1)),
    )
    pool = C.evaluate_recovery_pool(
        np.zeros(4, np.float32),
        np.zeros((0, 2), np.float32),
        np.zeros((0, 2), np.float32),
        0.5,
        family="v1",
    )
    assert pool["queried_count"] == 24
    assert len(pool["certified"]) == 2
    assert pool["best"]["query_rank"] == 0


def test_rollout_is_fail_closed_on_no_verified_plan(monkeypatch):
    monkeypatch.setattr(C.SS, "make_humans", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        C.SS,
        "collect_humans",
        lambda _humans: (
            np.zeros((0, 2), np.float32),
            np.zeros((0, 2), np.float32),
        ),
    )
    monkeypatch.setattr(C.SS, "advance_humans", lambda *args: None)
    monkeypatch.setattr(
        C,
        "evaluate_recovery_pool",
        lambda *args, **kwargs: {
            "best": None,
            "queried": [],
            "certified": [],
        },
    )
    result = C.rollout_recovery_controller(
        250001,
        0.5,
        T=5,
    )
    assert result["status"] == "nvp"
    assert result["steps"] == 0
    assert len(result["trace"]) == 1
