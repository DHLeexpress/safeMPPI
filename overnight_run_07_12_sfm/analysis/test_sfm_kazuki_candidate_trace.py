import numpy as np

import sfm_kazuki as K


def _fake_simulator(call_log=None):
    def simulate(_humans, state, plans, horizon):
        plans = np.asarray(plans, np.float32)
        count = len(plans)
        if call_log is not None:
            call_log.append(int(horizon))
        clear = 0.10 + 0.02 * np.arange(count, dtype=float)
        inside = np.ones(count, dtype=bool)
        terminal = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
        terminal[:, 0] += 0.01 * np.arange(count, dtype=np.float32)
        human_xy = np.zeros((count, 0, 2), dtype=np.float32)
        reach_step = np.full(count, int(horizon) + 1, dtype=int)
        if count > 2:
            reach_step[2] = int(horizon)
        return clear, inside, terminal, human_xy, reach_step

    return simulate


def _inputs():
    state = np.zeros(4, dtype=np.float32)
    nominal = np.zeros((10, 2), dtype=np.float32)
    candidate = np.zeros((10, 2), dtype=np.float32)
    candidate[:, 0] = 1.0
    # Include exact duplicates to exercise the function's canonical unique pool.
    candidates = np.stack([candidate, candidate, nominal])
    return state, nominal, candidates


def test_candidate_trace_is_opt_in_and_does_not_change_selection(monkeypatch):
    monkeypatch.setattr(K, "_simulate_sfm_plans", _fake_simulator())
    state, nominal, candidates = _inputs()
    kwargs = dict(
        margin=0.25,
        horizon=10,
        always_select=True,
        n_goal_plans=0,
        n_avoid_plans=0,
    )

    action_off, diag_off, plan_off = K.exact_sfm_horizon_filter_action(
        [], state, nominal, candidates, **kwargs)
    action_on, diag_on, plan_on = K.exact_sfm_horizon_filter_action(
        [], state, nominal, candidates, diagnostic_candidates=True, **kwargs)

    np.testing.assert_array_equal(action_on, action_off)
    np.testing.assert_array_equal(plan_on, plan_off)
    assert "candidate_pool" not in diag_off
    assert "selected_candidate_index" not in diag_off
    assert {
        key: value for key, value in diag_on.items()
        if key not in {"candidate_pool", "selected_candidate_index"}
    } == diag_off

    pool = diag_on["candidate_pool"]
    selected_index = diag_on["selected_candidate_index"]
    assert [row["candidate_index"] for row in pool] == list(range(len(pool)))
    assert len(pool) == diag_on["candidates_checked"]
    assert sum(row["selected"] for row in pool) == 1
    assert pool[selected_index]["selected"]
    np.testing.assert_array_equal(pool[selected_index]["controls"], plan_on)
    np.testing.assert_array_equal(pool[selected_index]["controls"][0], action_on)

    for row in pool:
        assert np.asarray(row["controls"]).shape == (10, 2)
        assert row["hard_margin_feasible"] == (
            row["recoverable"] and row["min_predicted_clearance"] >= kwargs["margin"])
        assert len(row["terminal"]) == 4
        assert isinstance(row["reach_step"], int)
        expected_arrival = row["reach_step"] if row["reach_step"] <= 10 else None
        assert row["predicted_arrival_step"] == expected_arrival


def test_nominal_early_return_trace_contains_the_simulated_unique_pool(monkeypatch):
    def all_safe(_humans, state, plans, horizon):
        count = len(plans)
        clear = np.full(count, 1.0)
        inside = np.ones(count, dtype=bool)
        terminal = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
        human_xy = np.zeros((count, 0, 2), dtype=np.float32)
        reach_step = np.full(count, int(horizon) + 1, dtype=int)
        return clear, inside, terminal, human_xy, reach_step

    monkeypatch.setattr(K, "_simulate_sfm_plans", all_safe)
    state, nominal, candidates = _inputs()
    action, diag, plan = K.exact_sfm_horizon_filter_action(
        [], state, nominal, candidates, margin=0.25, diagnostic_candidates=True)

    np.testing.assert_array_equal(action, nominal[0])
    np.testing.assert_array_equal(plan, nominal)
    assert diag["candidates_checked"] == 1
    assert len(diag["candidate_pool"]) > 1
    assert diag["selected_candidate_index"] == 0
    assert diag["candidate_pool"][0]["selected"]


def test_recursive_horizon_escalation_preserves_candidate_trace(monkeypatch):
    calls = []

    def escalating_simulator(_humans, state, plans, horizon):
        plans = np.asarray(plans, np.float32)
        count = len(plans)
        calls.append(int(horizon))
        # The first L=20 call audits direct continuations and rejects them.
        # The second L=20 call is the recursively regenerated candidate family.
        clear_value = 0.0 if calls == [10, 20] else 0.30
        clear = np.full(count, clear_value)
        inside = np.ones(count, dtype=bool)
        terminal = np.repeat(np.asarray(state, np.float32)[None], count, axis=0)
        human_xy = np.zeros((count, 0, 2), dtype=np.float32)
        reach_step = np.full(count, int(horizon) + 1, dtype=int)
        return clear, inside, terminal, human_xy, reach_step

    monkeypatch.setattr(K, "_simulate_sfm_plans", escalating_simulator)
    state, nominal, candidates = _inputs()
    action, diag, plan = K.exact_sfm_horizon_filter_action(
        [], state, nominal, candidates,
        margin=0.25,
        horizon=10,
        always_select=True,
        viability_lookahead=20,
        viability_escalate=True,
        viability_band=0.10,
        viability_escalation_band=0.10,
        diagnostic_candidates=True,
    )

    assert calls[:3] == [10, 20, 20]
    assert diag["selection_reason"] == "proactive_horizon_escalation"
    assert diag["horizon"] == 20
    assert diag["escalated_from_horizon"] == 10
    assert len(diag["candidate_pool"][0]["controls"]) == 20
    selected = diag["candidate_pool"][diag["selected_candidate_index"]]
    assert selected["selected"]
    np.testing.assert_array_equal(selected["controls"][0], action)
    np.testing.assert_array_equal(np.asarray(selected["controls"])[: len(plan)], plan)
