from types import SimpleNamespace

import numpy as np
import torch

import sfm_b1_kazuki_repair as R
import sfm_b1_kazuki_repair_audit as A
import sfm_scene as SS


class _Policy:
    d = 20
    H_pred = 10
    u_max = 2.0


class _DeterministicGuidancePolicy:
    H_pred = 10
    d = 20
    u_max = 2.0

    def _expand_ctx(self, context, count):
        return context.reshape(1, -1).expand(int(count), -1)

    def forward(self, value, tau, context):
        del tau, context
        return 0.15 * torch.tanh(value)


def test_locked_guidance_config_is_exact_and_b_sized():
    config = R.locked_guidance_config(0.1, n_sample=4)
    assert config.safe_coefs == (0.3,)
    assert config.goal_coef == 0.5
    assert config.safe_coef_gamma_span == 0.0
    assert config.goal_coef_gamma_span == 0.0
    assert config.n_sample == 4
    assert R.collector_ode_times(8) == tuple(index / 8 for index in range(9))


def test_same_latent_repair_reuses_exact_x0_and_has_no_refinement(monkeypatch):
    seen = {}

    def fake_predict(ped_xy, ped_vel, H, dt, device, dtype):
        del ped_xy, ped_vel, dt
        return torch.zeros(1, H, 1, 2, device=device, dtype=dtype)

    def fake_guided(
        policy, context, state, goal, ped_prediction, ped_velocity,
        radius, x0, times, config, collect_diagnostics,
    ):
        del (
            policy, context, state, goal, ped_prediction, ped_velocity,
            radius,
        )
        seen.update(
            x0=x0.detach().clone(),
            times=tuple(times),
            n_sample=config.n_sample,
            collect=collect_diagnostics,
        )
        trace = [dict(
            integrated_goal_guidance=torch.ones_like(x0),
            integrated_safety_guidance=2 * torch.ones_like(x0),
            component_semantics="test",
        )]
        return x0 + 0.1, trace, x0

    monkeypatch.setattr(R.KZ, "predict_pedestrians_t", fake_predict)
    monkeypatch.setattr(R.KZ, "guided_generate", fake_guided)
    x0 = torch.arange(80, dtype=torch.float32).reshape(4, 20) / 100
    context = torch.zeros(37)
    controls, diagnostics = R.same_latent_guided_controls(
        _Policy(),
        context,
        np.zeros(4, np.float32),
        np.zeros((1, 2), np.float32),
        np.zeros((1, 2), np.float32),
        0.5,
        x0,
        nfe=8,
    )
    torch.testing.assert_close(seen["x0"], x0)
    assert seen["times"] == tuple(index / 8 for index in range(9))
    assert seen["n_sample"] == 4 and seen["collect"]
    assert tuple(controls.shape) == (4, 10, 2)
    assert diagnostics["no_new_latents"]
    assert diagnostics["no_mppi_refinement"]
    np.testing.assert_allclose(
        diagnostics["goal_first_action"], np.full((4, 2), 2.0),
    )
    np.testing.assert_allclose(
        diagnostics["safety_first_action"], np.full((4, 2), 4.0),
    )


def test_guidance_diagnostics_do_not_change_guided_controls():
    torch.manual_seed(31)
    policy = _DeterministicGuidancePolicy()
    context = torch.zeros(3)
    state = np.zeros(4, np.float32)
    goal = torch.tensor(SS.GOAL, dtype=torch.float32)
    x0 = torch.randn(4, 20)
    ped_prediction = torch.full((1, 10, 2), 20.0)
    ped_velocity = torch.zeros(1, 2)
    config = R.locked_guidance_config(0.5, n_sample=4)
    times = R.collector_ode_times(8)
    plain, _, _ = R.KZ.guided_generate(
        policy,
        context,
        state,
        goal,
        ped_prediction,
        ped_velocity,
        SS.R_PED + config.collision_margin,
        x0.clone(),
        times,
        config,
        collect_diagnostics=False,
    )
    diagnostic, trace, _ = R.KZ.guided_generate(
        policy,
        context,
        state,
        goal,
        ped_prediction,
        ped_velocity,
        SS.R_PED + config.collision_margin,
        x0.clone(),
        times,
        config,
        collect_diagnostics=True,
    )
    torch.testing.assert_close(diagnostic, plain, rtol=0, atol=0)
    assert "integrated_goal_guidance" in trace[-1]
    assert "integrated_safety_guidance" in trace[-1]


def test_predictive_trap_matches_post_action_predicate_and_stops_at_three():
    states = [
        np.array([0.01 * index, 0.0, 0.0, 0.0], np.float32)
        for index in range(10)
    ]
    action = np.zeros(2, np.float32)
    assert R.predicted_trap(states, action)
    assert R.post_action_trap_matches(states, action)
    streak = 0
    for expected in (1, 2):
        streak, stop = R.next_trap_streak(streak, True)
        assert streak == expected and not stop
    streak, stop = R.next_trap_streak(streak, True)
    assert streak == 3 and stop
    assert R.next_trap_streak(streak, False) == (0, False)


def test_trigger_prefers_existing_trap_then_nvp_then_prediction(monkeypatch):
    replica = SimpleNamespace(states=[np.zeros(4, np.float32)])
    monkeypatch.setattr(A.FA, "_trap", lambda states: True)
    assert A._repair_trigger(replica, {"controls": np.zeros((10, 2))}) == "trap_streak"
    monkeypatch.setattr(A.FA, "_trap", lambda states: False)
    assert A._repair_trigger(replica, None) == "finite_B_NVP"
    monkeypatch.setattr(A.KR, "predicted_trap", lambda states, action: True)
    assert A._repair_trigger(
        replica, {"controls": np.zeros((10, 2))},
    ) == "predicted_trap"
    monkeypatch.setattr(A.KR, "predicted_trap", lambda states, action: False)
    assert A._repair_trigger(
        replica, {"controls": np.zeros((10, 2))},
    ) is None


def test_manifest_excludes_privileged_and_independent_fallbacks():
    manifest = R.manifest()
    assert manifest["safe_coef"] == 0.3
    assert manifest["goal_coef"] == 0.5
    assert "no privileged MPC" in manifest["exclusions"]
    assert "no independent raw fallback" in manifest["exclusions"]
    assert SS.DT > 0


def _negative_rows():
    rows = []
    for candidate_id in range(4):
        rows.append(dict(
            candidate_id=16 + candidate_id,
            parent_candidate_id=candidate_id,
            acquisition_step=candidate_id,
            controls=np.full((10, 2), candidate_id, np.float32),
            x0=np.full(20, candidate_id, np.float32),
            sigma=0.1 + candidate_id,
            mode="test",
            query_id=candidate_id,
            result=dict(
                resolved=True,
                y=0,
                full_h=True,
                terminal_step=10,
                taskspace=True,
                collision_free=True,
                certificate=False,
                diagnostics={},
            ),
        ))
    return rows


def _prepared():
    return dict(
        state=np.zeros(4, np.float32),
        hp10=torch.zeros(10, 16, 12),
        low=torch.zeros(5),
        hist=torch.zeros(16, 2),
        ped_xy=np.zeros((1, 2), np.float32),
        ped_vel=np.zeros((1, 2), np.float32),
    )


def test_neutral_margin_ranks_all_four_negatives_without_relabeling(
    monkeypatch,
):
    rows = _negative_rows()
    margins = iter((0.1, 0.7, 0.7, -0.2))
    monkeypatch.setattr(
        A.BC,
        "nominal_hp_margin",
        lambda *args: (next(margins), 1.0, 1.0),
    )
    chosen = A._select_neutral(rows, "margin", _prepared(), 0.5)
    assert chosen["candidate_id"] == 17
    assert all(row["result"]["y"] == 0 for row in rows)


def test_neutral_cost_ranks_all_four_negatives(monkeypatch):
    rows = _negative_rows()
    monkeypatch.setattr(
        A.BC,
        "nominal_hp_margin",
        lambda *args: (0.2, 1.0, 1.0),
    )
    monkeypatch.setattr(
        A.BC,
        "safemppi_proposal_cost",
        lambda *args, **kwargs: torch.tensor([3.0, 1.0, 1.0, 2.0]),
    )
    chosen = A._select_neutral(
        rows, "safemppi_cost", _prepared(), 0.5,
    )
    assert chosen["candidate_id"] == 17
    assert chosen["expert_cost"] == 1.0


def test_neutral_selection_requires_four_exact_full_h_negatives():
    rows = _negative_rows()
    rows[0]["result"]["resolved"] = False
    assert A._select_neutral(rows, "margin", _prepared(), 0.5) is None
    rows = _negative_rows()
    rows[0]["result"]["y"] = 1
    assert A._select_neutral(rows, "margin", _prepared(), 0.5) is None


def test_neutral_record_is_separate_and_nontraining(tmp_path):
    rows = _negative_rows()
    for row in rows:
        row["hp_margin"] = 0.2
    replica = SimpleNamespace(scenario_id=250001, gamma=0.5)
    record = A._neutral_record(
        0,
        replica,
        _prepared(),
        rows[0],
        step=9,
        selector="margin",
        repair_trigger="finite_B_NVP",
    )
    assert record["semantic_label"] == "neutral"
    assert record["verifier_y"] == 0
    assert not record["train_eligible"]
    assert not record["replay_default"]
    assert not record["gp_eligible"]
    path = tmp_path / "neutral_round.pt"
    marker = A._save_neutral_records(path, [record])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert marker["D0"] == 1
    assert payload["records"][0]["population"] == "D0"
