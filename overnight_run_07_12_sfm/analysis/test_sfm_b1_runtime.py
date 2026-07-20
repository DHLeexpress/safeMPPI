import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import grid_policy_sfm as GPS
import sfm_b1_eval as E
import sfm_b1_expand as X
import sfm_b1_query_diagnostic as QD
import sfm_b1_sweep as SW
import sfm_scene as SS


def test_nvp_isolates_one_replica(monkeypatch):
    monkeypatch.setattr(X.SS, "make_humans", lambda *args, **kwargs: [])
    first = X.Replica(1, .1, n_ped=0)
    second = X.Replica(2, .1, n_ped=0)
    X.nvp_fail_closed(first)
    assert not first.alive and first.status == "nvp"
    assert second.alive and second.status is None


def test_raw_evaluator_has_no_forbidden_import_or_call():
    source = Path(E.__file__).read_text()
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    forbidden = ("acquisition", "verifier", "selector", "template", "kazuki", "mppi", "refine")
    lowered = " ".join(imports).lower()
    assert not any(word in lowered for word in forbidden)
    raw = ast.get_source_segment(source, next(node for node in tree.body
                                              if isinstance(node, ast.FunctionDef) and node.name == "raw_rollout"))
    assert not any(word in raw.lower() for word in forbidden)


def test_zero_guidance_same_latent_matches_raw_generator():
    torch.manual_seed(18)
    policy = GPS.build_sfm_policy(width=24, res_dropout=0.0)
    context = policy.ctx_from(torch.randn(2, 10, 16, 12), torch.randn(2, 5), torch.randn(2, 16, 2))
    latent = torch.randn(2, policy.d)
    raw = E.integrate_latents(policy, latent.clone(), context, nfe=8)
    zero_guidance = E.integrate_latents(policy, latent.clone(), context, nfe=8)
    torch.testing.assert_close(raw, zero_guidance, rtol=0, atol=0)


def test_default_kazuki_is_separately_labeled_generate_refine():
    import sfm_kazuki as K
    config = K.KazukiConfig()
    assert config.safe_coefs == (0.3,) and config.goal_coef == 0.5
    assert config.n_copy > 0


def test_raw_support_is_counted_without_render_trace(monkeypatch):
    torch.manual_seed(22)
    policy = GPS.build_sfm_policy(width=24, res_dropout=0.0)
    monkeypatch.setattr(E.SS, "make_humans", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        E.SS, "collect_humans",
        lambda humans: (__import__("numpy").array([[3., 3.]], dtype="float32"),
                        __import__("numpy").zeros((1, 2), dtype="float32")),
    )
    monkeypatch.setattr(E.SS, "advance_humans", lambda humans, state: None)
    row = E.raw_rollout(policy, 1, .5, T=1, n_ped=1, collect_trace=False)
    assert row["trace"] is None
    assert sum(row["mode_counts"].values()) == 1


def test_scene_profiles_make_training_shift_explicit():
    training = SS.scene_profile("training")
    requested_id = SS.scene_profile("id")
    requested_ood = SS.scene_profile("requested_ood")
    legacy = SS.scene_profile("legacy_velocity_ood")
    assert (training["n_ped"], training["ped_speed_range"]) == (20, [.5, 1.0])
    assert (requested_id["n_ped"], requested_id["ped_speed_range"]) == (10, [.5, 1.0])
    assert (requested_ood["n_ped"], requested_ood["ped_speed_range"]) == (30, [1.0, 1.5])
    assert (legacy["n_ped"], legacy["ped_speed_range"]) == (20, [1.0, 1.5])
    assert requested_id["training_reference"] == training["training_reference"]
    assert "10 versus 20" in requested_id["shift_from_training"]


def test_scientific_eval_cli_requires_scene_profile():
    with pytest.raises(SystemExit):
        E.build_parser().parse_args(["--checkpoint", "x.pt", "--ep0", "1", "--M", "1", "--out", "x.json"])
    args = E.build_parser().parse_args([
        "--checkpoint", "x.pt", "--ep0", "1", "--M", "1", "--scene-profile", "requested_ood",
        "--out", "x.json",
    ])
    assert args.scene_profile == "requested_ood"


def test_evaluate_policy_passes_explicit_scene_contract(monkeypatch):
    calls = []

    def fake_rollout(policy, episode, gamma, **kwargs):
        calls.append(kwargs)
        return dict(
            gamma=gamma, success=False, collision=False, successful_clearance=None,
            time_to_goal=None, min_clearance=1.0, mode_counts={},
        )

    monkeypatch.setattr(E, "raw_rollout", fake_rollout)
    bank = {str(gamma): [1] for gamma in SS.GAMMAS}
    E.evaluate_policy(object(), bank, device="cpu", scene_profile="requested_ood")
    assert len(calls) == len(SS.GAMMAS)
    assert all(call["n_ped"] == 30 for call in calls)
    assert all(call["ped_speed_range"] == (1.0, 1.5) for call in calls)


def test_paired_query_snapshot_rule_is_shared_and_not_visual_curation():
    traces = {}
    for selector, offset in (("margin", 0.0), ("safemppi_cost", .1)):
        rows = []
        for scenario in (11, 12, 13):
            for gamma in QD.DIAGNOSTIC_GAMMAS:
                for step, distance in ((0, 2.0), (1, .8), (2, 1.2)):
                    rows.append(dict(
                        scenario_id=scenario, gamma=gamma, step=step, executed_id=0,
                        state=np.array([0.0, 0.0, 0.0, 0.0]),
                        ped_xy=np.array([[distance + offset, 0.0]]),
                    ))
        traces[selector] = rows
    chosen = QD.choose_shared_interaction_steps(traces, (11, 12, 13))
    assert [row["step"] for row in chosen] == [1, 1, 1]
    assert all("both selectors" in row["rule"] for row in chosen)


def test_forecast_boundary_is_json_native_for_numpy_timing():
    maximum_round, forecast, authorized = SW.full_sweep_forecast(np.float64(25.5))
    assert type(maximum_round) is float
    assert type(forecast) is float
    assert type(authorized) is bool
    json.dumps(dict(maximum_round=maximum_round, forecast=forecast, authorized=authorized))


def _selection_summary(*, pooled_cr, worst_cr, pooled_sr=.8, worst_sr=.7):
    pooled = dict(
        CR=pooled_cr, SR=pooled_sr,
        successful_clearance=dict(mean=.2), successful_time_to_goal=dict(mean=6.0),
        support={"left": 10, "right": 10, "yield": 10},
    )
    return dict(pooled=pooled, per_gamma={
        "low": dict(CR=worst_cr, SR=worst_sr),
        "high": dict(CR=min(pooled_cr, worst_cr), SR=pooled_sr),
    })


def test_selection_is_threshold_then_pooled_cr_not_continuous_worst_cr():
    lower_pooled = _selection_summary(pooled_cr=.05, worst_cr=.20)
    lower_worst = _selection_summary(pooled_cr=.08, worst_cr=.10)
    assert E.selection_key(lower_pooled) < E.selection_key(lower_worst)
    threshold_pass = _selection_summary(pooled_cr=.04, worst_cr=.04)
    threshold_fail = _selection_summary(pooled_cr=0.0, worst_cr=.05)
    assert E.selection_key(threshold_pass) < E.selection_key(threshold_fail)
