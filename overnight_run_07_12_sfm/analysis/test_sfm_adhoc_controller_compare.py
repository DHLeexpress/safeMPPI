import json

import numpy as np

import sfm_adhoc_controller_compare as C


def _claude_rollout(scenario, gamma):
    controls = np.zeros((10, 2), np.float32)
    result = {
        "segment": C.SM.rollout_positions(np.zeros(4, np.float32), controls),
    }
    candidate = {
        "query_rank": 0,
        "controls": controls,
        "certified": True,
        "verifier": result,
    }
    return {
        "controller": "claude_deterministic_recovery",
        "scenario_id": scenario,
        "gamma": gamma,
        "status": "nvp",
        "success": False,
        "collision": False,
        "nvp": True,
        "timeout": False,
        "steps": 1,
        "minimum_clearance": 0.2,
        "time_to_goal": None,
        "path": np.array([[0.0, 0.0], [0.01, 0.0]], np.float32),
        "trace": [{
            "state": np.zeros(4, np.float32),
            "ped_xy": np.array([[1.0, 1.0]], np.float32),
            "ped_vel": np.zeros((1, 2), np.float32),
            "pool": {
                "queried": [candidate],
                "best": candidate,
            },
        }],
    }


def _codex_rollout(scenario, gamma):
    controls = np.zeros((10, 2), np.float32)
    return {
        "controller": "codex_privileged_sfm",
        "scenario_id": scenario,
        "gamma": gamma,
        "status": "success",
        "success": True,
        "collision": False,
        "nvp": False,
        "timeout": False,
        "steps": 1,
        "minimum_clearance": 0.3,
        "time_to_goal": 0.1,
        "path": np.array([[0.0, 0.0], [0.01, 0.0]], np.float32),
        "trace": [{
            "state": np.zeros(4, np.float32),
            "ped_xy": np.array([[1.0, 1.0]], np.float32),
            "ped_vel": np.zeros((1, 2), np.float32),
            "output_filter": {
                "candidate_pool": [{
                    "controls": controls.tolist(),
                    "hard_margin_feasible": False,
                    "selected": True,
                }],
            },
        }],
    }


def test_privileged_config_matches_historical_wrapper():
    config = C.privileged_sfm_config()
    assert config.exact_sfm_step_filter
    assert config.step_filter_goal_plans == 12
    assert config.step_filter_avoid_plans == 18
    assert config.step_filter_viability_lookahead == 20
    assert config.step_filter_stagnation_horizon == 20
    assert config.safe_coef_by_gamma == (
        1.0, 0.3, 1.0, 0.3, 0.3, 0.3, 0.1,
    )
    assert config.goal_coef_by_gamma == (
        2.0, 0.5, 2.0, 0.5, 0.5, 0.5, 3.0,
    )


def test_render_and_delivery_keep_controller_semantics_separate(tmp_path):
    scenarios = [250001]
    gammas = [0.5]
    result = {
        "status": C.STATUS,
        "diagnostic_only": True,
        "changes_checkpoint": False,
        "enters_replay": False,
        "checkpoint": "/tmp/checkpoint.pt",
        "checkpoint_sha256": "abc",
        "scene": {"scene_profile": "double_density_velocity_ood"},
        "scenarios": scenarios,
        "gammas": gammas,
        "T": 180,
        "reach": 0.5,
        "sample_seed": 700000,
        "controllers": {
            "claude_v2_exact_socp": {
                "semantics": {"certificate": True},
                "rollouts": [_claude_rollout(250001, 0.5)],
            },
            "codex_privileged_sfm": {
                "semantics": {"certificate": False},
                "rollouts": [_codex_rollout(250001, 0.5)],
            },
        },
    }
    outdir = tmp_path / "delivery"
    payload = C.deliver(result, outdir=outdir)
    assert payload["controllers"]["claude_v2_exact_socp"]["metrics"]["nvp"] == 1
    assert payload["controllers"]["codex_privileged_sfm"]["metrics"]["success"] == 1
    assert (outdir / "controller_branch_comparison.png").stat().st_size > 0
    assert (outdir / "controller_traces.pt").stat().st_size > 0
    on_disk = json.loads((outdir / "metrics.json").read_text())
    assert on_disk["changes_checkpoint"] is False
    assert on_disk["enters_replay"] is False
    assert (
        on_disk["controllers"]["codex_privileged_sfm"]["semantics"]["certificate"]
        is False
    )
