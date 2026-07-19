from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_context as CX
import afe_core as AC
import grid_expand_afe_rbf as R


class _ReachedSourceGate(RuntimeError):
    pass


def _low7_policy():
    return R.HP.GridHPFlowPolicy(
        repr_dim=32,
        grid_hw=(32, 32),
        trunk_hidden=(160, 96),
        enc_depth=3,
        raw_condition_dim=7,
        conditioning_schema=CX.LOW7_SCHEMA,
    )


def _stub_checkpoint_loading(monkeypatch, calls):
    checkpoint = {"test": "low7"}
    digest = "a" * 64
    monkeypatch.setattr(R.AFE2, "_sha256_file", lambda _: digest)
    monkeypatch.setattr(R.HP, "load_hp", lambda *_args, **_kwargs: (_low7_policy(), checkpoint))

    def validate(profile_name, policy, actual_checkpoint, checkpoint_sha256):
        calls.update(
            profile_name=profile_name,
            policy=policy,
            checkpoint=actual_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
        return "model-hash", {"schema": "low7"}, "contract-hash"

    monkeypatch.setattr(R.AFE2, "validate_checkpoint_contract", validate)
    return digest, checkpoint


def test_low7_radius03_uses_the_same_declared_checkpoint_contract(monkeypatch) -> None:
    calls = {}
    digest, checkpoint = _stub_checkpoint_loading(monkeypatch, calls)
    monkeypatch.setattr(
        R.AFE2,
        "_git_state",
        lambda: (_ for _ in ()).throw(_ReachedSourceGate()),
    )
    monkeypatch.setattr(sys, "argv", [
        "grid_expand_afe_rbf.py",
        "--ckpt", "unused.pt",
        "--expected-ckpt-sha256", digest,
        "--scene-profile", "low7_radius03_canonical_v1",
        "--outdir", "unused-output",
        "--conditioning-schema", CX.LOW7_SCHEMA,
        "--freeze-visual-encoder",
    ])

    with pytest.raises(_ReachedSourceGate):
        R.main()

    profile = R.get_scene_profile("low7_radius03_canonical_v1")
    assert profile.interior_disk_radius == pytest.approx(0.3)
    assert profile.start == (0.3, 0.3)
    assert profile.goal == (4.7, 4.7)
    assert calls["profile_name"] == profile.name
    assert calls["checkpoint"] is checkpoint
    assert calls["checkpoint_sha256"] == digest
    assert CX.policy_contract(calls["policy"]).raw_condition_dim == 7
    assert all(
        not parameter.requires_grad
        for parameter in calls["policy"].enc_grid.parameters()
    )


def test_low7_radius03_refuses_an_unfrozen_visual_encoder(monkeypatch) -> None:
    calls = {}
    digest, _ = _stub_checkpoint_loading(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "grid_expand_afe_rbf.py",
        "--ckpt", "unused.pt",
        "--expected-ckpt-sha256", digest,
        "--scene-profile", "low7_radius03_canonical_v1",
        "--outdir", "unused-output",
        "--conditioning-schema", CX.LOW7_SCHEMA,
    ])

    with pytest.raises(ValueError, match="low7 closest-boundary conditioning"):
        R.main()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--K", "32"), "K=64, B=8, and batch=128"),
        (("--gp-replay-window", "0"), "GP replay window"),
        (("--lengthscale-multiplier", "0"), "length-scale multiplier"),
        (("--negative-alpha", "-0.1"), "negative alpha"),
    ],
)
def test_cli_rejects_sweep_contract_violations(monkeypatch, arguments, message) -> None:
    monkeypatch.setattr(sys, "argv", [
        "grid_expand_afe_rbf.py",
        "--ckpt", "unused.pt",
        "--expected-ckpt-sha256", "a" * 64,
        "--scene-profile", "low7_radius03_canonical_v1",
        "--outdir", "unused-output",
        *arguments,
    ])

    with pytest.raises(ValueError, match=message):
        R.main()


def test_config_defaults_preserve_the_legacy_execution_protocol() -> None:
    cfg = R.AFERBFConfig()
    assert cfg.execution_rule == "legacy_max_horizon_progress"
    assert cfg.gp_replay_window == 1
    assert cfg.lengthscale_multiplier == 1.0
    assert cfg.negative_alpha == 0.0
    assert cfg.conditioning_schema == CX.LOW5_SCHEMA
    assert cfg.freeze_visual_encoder is False


class _Policy:
    def __init__(self, candidates):
        self.candidates = torch.as_tensor(candidates, dtype=torch.float32)
        self.H_pred = int(self.candidates.shape[1])
        self.d = self.H_pred * 2

    def ctx_from(self, grid, low, hist):
        return torch.zeros(len(grid), 1)

    def sample(self, count, context, **_kwargs):
        assert count == len(self.candidates)
        return self.candidates.clone()

    def phi_s(self, candidates, context, *, s):
        assert s == pytest.approx(0.9)
        return torch.stack((torch.arange(len(candidates)), torch.ones(len(candidates))), 1)


class _GP:
    def sigma(self, features):
        return torch.ones(len(features))

    def sequential_acquire(self, features, steps, beta):
        chosen = list(range(steps))
        trace = [{
            "scores": torch.ones(len(features)),
            "remaining": None,
            "chosen": candidate_id,
            "chosen_score": 1.0,
            "ess_norm": 1.0,
            "entropy_norm": 1.0,
        } for candidate_id in chosen]
        return chosen, trace


class _Executor:
    def __init__(self, results):
        self.results = results

    def map(self, _function, tasks, chunksize):
        assert chunksize == 1
        return [
            (episode_id, candidate_id, dict(self.results[candidate_id]))
            for episode_id, candidate_id, _state, _controls, _gamma in tasks
        ]


def _result(*, y, exec_y=None, exec_prog=0.0, terminal_rescue=False):
    execution_label = int(y if exec_y is None else exec_y)
    return {
        "y": int(y),
        "margin": 0.4 if y else float("nan"),
        "resid": 0.0 if y else -0.1,
        "prog": float(exec_prog),
        "d0": 1.0,
        "reason": "ok" if y else "socp_fail",
        "n_socp_solve": 1,
        "verifier_seconds": 0.0,
        "exec_y": execution_label,
        "exec_prog": float(exec_prog),
        "exec_margin": 0.3 if execution_label else float("nan"),
        "terminal_hit": bool(terminal_rescue),
        "terminal_rescue": bool(terminal_rescue),
        "terminal_tau": 1 if terminal_rescue else None,
        "terminal_prog": float(exec_prog) if terminal_rescue else None,
        "terminal_resid": 0.0 if terminal_rescue else None,
        "terminal_reason": "ok" if terminal_rescue else None,
        "terminal_reverify": bool(terminal_rescue),
    }


def _run_one_step(
    monkeypatch,
    candidates,
    results,
    execution_rule,
    hp,
    *,
    query_budget=None,
    nvp_audit_all_k=False,
):
    count = len(candidates)
    env = SimpleNamespace(
        x0=torch.tensor((0.0, 0.0, 0.0, 0.0)),
        goal=torch.tensor((1.0, 0.0)),
        obstacles=torch.tensor(((10.0, 10.0, 0.1),)),
        r_robot=0.0,
        dt=1.0,
    )
    cfg = SimpleNamespace(
        gammas=(0.5,),
        taskspace_epsilon=2.0,
        conditioning_schema=CX.LOW5_SCHEMA,
        K=count,
        B=count if query_budget is None else int(query_budget),
        T=1,
        reach=0.01,
        seed=910,
        nfe=1,
        temp=1.0,
        s=0.9,
        beta=0.1,
        execution_rule=execution_rule,
        nvp_audit_all_k=bool(nvp_audit_all_k),
    )
    store = AC.DStore(conditioning_schema=CX.LOW5_SCHEMA, condition_dim=5)
    viz = []

    def contexts(episodes, _env, _cfg):
        batch = len(episodes)
        low = np.zeros((batch, 5), dtype=np.float32)
        low[:, -1] = [episode["gamma"] for episode in episodes]
        return (
            np.zeros((batch, 3, 32, 32), dtype=np.float32),
            low,
            np.zeros((batch, 16, 2), dtype=np.float32),
        )

    monkeypatch.setattr(R, "_context_arrays", contexts)
    monkeypatch.setattr(
        R,
        "_proposal_noise",
        lambda policy, active, cfg, purpose, round_i, control_t, device:
            torch.zeros(len(active) * cfg.K, policy.d),
    )
    monkeypatch.setattr(R, "_acquisition_stats", lambda *_args, **_kwargs: {
        "sig_all": [1.0] * count,
        "sig_sel": [1.0] * count,
    })
    monkeypatch.setattr(R.EX.GS, "mode1_config", lambda: {
        "barrier_activation_radius": 2.0,
        "polytope_nbase": 16,
        "predict_gain": 0.0,
    })
    monkeypatch.setattr(R.EX.GS, "planner_obstacles", lambda _env: _env.obstacles)
    monkeypatch.setattr(
        R.EX.GF,
        "polytope_HP",
        lambda *_args, **_kwargs: (hp, (None, None, None)),
    )

    episodes, timings = R.run_parallel_episodes(
        _Policy(candidates),
        _GP(),
        env,
        cfg,
        store,
        round_i=1,
        replicas=1,
        device="cpu",
        executor=_Executor(results),
        collect=True,
        viz=viz,
        purpose="gather",
    )
    return episodes[0], store, viz[0], timings


def test_legacy_selector_keeps_terminal_prefix_max_horizon_behavior(monkeypatch) -> None:
    candidates = np.asarray([
        [[0.4, 0.0]],
        [[0.8, 0.0]],
        [[0.0, 0.0]],
    ], dtype=np.float32)
    results = [
        _result(y=0, exec_y=1, exec_prog=0.9, terminal_rescue=True),
        _result(y=1, exec_y=1, exec_prog=0.5),
        _result(y=0, exec_y=0, exec_prog=2.0),
    ]
    monkeypatch.setattr(
        R.EX,
        "select_nominal_hp_execution",
        lambda *_args, **_kwargs: pytest.fail("legacy execution called the H_P helper"),
    )

    episode, store, frame, _ = _run_one_step(
        monkeypatch,
        candidates,
        results,
        "legacy_max_horizon_progress",
        lambda points: np.ones(len(points)),
    )

    assert episode["path"][1, 0] == pytest.approx(0.2)
    assert store.q_y == [0, 1, 0]
    assert store.pos_ids == [1]
    assert store.q_exec == [1, 0, 0]
    assert frame["sel"] == 0
    assert frame["exec_verified_hp_y"] is None


def test_nominal_selector_maps_exec_y_hp_gate_without_relabeling_dplus(monkeypatch) -> None:
    candidates = np.asarray([
        [[0.4, 0.0]],   # x1=.2: full positive and H_P eligible
        [[0.8, 0.0]],   # x1=.4: terminal prefix, eligible, best progress
        [[2.0, 0.0]],   # x1=1.: full positive but outside H_P level
    ], dtype=np.float32)
    results = [
        _result(y=1, exec_y=1, exec_prog=0.2),
        _result(y=0, exec_y=1, exec_prog=0.4, terminal_rescue=True),
        _result(y=1, exec_y=1, exec_prog=1.0),
    ]

    episode, store, frame, _ = _run_one_step(
        monkeypatch,
        candidates,
        results,
        R.EX.MAX_STEP_PROGRESS,
        lambda points: 1.0 - np.abs(np.asarray(points)[:, 0]),
    )

    assert episode["path"][1, 0] == pytest.approx(0.4)
    assert store.q_y == [1, 0, 1]
    assert store.pos_ids == [0, 2]
    assert store.q_exec == [0, 1, 0]
    assert frame["sel"] == 1
    assert frame["y"] == [1, 0, 1]
    assert frame["exec_y"] == [1, 1, 1]
    assert frame["exec_verified_hp_y"] == [1, 1, 0]
    stats = episode["step_stats"][0]
    assert stats["n_full_socp_positive"] == 2
    assert stats["n_exec_pos"] == 3
    assert stats["n_exec_verified_hp_positive"] == 2


def test_nominal_selector_nvp_is_fail_closed_and_keeps_full_labels(monkeypatch) -> None:
    candidates = np.asarray([
        [[0.4, 0.0]],
        [[0.8, 0.0]],
    ], dtype=np.float32)
    results = [_result(y=1, exec_y=1), _result(y=1, exec_y=1)]

    episode, store, frame, _ = _run_one_step(
        monkeypatch,
        candidates,
        results,
        R.EX.MAX_STEP_MARGIN,
        lambda points: np.where(np.isclose(np.asarray(points)[:, 0], 0.0), 1.0, 0.0),
    )

    assert episode["status"] == "nvp"
    assert episode["term_t"] == 0
    assert episode["nvp_reason"] == "no_exec_verified_nominal_hp_step"
    assert episode["path"].shape == (1, 2)
    assert store.q_y == [1, 1]
    assert store.pos_ids == [0, 1]
    assert store.q_exec == [0, 0]
    assert store.q_nvp_negative == [1, 1]
    assert frame["sel"] == -1
    assert frame["exec_verified_hp_y"] == [0, 0]


def test_all_k_nvp_audit_is_observation_only(monkeypatch) -> None:
    candidates = np.asarray([
        [[0.4, 0.0]],
        [[0.8, 0.0]],
        [[-0.2, 0.0]],  # eligible, but outside the selected first B
        [[1.2, 0.0]],
    ], dtype=np.float32)
    results = [_result(y=1, exec_y=1) for _ in candidates]

    def hp(points):
        x = np.asarray(points)[:, 0]
        return np.where(x < 0.0, 1.0, np.where(np.isclose(x, 0.0), 1.0, 0.0))

    without, store_without, frame_without, timing_without = _run_one_step(
        monkeypatch,
        candidates,
        results,
        R.EX.MAX_STEP_MARGIN,
        hp,
        query_budget=2,
        nvp_audit_all_k=False,
    )
    with_audit, store_with, frame_with, timing_with = _run_one_step(
        monkeypatch,
        candidates,
        results,
        R.EX.MAX_STEP_MARGIN,
        hp,
        query_budget=2,
        nvp_audit_all_k=True,
    )

    assert without["status"] == with_audit["status"] == "nvp"
    assert without["term_t"] == with_audit["term_t"] == 0
    np.testing.assert_array_equal(without["path"], with_audit["path"])
    assert store_without.q_sid == store_with.q_sid == [0, 0]
    assert store_without.pos_ids == store_with.pos_ids == [0, 1]
    assert store_without.q_exec == store_with.q_exec == [0, 0]
    assert store_without.q_nvp_negative == store_with.q_nvp_negative == [1, 1]
    assert frame_without["drawn"] == frame_with["drawn"] == [0, 1]
    assert frame_without["sel"] == frame_with["sel"] == -1
    assert frame_without["nvp_audit"] is None
    assert frame_with["nvp_audit"]["classification"] == (
        "selected_B_acquisition_miss"
    )
    assert frame_with["nvp_audit"]["audit_extra_verifications"] == 2
    assert timing_without["nvp_audit_verifier_wall"] == 0.0
    assert timing_with["nvp_audit_verifier_wall"] >= 0.0


def test_all_k_nvp_audit_does_not_run_after_selected_b_success(monkeypatch) -> None:
    candidates = np.asarray([
        [[-0.2, 0.0]],
        [[0.8, 0.0]],
        [[1.2, 0.0]],
    ], dtype=np.float32)
    results = [_result(y=1, exec_y=1) for _ in candidates]

    episode, store, frame, timings = _run_one_step(
        monkeypatch,
        candidates,
        results,
        R.EX.MAX_STEP_MARGIN,
        lambda points: np.ones(len(points)),
        query_budget=1,
        nvp_audit_all_k=True,
    )

    assert episode["status"] != "nvp"
    assert len(store.q_sid) == 1
    assert frame["nvp_audit"] is None
    assert timings["nvp_audit_verifier_wall"] == 0.0
