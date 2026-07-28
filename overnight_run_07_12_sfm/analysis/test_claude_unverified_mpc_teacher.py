import os

import numpy as np
import pytest
import torch

import claude_unverified_mpc_teacher as T
import claude_mpc_pool as MP
import sfm_b1_offline_store as OS
import sfm_scene as SS


def _result(y):
    return dict(
        resolved=True,
        y=int(y),
        taskspace=bool(y),
        collision_free=bool(y),
        certificate=bool(y),
        full_h=True,
        terminal_step=10,
        diagnostics={"slack": 0.1},
    )


def _shard(round_i=1):
    shard = OS.ExecutedRoundShard(round_i)
    for scenario, gamma in ((10, 0.1), (11, 0.1), (12, 1.0), (13, 1.0)):
        context_id = shard.add_context(
            scenario_id=scenario,
            gamma=gamma,
            step=0,
            state=np.zeros(4, np.float32),
            hp10=np.zeros((10, 16, 12), np.float32),
            low5=np.zeros(5, np.float32),
            hist=np.zeros((16, 2), np.float32),
            ped_xy=np.array([[2.0, 2.0]], np.float32),
            ped_vel=np.zeros((1, 2), np.float32),
        )
        shard.add_executed_window(
            context_id,
            np.zeros((10, 2), np.float32),
            np.zeros(20, np.float32),
            _result(context_id % 2),
            execution_source="unit_test",
            nvp_context=False,
        )
    return shard


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(20, 20, bias=False)
        self.d = 20
        self.u_max = 2.0

    def ctx_from(self, hp10, low, hist):
        del hp10, hist
        return low[:, :1]

    def cfm_loss(self, controls, context, weights=None):
        del context
        value = controls.reshape(len(controls), 20) / self.u_max
        per = (self.head(value) - value).square().mean(dim=1)
        if weights is None:
            return per.mean()
        return (per * weights).sum() / weights.sum()


class _InlineExecutor:
    def map(self, function, tasks):
        return [function(task) for task in tasks]


def _teacher_records(shard):
    records = []
    for teacher_id, context in enumerate(shard.contexts):
        records.append({
            "teacher_id": teacher_id,
            "round": shard.round_i,
            "context_id": context["context_id"],
            "scenario_id": context["scenario_id"],
            "gamma": context["gamma"],
            "episode_id": context["scenario_id"],
            "step": context["step"],
            "controls": np.full((10, 2), 0.25, np.float32),
            "source": T.TEACHER_SOURCE,
            "candidate_source": {
                "family": "constant_acceleration",
                "source_index": 0,
            },
            "controller_config_hash": "abc",
            "selector_diagnostics": {"filter_feasible": True},
            "pool_manifest": {},
            "clip": {
                "u_max": SS.U_MAX,
                "max_abs_before": 0.25,
                "max_clip_delta": 0.0,
            },
            "context_snapshot": T._context_snapshot(context),
            "socp_audit": None,
        })
    return records


def test_control_contract_rejects_shape_nan_and_large_clipping():
    controls, audit = T.validate_controls(
        np.full((10, 2), SS.U_MAX, np.float32),
    )
    assert controls.shape == (10, 2)
    assert audit["max_clip_delta"] == 0.0
    with pytest.raises(ValueError, match="shape"):
        T.validate_controls(np.zeros((9, 2), np.float32))
    bad = np.zeros((10, 2), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        T.validate_controls(bad)
    too_large = np.zeros((10, 2), np.float32)
    too_large[0, 0] = SS.U_MAX + 0.01
    with pytest.raises(ValueError, match="silently reinterpret"):
        T.validate_controls(too_large)


def test_candidate_provenance_keeps_first_family_when_plans_deduplicate():
    state = np.zeros(4, np.float32)
    plan = np.zeros((10, 2), np.float32)
    unique, sources = MP._deduplicate_plans_with_sources([
        (plan, {"family": "nominal", "source_index": 0, "state": state}),
        (plan.copy(), {"family": "refined", "source_index": 3, "state": state}),
    ])
    assert len(unique) == 1
    assert sources == [{"family": "nominal", "source_index": 0}]


def test_balanced_context_selection_and_hierarchical_mass():
    shard = _shard()
    chosen = T._balanced_context_ids(shard, shard.windows, max_contexts=2)
    assert len(chosen) == 2
    assert {
        round(float(shard.contexts[index]["gamma"]), 8) for index in chosen
    } == {0.1, 1.0}
    records = _teacher_records(shard)
    mass, accounting = T._teacher_mass(shard, records)
    assert accounting["total"] == pytest.approx(1.0)
    assert accounting["gamma"]["0.1"] == pytest.approx(0.5)
    assert accounting["gamma"]["1"] == pytest.approx(0.5)
    assert set(mass) == {0, 1, 2, 3}


def test_buffer_never_duck_types_ordinary_D_and_authenticates_shard(tmp_path):
    shard = _shard()
    shard_path = os.fspath(tmp_path / "round.pt")
    buffer_path = os.fspath(tmp_path / "teacher.pt")
    shard.save(shard_path)
    records = _teacher_records(shard)
    payload = T.save_buffer(
        buffer_path, shard_path, shard, records, {"counts": {"kept": 4}},
    )
    assert payload["status"] == T.BUFFER_STATUS
    assert payload["round_shard_sha256"] == OS.sha256_file(shard_path)
    forbidden = {"y", "query_id", "train_eligible", "x0"}
    assert all(not (forbidden & set(row)) for row in payload["records"])


def test_socp_negative_is_audit_only_and_teacher_is_still_kept(monkeypatch):
    shard = _shard()

    def fake_replay(*args, **kwargs):
        del args, kwargs
        return [object()], np.zeros(4, np.float32)

    monkeypatch.setattr(T.MP, "replay_prefix_humans", fake_replay)
    monkeypatch.setattr(
        T.SS,
        "collect_humans",
        lambda humans: (
            np.array([[2.0, 2.0]], np.float32),
            np.zeros((1, 2), np.float32),
        ),
    )
    monkeypatch.setattr(
        T,
        "select_teacher_plan",
        lambda *args, **kwargs: (
            np.full((10, 2), 0.2, np.float32),
            {
                "candidate_source": {
                    "family": "avoidance",
                    "source_index": 2,
                },
                "selector_diagnostics": {"filter_feasible": True},
                "pool_manifest": {},
                "clip": {
                    "u_max": SS.U_MAX,
                    "max_abs_before": 0.2,
                    "max_clip_delta": 0.0,
                },
            },
        ),
    )
    monkeypatch.setattr(
        T.SM,
        "verify_in_worker",
        lambda task: (
            task[0],
            task[1],
            {
                "resolved": True,
                "y": 0,
                "full_h": True,
                "diagnostics": {"slack": -0.2},
            },
        ),
    )
    records, audit = T.harvest_round(
        object(),
        shard,
        [shard.windows[0]],
        device="cpu",
        environment={"n_ped": 1, "ped_speed_range": (1.0, 2.0)},
        executor=_InlineExecutor(),
        audit_socp=True,
    )
    assert len(records) == 1
    assert records[0]["socp_audit"]["verifier_label"] == 0
    assert "y" not in records[0]
    assert audit["counts"]["kept"] == 1
    assert audit["counts"]["socp_audit_nonpositive"] == 1


def test_teacher_block_is_separate_and_uses_a_fresh_seed_per_step():
    shard = _shard()
    records = _teacher_records(shard)
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    before = policy.head.weight.detach().clone()
    result = T.distill_block(
        policy,
        optimizer,
        shard,
        records,
        epochs=2,
        batch=2,
        seed=7,
    )
    assert result["steps"] == 2
    assert result["optimizer_steps"] == 2
    assert result["sample_exposures"] == 8
    assert result["fresh_base_seed_count"] == 4
    assert not torch.equal(before, policy.head.weight)
    assert all("x0" not in record for record in records)
    noop = T.distill_block(
        policy,
        optimizer,
        shard,
        records,
        epochs=0,
        batch=2,
        seed=7,
    )
    assert noop["steps"] == 0
