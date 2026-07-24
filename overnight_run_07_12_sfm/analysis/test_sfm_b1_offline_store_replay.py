import json
import math

import numpy as np
import pytest
import torch

import sfm_b1_offline_exec as OE
import sfm_b1_offline_replay as OR
import sfm_b1_offline_store as OS


def _result(y, *, resolved=True, full_h=True):
    if not resolved:
        return dict(resolved=False, error="solver")
    return dict(
        resolved=True,
        y=int(y),
        taskspace=bool(y),
        collision_free=bool(y),
        certificate=bool(y),
        full_h=bool(full_h),
        terminal_step=10 if full_h else 4,
        diagnostics={"margin": 0.25},
    )


def _context(shard, *, scenario, gamma, step):
    return shard.add_context(
        scenario_id=scenario,
        gamma=gamma,
        step=step,
        state=np.zeros(4, np.float32),
        hp10=np.zeros((10, 16, 12), np.float32),
        low5=np.zeros(5, np.float32),
        hist=np.zeros((16, 2), np.float32),
        ped_xy=np.zeros((1, 2), np.float32),
        ped_vel=np.zeros((1, 2), np.float32),
    )


def _add_window(shard, *, scenario, gamma, step, y):
    context_id = _context(
        shard, scenario=scenario, gamma=gamma, step=step,
    )
    controls = np.full((10, 2), scenario + step / 100.0, np.float32)
    shard.add_executed_window(
        context_id,
        controls,
        _result(y),
        execution_source="selected_B" if y else "raw_continuation",
        nvp_context=not bool(y),
        candidate_id=0 if y else None,
        acquisition_step=0 if y else None,
        sigma=0.4 if y else None,
        hp_margin=0.2,
        mode="U" if scenario % 2 else "R",
    )
    return context_id


def _mixed_shard(positive=7, negative=3):
    shard = OS.ExecutedRoundShard(1)
    gammas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    for index in range(positive + negative):
        _add_window(
            shard,
            scenario=100 + index,
            gamma=gammas[index % len(gammas)],
            step=index,
            y=index < positive,
        )
    return shard


class _TinyPolicy(torch.nn.Module):
    """Minimal policy surface needed by the offline replay implementation."""

    def __init__(self):
        super().__init__()
        self.enc_grid = torch.nn.Linear(1, 1, bias=False)
        self.head = torch.nn.Linear(20, 20, bias=False)
        self.d = 20
        self.u_max = 2.0

    def ctx_from(self, grid, low, hist):
        del grid, hist
        return low[:, :1]

    def forward(self, value, tau, context):
        del tau, context
        return self.head(value)

    def cfm_loss(self, controls, context, weights=None):
        del context
        value = controls.reshape(len(controls), self.d) / self.u_max
        per = (self.head(value) - value).square().mean(dim=1)
        if weights is None:
            return per.mean()
        return (per * weights).sum() / weights.sum()

    def module_groups(self):
        return {"E_g": self.enc_grid, "head": self.head}


def _trainable(policy):
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)
    return [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad
    ]


def test_executed_store_has_one_window_per_context_and_exact_partition(tmp_path):
    shard = OS.ExecutedRoundShard(3)
    positive_context = _add_window(
        shard, scenario=11, gamma=0.1, step=2, y=1,
    )
    _add_window(shard, scenario=12, gamma=1.0, step=3, y=0)

    with pytest.raises(ValueError, match="at most one executed window"):
        shard.add_executed_window(
            positive_context,
            np.zeros((10, 2), np.float32),
            _result(1),
            execution_source="selected_B",
            nvp_context=False,
        )
    with pytest.raises(ValueError, match="exact full-H=10"):
        context_id = _context(
            shard, scenario=13, gamma=0.5, step=4,
        )
        shard.add_executed_window(
            context_id,
            np.zeros((10, 2), np.float32),
            _result(1, full_h=False),
            execution_source="selected_B",
            nvp_context=False,
        )
    with pytest.raises(ValueError, match=r"finite controls \[10,2\]"):
        context_id = _context(
            shard, scenario=14, gamma=0.5, step=5,
        )
        shard.add_executed_window(
            context_id,
            np.zeros((9, 2), np.float32),
            _result(1),
            execution_source="selected_B",
            nvp_context=False,
        )

    assert len(shard.D) == 2
    assert [row["y"] for row in shard.Dplus] == [1]
    assert [row["y"] for row in shard.Dminus] == [0]
    assert {row["window_id"] for row in shard.D} == {0, 1}
    assert {row["context_id"] for row in shard.D} == {0, 1}
    assert shard.validate() == {
        "round": 3,
        "contexts": 4,
        "D": 2,
        "Dplus": 1,
        "Dminus": 1,
        "errors": 0,
        "unresolved_contexts": 2,
    }

    path = tmp_path / "round_003.pt"
    manifest = shard.save(path)
    assert manifest["D"] == manifest["Dplus"] + manifest["Dminus"] == 2
    with open(str(path) + ".COMPLETE.json") as stream:
        marker = json.load(stream)
    assert marker["status"] == "OFFLINE_EXECUTED_ROUND_SHARD_COMPLETE"
    assert marker["sha256"] == OS.sha256_file(path)

    restored = OS.ExecutedRoundShard.load(path)
    assert restored.validate() == shard.validate()
    assert [row["execution_source"] for row in restored.D] == [
        "selected_B", "raw_continuation",
    ]
    np.testing.assert_array_equal(
        restored.Dminus[0]["controls"], shard.Dminus[0]["controls"],
    )


def test_stratified_batches_are_deterministic_and_exact_once():
    shard = _mixed_shard()
    left, left_positive, left_negative = OR.stratified_batches(
        shard, batch=4, seed=73,
    )
    right, _, _ = OR.stratified_batches(shard, batch=4, seed=73)
    left_ids = [
        (record[0].round_i, record[1]["window_id"])
        for batch in left for record in batch
    ]
    right_ids = [
        (record[0].round_i, record[1]["window_id"])
        for batch in right for record in batch
    ]
    assert left_ids == right_ids
    assert len(left_ids) == len(set(left_ids)) == len(shard.D)
    assert len(left_positive) == len(shard.Dplus) == 7
    assert len(left_negative) == len(shard.Dminus) == 3
    assert len(left) == math.ceil(len(shard.D) / 4)
    assert all(any(record[1]["y"] == 1 for record in batch) for batch in left)


@pytest.mark.parametrize("exposure_epochs", (1, 10, 100))
def test_replay_exact_exposure_counts_and_adam_step_formula(exposure_epochs):
    torch.manual_seed(14)
    shard = _mixed_shard()
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(_trainable(policy), lr=1.0e-4)
    report = OR.replay(
        policy,
        optimizer,
        shard,
        alpha=0.01,
        exposure_epochs=exposure_epochs,
        batch=4,
        device="cpu",
        seed=101,
    )
    steps_per_epoch = math.ceil(len(shard.D) / 4)
    assert report["batches_per_epoch"] == steps_per_epoch
    assert report["optimizer_steps"] == steps_per_epoch * exposure_epochs
    assert report["positive_total_visits"] == len(shard.Dplus) * exposure_epochs
    assert report["negative_total_visits"] == len(shard.Dminus) * exposure_epochs
    assert all(row["positive_visits"] == len(shard.Dplus) for row in report["epochs"])
    assert all(row["negative_visits"] == len(shard.Dminus) for row in report["epochs"])
    assert report["exact_once_per_exposure_epoch"]
    assert report["negative_used_for_training"]
    assert report["visual_encoder_sha_before"] == report["visual_encoder_sha_after"]


def test_alpha_zero_retains_and_counts_Dminus_but_never_uses_negative_gradient(
    monkeypatch,
):
    torch.manual_seed(15)
    shard = _mixed_shard()
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(_trainable(policy), lr=1.0e-4)
    original = OR._weighted_loss
    negative_loss_calls = []

    def audit_weighted_loss(policy, records, mass, population, device):
        if records and all(int(record[1]["y"]) == 0 for record in records):
            negative_loss_calls.append(len(records))
        return original(policy, records, mass, population, device)

    monkeypatch.setattr(OR, "_weighted_loss", audit_weighted_loss)
    report = OR.replay(
        policy,
        optimizer,
        shard,
        alpha=0.0,
        exposure_epochs=1,
        batch=4,
        device="cpu",
        seed=102,
    )
    assert len(shard.Dminus) == report["negative_eligible"] == 3
    assert report["negative_total_visits"] == 3
    assert not report["negative_used_for_training"]
    assert negative_loss_calls == []
    assert report["optimizer_steps"] == math.ceil(len(shard.D) / 4)
    assert all(row["negative_loss"] is None for row in report["epochs"])


def test_gp_cap_512_has_equal_gamma_quota_and_rotating_extra():
    shard = OS.ExecutedRoundShard(1)
    for gamma_index, gamma in enumerate(OE.SP.GAMMAS):
        for sample_index in range(74):
            _add_window(
                shard,
                scenario=1_000 + gamma_index,
                gamma=gamma,
                step=sample_index,
                y=1,
            )

    selected_round_2, report_round_2 = OE._gamma_balanced_records(
        shard, cap=512, round_i=2, seed=20,
    )
    selected_round_3, report_round_3 = OE._gamma_balanced_records(
        shard, cap=512, round_i=3, seed=20,
    )

    assert len(selected_round_2) == len(selected_round_3) == 512
    assert report_round_2["quota"] == report_round_3["quota"] == 73
    assert report_round_2["unique"] and report_round_3["unique"]
    assert report_round_2["rotating_extra_gamma"] == 0.1
    assert report_round_3["rotating_extra_gamma"] == 0.2
    assert report_round_2["per_gamma"]["0.1"] == 74
    assert report_round_3["per_gamma"]["0.2"] == 74
    for gamma in OE.SP.GAMMAS:
        expected_round_2 = 74 if gamma == 0.1 else 73
        expected_round_3 = 74 if gamma == 0.2 else 73
        assert report_round_2["per_gamma"][str(gamma)] == expected_round_2
        assert report_round_3["per_gamma"][str(gamma)] == expected_round_3


def test_gp_quota_fails_closed_instead_of_redistributing_gamma_shortfall():
    shard = OS.ExecutedRoundShard(1)
    for gamma_index, gamma in enumerate(OE.SP.GAMMAS):
        count = 72 if gamma == 0.1 else 74
        for sample_index in range(count):
            _add_window(
                shard,
                scenario=2_000 + gamma_index,
                gamma=gamma,
                step=sample_index,
                y=1,
            )
    with pytest.raises(RuntimeError, match="strict gamma-balanced GP quota"):
        OE._gamma_balanced_records(
            shard, cap=512, round_i=2, seed=21,
        )
