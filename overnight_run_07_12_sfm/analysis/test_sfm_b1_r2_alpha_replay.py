import copy

import numpy as np
import pytest
import torch

import grid_policy_sfm as GPS
import sfm_b1_r2_alpha_replay as R
import sfm_b1_store as BS


def _verifier_result(y):
    return dict(
        resolved=True, y=int(y), taskspace=bool(y),
        collision_free=bool(y), certificate=bool(y), full_h=True,
        terminal_step=10, train_eligible=bool(y),
        segment=np.zeros((11, 2), np.float32),
        pedestrian_prediction=np.zeros((11, 1, 2), np.float32),
        diagnostics={},
    )


def _recent(tmp_path):
    rng = np.random.RandomState(71)
    recent = BS.RecentRounds(tmp_path)
    for round_i in (1, 2):
        shard = BS.RoundShard(round_i)
        for gamma in (0.1, 1.0):
            context_id = shard.add_context(
                scenario_id=100 + round_i, gamma=gamma, step=0,
                state=np.zeros(4, np.float32),
                hp10=rng.randn(10, 16, 12).astype(np.float32),
                low5=rng.randn(5).astype(np.float32),
                hist=rng.randn(16, 2).astype(np.float32),
                ped_xy=np.zeros((1, 2), np.float32),
                ped_vel=np.zeros((1, 2), np.float32),
            )
            for candidate_id, y in enumerate((1, 1, 0)):
                shard.add_resolved_query(
                    context_id, candidate_id,
                    rng.randn(10, 2).astype(np.float32),
                    sigma=0.4, result=_verifier_result(y),
                    acquisition_step=candidate_id,
                )
        recent.append_and_save(shard)
    return recent


def _negative_only_recent(tmp_path):
    rng = np.random.RandomState(81)
    recent = BS.RecentRounds(tmp_path)
    shard = BS.RoundShard(1)
    context_id = shard.add_context(
        scenario_id=300, gamma=0.5, step=0,
        state=np.zeros(4, np.float32),
        hp10=rng.randn(10, 16, 12).astype(np.float32),
        low5=rng.randn(5).astype(np.float32),
        hist=rng.randn(16, 2).astype(np.float32),
        ped_xy=np.zeros((1, 2), np.float32),
        ped_vel=np.zeros((1, 2), np.float32),
    )
    for candidate_id in range(2):
        shard.add_resolved_query(
            context_id, candidate_id, rng.randn(10, 2).astype(np.float32),
            sigma=0.5, result=_verifier_result(0),
            acquisition_step=candidate_id,
        )
    recent.append_and_save(shard)
    return recent


def test_declared_grid_is_exact_and_other_knobs_fail_closed():
    names = set()
    for alpha in R.ALPHAS:
        for epochs in R.REPLAY_EPOCHS:
            cfg = R.ExperimentConfig(alpha=alpha, replay_epochs=epochs)
            assert cfg.validate() is cfg
            names.add(cfg.arm_name)
    assert len(names) == 9
    with pytest.raises(ValueError):
        R.ExperimentConfig(alpha=0.001, replay_epochs=1).validate()
    with pytest.raises(ValueError):
        R.ExperimentConfig(alpha=0.0, replay_epochs=4).validate()
    with pytest.raises(ValueError):
        R.ExperimentConfig(alpha=0.0, replay_epochs=1, lr=1e-5).validate()
    required_by_gather = (
        "K", "B", "T", "H", "nfe", "temp", "phi_s", "selector",
    )
    assert all(hasattr(R.ExperimentConfig(0.0, 1), key) for key in required_by_gather)


def test_fixed_probe_is_deterministic(tmp_path):
    recent = _recent(tmp_path)
    policy = GPS.build_sfm_policy(width=16, res_dropout=0.0)
    positives = recent.positive_records()
    left = R._fixed_probe_loss(
        policy, positives, batch=3, device="cpu", seed=123,
    )
    right = R._fixed_probe_loss(
        policy, positives, batch=3, device="cpu", seed=123,
    )
    assert left == right


def test_alpha_zero_never_reads_negative_and_one_epoch_is_complete(tmp_path, monkeypatch):
    torch.manual_seed(17)
    recent = _recent(tmp_path)
    policy = GPS.build_sfm_policy(width=16, res_dropout=0.0)
    BS.configure_expansion_trainability(policy)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=R.LEARNING_RATE,
    )
    monkeypatch.setattr(
        recent, "negative_records",
        lambda: (_ for _ in ()).throw(AssertionError("alpha=0 read D-")),
    )
    cfg = R.ExperimentConfig(alpha=0.0, replay_epochs=1)
    report = R.repeat_complete_replay(
        policy, optimizer, recent, cfg, device="cpu", round_i=1,
    )
    assert report["optimizer_steps"] == 1
    assert report["positive_total_visits"] == report["positive_eligible"]
    assert report["negative_eligible"] == 0
    assert report["negative_total_visits"] == 0
    assert not report["negative_used_for_training"]
    assert report["fixed_probe"]["before"]["negative"] is None
    assert report["fixed_probe"]["after"]["negative"] is None
    assert report["epochs"][0]["positive_coverage"]["exact_once"]
    assert report["visual_encoder_sha_before"] == report["visual_encoder_sha_after"]
    assert report["module_relative_parameter_drift"]["E_g"] == 0.0


def test_signed_replay_repeats_complete_support_per_epoch(tmp_path):
    torch.manual_seed(19)
    recent = _recent(tmp_path)
    policy = GPS.build_sfm_policy(width=16, res_dropout=0.0)
    initial = copy.deepcopy(policy)
    BS.configure_expansion_trainability(policy)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=R.LEARNING_RATE,
    )
    cfg = R.ExperimentConfig(alpha=0.1, replay_epochs=10)
    report = R.repeat_complete_replay(
        policy, optimizer, recent, cfg, device="cpu", round_i=2,
    )
    assert report["optimizer_steps"] == 10
    assert report["positive_total_visits"] == 10 * report["positive_eligible"]
    assert report["negative_total_visits"] == 10 * report["negative_eligible"]
    assert all(row["positive_coverage"]["exact_once"] for row in report["epochs"])
    assert all(row["negative_coverage"]["exact_once"] for row in report["epochs"])
    assert all(row["gradient_cosine"] is not None for row in report["epochs"])
    assert report["module_relative_parameter_drift"]["E_g"] == 0.0
    assert report["visual_encoder_sha_before"] == report["visual_encoder_sha_after"]
    assert any(
        not torch.equal(value, initial.state_dict()[name])
        for name, value in policy.state_dict().items()
        if not name.startswith("enc_grid.")
    )


def test_signed_negative_only_audits_every_record_and_takes_no_step(tmp_path):
    torch.manual_seed(29)
    recent = _negative_only_recent(tmp_path)
    policy = GPS.build_sfm_policy(width=16, res_dropout=0.0)
    BS.configure_expansion_trainability(policy)
    initial = copy.deepcopy(policy.state_dict())
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=R.LEARNING_RATE,
    )
    cfg = R.ExperimentConfig(alpha=0.1, replay_epochs=10)
    report = R.repeat_complete_replay(
        policy, optimizer, recent, cfg, device="cpu", round_i=1,
    )
    assert report["positive_eligible"] == 0
    assert report["optimizer_steps"] == 0
    assert report["negative_total_visits"] == 10 * report["negative_eligible"]
    assert all(row["path"] == "signed_no_positive" for row in report["epochs"])
    assert all(row["negative_coverage"]["exact_once"] for row in report["epochs"])
    for name, value in policy.state_dict().items():
        assert torch.equal(value, initial[name])
