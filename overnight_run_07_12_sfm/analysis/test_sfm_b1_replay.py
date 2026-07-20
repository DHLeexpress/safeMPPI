import copy
import numpy as np
import torch

import grid_policy_sfm as GPS
import sfm_b1_store as S


def _result(y):
    return dict(
        resolved=True, y=y, taskspace=bool(y), collision_free=bool(y), certificate=bool(y),
        full_h=True, terminal_step=10, train_eligible=bool(y), segment=np.zeros((11, 2)),
        pedestrian_prediction=np.zeros((11, 1, 2)), diagnostics={},
    )


def _recent(tmp_path):
    recent = S.RecentRounds(tmp_path)
    for round_i in (1, 2):
        shard = S.RoundShard(round_i)
        for gamma in (.1, .5):
            for scenario in (10, 11):
                cid = shard.add_context(
                    scenario_id=scenario, gamma=gamma, step=0, state=np.zeros(4),
                    hp10=np.random.randn(10, 16, 12), low5=np.random.randn(5),
                    hist=np.random.randn(16, 2), ped_xy=np.zeros((1, 2)), ped_vel=np.zeros((1, 2)),
                )
                for q in range(3):
                    shard.add_resolved_query(cid, q, np.random.randn(10, 2), .2, _result(q != 2), acquisition_step=q)
        recent.append_and_save(shard)
    return recent


def test_exact_hierarchical_mass_and_no_duplicate_positives(tmp_path):
    recent = _recent(tmp_path)
    records = recent.positive_records()
    ordered = S.hierarchical_order(records, seed=9)
    identities = [(shard.round_i, row["query_id"]) for shard, row in ordered]
    assert len(identities) == len(set(identities)) == len(records)
    mass, diagnostics = S.hierarchy_mass(ordered)
    assert abs(diagnostics["total"] - 1) < 1e-12
    assert max(diagnostics["gamma"].values()) - min(diagnostics["gamma"].values()) < 1e-12


def test_alpha_zero_is_bitwise_positive_only_and_never_reads_negatives(tmp_path, monkeypatch):
    torch.manual_seed(11)
    recent = _recent(tmp_path)
    policy_a = GPS.build_sfm_policy(width=24, res_dropout=0.0)
    policy_b = copy.deepcopy(policy_a)
    S.configure_expansion_trainability(policy_a)
    S.configure_expansion_trainability(policy_b)
    optimizer_a = torch.optim.Adam([p for p in policy_a.parameters() if p.requires_grad], lr=1e-5)
    optimizer_b = torch.optim.Adam([p for p in policy_b.parameters() if p.requires_grad], lr=1e-5)
    torch.manual_seed(991)
    expected = S.positive_only_update(policy_a, optimizer_a, recent, batch=128, seed=4)
    monkeypatch.setattr(recent, "negative_records", lambda: (_ for _ in ()).throw(AssertionError("D- read")))
    torch.manual_seed(991)
    actual = S.signed_update(policy_b, optimizer_b, recent, alpha=0.0, batch=128, seed=4)
    assert expected["visited"] == actual["visited"]
    for left, right in zip(policy_a.state_dict().values(), policy_b.state_dict().values()):
        assert torch.equal(left, right)


def test_signed_visits_every_eligible_negative_once(tmp_path):
    torch.manual_seed(12)
    recent = _recent(tmp_path)
    policy = GPS.build_sfm_policy(width=24, res_dropout=0.0)
    S.configure_expansion_trainability(policy)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    report = S.signed_update(policy, optimizer, recent, alpha=.01, batch=128, seed=8)
    assert len(report["negative_visited"]) == report["negative_eligible"]
    assert len(set(report["negative_visited"])) == report["negative_eligible"]
    assert abs(report["negative_mass"]["total"] - 1) < 1e-12
