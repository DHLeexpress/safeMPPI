import numpy as np
import pytest
import torch

import claude_continuation as CC
import sfm_b1_offline_store as OS


def _result(y):
    return dict(resolved=True, y=int(y), taskspace=bool(y),
                collision_free=bool(y), certificate=bool(y), full_h=True,
                terminal_step=10, diagnostics={"m": 1})


def _shard(positive=9, negative=4):
    shard = OS.ExecutedRoundShard(2)
    gammas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    for index in range(positive + negative):
        context_id = shard.add_context(
            scenario_id=500 + index % 4, gamma=gammas[index % 7],
            step=index, state=np.zeros(4, np.float32),
            hp10=np.zeros((10, 16, 12), np.float32),
            low5=np.zeros(5, np.float32),
            hist=np.zeros((16, 2), np.float32),
            ped_xy=np.ones((2, 2), np.float32) * 3,
            ped_vel=np.zeros((2, 2), np.float32),
        )
        shard.add_executed_window(
            context_id, np.full((10, 2), 0.1 * index, np.float32),
            np.zeros(20, np.float32), _result(index < positive),
            execution_source="selected_B", nvp_context=False,
            candidate_id=0, acquisition_step=0, sigma=0.1, hp_margin=0.1,
            mode="U",
        )
    return shard


def _anchor(n=6):
    records = []
    gammas = (0.1, 0.5, 1.0)
    for index in range(n):
        records.append(dict(
            episode=380000 + index, gamma=gammas[index % 3], step=index,
            context=dict(
                state=np.zeros(4, np.float32),
                hp10=np.zeros((10, 16, 12), np.float32),
                low5=np.zeros(5, np.float32),
                hist=np.zeros((16, 2), np.float32),
                ped_xy=np.ones((2, 2), np.float32) * 3,
                ped_vel=np.zeros((2, 2), np.float32),
            ),
            controls=np.full((10, 2), 0.5, np.float32),
        ))
    return CC.AnchorShard(dict(records=records))


def test_interleave_covers_exactly_once_and_is_deterministic():
    a = [f"a{i}" for i in range(5)]
    b = [f"b{i}" for i in range(2)]
    c = [f"c{i}" for i in range(9)]
    first = CC._interleave([a, b, c])
    second = CC._interleave([a, b, c])
    assert first == second
    assert sorted(first) == sorted(a + b + c)


def test_positive_mass_shares_sum_to_declared_composition():
    shard = _shard()
    anchor = _anchor()
    recovery = [dict(
        context_id=0, controls=np.full((10, 2), 0.2, np.float32), y=1,
        query_id=0, execution_source="synthetic_certified_recovery",
        nvp_context=False, candidate_id=None, acquisition_step=None,
        sigma=None, hp_margin=None, mode="recovery",
        verifier_diagnostics={},
    )]
    for anchor_mass in CC.ANCHOR_GRID:
        populations = CC.build_positive_populations(
            shard, recovery, anchor, anchor_mass,
        )
        weights, membership, total = CC.build_weights(
            populations, [(shard, r) for r in shard.Dminus],
        )
        sums = {}
        for name, _, records, share in populations:
            s = sum(
                weights[(id(h), int(r["query_id"]))] for h, r in records
            )
            sums[name] = s / total
            assert s / total == pytest.approx(share, abs=1e-9)
        assert sum(sums.values()) == pytest.approx(1.0, abs=1e-9)
        assert sums["recovery"] == pytest.approx(0.05, abs=1e-9)
        assert sums["anchor"] == pytest.approx(anchor_mass, abs=1e-9)


def test_epoch_batches_exact_once_positive_seeded_deterministic():
    shard = _shard()
    anchor = _anchor()
    populations = CC.build_positive_populations(shard, [], anchor, 0.25)
    negatives = [(shard, r) for r in shard.Dminus]
    first = CC.epoch_batches(
        populations, negatives, batch=4, seed=7, epoch=0,
    )
    second = CC.epoch_batches(
        populations, negatives, batch=4, seed=7, epoch=0,
    )
    def ids(batches):
        return [[(id(h), r["query_id"]) for h, r in b] for b in batches]
    assert ids(first) == ids(second)
    flat = [x for b in ids(first) for x in b]
    assert len(flat) == len(set(flat))
    total = sum(len(records) for _, _, records, _ in populations) \
        + len(negatives)
    assert len(flat) == total
    weights, membership, _ = CC.build_weights(populations, negatives)
    for b in first:
        assert any(
            membership[(id(h), int(r["query_id"]))] != "negative"
            for h, r in b
        )
    # a different epoch produces a different (but deterministic) order
    third = CC.epoch_batches(
        populations, negatives, batch=4, seed=7, epoch=1,
    )
    assert ids(third) != ids(first)


def test_admissibility_hard_gates():
    r1 = dict(SR=0.7, CR=0.3, timeout=0.0, Validity=0.7, clearance=0.12,
              time=10.0, g01=dict(clearance=0.15, time=13.0),
              g10=dict(clearance=0.11, time=10.0))
    good = dict(r1)
    ok, checks = CC.admissible(good, r1)
    assert ok, checks
    worse_cr = dict(r1, CR=0.31)
    assert not CC.admissible(worse_cr, r1)[0]
    worse_v = dict(r1, Validity=0.699)
    assert not CC.admissible(worse_v, r1)[0]
    sr_within_tol = dict(r1, SR=0.66)
    assert CC.admissible(sr_within_tol, r1)[0]
    sr_beyond = dict(r1, SR=0.64)
    assert not CC.admissible(sr_beyond, r1)[0]
    bad_trend = dict(r1, g01=dict(clearance=0.10, time=13.0))
    assert not CC.admissible(bad_trend, r1)[0]
    no_success_cell = dict(r1, g01=dict(clearance=None, time=None))
    assert not CC.admissible(no_success_cell, r1)[0]


class _TinyPolicy(torch.nn.Module):
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


def test_continuation_replay_runs_and_logs_population_losses():
    shard = _shard()
    anchor = _anchor()
    policy = _TinyPolicy()
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3,
    )
    populations = CC.build_positive_populations(shard, [], anchor, 0.5)
    negatives = [(shard, r) for r in shard.Dminus]
    before = policy.head.weight.detach().clone()
    log = CC.continuation_replay(
        policy, optimizer, populations, negatives, epochs=2, batch=4,
        seed=3, device="cpu",
    )
    assert log["steps"] > 0
    assert log["unique_positive"] == len(shard.Dplus) + len(anchor.windows)
    assert log["losses"]["anchor"] is not None
    assert log["losses"]["new"] is not None
    assert not torch.equal(before, policy.head.weight)
    assert torch.equal(
        _TinyPolicy().enc_grid.weight * 0 + policy.enc_grid.weight,
        policy.enc_grid.weight,
    )
