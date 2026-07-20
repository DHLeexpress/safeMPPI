from __future__ import annotations

import torch

from afe_restart.stage3_low7_pretrain import (
    GAMMAS,
    Low7Pool,
    _canonical_gamma,
    _objective_weights,
    paired_split,
    polar_reflection_indices,
    reflection_paired_cfm_terms,
    reflect_low7_batch,
)


def _pool() -> Low7Pool:
    pairs = 4
    gamma = torch.tensor(
        [value for value in GAMMAS for _ in range(pairs)], dtype=torch.float64
    )
    pair_ids = torch.tensor(list(range(pairs)) * len(GAMMAS), dtype=torch.long)
    trajectory_ids = torch.arange(len(gamma), dtype=torch.long)
    count = len(gamma)
    return Low7Pool(
        grid=torch.zeros(count, 3, 32, 32),
        low7=torch.zeros(count, 7),
        hist=torch.zeros(count, 16, 2),
        plans=torch.zeros(count, 10, 2),
        gamma=gamma,
        pair_ids=pair_ids,
        trajectory_ids=trajectory_ids,
        trajectory_weight=torch.ones(count, dtype=torch.float64),
        trajectory_rows=tuple(
            {"trajectory_id": index} for index in range(count)
        ),
        query_hashes=tuple("0" * 64 for _ in range(count)),
        declared_pair_ids=tuple(range(pairs)),
        source={},
    )


def test_pair_split_is_disjoint_across_every_gamma() -> None:
    pool = _pool()
    train, validation, audit = paired_split(
        pool, validation_pairs=1, seed=17
    )

    train_pairs = set(pool.pair_ids[train].tolist())
    validation_pairs = set(pool.pair_ids[validation].tolist())
    assert train_pairs.isdisjoint(validation_pairs)
    assert audit["pair_leakage"] == 0
    for gamma in GAMMAS:
        entry = audit["per_gamma"][f"{gamma:g}"]
        assert entry == {
            "train_trajectories": 3,
            "validation_trajectories": 1,
        }


def test_objective_has_equal_gamma_mass() -> None:
    pool = _pool()
    rows = torch.arange(len(pool))
    weights = _objective_weights(pool, rows)

    masses = []
    for gamma in GAMMAS:
        mask = torch.isclose(
            pool.gamma, torch.tensor(gamma, dtype=torch.float64), atol=5e-7
        )
        masses.append(float(weights[mask].sum()))
    assert masses == [1.0] * len(GAMMAS)


def test_float32_gamma_is_canonicalized_before_identity_hashing() -> None:
    assert _canonical_gamma(float(torch.tensor(0.1, dtype=torch.float32))) == 0.1


def test_pair_split_uses_bank_before_success_missingness() -> None:
    pool = _pool()
    pool = Low7Pool(
        **{
            **pool.__dict__,
            "declared_pair_ids": tuple(range(6)),
        }
    )
    _train, _validation, audit = paired_split(pool, validation_pairs=2, seed=17)

    assert len(audit["train_pair_ids"]) == 4
    assert len(audit["validation_pair_ids"]) == 2
    assert set(audit["train_pairs_without_targets"]) | set(
        audit["validation_pairs_without_targets"]
    ) == {4, 5}


def test_low7_reflection_is_an_involution_and_preserves_gamma() -> None:
    grid = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(
        2, 3, 32, 32
    )
    low7 = torch.tensor(
        ((1, 2, 3, 4, 5, 6, 0.1), (7, 8, 9, 10, 11, 12, 1.0)),
        dtype=torch.float32,
    )
    hist = torch.arange(2 * 16 * 2, dtype=torch.float32).reshape(2, 16, 2)
    plans = torch.arange(2 * 10 * 2, dtype=torch.float32).reshape(2, 10, 2)

    reflected = reflect_low7_batch(grid, low7, hist, plans)
    restored = reflect_low7_batch(*reflected)

    for actual, expected in zip(restored, (grid, low7, hist, plans)):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(reflected[1][:, -1], low7[:, -1])
    assert sorted(polar_reflection_indices().tolist()) == list(range(32))


class _Velocity(torch.nn.Module):
    u_max = 1.0
    d = 20

    def __init__(self, bias: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer("bias", torch.tensor(bias).repeat(10))

    def ctx_from(self, grid, low7, hist):
        return torch.zeros(len(grid), 1, device=grid.device)

    def forward(self, x, tau, context):
        return x + self.bias


def test_direct_equivariance_term_detects_coordinate_bias() -> None:
    grid = torch.zeros(3, 3, 32, 32)
    low7 = torch.zeros(3, 7)
    hist = torch.zeros(3, 16, 2)
    plans = torch.randn(3, 10, 2)
    symmetric = reflection_paired_cfm_terms(
        _Velocity((0.0, 0.0)),
        grid,
        low7,
        hist,
        plans,
        generator=torch.Generator().manual_seed(5),
    )[1]
    biased = reflection_paired_cfm_terms(
        _Velocity((1.0, 0.0)),
        grid,
        low7,
        hist,
        plans,
        generator=torch.Generator().manual_seed(5),
    )[1]

    torch.testing.assert_close(symmetric, torch.zeros_like(symmetric))
    assert float(biased) > 0.5
