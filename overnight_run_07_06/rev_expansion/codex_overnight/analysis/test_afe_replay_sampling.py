from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_core as AC
import grid_expand_afe2 as AFE2


def _imbalanced_store() -> AC.DStore:
    store = AC.DStore()
    # round, gamma, episode(replica), number of positive queries at one context
    cells = (
        (1, 0.1, 0, 100),
        (1, 0.1, 1, 1),
        (1, 0.5, 2, 1),
        (2, 0.1, 0, 1),
        (2, 0.5, 2, 1),
    )
    for query_round, gamma, episode, count in cells:
        context_id = len(store.ctx_meta)
        store.ctx_meta.append((query_round, episode, 0))
        for _ in range(count):
            query_id = len(store.q_sid)
            store.q_sid.append(context_id)
            store.q_round.append(query_round)
            store.q_gamma.append(gamma)
            store.pos_ids.append(query_id)
    return store


def test_query_uniform_positive_replay_preserves_the_legacy_rng_draw() -> None:
    store = _imbalanced_store()
    population = list(store.pos_ids)
    expected_rng = np.random.default_rng(17)
    expected = [
        population[index]
        for index in expected_rng.integers(0, len(population), 200)
    ]

    actual = store.sample_positive_ids(
        200,
        np.random.default_rng(17),
        eligible_ids=population,
    )

    assert actual == expected


def test_hierarchical_positive_replay_neutralizes_query_count_dominance() -> None:
    store = _imbalanced_store()
    population = list(store.pos_ids)
    hierarchy = store.positive_replay_hierarchy(eligible_ids=population)

    draws = store.sample_positive_ids(
        20_000,
        np.random.default_rng(23),
        eligible_ids=population,
        sampling="round_gamma_replica_context",
        hierarchy=hierarchy,
    )

    rounds = np.asarray([store.q_round[query_id] for query_id in draws])
    # The first context owns 100/104 queries, but only one of the equally sampled
    # round->gamma->replica->context leaves.
    dominant = np.mean(np.asarray(draws) < 100)
    assert np.mean(rounds == 1) == pytest.approx(0.5, abs=0.02)
    assert dominant == pytest.approx(0.125, abs=0.02)


def test_hierarchical_positive_replay_respects_the_eligible_window() -> None:
    store = _imbalanced_store()
    eligible = [
        query_id for query_id in store.pos_ids
        if store.q_round[query_id] == 2
    ]
    hierarchy = store.positive_replay_hierarchy(eligible_ids=eligible)

    draws = store.sample_positive_ids(
        100,
        np.random.default_rng(9),
        eligible_ids=eligible,
        sampling="round_gamma_replica_context",
        hierarchy=hierarchy,
    )

    assert {store.q_round[query_id] for query_id in draws} == {2}
    assert set(draws).issubset(set(eligible))


def test_hierarchical_positive_replay_rejects_query_context_round_mismatch() -> None:
    store = _imbalanced_store()
    store.ctx_meta[0] = (99, 0, 0)
    with pytest.raises(RuntimeError, match="query/context round mismatch"):
        store.positive_replay_hierarchy(eligible_ids=store.pos_ids)


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.u_max = 1.0
        self.d = 2

    def module_groups(self):
        return {"trunk": self}

    def ctx_from(self, grid, low, hist):
        del grid, hist
        return low[:, :1]

    def _expand_ctx(self, context, count):
        assert len(context) == count
        return context

    def forward(self, values, time, context):
        del time
        return values + self.weight + context

    def cfm_loss(self, controls, context):
        target = controls.mean(dim=1)
        return (self.weight + context - target).square().mean()


def _trainable_store() -> AC.DStore:
    store = AC.DStore()
    for query_round, gamma, episode in ((1, 0.1, 0), (1, 0.5, 1), (2, 0.1, 0)):
        context_id = len(store.ctx_meta)
        store.ctx_meta.append((query_round, episode, 0))
        store.ctx_hp.append(np.zeros((1, 2, 2), np.float32))
        store.ctx_low5.append(np.asarray((1.0, 0.0, 0.0, 0.0, gamma), np.float32))
        store.ctx_hist.append(np.zeros((1, 2), np.float32))
        query_id = len(store.q_sid)
        store.q_sid.append(context_id)
        store.q_round.append(query_round)
        store.q_gamma.append(gamma)
        store.q_U.append(np.asarray([[1.0, 0.0]], np.float32))
        store.pos_ids.append(query_id)
    return store


def test_update_round_routes_the_hierarchical_replay_option(monkeypatch) -> None:
    store = _trainable_store()
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    cfg = SimpleNamespace(
        arm="afe",
        replay_window=2,
        replay_sampling="round_gamma_replica_context",
        batch=2,
        afe_steps=1,
        grad_clip=0.0,
    )
    original = store.sample_pos
    calls = []

    def sample_spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "sample_pos", sample_spy)
    result = AFE2.update_round(
        policy,
        optimizer,
        store,
        cfg,
        torch.device("cpu"),
        np.random.default_rng(3),
        round_i=2,
    )

    assert result["replay_sampling"] == "round_gamma_replica_context"
    assert calls[0]["sampling"] == "round_gamma_replica_context"
    assert calls[0]["hierarchy"]
