from types import SimpleNamespace
import os
import sys

import pytest
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import grid_expand_afe_ensemble as E


class FakeStore:
    def __init__(self, labels, gammas):
        self.q_y = list(labels)
        self.q_gamma = list(gammas)

    def __len__(self):
        return len(self.q_y)


def config():
    return SimpleNamespace(
        ensemble_members=3,
        ensemble_hidden=5,
        ensemble_dropout=0.1,
        ensemble_train_fraction=0.9,
        ensemble_lr=1.0e-3,
        ensemble_steps=3,
        ensemble_early_window=1,
        gammas=(0.1, 0.5),
        seed=910,
    )


def test_refit_uses_every_successful_query_and_both_labels(monkeypatch):
    store = FakeStore(
        labels=[0, 1, 1, 0, 1, 0],
        gammas=[0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
    )
    features = torch.randn(len(store), 32)

    def embed_queries(_policy, _store, _cfg, _device, ids):
        assert ids == list(range(len(store)))
        return features

    monkeypatch.setattr(E.AFE2, "embed_queries", embed_queries)
    estimator, diagnostics = E._fit_estimator(None, store, config(), "cpu", round_i=2)
    assert estimator.n == len(store)
    assert diagnostics["positive_fraction"] == pytest.approx(0.5)
    assert diagnostics["per_gamma_labels"]["0.1"] == {
        "total": 3, "positive": 2, "negative": 1,
    }
    assert diagnostics["per_gamma_labels"]["0.5"] == {
        "total": 3, "positive": 1, "negative": 2,
    }


def test_constant_label_archive_matches_reference_behavior(monkeypatch):
    store = FakeStore(labels=[1, 1, 1], gammas=[0.1, 0.1, 0.5])
    monkeypatch.setattr(
        E.AFE2, "embed_queries", lambda *_args, **_kwargs: torch.randn(3, 32)
    )
    _, diagnostics = E._fit_estimator(None, store, config(), "cpu", round_i=1)
    assert diagnostics["positive_fraction"] == 1.0
    assert diagnostics["label_unique_count"] == 1
    assert diagnostics["label_std"] == 0.0


def test_refit_seed_is_reproducible_and_restores_global_rng(monkeypatch):
    store = FakeStore(
        labels=[0, 1, 1, 0, 1, 0],
        gammas=[0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
    )
    features = torch.arange(len(store) * 32, dtype=torch.float32).reshape(len(store), 32)
    monkeypatch.setattr(
        E.AFE2, "embed_queries", lambda *_args, **_kwargs: features.clone()
    )
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first, _ = E._fit_estimator(None, store, config(), "cpu", round_i=3)
    after = torch.random.get_rng_state().clone()
    second, _ = E._fit_estimator(None, store, config(), "cpu", round_i=3)
    query = torch.randn(4, 32)
    assert torch.equal(before, after)
    assert torch.equal(first.sigma(query), second.sigma(query))
