import os
import sys

import pytest
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import afe_ensemble_core as EC


def test_reference_architecture_and_fit_contract():
    torch.manual_seed(7)
    estimator = EC.DeepEnsembleSigma(
        feature_dim=4,
        max_steps=4,
        early_window=2,
        device="cpu",
    )
    features = torch.randn(20, 4)
    labels = torch.tensor([0.0, 1.0] * 10)
    diagnostics = estimator.fit(features, labels)

    assert len(estimator.models) == 5
    layers = list(estimator.models[0].net)
    assert [type(layer) for layer in layers] == [
        torch.nn.Linear,
        torch.nn.Dropout,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.Dropout,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]
    assert layers[0].in_features == 4
    assert layers[0].out_features == 100
    assert layers[3].in_features == 100
    assert layers[3].out_features == 100
    assert layers[6].out_features == 1
    assert diagnostics["n"] == 20
    assert diagnostics["positive_fraction"] == pytest.approx(0.5)
    assert len(diagnostics["member_steps"]) == 5


def test_sigma_is_population_std_of_raw_member_predictions():
    torch.manual_seed(11)
    estimator = EC.DeepEnsembleSigma(
        feature_dim=3,
        members=3,
        hidden_dim=5,
        max_steps=3,
        early_window=1,
    )
    features = torch.randn(12, 3)
    labels = torch.tensor([0.0] * 6 + [1.0] * 6)
    estimator.fit(features, labels)
    query = torch.randn(7, 3)
    normalized = EC.l2_normalize(query)
    direct = torch.stack([
        model(normalized).squeeze(-1) for model in estimator.models
    ]).std(dim=0, correction=0)
    assert torch.allclose(estimator.sigma(query), direct)


def test_acquisition_is_without_replacement_and_has_no_fake_conditioning():
    torch.manual_seed(19)
    estimator = EC.DeepEnsembleSigma(
        feature_dim=3,
        members=3,
        hidden_dim=5,
        max_steps=3,
        early_window=1,
    )
    fit_features = torch.randn(16, 3)
    labels = torch.tensor([0.0, 1.0] * 8)
    estimator.fit(fit_features, labels)
    candidates = torch.randn(8, 3)
    order = torch.tensor([3, 1, 7, 0, 2, 4, 5, 6])
    vectors = estimator.sequential_score_vectors(candidates, order, steps=3)
    full = estimator.sigma(candidates)
    assert torch.allclose(vectors[0], full)
    assert torch.allclose(vectors[1], full[torch.tensor([0, 1, 2, 4, 5, 6, 7])])

    selected, trace = estimator.sequential_acquire(candidates, steps=5, beta=0.1)
    assert len(selected) == len(set(selected)) == 5
    assert [len(row["scores"]) for row in trace] == [8, 7, 6, 5, 4]


def test_unfit_estimator_is_explicitly_uniform_and_cannot_guide():
    estimator = EC.DeepEnsembleSigma(feature_dim=3)
    query = torch.randn(4, 3)
    assert torch.equal(estimator.sigma(query), torch.zeros(4))
    with pytest.raises(RuntimeError, match="fitted"):
        estimator.sequential_acquire(query, steps=2, beta=0.1)


def test_estimator_state_roundtrip_preserves_predictions():
    torch.manual_seed(29)
    estimator = EC.DeepEnsembleSigma(
        feature_dim=3,
        members=3,
        hidden_dim=5,
        max_steps=3,
        early_window=1,
    )
    features = torch.randn(14, 3)
    labels = torch.tensor([0.0, 1.0] * 7)
    estimator.fit(features, labels)
    query = torch.randn(6, 3)
    before_mean, before_sigma = estimator.mean_and_sigma(query)

    restored = EC.DeepEnsembleSigma.from_state_dict(estimator.state_dict())
    after_mean, after_sigma = restored.mean_and_sigma(query)
    assert torch.equal(before_mean, after_mean)
    assert torch.equal(before_sigma, after_sigma)
    assert restored.diagnostics() == estimator.diagnostics()
