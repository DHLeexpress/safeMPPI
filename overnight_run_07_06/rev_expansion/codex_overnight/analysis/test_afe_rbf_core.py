from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_rbf_core as RC


def test_rbf_sigma_is_small_on_buffer_and_large_for_a_distant_feature() -> None:
    gp = RC.RBFGPSigma(lengthscale=0.25, lam=1.0e-4)
    buffer = torch.tensor([[1.0, 0.0], [0.98, 0.2]], dtype=torch.float32)
    gp.set_buffer(buffer)
    values = gp.sigma(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    assert float(values[0]) < 0.02
    assert float(values[1]) > 0.95
    assert gp.diagnostics()["kernel_effective_rank"] > 1.0


def test_lengthscale_is_mean_pairwise_normalized_distance() -> None:
    features = torch.tensor([[2.0, 0.0], [0.0, 3.0], [-4.0, 0.0]])
    expected = (np.sqrt(2.0) + 2.0 + np.sqrt(2.0)) / 3.0
    assert RC.mean_pairwise_lengthscale(features) == pytest.approx(expected)


def test_batch_conditional_variance_penalizes_near_duplicates() -> None:
    gp = RC.RBFGPSigma(lengthscale=0.25, lam=1.0e-3)
    candidates = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    conditional = gp.conditional_variance(candidates)

    assert float(conditional[0]) == pytest.approx(float(conditional[1]), rel=1.0e-5)
    assert float(conditional[0]) < 0.01
    assert float(conditional[2]) > 0.9


class _Store:
    def __init__(self):
        self.pos_ids = list(range(36))
        self.q_round = [1] * 28 + [2] * 8
        self.q_gamma = [0.1] * 20 + [0.5] * 8 + [0.1] * 4 + [0.5] * 4


def test_previous_round_buffer_is_capped_balanced_and_round_local() -> None:
    store = _Store()
    selected = RC.previous_round_positive_ids(
        store, round_i=1, cap=10, gammas=(0.1, 0.5), seed=7
    )
    selected_gamma = [store.q_gamma[index] for index in selected]

    assert len(selected) == len(set(selected)) == 10
    assert all(store.q_round[index] == 1 for index in selected)
    assert selected_gamma.count(0.1) == 5
    assert selected_gamma.count(0.5) == 5
