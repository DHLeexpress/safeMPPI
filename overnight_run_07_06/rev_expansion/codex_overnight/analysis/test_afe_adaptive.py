from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_adaptive as AD
import afe_core as AC


def test_positive_replay_window_preserves_archive_and_limits_population() -> None:
    store = AC.DStore()
    store.pos_ids = list(range(12))
    store.q_round = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

    assert store.positive_ids() == list(range(12))
    assert store.positive_ids(round_i=6, replay_window=1) == [10, 11]
    assert store.positive_ids(round_i=6, replay_window=5) == list(range(2, 12))
    assert store.pos_ids == list(range(12))


def test_round_local_beta_calibration_hits_requested_target() -> None:
    generator = torch.Generator().manual_seed(7)
    buffer = torch.randn(64, 8, generator=generator)
    pools = [torch.randn(64, 8, generator=generator) for _ in range(4)]
    from afe_rbf_core import RBFGPSigma

    gp = RBFGPSigma(lengthscale=0.7, lam=1.0e-2)
    gp.set_buffer(buffer)
    cfg = SimpleNamespace(K=64, B=8, seed=910)
    result = AD.calibrate_from_pools(gp, pools, cfg, round_i=3, target=0.5)

    assert result["target"] == 0.5
    assert result["solution"]["achieved"]["ess_med"] == pytest.approx(
        0.5, abs=1.0e-4
    )
    assert result["verifier_queries"] == 0


def test_rbf_counterfactual_sweep_is_read_only_and_complete() -> None:
    generator = torch.Generator().manual_seed(11)
    buffer = torch.randn(512, 8, generator=generator)
    before = buffer.clone()
    pools = [torch.randn(64, 8, generator=generator) for _ in range(3)]
    cfg = SimpleNamespace(K=64, B=8, seed=910, gp_lam=1.0e-2)

    rows = AD.rbf_counterfactual_sweep(
        pools,
        buffer,
        cfg,
        round_i=2,
        target=0.5,
        lengthscale=0.7,
    )

    assert len(rows) == 6
    assert {(row["cap"], row["lengthscale_multiplier"]) for row in rows} == {
        (cap, multiplier)
        for cap in (128, 512)
        for multiplier in (0.5, 1.0, 2.0)
    }
    assert all(row["achieved"]["ess_med"] == pytest.approx(0.5, abs=1.0e-4)
               for row in rows)
    assert torch.equal(buffer, before)
