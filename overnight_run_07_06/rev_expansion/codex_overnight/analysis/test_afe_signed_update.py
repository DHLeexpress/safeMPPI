from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_signed_update as SU


def test_alpha_zero_delegates_before_inspecting_or_drawing_negatives(monkeypatch) -> None:
    expected = object()
    seen = {}

    def baseline(policy, opt, store, cfg, device, rng, round_i):
        seen["args"] = (policy, opt, store, cfg, device, rng, round_i)
        return expected

    monkeypatch.setattr(SU.AFE2, "update_round", baseline)

    class PositiveOnlyStore:
        @property
        def q_y(self):
            raise AssertionError("alpha=0 inspected the negative archive")

    arguments = (
        object(), object(), PositiveOnlyStore(), object(), torch.device("cpu"), object(), 7
    )
    result = SU.update_round_signed(*arguments, alpha=0.0)

    assert result is expected
    assert seen["args"] == arguments


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_grid = torch.nn.Linear(1, 1, bias=False)
        self.trunk = torch.nn.Linear(1, 2, bias=False)
        torch.nn.init.constant_(self.enc_grid.weight, 0.5)
        torch.nn.init.zeros_(self.trunk.weight)
        self.enc_grid.weight.requires_grad_(False)
        self.u_max = 1.0
        self.d = 2

    def module_groups(self):
        return {"E_g": self.enc_grid, "trunk": self.trunk}

    def ctx_from(self, grid, low, hist):
        del grid, hist
        ones = torch.ones((low.shape[0], 1), dtype=low.dtype, device=low.device)
        return low[:, :1] + self.enc_grid(ones)

    def _expand_ctx(self, context, count):
        assert context.shape[0] == count
        return context

    def forward(self, values, time, context):
        del time
        return 0.1 * values + self.trunk(context)

    def cfm_loss(self, controls, context):
        target = controls.mean(dim=1)
        return (self.trunk(context) - target).square().mean()


class _Store:
    def __init__(self):
        self.q_y = [1, 0, 1, 0, 1, 0, 1, 0]
        # Query 5 is a full-window negative with a certified terminal prefix;
        # it must remain neutral rather than enter signed negative replay.
        self.q_exec_y = [1, 0, 1, 0, 1, 1, 1, 0]
        # Only queries 5 and 6 came from terminal NVP contexts. Query 6 is also
        # full-H positive, exercising the deliberately overlapping task labels.
        self.q_nvp_negative = [0, 0, 0, 0, 0, 1, 1, 0]
        self.q_round = [1, 1, 2, 2, 3, 3, 4, 4]
        self.q_gamma = [0.1] * len(self.q_y)
        self.q_sid = list(range(len(self.q_y)))
        self.q_U = [
            np.asarray([[1.0, 1.0] if label else [-1.0, -1.0]], np.float32)
            for label in self.q_y
        ]
        self.ctx_low5 = [np.asarray([1.0], np.float32) for _ in self.q_y]
        self.ctx_hist = [np.zeros((1, 2), np.float32) for _ in self.q_y]
        self.pos_ids = [index for index, label in enumerate(self.q_y) if label]

    def positive_ids(self, *, round_i=None, replay_window=None):
        if replay_window is None:
            return list(self.pos_ids)
        first_round = max(1, int(round_i) - int(replay_window) + 1)
        return [
            query_id for query_id in self.pos_ids
            if first_round <= self.q_round[query_id] <= round_i
        ]

    def sample_pos(self, batch, rng, *, eligible_ids=None):
        population = self.pos_ids if eligible_ids is None else eligible_ids
        positions = rng.integers(0, len(population), batch)
        ids = [population[int(position)] for position in positions]
        return (*self._batch(ids), ids)

    def _batch(self, ids):
        sids = [self.q_sid[query_id] for query_id in ids]
        return (
            self.grid3_of(sids),
            torch.stack([torch.from_numpy(self.ctx_low5[sid]) for sid in sids]),
            torch.stack([torch.from_numpy(self.ctx_hist[sid]) for sid in sids]),
            torch.stack([torch.from_numpy(self.q_U[query_id]) for query_id in ids]),
        )

    def grid3_of(self, sids):
        return torch.zeros((len(sids), 3, 2, 2), dtype=torch.float32)


def test_signed_update_uses_separate_recent_batches_and_normalized_gradients() -> None:
    policy = _Policy()
    store = _Store()
    cfg = SimpleNamespace(
        arm="afe",
        replay_window=2,
        batch=2,
        afe_steps=2,
        grad_clip=0.25,
        seed=19,
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    encoder_before = policy.enc_grid.weight.detach().clone()

    result = SU.update_round_signed(
        policy,
        optimizer,
        store,
        cfg,
        torch.device("cpu"),
        np.random.default_rng(5),
        round_i=4,
        alpha=0.25,
        negative_rng=np.random.default_rng(11),
    )

    assert result["signed_active"] is True
    assert result["replay_eligible"] == 2
    assert result["negative_replay_eligible"] == 2
    assert sum(result["drawn_ids"].values()) == cfg.batch * cfg.afe_steps
    assert sum(result["negative_drawn_ids"].values()) == cfg.batch * cfg.afe_steps
    assert all(store.q_y[query_id] == 1 for query_id in result["drawn_ids"])
    assert all(
        store.q_nvp_negative[query_id] == 1
        for query_id in result["negative_drawn_ids"]
    )
    assert set(result["negative_drawn_ids"]).issubset({5, 6})
    assert all(store.q_round[query_id] in (3, 4) for query_id in result["drawn_ids"])
    assert all(
        store.q_round[query_id] in (3, 4)
        for query_id in result["negative_drawn_ids"]
    )
    assert torch.equal(policy.enc_grid.weight, encoder_before)
    assert result["positive_grad_norm_by_group"]["E_g"] == 0.0
    assert result["negative_grad_norm_by_group"]["E_g"] == 0.0
    assert result["post_clip_grad_norm"] <= cfg.grad_clip + 1.0e-6
    assert len(result["signed_step_diagnostics"]) == cfg.afe_steps
    for step in result["signed_step_diagnostics"]:
        expected_scaled = 0.25 * step["positive_grad_norm"]
        assert step["scaled_negative_grad_norm"] == pytest.approx(expected_scaled)
        assert -1.0 <= step["gradient_cosine"] <= 1.0
        assert step["rho"] > 0.0
