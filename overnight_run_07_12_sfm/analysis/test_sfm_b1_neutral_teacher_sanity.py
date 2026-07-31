import copy

import numpy as np
import pytest
import torch

import sfm_b1_neutral_teacher_sanity as N
import sfm_b1_store as BS


def _payload():
    records = []
    for index, gamma in enumerate((0.1, 1.0)):
        records.append({
            "neutral_id": index,
            "population": "D0",
            "semantic_label": "neutral",
            "round": 1,
            "scenario_id": 250_000 + index,
            "gamma": gamma,
            "step": index,
            "state": np.zeros(4, np.float32),
            "hp10": np.zeros((10, 16, 12), np.float32),
            "low5": np.asarray([1 + index, 0, 0, 0, gamma], np.float32),
            "hist": np.zeros((16, 2), np.float32),
            "ped_xy": np.zeros((40, 2), np.float32),
            "ped_vel": np.zeros((40, 2), np.float32),
            "controls": np.full((10, 2), 0.2 + index, np.float32),
            "x0": np.zeros(20, np.float32),
            "verifier_result": {"resolved": True, "y": 0},
            "verifier_y": 0,
            "train_eligible": False,
            "replay_default": False,
            "gp_eligible": False,
        })
    return {
        "status": "SFM_B1_NEUTRAL_ROUND_COMPLETE",
        "round": 1,
        "records": records,
        "summary": {"D0": len(records)},
    }


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_grid = torch.nn.Linear(1, 1, bias=False)
        self.enc_grid.requires_grad_(False)
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def ctx_from(self, grid, low, hist):
        return low[:, :1]

    def cfm_loss(self, controls, context, weights=None):
        target = controls.reshape(len(controls), -1).mean(dim=1)
        prediction = self.scale * context[:, 0]
        per = (prediction - target).square()
        return per.mean() if weights is None else (per * weights).mean()


def test_neutral_conversion_preserves_exact_negative_isolation():
    payload = _payload()
    original = copy.deepcopy(payload)
    holder, records = N._neutral_records(payload)
    assert len(holder.contexts) == len(records) == 2
    assert all(row["y"] == 0 for _, row in records)
    assert all(row["semantic_label"] == "neutral" for _, row in records)
    mass, accounting = BS.hierarchy_mass(records)
    assert sum(mass.values()) == pytest.approx(1.0)
    assert accounting["gamma"] == pytest.approx({"0.1": 0.5, "1.0": 0.5})
    for source in payload["records"]:
        assert not source["train_eligible"]
        assert not source["replay_default"]
        assert not source["gp_eligible"]
        assert source["verifier_y"] == 0
    for actual, expected in zip(payload["records"], original["records"]):
        assert actual.keys() == expected.keys()
        np.testing.assert_array_equal(actual["controls"], expected["controls"])
        np.testing.assert_array_equal(actual["x0"], expected["x0"])


def test_neutral_update_uses_whole_support_once_per_inner_step():
    _, records = N._neutral_records(_payload())
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(
        [policy.scale], lr=1.0e-2
    )
    before = float(policy.scale.detach())
    report = N._neutral_update(
        policy,
        optimizer,
        records,
        steps=4,
        batch=1,
        device="cpu",
        seed=7,
    )
    assert report["optimizer_steps"] == 4
    assert report["sample_exposures"] == 8
    assert report["exact_once_per_inner_step"]
    assert report["stored_x0_used_for_training"] is False
    assert len(report["losses"]) == 4
    assert float(policy.scale.detach()) != before
    assert all(row["y"] == 0 for _, row in records)


def test_neutral_conversion_rejects_relabeling():
    payload = _payload()
    payload["records"][0]["verifier_y"] = 1
    with pytest.raises(RuntimeError, match="D0 semantics"):
        N._neutral_records(payload)
