import json

import numpy as np
import pytest
import torch

import claude_corrected_distill as CD
import grid_policy_sfm as GPS
import sfm_b1_offline_store as OS


def _result(y):
    return dict(resolved=True, y=int(y), taskspace=bool(y),
                collision_free=bool(y), certificate=bool(y), full_h=True,
                terminal_step=10, diagnostics={"m": 1})


def _shard(round_i, positive=6, negative=2, base=100):
    shard = OS.ExecutedRoundShard(round_i)
    gammas = (0.1, 0.3, 0.5, 1.0)
    rng = np.random.default_rng(round_i)
    for index in range(positive + negative):
        context_id = shard.add_context(
            scenario_id=base + index % 3, gamma=gammas[index % 4],
            step=index, state=rng.normal(size=4).astype(np.float32),
            hp10=rng.normal(size=(10, 16, 12)).astype(np.float32),
            low5=rng.normal(size=5).astype(np.float32),
            hist=rng.normal(size=(16, 2)).astype(np.float32),
            ped_xy=np.ones((2, 2), np.float32) * 3,
            ped_vel=np.zeros((2, 2), np.float32),
        )
        shard.add_executed_window(
            context_id,
            rng.uniform(-1, 1, size=(10, 2)).astype(np.float32),
            np.zeros(20, np.float32), _result(index < positive),
            execution_source="selected_B", nvp_context=False,
            candidate_id=0, acquisition_step=0, sigma=0.1, hp_margin=0.1,
            mode="U",
        )
    return shard


def test_w2_holder_fail_closed():
    with pytest.raises(RuntimeError, match="W=2 requires BOTH"):
        CD.CorrectedRecent([_shard(1)])
    with pytest.raises(RuntimeError, match="distinct rounds"):
        CD.CorrectedRecent([_shard(1), _shard(1, base=200)])
    recent = CD.CorrectedRecent([_shard(1), _shard(2, base=200)])
    assert len(recent.positive_records()) == 12
    assert len(recent.negative_records()) == 4


def test_production_policy_chunked_gradient_matches_direct():
    torch.manual_seed(0)
    policy = GPS.build_sfm_policy(device="cpu")
    policy.train()
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)
    shard = _shard(3, positive=9, negative=0)
    records = [(shard, row) for row in shard.Dplus]
    mass, _ = CD.normalized_teacher_mass(records)
    batch, seed = 4, 777

    # production chunked-accumulated path
    policy.zero_grad(set_to_none=True)
    chunked_total = CD.teacher_epoch_loss(
        policy, records, mass, batch=batch, device="cpu", seed=seed,
        backward=True,
    )
    chunked_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }

    # independent direct implementation replaying the identical chunk noise
    policy.zero_grad(set_to_none=True)
    direct = CD.direct_teacher_loss(
        policy, records, mass, batch=batch, device="cpu", seed=seed,
    )
    direct.backward()
    direct_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }

    assert chunked_total == pytest.approx(float(direct.detach()), abs=1e-6)
    assert set(chunked_gradients) == set(direct_gradients)
    worst = max(
        float((chunked_gradients[name] - direct_gradients[name]).abs().max())
        for name in chunked_gradients
    )
    assert worst < 1e-5, f"gradient disagreement {worst}"
    # the objective actually moves parameters (non-trivial gradients)
    assert max(
        float(g.abs().max()) for g in chunked_gradients.values()
    ) > 0.0


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
        if weights is not None:
            per = per * weights
        return per.mean()

    def module_groups(self):
        return {"E_g": self.enc_grid, "head": self.head}


def test_corrected_ordinary_is_one_adam_step_per_epoch():
    recent = CD.CorrectedRecent([_shard(1), _shard(2, base=200)])
    policy = _TinyPolicy()
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3,
    )
    out = CD.corrected_ordinary_epochs(
        policy, optimizer, recent, alpha=0.01, epochs=3, batch=4,
        device="cpu", seed=9,
    )
    assert out["adam_steps"] == 3
    assert all(row["optimizer_steps"] == 1 for row in out["per_epoch"])


def test_pooled_by_label_reads_the_right_record(tmp_path):
    def _cell(sr):
        return dict(summary=dict(
            pooled=dict(
                SR=sr, CR=0.2, timeout=0.0,
                Validity=dict(mean=0.7),
                successful_clearance=dict(mean=0.1),
                successful_time_to_goal=dict(mean=9.0),
            ),
            per_gamma={
                g: dict(successful_clearance=dict(mean=0.1),
                        successful_time_to_goal=dict(mean=9.0))
                for g in ("0.1", "1.0")
            },
        ))
    payload = dict(records=[
        dict(label="r0", cell=_cell(0.5)),
        dict(label="r1", cell=_cell(0.9)),
    ])
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    assert CD.pooled_by_label(path, "r1")["SR"] == 0.9
    assert CD.pooled_by_label(path, "r0")["SR"] == 0.5
    with pytest.raises(KeyError):
        CD.pooled_by_label(path, "r7")
