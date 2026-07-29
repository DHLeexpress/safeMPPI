import numpy as np
import pytest
import torch

import claude_afe_fidelity as AF


def test_golden_rule_gate():
    ok = dict(resolved=True, y=1, full_h=True, terminal_step=10)
    assert AF.certified(ok)
    for bad in (dict(ok, y=0), dict(ok, full_h=False),
                dict(ok, terminal_step=9), dict(ok, resolved=False)):
        assert not AF.certified(bad)


def _shard(counts_by_context, round_i=1):
    shard = AF.CertifiedQueryShard(round_i)
    gammas = (0.1, 0.5, 1.0)
    rng = np.random.default_rng(0)
    for c, (n_pos, js) in enumerate(counts_by_context):
        cid = shard.add_context(
            scenario_id=100 + c % 2, gamma=gammas[c % 3], step=c,
            state=np.zeros(4, np.float32),
            hp10=np.zeros((10, 16, 12), np.float32),
            low5=np.zeros(5, np.float32),
            hist=np.zeros((16, 2), np.float32),
            ped_xy=np.ones((1, 2), np.float32),
            ped_vel=np.zeros((1, 2), np.float32),
        )
        for j in range(n_pos):
            shard.add_query(
                cid, rng.uniform(-1, 1, (10, 2)).astype(np.float32),
                np.zeros(20, np.float32),
                dict(resolved=True, y=1, full_h=True, terminal_step=10),
                J=js[j], sigma=0.1, executed=(j == 0), source="pool",
            )
    return shard


def test_shard_rejects_uncertified_positive():
    shard = AF.CertifiedQueryShard(1)
    cid = shard.add_context(
        scenario_id=1, gamma=0.5, step=0, state=np.zeros(4, np.float32),
        hp10=np.zeros((10, 16, 12), np.float32),
        low5=np.zeros(5, np.float32), hist=np.zeros((16, 2), np.float32),
        ped_xy=np.ones((1, 2), np.float32),
        ped_vel=np.zeros((1, 2), np.float32),
    )
    with pytest.raises(ValueError):
        shard.add_query(
            cid, np.zeros((10, 2), np.float32), np.zeros(20, np.float32),
            dict(resolved=True, y=1, full_h=True, terminal_step=9),
            J=0.0, sigma=None, executed=False, source="pool",
        )


def test_tau_calibration_hits_median_ess_half():
    shard = _shard([(4, [0.0, 1.0, 2.0, 3.0]), (3, [0.0, 0.5, 4.0]),
                    (5, [0.0, 2.0, 2.5, 3.0, 9.0])])
    tau, ess = AF.calibrate_tau_j(shard)
    assert tau is not None
    assert ess == pytest.approx(0.5, abs=1e-3)


def test_objective_weights_sum_to_one_and_G_is_contextwise_gibbs():
    shard = _shard([(3, [0.0, 1.0, 2.0]), (2, [0.0, 5.0]), (1, [0.0])])
    for objective in ("U", "G"):
        records, weights, info = AF.objective_weights([shard], objective)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert info["mass_residual"] < 1e-9
    _, wU, _ = AF.objective_weights([shard], "U")
    _, wG, infoG = AF.objective_weights([shard], "G")
    rows0 = [r for r in shard.Dplus if r["context_id"] == 0]
    u_vals = [wU[(id(shard), r["query_id"])] for r in rows0]
    assert max(u_vals) == pytest.approx(min(u_vals))
    g_vals = [(r["J"], wG[(id(shard), r["query_id"])]) for r in rows0]
    g_sorted = sorted(g_vals)
    assert g_sorted[0][1] > g_sorted[-1][1]


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_grid = torch.nn.Linear(1, 1, bias=False)
        self.head = torch.nn.Linear(20, 20, bias=False)
        self.d, self.u_max = 20, 2.0

    def ctx_from(self, g, l, h):
        return l[:, :1]

    def forward(self, v, t, c):
        return self.head(v)

    def cfm_loss(self, controls, ctx, weights=None):
        per = (self.head(controls.reshape(len(controls), 20) / 2.0)
               - controls.reshape(len(controls), 20) / 2.0).square().mean(1)
        return (per * weights).mean() if weights is not None else per.mean()

    def module_groups(self):
        return {"head": self.head}


def test_certified_replay_one_adam_step_per_epoch():
    shard = _shard([(3, [0.0, 1.0, 2.0]), (2, [0.0, 5.0])])
    policy = _Tiny()
    policy.enc_grid.weight.requires_grad_(False)
    opt = torch.optim.Adam([policy.head.weight], lr=1e-3)
    info = AF.certified_replay(
        policy, opt, [shard], objective="G", epochs=4, batch=2,
        device="cpu", seed=1,
    )
    assert info["adam_steps"] == 4 and len(info["losses"]) == 4
