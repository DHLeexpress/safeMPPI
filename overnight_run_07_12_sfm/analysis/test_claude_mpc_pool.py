import numpy as np
import torch

import claude_mpc_pool as MP
import sfm_b1_offline_store as OS
import sfm_scene as SS


def _result_positive():
    return dict(
        resolved=True, y=1, taskspace=True, collision_free=True,
        certificate=True, full_h=True, terminal_step=10,
        diagnostics={"slack": 0.1},
    )


def _record_episode_shard(scenario=123, gamma=0.5, steps=4, n_ped=6):
    """Simulate a short episode and store its exact contexts/windows."""
    speed_range = (1.0, 2.0)
    humans = SS.make_humans(scenario, 0, n_ped, speed_range)
    state = np.zeros(4, np.float32)
    shard = OS.ExecutedRoundShard(1)
    rng = np.random.default_rng(0)
    for step in range(steps):
        ped_xy, ped_vel = SS.collect_humans(humans)
        context_id = shard.add_context(
            scenario_id=scenario, gamma=gamma, step=step, state=state.copy(),
            hp10=np.zeros((10, 16, 12), np.float32),
            low5=np.zeros(5, np.float32),
            hist=np.zeros((16, 2), np.float32),
            ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
        )
        controls = np.tile(
            rng.uniform(-0.5, 0.5, size=(1, 2)).astype(np.float32), (10, 1),
        )
        shard.add_executed_window(
            context_id, controls, np.zeros(20, np.float32),
            _result_positive(), execution_source="selected_B",
            nvp_context=False, candidate_id=0, acquisition_step=0,
            sigma=0.1, hp_margin=0.1, mode="U",
        )
        action = controls[0]
        state[:2] = state[:2] + SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
        state[2:4] = state[2:4] + SS.DT * action
        SS.advance_humans(humans, state)
    return shard


def test_prefix_replay_reconstructs_stored_context_exactly():
    shard = _record_episode_shard()
    environment = dict(n_ped=6, ped_speed_range=(1.0, 2.0))
    for target_step in (0, 2, 3):
        context = next(
            c for c in shard.contexts if int(c["step"]) == target_step
        )
        humans, state = MP.replay_prefix_humans(
            shard, 123, 0.5, target_step, environment,
        )
        ped_xy, ped_vel = SS.collect_humans(humans)
        assert np.allclose(ped_xy, context["ped_xy"], atol=1e-6)
        assert np.allclose(ped_vel, context["ped_vel"], atol=1e-6)
        assert np.allclose(state, context["state"], atol=1e-6)


def test_privileged_config_matches_codex_source():
    cfg = MP.privileged_sfm_config()
    assert cfg.exact_sfm_step_filter is True
    assert cfg.step_filter_margin == 0.22
    assert cfg.step_filter_goal_plans == 12
    assert cfg.step_filter_avoid_plans == 18
    assert cfg.safe_coef_by_gamma == (1.0, 0.3, 1.0, 0.3, 0.3, 0.3, 0.1)


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
        if weights is None:
            return per.mean()
        return (per * weights).sum() / weights.sum()


def test_distill_block_trains_only_on_mpc_records_and_moves_params():
    shard = _record_episode_shard(steps=3)
    records = [
        dict(
            context_id=int(c["context_id"]), y=1, query_id=i,
            controls=np.full((10, 2), 0.3, np.float32),
            source="codex_privileged_mpc_pool", rank=0,
            privileged_clearance=0.3, privileged_margin=0.22,
            safemppi_cost=1.0, verifier_diagnostics={},
        )
        for i, c in enumerate(shard.contexts)
    ]
    policy = _TinyPolicy()
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3,
    )
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    result = MP.distill_block(
        policy, optimizer, shard, records, epochs=2, batch=2, seed=5,
    )
    assert result["steps"] == 2 * 2  # ceil(3/2)=2 batches x 2 epochs
    assert result["records"] == 3
    assert not torch.equal(before["head.weight"], policy.state_dict()["head.weight"])
    assert torch.equal(before["enc_grid.weight"], policy.state_dict()["enc_grid.weight"])
    # empty buffer is a no-op
    empty = MP.distill_block(
        policy, optimizer, shard, [], epochs=2, batch=2, seed=5,
    )
    assert empty["steps"] == 0
