import copy

import numpy as np
import torch

import claude_offline_aug as AUG
import sfm_b1_offline_replay as OR
import sfm_b1_offline_store as OS
import sfm_metrics2 as SM


def _result(y):
    return dict(
        resolved=True, y=int(y), taskspace=bool(y), collision_free=bool(y),
        certificate=bool(y), full_h=True, terminal_step=10,
        diagnostics={"margin": 0.25},
    )


def _context(shard, *, scenario, gamma, step, state, ped_xy, ped_vel):
    return shard.add_context(
        scenario_id=scenario, gamma=gamma, step=step,
        state=np.asarray(state, np.float32),
        hp10=np.zeros((10, 16, 12), np.float32),
        low5=np.zeros(5, np.float32),
        hist=np.zeros((16, 2), np.float32),
        ped_xy=np.asarray(ped_xy, np.float32).reshape(-1, 2),
        ped_vel=np.asarray(ped_vel, np.float32).reshape(-1, 2),
    )


def _add(shard, *, scenario, gamma, step, y, state, ped_xy, ped_vel,
         controls, collision_after=False, trap=False, result=None):
    context_id = _context(
        shard, scenario=scenario, gamma=gamma, step=step, state=state,
        ped_xy=ped_xy, ped_vel=ped_vel,
    )
    window_id = shard.add_executed_window(
        context_id, np.asarray(controls, np.float32),
        np.zeros(20, np.float32), result or _result(y),
        execution_source="selected_B" if y else "raw_continuation",
        nvp_context=not bool(y), candidate_id=0 if y else None,
        acquisition_step=0 if y else None, sigma=0.4 if y else None,
        hp_margin=0.2, mode="U",
    )
    shard.windows[window_id].update(
        collision_after_action=bool(collision_after), trap_event=bool(trap),
    )
    return window_id


FAST = np.full((10, 2), 1.2, np.float32)      # strong forward motion
STILL = np.zeros((10, 2), np.float32)          # no displacement
FAR_PED = [[5.5, 0.5]]
NEAR_PED = [[0.35, 0.12]]
ZERO_VEL = [[0.0, 0.0]]


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

    def module_groups(self):
        return {"E_g": self.enc_grid, "head": self.head}


def _freeze_encoder(policy):
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)


def _mixed_shard():
    shard = OS.ExecutedRoundShard(1)
    gammas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    for index in range(10):
        _add(
            shard, scenario=100 + index, gamma=gammas[index % 7], step=index,
            y=index < 7, state=[0.5, 0.5, 0.5, 0.5],
            ped_xy=NEAR_PED if index % 2 else FAR_PED, ped_vel=ZERO_VEL,
            controls=FAST if index < 7 else STILL,
            collision_after=(index == 8), trap=(index == 9),
        )
    return shard


class _InlineExecutor:
    def map(self, fn, tasks):
        return [fn(task) for task in tasks]


def test_original_mode_returns_untouched_shard():
    shard = _mixed_shard()
    view, report = AUG.build_replay_view(shard, "original")
    assert view is shard
    assert report["mode"] == "original"


def test_original_mode_replay_is_bitwise_identical_to_control():
    shard = _mixed_shard()
    torch.manual_seed(0)
    policy_a = _TinyPolicy()
    policy_b = copy.deepcopy(policy_a)
    _freeze_encoder(policy_a)
    _freeze_encoder(policy_b)
    opt_a = torch.optim.Adam(
        [p for p in policy_a.parameters() if p.requires_grad], lr=1e-3,
    )
    opt_b = torch.optim.Adam(
        [p for p in policy_b.parameters() if p.requires_grad], lr=1e-3,
    )
    control = OR.replay(
        policy_a, opt_a, shard, alpha=0.01, exposure_epochs=1, batch=4,
        device="cpu", seed=7,
    )
    treated = AUG.replay_with_mode(
        policy_b, opt_b, shard, mode="original", alpha=0.01,
        exposure_epochs=1, batch=4, device="cpu", seed=7,
    )
    for key, value in policy_a.state_dict().items():
        assert torch.equal(value, policy_b.state_dict()[key]), key
    assert control["optimizer_steps"] == treated["optimizer_steps"]


def test_population_tagging_follows_declared_rules():
    shard = OS.ExecutedRoundShard(2)
    # near + moving certified positive -> pop A
    _add(shard, scenario=1, gamma=0.5, step=0, y=1,
         state=[0, 0, 1.0, 0], ped_xy=[[0.8, 0.6]], ped_vel=ZERO_VEL,
         controls=FAST)
    # far certified positive -> excluded from pop A
    _add(shard, scenario=2, gamma=0.5, step=0, y=1,
         state=[0, 0, 1.0, 0], ped_xy=FAR_PED, ped_vel=ZERO_VEL,
         controls=np.full((10, 2), 0.4, np.float32))
    # certified but still (slow) positive -> excluded from A, NEVER in B
    _add(shard, scenario=3, gamma=0.5, step=0, y=1,
         state=[0, 0, 0, 0], ped_xy=[[0.8, 0.6]], ped_vel=ZERO_VEL,
         controls=STILL)
    # negative with actual collision -> pop B
    _add(shard, scenario=4, gamma=0.5, step=0, y=0,
         state=[0, 0, 1.0, 0], ped_xy=NEAR_PED, ped_vel=ZERO_VEL,
         controls=FAST, collision_after=True)
    # negative, no collision flags, but no displacement -> pop B (no progress)
    _add(shard, scenario=5, gamma=0.5, step=0, y=0,
         state=[0, 0, 0, 0], ped_xy=FAR_PED, ped_vel=ZERO_VEL,
         controls=STILL)
    pop_a, pop_b, stats = AUG.tag_populations(shard)
    assert [w["context_id"] for w in pop_a] == [0]
    assert sorted(w["context_id"] for w in pop_b) == [3, 4]
    assert stats["popA"] == 1 and stats["popB"] == 2
    # the certified-slow window is not relabeled
    assert all(int(w["y"]) == 0 for w in pop_b)


def test_recovery_candidates_are_deterministic_and_bounded():
    state = [1.0, 1.0, 1.5, -0.5]
    first = AUG.recovery_candidates(state)
    second = AUG.recovery_candidates(state)
    assert len(first) == 1 + len(AUG.BRAKE_STEPS) * AUG.K_DIR * len(AUG.ACCELS)
    for (controls_a, prov_a), (controls_b, prov_b) in zip(first, second):
        assert np.array_equal(controls_a, controls_b)
        assert prov_a == prov_b
        assert controls_a.shape == (10, 2)
        assert float(np.abs(controls_a).max()) <= 2.0 + 1e-6


def test_recovery_records_are_exactly_certified_with_provenance():
    shard = OS.ExecutedRoundShard(3)
    _add(shard, scenario=9, gamma=0.5, step=4, y=0,
         state=[2.0, 2.0, 0.8, 0.0], ped_xy=[[2.9, 2.0]],
         ped_vel=[[-0.5, 0.0]], controls=FAST, collision_after=True)
    records, audit = AUG.build_recovery_records(
        shard, shard.Dminus, _InlineExecutor(),
    )
    assert records, "an escapable context must yield certified recovery"
    assert len(records) <= AUG.RECOVERY_KEEP
    assert audit["certified_kept"] == len(records)
    for record in records:
        context = shard.contexts[record["context_id"]]
        recheck = SM.verify_query(
            context["state"], record["controls"], context["ped_xy"],
            context["ped_vel"], context["gamma"],
        )
        assert recheck["resolved"] and int(recheck["y"]) == 1
        assert record["execution_source"] == "synthetic_certified_recovery"
        prov = record["recovery_provenance"]
        assert prov["parent_context_id"] == record["context_id"]
        assert "generator" in prov and "objective" in prov
        assert prov["family"] == "v1"


def test_orig_plus_recovery_keeps_full_population_and_appends():
    shard = _mixed_shard()
    view, report = AUG.build_replay_view(
        shard, "orig_plus_recovery", executor=_InlineExecutor(),
    )
    added = report["recovery_audit"]["certified_kept"]
    assert len(view.windows) == len(shard.windows) + added
    assert len(view.Dminus) == len(shard.Dminus)
    assert len(view.Dplus) == len(shard.Dplus) + added
    synthetic = [
        w for w in view.windows
        if w["execution_source"] == "synthetic_certified_recovery"
    ]
    assert len(synthetic) == added


def test_v2_family_is_deterministic_goal_directed_and_certifiable():
    state = [1.0, 1.0, 0.4, -0.6]
    first = AUG.recovery_candidates_v2(state)
    second = AUG.recovery_candidates_v2(state)
    assert len(first) == len(second) == (
        len(AUG.CRUISE_SPEEDS)
        + 2 * AUG.K_DIR * len(AUG.V2_ACCELS) * len(AUG.CRUISE_SPEEDS)
    )
    import sfm_metrics2 as SM2
    for (ca, pa), (cb, pb) in zip(first, second):
        assert np.array_equal(ca, cb) and pa == pb
        assert ca.shape == (10, 2)
        assert float(np.abs(ca).max()) <= 2.0 + 1e-6
    # pure-cruise candidate ends moving toward the goal
    controls = first[0][0]
    seg = SM2.rollout_positions(state, controls)
    velocity = np.asarray(state, np.float32)[2:4].copy()
    for action in controls:
        velocity = velocity + 0.1 * action
    toward = velocity @ ((np.array([6.0, 6.0]) - seg[-1])
                         / np.linalg.norm(np.array([6.0, 6.0]) - seg[-1]))
    assert toward > 0.5


def test_v2_recovery_records_certified_and_tagged():
    shard = OS.ExecutedRoundShard(4)
    # pedestrian behind the robot, receding: a goal-directed certified
    # escape clearly exists
    _add(shard, scenario=9, gamma=0.5, step=4, y=0,
         state=[2.0, 2.0, 0.6, 0.6], ped_xy=[[1.2, 2.0]],
         ped_vel=[[-0.6, 0.0]], controls=STILL, trap=True)
    records, audit = AUG.build_recovery_records(
        shard, shard.Dminus, _InlineExecutor(), family="v2",
    )
    assert audit["family"] == "v2"
    assert records
    for record in records:
        assert record["recovery_provenance"]["family"] == "v2"
        context = shard.contexts[record["context_id"]]
        recheck = SM.verify_query(
            context["state"], record["controls"], context["ped_xy"],
            context["ped_vel"], context["gamma"],
        )
        assert recheck["resolved"] and int(recheck["y"]) == 1


def test_hard_recovery_replay_respects_exact_once_accounting():
    shard = _mixed_shard()
    policy = _TinyPolicy()
    _freeze_encoder(policy)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3,
    )
    result = AUG.replay_with_mode(
        policy, optimizer, shard, mode="hard_recovery", alpha=0.01,
        exposure_epochs=1, batch=4, device="cpu", seed=11,
        executor=_InlineExecutor(),
    )
    report = result["replay_intervention"]
    assert report["mode"] == "hard_recovery"
    assert "recovery_audit" in report
    view = report["view"]
    assert view["D"] == view["Dplus"] + view["Dminus"]
    assert result["positive_eligible"] == view["Dplus"]
    assert result["negative_eligible"] == view["Dminus"]
    assert result["exact_once_per_exposure_epoch"] is True
