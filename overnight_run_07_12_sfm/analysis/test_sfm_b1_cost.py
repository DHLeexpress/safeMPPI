from dataclasses import asdict, replace

import numpy as np
import torch

from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
import sfm_b1_cost as C
import sfm_scene as SS


def test_safemppi_scorer_parity_on_identical_inputs():
    torch.manual_seed(14)
    controls = torch.randn(7, 10, 2).clamp(-.35, .35)
    state = torch.tensor([0., 0., .1, .05])
    ped_xy = torch.tensor([[4.5, 4.0], [5.0, 3.8]])
    ped_vel = torch.tensor([[-.1, .05], [.0, -.1]])
    config = replace(C.frozen_expert_config(), proposal_gaussian_mix=0, warm_start=False)
    score = C.safemppi_proposal_cost(state, controls, SS.GOAL, ped_xy, ped_vel, config)
    adapter = SafeMPPIAdapter(**asdict(config))
    obstacles = torch.cat([ped_xy, torch.full((2, 1), SS.R_PED)], dim=1)
    _, info = adapter.plan(
        state, torch.as_tensor(SS.GOAL), obstacles, gamma=.5,
        obstacle_velocities=ped_vel, proposal_controls=controls, return_rollouts=True, seed=8,
    )
    best = controls[int(torch.argmin(score))]
    torch.testing.assert_close(torch.as_tensor(info["best_sequence"]), best, rtol=0, atol=1e-6)


def test_gate_precedes_selector_and_nvp_when_none():
    rows = [dict(
        candidate_id=0, query_id=0, controls=np.zeros((10, 2), np.float32),
        result={"resolved": True, "y": 0}, mode="yield",
    )]
    assert C.select_admissible(
        rows, selector="margin", state=np.zeros(4), ped_xy=np.array([[3., 3.]]),
        ped_vel=np.zeros((1, 2)), gamma=.5,
    ) is None
