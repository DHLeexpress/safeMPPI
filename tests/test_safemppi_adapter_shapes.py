import torch

from cfm_mppi.safegpc_adapter import SafeMPPIAdapter


def test_safemppi_adapter_returns_bounded_control():
    adapter = SafeMPPIAdapter(horizon=5, num_samples=16, gamma=0.5, dynamics_type="doubleintegrator")
    action, info = adapter.plan(
        torch.zeros(4),
        torch.tensor([1.0, 1.0]),
        torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float32),
        seed=0,
    )
    assert action.shape == (2,)
    assert torch.all(action <= 2.0)
    assert torch.all(action >= -2.0)
    assert "min_barrier_h" in info
