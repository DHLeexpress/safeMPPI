import torch

from cfm_mppi.models.drifting_generator import DriftingGenerator
from cfm_mppi.training.drift_loss_torch import drifting_loss


def _batch(batch=2, horizon=10):
    return {
        "states": torch.zeros(batch, horizon + 1, 4),
        "controls_si": torch.zeros(batch, horizon, 2),
        "controls_dyn": torch.zeros(batch, horizon, 2),
        "start": torch.zeros(batch, 2),
        "goal": torch.ones(batch, 2),
        "ego_history": torch.zeros(batch, 5, 4),
        "action_history": torch.zeros(batch, 5, 2),
        "nearest_obstacle_history": torch.zeros(batch, 5, 4),
        "gamma": torch.ones(batch) * 0.1,
        "safety_margin": torch.ones(batch) * 0.5,
    }


def test_drifting_loss_gradient_and_generator_shape():
    model = DriftingGenerator.from_mizuta_defaults(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_len=64,
        history_len=5,
    )
    batch = _batch()
    noise = torch.randn(2, 2, 10)
    gen = model.forward_batch(noise, batch)
    assert gen.shape == (2, 2, 10)
    target = torch.randn_like(gen)
    loss = drifting_loss(gen, target, fixed_neg=noise)
    loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert grad > 0.0
