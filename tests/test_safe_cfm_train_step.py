import torch

from cfm_mppi.models.contextual_transformer import ContextualTransformerModel
from cfm_mppi.training.train_loop_safe_cfm import safe_cfm_loss


def test_safe_cfm_one_step_finite_gradients():
    model = ContextualTransformerModel(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_len=64,
        history_len=5,
    )
    batch = {
        "states": torch.zeros(2, 11, 4),
        "controls_si": torch.randn(2, 10, 2),
        "controls_dyn": torch.randn(2, 10, 2),
        "start": torch.zeros(2, 2),
        "goal": torch.ones(2, 2),
        "ego_history": torch.zeros(2, 5, 4),
        "action_history": torch.zeros(2, 5, 2),
        "nearest_obstacle_history": torch.zeros(2, 5, 4),
        "gamma": torch.ones(2) * 0.1,
        "safety_margin": torch.ones(2) * 0.5,
    }
    loss, _ = safe_cfm_loss(model, batch, torch.device("cpu"))
    loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert grad > 0.0
