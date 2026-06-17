import torch

from cfm_mppi.models.context_encoder import ContextEncoder


def test_context_encoder_tokens():
    enc = ContextEncoder(d_model=32, history_len=5)
    tokens = enc(
        start=torch.zeros(3, 2),
        goal=torch.ones(3, 2),
        ego_history=torch.zeros(3, 5, 4),
        action_history=torch.zeros(3, 5, 2),
        nearest_obstacle_history=torch.zeros(3, 5, 4),
        gamma=torch.tensor([0.1, 0.2, float("nan")]),
        safety_margin=torch.ones(3) * 0.5,
    )
    assert tokens.shape == (3, 7, 32)
    assert torch.isfinite(tokens).all()
