import importlib

import torch

from cfm_mppi.models.contextual_transformer import ContextualTransformerModel
from cfm_mppi.models.transformer import TransformerModel


def _ctx(batch, hist=5):
    return dict(
        start=torch.zeros(batch, 2),
        goal=torch.ones(batch, 2),
        ego_history=torch.zeros(batch, hist, 4),
        action_history=torch.zeros(batch, hist, 2),
        nearest_obstacle_history=torch.zeros(batch, hist, 4),
        gamma=torch.ones(batch) * 0.1,
        safety_margin=torch.ones(batch) * 0.5,
    )


def test_mizuta_transformer_instantiates_and_train_imports():
    model = TransformerModel()
    assert sum(p.numel() for p in model.parameters()) > 0
    importlib.import_module("cfm_mppi.train")


def test_contextual_transformer_t10_t80():
    model = ContextualTransformerModel(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_len=128,
        history_len=5,
    )
    for horizon in (10, 80):
        x = torch.randn(2, 2, horizon)
        out = model(x, torch.rand(2), **_ctx(2))
        assert out.shape == (2, 2, horizon)
        assert torch.isfinite(out).all()
