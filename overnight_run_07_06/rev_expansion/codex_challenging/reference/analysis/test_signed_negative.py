#!/usr/bin/env python3
"""Regression gate for the optional signed rejected-sample update."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent
WORK = REV.parent
RUN = WORK.parent
sys.path[:0] = [str(ROOT), str(REV), str(WORK), str(RUN)]

import grid_expand_hardtail as HT  # noqa: E402
import grid_feats as GF  # noqa: E402
import grid_hp_expt as HP  # noqa: E402


def main() -> None:
    torch.manual_seed(2718)
    np.random.seed(2718)
    policy = HP.GridHPFlowPolicy(width=64, depth=1, repr_dim=32, grid_hw=(32, 32))
    n = 8
    fresh = {
        "grid": torch.randn(n, 3, 32, 32),
        "low5": torch.randn(n, 5),
        "hist": torch.randn(n, GF.K_HIST, 2),
        "U": torch.randn(n, policy.T, 2).clamp(-1, 1),
        "gamma": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 0.5]),
        "rid": np.arange(n),
        "mode": np.asarray(["m"] * n, dtype=object),
        "negative": {
            "grid": torch.randn(n, 3, 32, 32),
            "low5": torch.randn(n, 5),
            "hist": torch.randn(n, GF.K_HIST, 2),
            "U": torch.randn(n, policy.T, 2).clamp(-1, 1),
        },
    }
    cfg = HT.CurConfig(batch_cap=8, alpha=5e-4, neg_batch=4, hard_quota=0,
                       guard_quota=0, max_functional_step=0.0, lwf_eta=0.0,
                       field_grad_clip=1.0, enc_grad_clip=5.0)
    field = list(policy.trunk.parameters()) + list(policy.head.parameters())
    enc = policy.encoder_modules()
    opt = torch.optim.Adam([{"params": field}, {"params": enc}], lr=0.0)
    result = HT.update_flow_fresh(
        policy, opt, fresh, np.arange(4), np.arange(4, 8), (0.5, 0.5), 1, cfg,
        field, enc, "cpu",
    )
    assert result is not None
    assert math.isclose(
        result["loss"], result["positive_loss"] - cfg.alpha * result["negative_loss"],
        rel_tol=1e-6, abs_tol=1e-6,
    )
    assert result["negative_pool"] == n
    assert result["negative_grad_rms_field"] > 0.0
    assert result["negative_grad_rms_encoder"] > 0.0
    assert "alpha" not in HT._resume_signature(HT.CurConfig(alpha=0.0), True, 0.0)
    signed_sig = HT._resume_signature(cfg, False, 0.3)
    assert signed_sig["alpha"] == cfg.alpha and signed_sig["neg_batch"] == cfg.neg_batch
    payload = {
        "status": "PASS",
        "alpha": cfg.alpha,
        "signed_loss": result["loss"],
        "positive_loss": result["positive_loss"],
        "negative_loss": result["negative_loss"],
        "negative_grad_rms_field": result["negative_grad_rms_field"],
        "negative_grad_rms_encoder": result["negative_grad_rms_encoder"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
