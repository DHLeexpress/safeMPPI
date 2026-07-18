from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import afe_context as CX
import grid_expand_afe_rbf as RBF


def _v2_args(**overrides):
    values = {
        "protocol_profile": "v2_smoke",
        "scene_profile": "low7_radius1_canonical_v1",
        "rounds": 10,
        "rollout_replicas": 8,
        "K": 16,
        "B": 4,
        "T": 300,
        "M_eval": 0,
        "batch": 128,
        "afe_steps": 0,
        "afe_lr": 1.0e-5,
        "gp_cap": 512,
        "gp_lam": 1.0e-2,
        "acquisition_mode": "sequential",
        "adaptive_ess_target": 0.5,
        "adaptive_beta_contexts_per_gamma": 64,
        "adaptive_beta_equalize_gammas": True,
        "replay_window": 2,
        "replay_sampling": "round_gamma_replica_context",
        "replay_update_mode": "one_epoch_without_replacement",
        "gp_replay_window": 2,
        "gp_replay_sampling": "round_gamma_replica_context",
        "lengthscale_multiplier": 1.0,
        "negative_alpha": 0.0,
        "execution_rule": "nominal_hp_max_step_margin_only",
        "conditioning_schema": CX.LOW7_SCHEMA,
        "freeze_visual_encoder": True,
        "skip_training_probes": True,
        "calibration_replicas": 8,
        "calibration_control_steps": 4,
        "sweep_compact_artifacts": True,
        "compact_checkpoint_every": 1,
        "route_metric_steps": 10,
        "route_ambiguity_band": 0.05,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_v2_smoke_contract_accepts_only_the_declared_end_to_end_recipe() -> None:
    RBF.validate_protocol_args(_v2_args())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("K", 64),
        ("B", 8),
        ("rollout_replicas", 2),
        ("afe_lr", 1.0e-4),
        ("replay_sampling", "query_uniform"),
        ("replay_update_mode", "fixed_steps_with_replacement"),
        ("execution_rule", "nominal_hp_max_step_progress"),
        ("compact_checkpoint_every", 10),
    ],
)
def test_v2_smoke_contract_rejects_silent_recipe_drift(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        RBF.validate_protocol_args(_v2_args(**{name: value}))


def test_v1_contract_remains_backward_compatible() -> None:
    RBF.validate_protocol_args(SimpleNamespace(
        protocol_profile="v1", K=64, B=8, batch=128
    ))
    with pytest.raises(ValueError, match="first RBF study"):
        RBF.validate_protocol_args(SimpleNamespace(
            protocol_profile="v1", K=16, B=8, batch=128
        ))
