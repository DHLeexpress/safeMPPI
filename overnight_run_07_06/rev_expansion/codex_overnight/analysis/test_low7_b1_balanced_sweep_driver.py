from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import low7_b1_balanced_sweep_driver as DRIVER


def value(command, flag):
    return command[command.index(flag) + 1]


def test_b1_matrix_is_the_declared_24_arm_factorial() -> None:
    assert len(DRIVER.ARMS) == 24
    assert len({arm.arm_id for arm in DRIVER.ARMS}) == 24
    assert {arm.gp_cap for arm in DRIVER.ARMS} == {512, 1024}
    assert {arm.ess_target for arm in DRIVER.ARMS} == {0.25, 0.5}
    assert {arm.alpha for arm in DRIVER.ARMS} == {0.0, 0.001, 0.01}
    assert {arm.execution_rule for arm in DRIVER.ARMS} == {
        "nominal_hp_max_step_margin", "nominal_hp_safemppi_cost"
    }


def test_b1_command_preserves_sample_complete_b1_and_qualification(tmp_path) -> None:
    delivery = tmp_path / "DELIVERY_COMPLETE.json"
    args = SimpleNamespace(
        python="python",
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_sha256="a" * 64,
        pretrain_delivery=delivery,
        verifier_workers=48,
    )
    arm = DRIVER.Arm(1024, 0.25, 0.001, "nominal_hp_safemppi_cost")
    command = DRIVER.trainer_command(args, arm, tmp_path / "run", preflight=False)

    assert value(command, "--protocol-profile") == "b1_balanced_r0_sweep"
    assert value(command, "--rounds") == "20"
    assert value(command, "--rollout-replicas") == "8"
    assert value(command, "--K") == "16"
    assert value(command, "--B") == "4"
    assert value(command, "--gp-cap") == "1024"
    assert value(command, "--adaptive-ess-target") == "0.25"
    assert value(command, "--negative-alpha") == "0.001"
    assert value(command, "--execution-rule") == "nominal_hp_safemppi_cost"
    assert value(command, "--replay-window") == "2"
    assert value(command, "--gp-replay-window") == "2"
    assert value(command, "--replay-update-mode") == "one_epoch_without_replacement"
    assert value(command, "--replay-loss-weighting") == (
        "gamma_episode_context_query_equal_mass"
    )
    assert value(command, "--lengthscale-multiplier") == "1.0"
    assert value(command, "--balanced-r0-delivery") == str(delivery.resolve())
    assert "--nvp-audit-all-k" in command
    assert "--freeze-visual-encoder" in command
    assert "--skip-training-probes" in command


def test_global_selection_prioritizes_balanced_success_coverage() -> None:
    better_sr = {
        "arm": {"arm_id": "high_sr"},
        "best": {"J": 0.2, "SR": 0.9, "CR": 0.1, "timeout": 0.0,
                 "minimum_clearance": 0.1, "round": 5},
    }
    better_coverage = {
        "arm": {"arm_id": "high_j"},
        "best": {"J": 0.4, "SR": 0.7, "CR": 0.3, "timeout": 0.0,
                 "minimum_clearance": 0.1, "round": 5},
    }
    assert min((better_sr, better_coverage), key=DRIVER.global_key) is better_coverage
