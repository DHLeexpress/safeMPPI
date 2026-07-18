from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import low7_rbf_v2_smoke_driver as DR


def test_v2_driver_pins_the_qualification_recipe(tmp_path: Path) -> None:
    args = SimpleNamespace(
        python="python",
        ckpt=tmp_path / "checkpoint.pt",
        expected_ckpt_sha256="a" * 64,
        verifier_workers=32,
    )
    command = DR.trainer_command(args, tmp_path / "run")

    def value(flag: str) -> str:
        index = command.index(flag)
        return command[index + 1]

    assert value("--protocol-profile") == "v2_smoke"
    assert value("--scene-profile") == "low7_radius1_canonical_v1"
    assert value("--rounds") == "10"
    assert value("--rollout-replicas") == "8"
    assert value("--K") == "16"
    assert value("--B") == "4"
    assert value("--adaptive-beta-contexts-per-gamma") == "64"
    assert "--adaptive-beta-equalize-gammas" in command
    assert value("--replay-window") == "2"
    assert value("--replay-sampling") == "round_gamma_replica_context"
    assert value("--replay-update-mode") == "one_epoch_without_replacement"
    assert value("--gp-replay-window") == "2"
    assert value("--gp-replay-sampling") == "round_gamma_replica_context"
    assert value("--lengthscale-multiplier") == "1.0"
    assert value("--afe-steps") == "0"
    assert value("--afe-lr") == "1e-5"
    assert value("--execution-rule") == "nominal_hp_max_step_margin_only"
    assert value("--compact-checkpoint-every") == "1"
    assert value("--route-metric-steps") == "10"
    assert "--skip-training-probes" in command
    assert "--freeze-visual-encoder" in command


def test_v2_wrapper_enforces_exclusive_gpu_and_single_driver() -> None:
    wrapper = (ROOT / "run_low7_rbf_v2_smoke.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=$PHYSICAL_INDEX" in wrapper
    assert "query-compute-apps=pid" in wrapper
    assert "low7_rbf_v2_smoke_driver.py" in wrapper
    assert (ROOT / "video_afe2.py").is_file()
