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
        study_profile=DR.BASELINE_STUDY,
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
    assert value("--replay-loss-weighting") == "query_uniform"
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
    assert "--nvp-audit-all-k" not in command


def test_v2_lineage_mass_driver_pins_the_structural_smoke(tmp_path: Path) -> None:
    args = SimpleNamespace(
        python="python",
        ckpt=tmp_path / "checkpoint.pt",
        expected_ckpt_sha256="a" * 64,
        verifier_workers=64,
        study_profile=DR.LINEAGE_MASS_STUDY,
    )
    command = DR.trainer_command(args, tmp_path / "run")

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--protocol-profile") == "v2_lineage_mass_smoke"
    assert value("--replay-loss-weighting") == (
        "gamma_episode_context_query_equal_mass"
    )
    assert value("--execution-rule") == "nominal_hp_max_step_margin"
    assert "--nvp-audit-all-k" in command


def test_v2_wrapper_enforces_exclusive_gpu_and_single_driver() -> None:
    wrapper = (ROOT / "run_low7_rbf_v2_smoke.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=$PHYSICAL_INDEX" in wrapper
    assert "query-compute-apps=pid" in wrapper
    assert "low7_rbf_v2_smoke_driver.py" in wrapper
    assert "STUDY_PROFILE" in wrapper
    assert '${STUDY_PROFILE:-baseline}' not in wrapper
    assert 'set STUDY_PROFILE explicitly' in wrapper
    assert (ROOT / "video_afe2.py").is_file()


def test_v2_driver_delivers_true_evaluation_report() -> None:
    source = (ROOT / "analysis" / "low7_rbf_v2_smoke_driver.py").read_text()
    assert '"--render-only"' in source
    assert '"--presentation-outdir"' in source
    assert 'destination = args.out / f"report.{suffix}"' in source
    assert '"true_evaluation_reports": report_records' in source
