from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import low7_rbf_sweep_driver as DR


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _summary(sr, cr, timeout, clearance, round_i, *, probe_sr=0.0):
    return {
        "status": "AFE_RBF_RAW_M50_SWEEP_COMPLETE",
        "M": 50,
        "post_hoc_best_round": round_i,
        "post_hoc_ranking": [{
            "rank": 1,
            "round": round_i,
            "SR": sr,
            "CR": cr,
            "timeout": timeout,
            "mean_minimum_clearance": clearance,
        }],
        "trainer_probe_SR": probe_sr,
    }


def test_sweep_matrix_is_exactly_the_declared_48_unique_arms() -> None:
    arms = DR.sweep_arms()

    assert len(arms) == len({arm.arm_id for arm in arms}) == 48
    assert {arm.lengthscale_multiplier for arm in arms} == {0.5, 1.0}
    assert {arm.negative_alpha for arm in arms} == {0.0, 0.001, 0.005}
    assert {arm.afe_steps for arm in arms} == {4, 16, 32, 64}
    assert {arm.execution_rule for arm in arms} == {
        "nominal_hp_max_step_progress",
        "nominal_hp_max_step_margin",
    }


def test_pipeline_commands_pin_training_recipe_and_invoke_evaluator_directly(
    tmp_path: Path,
) -> None:
    arm = DR.Arm(0.5, 0.001, 16, "nominal_hp_max_step_margin")
    commands = DR.pipeline_commands(
        arm,
        python="/env/python",
        checkpoint=tmp_path / "pretrained.pt",
        checkpoint_sha256="a" * 64,
        scene_profile="low7_radius1_canonical_v1",
        output_root=tmp_path / "out",
        verifier_workers=8,
    )
    train = commands["train"]

    expected = {
        "--rounds": "100",
        "--rollout-replicas": "2",
        "--K": "64",
        "--B": "8",
        "--T": "300",
        "--M-eval": "0",
        "--batch": "128",
        "--afe-steps": "16",
        "--afe-lr": "1e-4",
        "--adaptive-ess-target": "0.5",
        "--replay-window": "5",
        "--gp-replay-window": "5",
        "--lengthscale-multiplier": "0.5",
        "--negative-alpha": "0.001",
        "--execution-rule": "nominal_hp_max_step_margin",
        "--conditioning-schema": "low7_closest_boundary",
        "--calibration-replicas": "32",
        "--calibration-control-steps": "1",
        "--verifier-workers": "8",
        "--seed": "910",
    }
    assert {flag: _option(train, flag) for flag in expected} == expected
    assert "--freeze-visual-encoder" in train
    assert "--skip-training-probes" in train
    assert "--sweep-compact-artifacts" in train
    evaluate = commands["evaluate"]
    assert evaluate[:2] == ["/env/python", str(DR.EVALUATOR)]
    assert "--run-root" in evaluate and "--outdir" in evaluate
    assert _option(evaluate, "--verifier-workers") == "8"
    assert commands["validate_evaluation"][-1] == "--validate-only"


def test_global_ranking_uses_true_pooled_evaluation_not_trainer_probe() -> None:
    arms = DR.sweep_arms()[:3]
    summaries = [
        (arms[0], _summary(0.4, 0.2, 0.4, 0.1, 30, probe_sr=0.0)),
        (arms[1], _summary(0.3, 0.0, 0.7, 0.5, 10, probe_sr=1.0)),
        (arms[2], _summary(0.4, 0.1, 0.5, 0.0, 40, probe_sr=0.0)),
    ]

    ranking = DR.global_ranking(summaries)

    assert [row["arm_id"] for row in ranking] == [
        arms[2].arm_id,
        arms[0].arm_id,
        arms[1].arm_id,
    ]
    assert [row["overall_rank"] for row in ranking] == [1, 2, 3]


def test_existing_output_root_is_rejected_without_deletion(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "partial.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="absent/new"):
        DR.prepare_output_root(output)

    assert marker.read_text() == "keep"


def test_ffprobe_video_requires_exact_declared_frame_count(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "streams": [{
            "codec_name": "h264",
            "width": 1994,
            "height": 1008,
            "nb_read_frames": "19",
        }]
    }
    monkeypatch.setattr(
        DR.subprocess,
        "check_output",
        lambda *_args, **_kwargs: json.dumps(payload),
    )
    assert DR.ffprobe_video(tmp_path / "video.mp4", 19)["frames"] == 19
    with pytest.raises(RuntimeError, match="declared 18-frame"):
        DR.ffprobe_video(tmp_path / "video.mp4", 18)


def test_gpu_gate_rejects_an_active_compute_pid(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    replies = iter(("3, GPU-expected, H100, 550.1, 95830\n", "12345\n"))
    monkeypatch.setattr(DR.subprocess, "check_output", lambda *args, **kwargs: next(replies))

    with pytest.raises(RuntimeError, match="active compute PIDs"):
        DR.gpu_record(3, "GPU-expected")


def test_bounded_runner_does_not_retry_or_submit_after_failure() -> None:
    arms = DR.sweep_arms()[:3]
    seen = []

    def worker(arm):
        seen.append(arm.arm_id)
        if arm == arms[1]:
            raise RuntimeError("failed once")
        return {"arm_id": arm.arm_id}

    with pytest.raises(RuntimeError, match="failed once"):
        DR.run_bounded(arms, 1, worker)

    assert seen == [arms[0].arm_id, arms[1].arm_id]


def test_shell_wrapper_has_six_argument_gpu_and_thread_gate() -> None:
    wrapper = (ROOT / "run_low7_rbf_sweep.sh").read_text()

    assert "[[ $# -ne 6 ]]" in wrapper
    assert "CUDA_VISIBLE_DEVICES=$PHYSICAL_INDEX" in wrapper
    assert "OMP_NUM_THREADS=1" in wrapper
    assert "MKL_NUM_THREADS=1" in wrapper
    assert "MAX_JOBS=${MAX_JOBS:-2}" in wrapper
    assert "VERIFIER_WORKERS=${VERIFIER_WORKERS:-8}" in wrapper
    assert "--query-compute-apps=pid" in wrapper
    assert "already has active compute PIDs" in wrapper
    assert "for REQUIRED_COMMAND in nvidia-smi ffmpeg ffprobe" in wrapper


def test_completion_manifest_inventory_is_lightweight_and_complete(tmp_path: Path) -> None:
    arms = DR.sweep_arms()
    paths = DR.completion_artifacts(tmp_path, arms)

    assert len(paths) == 4 + 3 * 48
    assert len(set(paths)) == len(paths)
    assert tmp_path / "sweep_contract.json" in paths
    assert tmp_path / "best_training.mp4" in paths
