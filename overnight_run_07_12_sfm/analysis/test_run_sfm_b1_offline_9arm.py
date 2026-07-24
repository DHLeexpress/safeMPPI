from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_sfm_b1_offline_9arm as L  # noqa: E402


def _gpu(index: int) -> L.BASE.GPU:
    return L.BASE.GPU(
        index=str(index),
        uuid=f"GPU-{index}",
        name="test",
        memory_total_mib=100,
        memory_used_mib=0,
        utilization_percent=0,
        pci_bus_id=f"0000:0{index}:00.0",
    )


def test_arm_grid_and_four_gpu_allocation():
    arms = list(L.arm_grid())
    assert len(arms) == 9
    assert len({arm.name for arm in arms}) == 9
    assert {
        (arm.alpha, arm.exposure_epochs) for arm in arms
    } == {
        (alpha, epochs)
        for alpha in L.ALPHAS
        for epochs in L.EXPOSURE_EPOCHS
    }
    allocation = L.allocate_arms(arms, [_gpu(i) for i in range(4)])
    assert sorted(map(len, allocation.values())) == [2, 2, 2, 3]
    assert set().union(*map(set, allocation.values())) == set(arms)
    assert {
        arm.exposure_epochs for arm in allocation["GPU-0"]
    } == {1}


def test_output_root_must_be_new_and_under_research1(tmp_path, monkeypatch):
    root = tmp_path / "research1"
    root.mkdir()
    monkeypatch.setattr(L, "RESEARCH_ROOT", root)
    target = root / "new-study"
    assert L._validated_output_root(target) == target.resolve()
    target.mkdir()
    with pytest.raises(FileExistsError):
        L._validated_output_root(target)
    with pytest.raises(ValueError):
        L._validated_output_root(tmp_path / "elsewhere")


def test_commands_cover_declared_rounds_and_raw_common_bank(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    args = type("Args", (), {
        "checkpoint": str(checkpoint),
        "verifier_workers": 8,
        "seed": 20260724,
        "eval_ep0": 260000,
        "eval_noise_seed": 20260723,
    })()
    arm = L.Arm(0.01, 10)
    train = L._trainer_command(args, arm, tmp_path / "train")
    assert train[train.index("--rounds") + 1] == "10"
    assert train[train.index("--exposure-epochs") + 1] == "10"
    evaluate = L._evaluation_command(
        args,
        arm,
        tmp_path / "train",
        tmp_path / "eval",
        cache_dir=tmp_path / "common_cache",
    )
    checkpoints_start = evaluate.index("--checkpoints") + 1
    checkpoints_end = evaluate.index("--labels")
    assert evaluate[checkpoints_start] == str(checkpoint.resolve())
    assert evaluate[checkpoints_start + 1].endswith("round_01.pt")
    labels_start = evaluate.index("--labels") + 1
    labels_end = evaluate.index("--scene-profile")
    assert evaluate[labels_start:labels_end] == [
        f"r{round_i}" for round_i in range(11)
    ]
    assert evaluate[evaluate.index("--ep0") + 1] == "260000"
    assert evaluate[evaluate.index("--device") + 1] == "cuda:0"
    assert evaluate[evaluate.index("--cache-dir") + 1] == str(
        (tmp_path / "common_cache").resolve()
    )
    common = L._common_r0_command(args, tmp_path / "common")
    assert common[common.index("--labels") + 1] == "r0"
    assert common[common.index("--checkpoints") + 1] == str(
        checkpoint.resolve()
    )


def test_screening_key_is_safety_first():
    base = {
        "CR": 0.1,
        "Validity": 0.5,
        "SR": 0.8,
        "clearance": 0.1,
        "time_to_goal": 9.0,
        "round": 1,
        "exposure_epochs": 1,
        "alpha": 0.0,
    }
    lower_collision = {**base, "CR": 0.09, "Validity": 0.0}
    higher_validity = {**base, "Validity": 0.6, "SR": 0.0}
    assert L._screening_key(lower_collision) < L._screening_key(base)
    assert L._screening_key(higher_validity) < L._screening_key(base)


def test_validate_sidecar_authenticates_digest(tmp_path):
    artifact = tmp_path / "round_00.pt"
    artifact.write_bytes(b"checkpoint")
    sidecar = Path(str(artifact) + ".COMPLETE.json")
    sidecar.write_text(json.dumps({
        "status": "COMPLETE",
        "sha256": L.BASE.sha256_file(artifact),
    }))
    observed = L._validate_sidecar(artifact)
    assert observed["sha256"] == L.BASE.sha256_file(artifact)
    sidecar.write_text(json.dumps({
        "status": "COMPLETE",
        "sha256": "0" * 64,
    }))
    with pytest.raises(RuntimeError):
        L._validate_sidecar(artifact)
