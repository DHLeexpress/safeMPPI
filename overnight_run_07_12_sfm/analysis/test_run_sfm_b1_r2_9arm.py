import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_sfm_b1_r2_9arm as R


def _gpus(n=4):
    return [
        R.GPU(
            index=str(index), uuid=f"GPU-{index}", name="H100", memory_total_mib=95830,
            memory_used_mib=20, utilization_percent=0, pci_bus_id=f"0000:{index:02x}:00.0",
        )
        for index in range(n)
    ]


def test_four_gpu_assignment_is_declared_three_two_two_two():
    allocation = R.assign_arms(list(R.arm_grid()), _gpus())
    by_index = {
        gpu.index: [arm.name for arm in allocation[gpu.uuid]] for gpu in _gpus()
    }
    assert [len(by_index[str(index)]) for index in range(4)] == [3, 2, 2, 2]
    assert {arm.replay_epochs for arm in allocation["GPU-0"]} == {1}
    for index in ("1", "2", "3"):
        assert {
            arm.replay_epochs for arm in allocation[f"GPU-{index}"]
        } == {10, 100}
    assert sorted(sum(by_index.values(), [])) == sorted(arm.name for arm in R.arm_grid())


def test_assignment_fails_when_capacity_is_insufficient():
    with pytest.raises(RuntimeError, match="exceed"):
        R.assign_arms(list(R.arm_grid()), _gpus(2), max_arms_per_gpu=3)


def test_explicit_busy_gpu_fails_closed():
    gpus = _gpus(2)
    with pytest.raises(RuntimeError, match="not idle"):
        R.select_idle_gpus(
            gpus, [{"gpu_uuid": "GPU-1"}], "0,1",
            max_memory_mib=1024, max_utilization=5,
        )
    selected = R.select_idle_gpus(
        gpus, [{"gpu_uuid": "GPU-1"}], "auto",
        max_memory_mib=1024, max_utilization=5,
    )
    assert [gpu.index for gpu in selected] == ["0"]


def test_complete_marker_requires_contract_and_checkpoint_hashes(tmp_path):
    arm = R.Arm(0.01, 10)
    arm_dir = tmp_path / arm.name
    arm_dir.mkdir()
    history = []
    for round_i in range(3):
        path = arm_dir / f"round_{round_i:02d}.pt"
        path.write_bytes(f"round-{round_i}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (arm_dir / f"round_{round_i:02d}.pt.COMPLETE.json").write_text(json.dumps(
            dict(status="COMPLETE", path=str(path), sha256=digest)
        ))
        if round_i:
            history.append(dict(round=round_i, checkpoint_sha256=digest))
    contract = R._expected_arm_contract(
        arm, checkpoint_sha256="source", scene_profile="double_density_velocity_ood",
        ell=R.ELL, cap=R.CAP, seed=7, verifier_workers=8,
    )
    marker = dict(
        status=R.ARM_STATUS, experiment=arm.name,
        recipe=dict(
            alpha=contract["alpha"],
            replay_epochs=contract["replay_epochs"],
            rounds=contract["rounds"], scene_profile=contract["scene_profile"],
            seed=contract["seed"], verifier_workers=contract["verifier_workers"],
            lr=contract["lr"],
        ),
        constants=dict(ell=contract["ell"], cap=contract["cap"]),
        source_checkpoint_sha256="source", history=history,
    )
    (arm_dir / "COMPLETE.json").write_text(json.dumps(marker))
    result = R.validate_complete_arm(
        arm_dir, arm, checkpoint_sha256="source",
        scene_profile="double_density_velocity_ood",
        ell=R.ELL, cap=R.CAP, seed=7, verifier_workers=8,
    )
    assert [row["round"] for row in result["checkpoints"]] == [0, 1, 2]
    (arm_dir / "round_01.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="mismatch"):
        R.validate_complete_arm(
            arm_dir, arm, checkpoint_sha256="source",
            scene_profile="double_density_velocity_ood",
            ell=R.ELL, cap=R.CAP, seed=7, verifier_workers=8,
        )


def test_incomplete_nonempty_arm_is_not_overwritten(tmp_path):
    arm = R.Arm(0.0, 1)
    arm_dir = tmp_path / arm.name
    arm_dir.mkdir()
    (arm_dir / "partial.log").write_text("preserve")
    with pytest.raises(RuntimeError, match="incomplete nonempty"):
        R.validate_complete_arm(
            arm_dir, arm, checkpoint_sha256="source",
            scene_profile="double_density_velocity_ood",
            ell=R.ELL, cap=R.CAP, seed=7, verifier_workers=8,
        )


def test_trainer_command_uses_complete_replay_epoch_cli(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = type("Args", (), dict(
        checkpoint=str(checkpoint), verifier_workers=8, seed=17,
    ))()
    command = R._trainer_command(args, R.Arm(0.1, 100), tmp_path / "arm")
    assert command[command.index("--alpha") + 1] == "0.1"
    assert command[command.index("--replay-epochs") + 1] == "100"
    assert command[command.index("--verifier-workers") + 1] == "8"
    assert "--adam-steps" not in command
