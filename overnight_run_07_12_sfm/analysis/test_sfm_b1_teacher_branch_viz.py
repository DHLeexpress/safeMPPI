import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import sfm_b1_offline_store as OS
import sfm_b1_teacher_branch_viz as V
import sfm_scene as SS


def _result(y):
    return dict(
        resolved=True, y=int(y), full_h=True, terminal_step=10,
        taskspace=bool(y), collision_free=bool(y), certificate=bool(y),
        diagnostics=dict(slack=0.1),
    )


def _shard(path):
    shard = OS.ExecutedRoundShard(1)
    for scenario in (10, 11, 12):
        for gamma in SS.GAMMAS:
            state = np.array([0.2, 0.3, 0.0, 0.0], np.float32)
            ped_xy = np.array([[1.0, 1.0]], np.float32)
            ped_vel = np.array([[0.1, 0.0]], np.float32)
            context_id = shard.add_context(
                scenario_id=scenario, gamma=gamma, step=0, state=state,
                hp10=np.zeros((10, 16, 12), np.float32),
                low5=np.zeros(5, np.float32),
                hist=np.zeros((16, 2), np.float32),
                ped_xy=ped_xy, ped_vel=ped_vel,
            )
            y = int(gamma >= 0.3)
            shard.add_executed_window(
                context_id, np.zeros((10, 2), np.float32),
                np.zeros(20, np.float32), _result(y),
                execution_source="unit_test", nvp_context=False,
            )
    shard.save(path)
    return shard


def _teacher(path, shard_path, shard, *, mismatch=False, forbidden=False):
    context = shard.contexts[0]
    snapshot = {
        field: (
            np.asarray(context[field]).copy()
            if field in ("state", "ped_xy", "ped_vel")
            else context[field]
        )
        for field in V.CONTEXT_FIELDS
    }
    if mismatch:
        snapshot["state"][0] += 0.1
    record = dict(
        teacher_id=0, context_id=0,
        controls=np.full((10, 2), 1.5, np.float32),
        source=V.TEACHER_SOURCE,
        candidate_family="constant_acceleration_escape",
        context_snapshot=snapshot,
    )
    if forbidden:
        record["y"] = 1
    torch.save(dict(
        status=V.TEACHER_STATUS, version=1, round=1,
        round_shard_sha256=V._sha256(shard_path),
        records=[record],
        provenance=dict(unit_test=True),
    ), path)


def test_load_inputs_enforces_exact_context_and_no_safety_label(tmp_path):
    shard_path = os.fspath(tmp_path / "round.pt")
    teacher_path = os.fspath(tmp_path / "teacher.pt")
    shard = _shard(shard_path)
    _teacher(teacher_path, shard_path, shard)
    loaded, _, by_context, provenance = V.load_inputs(
        shard_path, teacher_path,
    )
    assert len(loaded.Dplus) == 15
    assert len(loaded.Dminus) == 6
    assert by_context[0][0]["source"] == V.TEACHER_SOURCE
    assert provenance["teacher_records"] == 1
    assert provenance["teacher_families"] == {
        "constant_acceleration_escape": 1,
    }

    _teacher(teacher_path, shard_path, shard, mismatch=True)
    with pytest.raises(ValueError, match="does not match"):
        V.load_inputs(shard_path, teacher_path)
    _teacher(teacher_path, shard_path, shard, forbidden=True)
    with pytest.raises(ValueError, match="safety-label fields"):
        V.load_inputs(shard_path, teacher_path)


def test_draw_cell_uses_purple_teacher_without_relabeling(tmp_path):
    shard_path = os.fspath(tmp_path / "round.pt")
    teacher_path = os.fspath(tmp_path / "teacher.pt")
    shard = _shard(shard_path)
    _teacher(teacher_path, shard_path, shard)
    _, _, teachers, _ = V.load_inputs(shard_path, teacher_path)
    rows = V._lineages(shard)[(10, 0.1)]
    figure, axis = plt.subplots()
    V.draw_cell(axis, rows, teachers, through_step=0)
    colors = [line.get_color() for line in axis.lines]
    teacher_lines = [
        line for line in axis.lines if line.get_color() == V.TEACHER_COLOR
    ]
    assert V.TEACHER_COLOR in colors
    assert all(line.get_linestyle() == "--" for line in teacher_lines)
    assert any(line.get_color() == "#111111" for line in axis.lines)
    plt.close(figure)


def test_render_final_png_and_report(tmp_path):
    shard_path = os.fspath(tmp_path / "round.pt")
    teacher_path = os.fspath(tmp_path / "teacher.pt")
    png = os.fspath(tmp_path / "teacher_branches.png")
    report_path = os.fspath(tmp_path / "teacher_branches.json")
    shard = _shard(shard_path)
    _teacher(teacher_path, shard_path, shard)
    report = V.render(
        shard_path, teacher_path, png, report_path,
        scenarios=(10, 11, 12),
    )
    assert report["status"] == "SFM_B1_TEACHER_D_BRANCH_VIZ_COMPLETE"
    assert report["teacher_counts"] == dict(
        records=1, contexts=1,
        families={"constant_acceleration_escape": 1},
    )
    assert os.path.getsize(png) > 0
    assert os.path.getsize(report_path) > 0
