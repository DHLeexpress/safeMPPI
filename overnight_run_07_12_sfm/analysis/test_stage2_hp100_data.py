import json

import numpy as np
import pytest
import torch

import sfm_hp100_dynamics as D
import stage2_hp100_data as S

_ORIGINAL_HP100_FRAME = S.HPF.hp100_frame


class _FakePlanner:
    def plan(self, state, goal, obstacles, **kwargs):
        _, geometry = _ORIGINAL_HP100_FRAME(
            state[:2],
            obstacles,
            sensing=S.SS.R_SENSE,
            n_base=S.N_BASE,
            obstacle_velocities=kwargs["obstacle_velocities"],
            robot_velocity=state[2:4],
            predict_gain=S.locked_expert_config()["predict_gain"],
            predict_tau=S.HORIZON * D.DT,
            return_geometry=True,
        )
        polytope = tuple(geometry[key] for key in ("A", "b", "ref", "margins"))
        return torch.tensor([9.0, -9.0]), {"polytope": polytope}


def test_rollout_collects_fresh_hp_and_future_executed_windows(monkeypatch):
    monkeypatch.setattr(S.SS, "make_humans", lambda *args, **kwargs: [object()] * S.N_PED)
    monkeypatch.setattr(
        S.SS,
        "collect_humans",
        lambda humans: (
            np.full((S.N_PED, 2), 20.0, np.float32),
            np.zeros((S.N_PED, 2), np.float32),
        ),
    )
    monkeypatch.setattr(S.SS, "advance_humans", lambda humans, state: None)
    hp_calls = []

    original_hp = S.HPF.hp100_frame

    def fresh_hp(robot_xy, obstacles, **kwargs):
        hp_calls.append((
            np.asarray(robot_xy).copy(), np.asarray(obstacles).copy(),
            kwargs["sensing"], kwargs["n_base"],
            np.asarray(kwargs["obstacle_velocities"]).copy(),
            np.asarray(kwargs["robot_velocity"]).copy(),
        ))
        return original_hp(robot_xy, obstacles, **kwargs)

    monkeypatch.setattr(S.HPF, "hp100_frame", fresh_hp)
    records, summary = S.rollout_episode(
        3, 0.5, device="cpu", planner=_FakePlanner(), T_max=2
    )
    assert summary["timeout"] and len(records) == 2
    assert len(hp_calls) == 2
    assert all(call[3] == 16 for call in hp_calls)
    assert all(call[4].shape == (S.N_PED, 2) for call in hp_calls)
    assert records[0]["hp"].shape == (32, 100)
    assert records[0]["hp"].dtype == np.float32
    np.testing.assert_allclose(records[0]["executed_action"], [2.0, -2.0])
    np.testing.assert_allclose(records[1]["state"][2:], [0.2, -0.2])
    assert records[0]["U"].shape == (10, 2)
    np.testing.assert_allclose(records[0]["U"], np.tile([2.0, -2.0], (10, 1)))
    assert np.max(np.abs(records[0]["U"])) <= D.U_MAX


def _record(episode=0, step=0):
    return dict(
        hp=np.zeros((32, 100), np.float32),
        low5=np.zeros(5, np.float32),
        hist=np.zeros((16, 2), np.float32),
        U=np.zeros((10, 2), np.float32),
        episode=np.int64(episode),
        step=np.int64(step),
        state=np.zeros(4, np.float32),
        ped_xy=np.zeros((20, 2), np.float32),
        ped_vel=np.zeros((20, 2), np.float32),
        executed_action=np.zeros(2, np.float32),
    )


def test_small_cpu_dataset_smoke_writes_auditable_manifest(tmp_path):
    def fake_rollout(episode, gamma, **kwargs):
        success = episode == 1
        return [_record(episode, 0)], dict(
            episode=episode, gamma=gamma, success=success, collision=not success,
            timeout=False, steps=1, min_clearance=(1.0 if success else -0.1),
        )

    manifest = S.generate_dataset(
        tmp_path,
        episode_start=0,
        successes_per_gamma=1,
        max_attempts_per_gamma=2,
        gammas=(0.5,),
        device="cpu",
        T_max=2,
        rollout_fn=fake_rollout,
    )
    assert manifest["status"] == "HP100_ID_DATASET_COMPLETE"
    assert not manifest["canonical_full_run"]
    assert manifest["dynamics"]["action_cap"]["maximum"] == 2.0
    assert manifest["dynamics"]["velocity_cap"]["maximum"] == 2.0
    assert manifest["feature"]["nominal_polytope_n_base"] == 16
    assert manifest["feature"]["velocity_aware"] is True
    assert manifest["feature"]["predict_tau"] == 1.0
    assert manifest["total_successful_lineages"] == 1
    assert "old-grid upsample" in manifest["feature"]["construction"]
    assert manifest["files"][0]["successful_episodes"] == [1]
    assert manifest["files"][0]["rejected_episodes"] == [0]
    assert manifest["files"][0]["episode_range"] == [0, 2]
    data_path = tmp_path / "sfm_hp100_windows_g0.5.pt"
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    assert payload["hp"].shape == (1, 32, 100)
    assert payload["state"].shape == (1, 4)
    assert payload["ped_xy"].shape == (1, 20, 2)
    assert payload["ped_vel"].shape == (1, 20, 2)
    assert manifest["files"][0]["sha256"] == S.sha256_file(data_path)
    disk_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert disk_manifest["source_hashes"]["dynamics"]["sha256"]


def test_success_quota_fails_closed_at_attempt_cap(tmp_path):
    def rejected(episode, gamma, **kwargs):
        return [_record(episode, 0)], dict(
            episode=episode, gamma=gamma, success=False, collision=True,
            timeout=False, steps=1, min_clearance=-0.1,
        )

    with pytest.raises(RuntimeError, match="0/1 successful trajectories in 2 attempts"):
        S.generate_dataset(
            tmp_path,
            successes_per_gamma=1,
            max_attempts_per_gamma=2,
            gammas=(0.5,),
            T_max=2,
            rollout_fn=rejected,
        )
