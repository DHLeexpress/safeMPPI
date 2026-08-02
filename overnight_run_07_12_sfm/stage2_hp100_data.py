"""Generate fresh Hp100 ID demonstrations with one shared capped dynamics.

This is an additive replacement for the unavailable Hp10 stage-2 generator;
it never reads, interpolates, or upsamples the old ``16 x 12`` grid files.
The canonical run fills exactly 500 successful trajectories for each of seven
gammas by advancing deterministic episode IDs from zero.  It uses the locked
SafeMPPI expert in the matched training environment (20 pedestrians,
0.5--1.0 m/s).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import multiprocessing as mp

import numpy as np
import torch

import _paths  # noqa: F401
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
import sfm_b1_expert as EXPERT
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_scene as SS


SCHEMA_VERSION = "sfm_hp100_id_demonstrations_v1"
HORIZON = 10
N_BASE = 16
N_PED = 20
PED_SPEED_RANGE = (0.5, 1.0)
EPISODE_START = 0
SUCCESSES_PER_GAMMA = 500
MAX_ATTEMPTS_PER_GAMMA = 5000
T = 180
REACH = 0.5


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CappedSafeMPPIAdapter(SafeMPPIAdapter):
    """Locked expert whose internal rollouts use the Hp100 dynamics contract."""

    def _step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if self.config.dynamics_type != "doubleintegrator":
            raise ValueError("Hp100 demonstrations require double-integrator dynamics")
        return DYN.step_torch(
            state,
            control,
            dt=float(self.config.dt),
            u_max=DYN.U_MAX,
            v_max=DYN.V_MAX,
        )


def locked_expert_config() -> dict:
    config = asdict(EXPERT.demonstration_config())
    required = dict(
        horizon=HORIZON,
        dt=DYN.DT,
        polytope_nbase=N_BASE,
        barrier_activation_radius=SS.R_SENSE,
        predict_gain=HPF.PREDICT_GAIN,
        dynamics_type="doubleintegrator",
    )
    for key, expected in required.items():
        if config[key] != expected:
            raise RuntimeError(
                f"locked SafeMPPI expert drifted at {key}: {config[key]} != {expected}"
            )
    if tuple(map(float, config["u_min"])) != (-DYN.U_MAX, -DYN.U_MAX):
        raise RuntimeError("locked expert action lower bound disagrees with Hp100 dynamics")
    if tuple(map(float, config["u_max"])) != (DYN.U_MAX, DYN.U_MAX):
        raise RuntimeError("locked expert action upper bound disagrees with Hp100 dynamics")
    return config


def _obstacles(ped_xy) -> np.ndarray:
    positions = np.asarray(ped_xy, dtype=np.float32).reshape(-1, 2)
    return np.concatenate(
        (positions, np.full((len(positions), 1), SS.R_PED, np.float32)), axis=1
    )


def _assert_feature_matches_planner(hp, feature_geometry, planner_polytope, robot_xy):
    """Fail closed unless the stored Hp raster is the planner's exact geometry."""
    if planner_polytope is None:
        raise RuntimeError("locked SafeMPPI expert did not expose its nominal polytope")
    planner_geometry = dict(zip(("A", "b", "ref", "margins"), planner_polytope))
    for key, expected_value in planner_geometry.items():
        expected = np.asarray(expected_value, np.float32)
        actual = np.asarray(feature_geometry[key], np.float32)
        if expected.shape != actual.shape or not np.allclose(
            actual, expected, rtol=1.0e-6, atol=1.0e-6
        ):
            delta = (
                float(np.max(np.abs(actual - expected)))
                if actual.shape == expected.shape else float("inf")
            )
            raise RuntimeError(
                f"Hp100/planner nominal polytope mismatch at {key}: max_delta={delta}"
            )

    center = np.asarray(robot_xy, np.float64).reshape(2)
    n_theta, n_r = HPF.HP100_SHAPE
    theta = -np.pi + (np.arange(n_theta) + 0.5) * 2.0 * np.pi / n_theta
    radius = (np.arange(n_r) + 0.5) * SS.R_SENSE / n_r
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    points = center[None, None] + directions[:, None] * radius[None, :, None]
    A = np.asarray(planner_geometry["A"], np.float64)
    b = np.asarray(planner_geometry["b"], np.float64)
    margins = np.asarray(planner_geometry["margins"], np.float64)
    expected_hp = ((b[None] - points.reshape(-1, 2) @ A.T) / margins[None]).min(axis=1)
    expected_hp = np.clip(expected_hp, -1.0, 1.0).reshape(n_theta, n_r)
    if not np.allclose(np.asarray(hp), expected_hp, rtol=1.0e-5, atol=1.0e-5):
        delta = float(np.max(np.abs(np.asarray(hp) - expected_hp)))
        raise RuntimeError(
            f"stored Hp100 raster disagrees with planner geometry: max_delta={delta}"
        )


def rollout_episode(
    episode: int,
    gamma: float,
    *,
    device: str = "cpu",
    planner=None,
    T_max: int = T,
) -> tuple[list[dict], dict]:
    """Collect current-context H10 targets from one fresh expert rollout."""
    expert_config = locked_expert_config()
    if planner is None:
        planner = CappedSafeMPPIAdapter(**expert_config)
    humans = SS.make_humans(
        int(episode), seed=0, n_ped=N_PED, speed_range=PED_SPEED_RANGE
    )
    state = np.zeros(4, np.float32)
    control_history: list[np.ndarray] = []
    records: list[dict] = []
    collision = reached = False
    minimum_clearance = float("inf")
    goal = torch.as_tensor(SS.GOAL, dtype=torch.float32, device=device)
    for step in range(int(T_max)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        ped_xy = np.asarray(ped_xy, np.float32)
        ped_vel = np.asarray(ped_vel, np.float32)
        if ped_xy.shape != (N_PED, 2) or ped_vel.shape != (N_PED, 2):
            raise ValueError(
                f"expected {N_PED} pedestrian states, got {ped_xy.shape}/{ped_vel.shape}"
            )
        clearance = float(
            np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED
        )
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < REACH:
            reached = True
            break

        obstacles = _obstacles(ped_xy)
        hp, feature_geometry = HPF.hp100_frame(
            state[:2],
            obstacles,
            sensing=SS.R_SENSE,
            n_base=N_BASE,
            obstacle_velocities=ped_vel,
            robot_velocity=state[2:4],
            predict_gain=float(expert_config["predict_gain"]),
            predict_tau=HORIZON * DYN.DT,
            return_geometry=True,
        )
        hp = np.asarray(hp, dtype=np.float32)
        if hp.shape != (32, 100):
            raise ValueError(f"fresh Hp100 feature must be [32,100], got {hp.shape}")
        low5 = np.asarray(HPF.low5(state, SS.GOAL, float(gamma)), np.float32)
        hist = np.asarray(HPF.hist_pad(control_history), np.float32)

        action, info = planner.plan(
            torch.as_tensor(state, dtype=torch.float32, device=device),
            goal,
            torch.as_tensor(obstacles, dtype=torch.float32, device=device),
            gamma=float(gamma),
            obstacle_velocities=torch.as_tensor(
                ped_vel, dtype=torch.float32, device=device
            ),
            seed=int(episode) * 200 + int(step),
            return_rollouts=False,
        )
        _assert_feature_matches_planner(
            hp, feature_geometry, info.get("polytope"), state[:2]
        )
        action = DYN.clip_action_numpy(
            action.detach().cpu().numpy().astype(np.float32).reshape(2)
        ).astype(np.float32, copy=False)
        records.append(dict(
            hp=hp.copy(),
            low5=low5.copy(),
            hist=hist.copy(),
            episode=np.int64(episode),
            step=np.int64(step),
            state=state.copy(),
            ped_xy=ped_xy.copy(),
            ped_vel=ped_vel.copy(),
            executed_action=action.copy(),
        ))
        state = DYN.step_numpy(state, action).astype(np.float32, copy=False)
        control_history.append(action.copy())
        SS.advance_humans(humans, state)

    if not collision and not reached:
        ped_xy, _ = SS.collect_humans(humans)
        clearance = float(
            np.linalg.norm(np.asarray(ped_xy) - state[:2][None], axis=1).min()
            - SS.R_PED
        )
        minimum_clearance = min(minimum_clearance, clearance)
        collision = clearance < 0.0
        reached = bool(
            not collision and float(np.linalg.norm(state[:2] - SS.GOAL)) < REACH
        )
    # Match the original SafeMPPI demonstration contract: U is the future
    # *executed* receding-horizon control window, not the planner's current
    # reward-weighted mean sequence.  The final available action is repeated
    # to keep every supervised target exactly H=10.
    controls = np.asarray(control_history, np.float32)
    for index, record in enumerate(records):
        target = controls[index:index + HORIZON]
        if len(target) < HORIZON:
            target = np.concatenate(
                (target, np.repeat(target[-1:], HORIZON - len(target), axis=0)),
                axis=0,
            )
        record["U"] = target.astype(np.float32, copy=False)
    return records, dict(
        episode=int(episode),
        gamma=float(gamma),
        success=bool(reached and not collision),
        collision=bool(collision),
        timeout=bool(not reached and not collision),
        steps=len(records),
        min_clearance=float(minimum_clearance),
    )


def pack_records(records: list[dict]) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot pack an empty Hp100 demonstration set")
    array_keys = (
        "hp", "low5", "hist", "U", "state", "ped_xy", "ped_vel",
        "executed_action",
    )
    payload = {
        key: torch.from_numpy(np.stack([row[key] for row in records])).to(torch.float32)
        for key in array_keys
    }
    payload["episode"] = torch.as_tensor(
        [row["episode"] for row in records], dtype=torch.int64
    )
    payload["step"] = torch.as_tensor(
        [row["step"] for row in records], dtype=torch.int64
    )
    expected = dict(
        hp=(32, 100), low5=(5,), hist=(16, 2), U=(HORIZON, 2),
        state=(4,), ped_xy=(N_PED, 2), ped_vel=(N_PED, 2),
        executed_action=(2,),
    )
    for key, trailing in expected.items():
        if tuple(payload[key].shape[1:]) != trailing:
            raise ValueError(f"packed {key} has shape {tuple(payload[key].shape)}")
    return payload


def _atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    paths = {
        "generator": Path(__file__).resolve(),
        "dynamics": Path(inspect.getsourcefile(DYN)).resolve(),
        "features": Path(inspect.getsourcefile(HPF)).resolve(),
        "expert_config": Path(inspect.getsourcefile(EXPERT)).resolve(),
        "safemppi_adapter": Path(inspect.getsourcefile(SafeMPPIAdapter)).resolve(),
        "scene": Path(inspect.getsourcefile(SS)).resolve(),
    }
    return {
        name: dict(path=str(path), sha256=sha256_file(path))
        for name, path in paths.items()
    }


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return dict(root=str(root), head=head, clean=not bool(status), status=status)


def _assert_provenance_unchanged(initial_git: dict, initial_hashes: dict) -> dict:
    final_git = _git_provenance()
    final_hashes = _source_hashes()
    if final_git != initial_git:
        raise RuntimeError(
            "source Git provenance changed during HP100 collection; "
            "refusing to publish a manifest"
        )
    if final_hashes != initial_hashes:
        raise RuntimeError(
            "source file hashes changed during HP100 collection; "
            "refusing to publish a manifest"
        )
    return final_git


def _collect_gamma(payload, rollout_fn=rollout_episode) -> tuple[dict, list[dict]]:
    """Worker-safe collection for one gamma; writes only its own tensor file."""
    (
        output_dir, gamma, episode_start, successes_per_gamma,
        max_attempts_per_gamma, device, T_max,
    ) = payload
    output = Path(output_dir).resolve()
    path = output / f"sfm_hp100_windows_g{float(gamma)}.pt"
    progress_path = output / f"collection_progress_g{float(gamma)}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing data: {path}")
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress: {progress_path}")
    accepted, summaries, successful_ids = [], [], []
    episode = int(episode_start)
    while (
        len(successful_ids) < int(successes_per_gamma)
        and len(summaries) < int(max_attempts_per_gamma)
    ):
        records, summary = rollout_fn(
            int(episode), float(gamma), device=device, T_max=int(T_max)
        )
        summaries.append(summary)
        if summary["success"]:
            accepted.extend(records)
            successful_ids.append(int(episode))
        _atomic_json_save(dict(
            status="HP100_GAMMA_COLLECTION_IN_PROGRESS",
            gamma=float(gamma), accepted_successes=len(successful_ids),
            target_successes=int(successes_per_gamma), attempted_episodes=len(summaries),
            max_attempts=int(max_attempts_per_gamma), latest_episode=int(episode),
            latest_outcome={
                key: summary[key]
                for key in ("success", "collision", "timeout", "steps", "min_clearance")
            },
        ), progress_path)
        episode += 1
    if len(successful_ids) != int(successes_per_gamma):
        raise RuntimeError(
            f"gamma {gamma} obtained {len(successful_ids)}/{successes_per_gamma} "
            f"successful trajectories in {max_attempts_per_gamma} attempts"
        )
    tensors = pack_records(accepted)
    successful_episodes = sorted({int(value) for value in tensors["episode"].tolist()})
    if successful_episodes != successful_ids:
        raise RuntimeError(f"gamma {gamma} packed lineage IDs disagree with success ledger")
    packed = dict(
        schema_version=SCHEMA_VERSION, success_only=True, gamma=float(gamma),
        n_traj=len(successful_episodes), n_seeds=len(summaries),
        episode_start=int(episode_start), episode_stop_exclusive=int(episode),
        dynamics=DYN.contract(), **tensors,
    )
    _atomic_torch_save(packed, path)
    _atomic_json_save(dict(
        status="HP100_GAMMA_COLLECTION_COMPLETE",
        gamma=float(gamma), accepted_successes=len(successful_ids),
        target_successes=int(successes_per_gamma), attempted_episodes=len(summaries),
        episode_range=[int(episode_start), int(episode)], data_file=path.name,
        data_sha256=sha256_file(path),
    ), progress_path)
    row = dict(
        gamma=float(gamma), file=path.name, sha256=sha256_file(path),
        bytes=path.stat().st_size, windows=len(tensors["episode"]),
        n_traj=len(successful_episodes), successful_episodes=successful_episodes,
        attempted_episodes=len(summaries),
        rejected_episodes=[
            int(item["episode"]) for item in summaries if not bool(item["success"])
        ],
        episode_range=[int(episode_start), int(episode)],
        progress_file=progress_path.name,
        progress_sha256=sha256_file(progress_path),
    )
    return row, summaries


def generate_dataset(
    output_dir,
    *,
    episode_start: int = EPISODE_START,
    successes_per_gamma: int = SUCCESSES_PER_GAMMA,
    max_attempts_per_gamma: int = MAX_ATTEMPTS_PER_GAMMA,
    gammas=SS.GAMMAS,
    device: str = "cpu",
    T_max: int = T,
    rollout_fn=rollout_episode,
    expected_source_commit: str | None = None,
    jobs: int = 1,
) -> dict:
    """Generate per-gamma successful-only files and an authenticated manifest."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    if int(successes_per_gamma) <= 0:
        raise ValueError("successes_per_gamma must be positive")
    if int(max_attempts_per_gamma) < int(successes_per_gamma):
        raise ValueError("max_attempts_per_gamma must be at least successes_per_gamma")
    git = _git_provenance()
    source_hashes = _source_hashes()
    if expected_source_commit is not None and git["head"] != str(expected_source_commit):
        raise RuntimeError(
            f"source commit {git['head']} != expected {expected_source_commit}"
        )
    canonical_request = bool(
        int(episode_start) == EPISODE_START
        and int(successes_per_gamma) == SUCCESSES_PER_GAMMA
        and tuple(map(float, gammas)) == tuple(map(float, SS.GAMMAS))
        and int(T_max) == T
    )
    if canonical_request and not git["clean"]:
        raise RuntimeError("canonical HP100 collection requires a clean frozen worktree")
    if canonical_request and expected_source_commit is None:
        raise RuntimeError("canonical HP100 collection requires --expected-source-commit")
    gamma_values = tuple(map(float, gammas))
    if int(jobs) < 1:
        raise ValueError("jobs must be positive")
    worker_payloads = [(
        str(output), gamma, int(episode_start), int(successes_per_gamma),
        int(max_attempts_per_gamma), str(device), int(T_max),
    ) for gamma in gamma_values]
    if int(jobs) > 1:
        if rollout_fn is not rollout_episode:
            raise ValueError("parallel collection requires the canonical rollout function")
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(jobs), len(worker_payloads)), mp_context=context,
        ) as executor:
            results = list(executor.map(_collect_gamma, worker_payloads))
    else:
        results = [
            _collect_gamma(worker_payload, rollout_fn=rollout_fn)
            for worker_payload in worker_payloads
        ]
    results.sort(key=lambda result: result[0]["gamma"])
    final_git = _assert_provenance_unchanged(git, source_hashes)
    file_rows = [result[0] for result in results]
    rollout_summaries = {
        str(result[0]["gamma"]): result[1] for result in results
    }
    manifest = dict(
        status="HP100_ID_DATASET_COMPLETE",
        schema_version=SCHEMA_VERSION,
        canonical_full_run=bool(
            canonical_request
            and sum(row["n_traj"] for row in file_rows)
            == SUCCESSES_PER_GAMMA * len(SS.GAMMAS)
        ),
        role="successful SafeMPPI ID demonstrations for fresh Hp100 pretraining",
        total_successful_lineages=sum(row["n_traj"] for row in file_rows),
        episode_allocation=dict(
            start=int(episode_start),
            successful_trajectories_per_gamma=int(successes_per_gamma),
            max_attempts_per_gamma=int(max_attempts_per_gamma),
            terminal_ranges={
                str(row["gamma"]): row["episode_range"] for row in file_rows
            },
        ),
        environment=dict(
            n_ped=N_PED,
            ped_speed_range=list(PED_SPEED_RANGE),
            gammas=list(map(float, gammas)),
            horizon=HORIZON,
            T=int(T_max),
            goal=np.asarray(SS.GOAL, float).tolist(),
            task_bounds=[float(SS.TASK_LO), float(SS.TASK_HI)],
            pedestrian_radius=float(SS.R_PED),
            sensing_radius=float(SS.R_SENSE),
        ),
        expert=dict(
            name=EXPERT.EXPERT_NAME,
            config=locked_expert_config(),
            execution="CappedSafeMPPIAdapter using sfm_hp100_dynamics for internal and real steps",
            supervised_target="next H=10 executed controls; repeat final action at terminal prefix",
        ),
        feature=dict(
            shape=[32, 100],
            dtype="float32",
            temporal_storage="current Hp frame; Hp10 history is built trajectory-locally by the loader",
            radial_bins=100,
            angular_bins=32,
            radial_pooling="none",
            nominal_polytope_n_base=N_BASE,
            velocity_aware=True,
            predict_gain=float(locked_expert_config()["predict_gain"]),
            predict_tau=float(HORIZON * DYN.DT),
            planner_geometry_runtime_assertion="A,b,ref,margins and raster checked at every context",
            construction="fresh from raw state and pedestrian geometry; no interpolation or old-grid upsample",
            contract=(HPF.contract() if hasattr(HPF, "contract") else None),
        ),
        dynamics=DYN.contract(),
        files=file_rows,
        source_hashes=source_hashes,
        source_git=git,
        source_completion_audit=dict(
            git=final_git,
            source_hashes_equal=True,
            manifest_published_only_after_all_workers_completed=True,
        ),
        parallelism=dict(
            jobs=int(jobs), start_method=("spawn" if int(jobs) > 1 else "none"),
            device=str(device), gamma_workers_are_independent=True,
        ),
        rollout_summaries=rollout_summaries,
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    with open(temporary, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-start", type=int, default=EPISODE_START)
    parser.add_argument(
        "--successes-per-gamma", type=int, default=SUCCESSES_PER_GAMMA
    )
    parser.add_argument(
        "--max-attempts-per-gamma", type=int, default=MAX_ATTEMPTS_PER_GAMMA
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=SS.GAMMAS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--expected-source-commit", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fill one successful gamma=0.5 trajectory; still uses T=180 and the locked expert",
    )
    args = parser.parse_args()
    if args.smoke:
        args.episode_start = EPISODE_START
        args.successes_per_gamma = 1
        args.max_attempts_per_gamma = 20
        args.gammas = [0.5]
    manifest = generate_dataset(
        args.output_dir,
        episode_start=args.episode_start,
        successes_per_gamma=args.successes_per_gamma,
        max_attempts_per_gamma=args.max_attempts_per_gamma,
        gammas=args.gammas,
        device=args.device,
        expected_source_commit=args.expected_source_commit,
        jobs=args.jobs,
    )
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
        "canonical_full_run": manifest["canonical_full_run"],
    }), flush=True)


if __name__ == "__main__":
    main()
