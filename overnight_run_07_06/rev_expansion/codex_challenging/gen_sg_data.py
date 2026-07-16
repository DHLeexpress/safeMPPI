#!/usr/bin/env python3
"""Generate goal-aware SafeMPPI demonstrations for random start/goal pairs.

Subcommands provide a resumable pipeline:

* ``manifest`` selects unique random pairs from the approved blue/red pools;
* ``worker`` runs one contiguous pair shard on CPU or CUDA;
* ``merge`` assembles worker shards into ``w8sg_windows_g{gamma}.pt``.

Every training record includes the episode start and goal.  ``low5`` is built
with the episode goal, never the environment's canonical (5,5) goal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
STAGE_DIR = HERE / "stage_results" / "02_demos"
DATA_DIR = STAGE_DIR / "data"
WORKER_DIR = DATA_DIR / "workers"
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _paths  # noqa: F401,E402
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import grid_feats as GF  # noqa: E402
import grid_scene as GS  # noqa: E402
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter  # noqa: E402

import gen_uniform_data as SEEDS  # noqa: E402
from viz_style import GAMMAS  # noqa: E402


def gamma_tag(gamma: float) -> str:
    return str(float(gamma))


def create_manifest(output: Path, n_pairs: int, seed: int, overwrite: bool = False) -> dict[str, Any]:
    """Choose unique Cartesian-product pairs, identically reused by every gamma."""
    if output.exists() and not overwrite:
        with np.load(output) as saved:
            if len(saved["starts"]) != n_pairs or int(saved["seed"]) != seed:
                raise ValueError(f"existing manifest {output} does not match n={n_pairs}, seed={seed}")
            return {"n_pairs": len(saved["starts"]), "seed": int(saved["seed"]), "reused": True}

    env = SEEDS.make_walled_env(8)
    blue, red = SEEDS.start_goal_pools(env)
    total = len(blue) * len(red)
    if n_pairs > total:
        raise ValueError(f"requested {n_pairs} unique pairs but only {total} exist")
    rng = np.random.default_rng(seed)
    flat = rng.choice(total, size=n_pairs, replace=False)
    start_indices = flat // len(red)
    goal_indices = flat % len(red)
    starts = blue[start_indices].astype(np.float32)
    goals = red[goal_indices].astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        starts=starts,
        goals=goals,
        start_indices=start_indices.astype(np.int32),
        goal_indices=goal_indices.astype(np.int32),
        pair_indices=np.arange(n_pairs, dtype=np.int32),
        seed=np.int64(seed),
        wall_plugs=np.int32(8),
    )
    return {
        "n_pairs": n_pairs,
        "seed": seed,
        "start_pool": len(blue),
        "goal_pool": len(red),
        "unique_pairs": len(np.unique(flat)),
        "reused": False,
    }


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def goal_windows(
    states: np.ndarray,
    controls: np.ndarray,
    env,
    goal: np.ndarray,
    gamma: float,
    *,
    K: int = GF.K_HIST,
    H: int = GF.H_PRED,
):
    """Slice one episode using its sampled goal in every low-dimensional context."""
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    grids, lows, histories, targets = [], [], [], []
    for step in range(len(controls)):
        grids.append(GF.axis_grid(states[step, :2], obs, rr))
        lows.append(GF.low5(states[step], goal, gamma))
        histories.append(GF.hist_pad(controls[max(0, step - K):step], K))
        target = controls[step:step + H]
        if len(target) < H:
            target = np.concatenate((target, np.repeat(target[-1:], H - len(target), axis=0)), axis=0)
        targets.append(target.astype(np.float32))
    return grids, lows, histories, targets


def _torch_di_step(state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
    result = state.clone()
    result[0] = state[0] + dt * state[2] + 0.5 * dt * dt * action[0]
    result[1] = state[1] + dt * state[3] + 0.5 * dt * dt * action[1]
    result[2] = state[2] + dt * action[0]
    result[3] = state[3] + dt * action[1]
    return result


def classify_path(path: np.ndarray, env, goal: np.ndarray, reach: float) -> dict[str, Any]:
    xy = np.asarray(path, dtype=float)[:, :2]
    obs = env.obstacles.detach().cpu().numpy()
    clearance = (
        np.linalg.norm(xy[:, None, :] - obs[None, :, :2], axis=2)
        - obs[None, :, 2]
        - float(env.r_robot)
    )
    collision = bool((clearance.min(axis=1) < 0.0).any())
    reached = bool(np.linalg.norm(xy[-1] - goal) < reach)
    in_taskspace = bool(((xy >= 0.0) & (xy <= 5.0)).all())
    return {
        "success": reached and not collision and in_taskspace,
        "reached": reached,
        "collision": collision,
        "in_taskspace": in_taskspace,
        "min_clearance": float(clearance.min()),
        "endpoint_distance": float(np.linalg.norm(xy[-1] - goal)),
    }


def rollout_pair(
    env,
    cfg: dict,
    start: np.ndarray,
    goal: np.ndarray,
    gamma: float,
    pair_index: int,
    *,
    device: torch.device,
    reach: float,
    seed_base: int,
    max_retries: int,
):
    """Run an episode, retrying only the planner randomness for the same pair."""
    obs_plan = GS.planner_obstacles(env).to(device)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
    last = None
    for retry in range(max_retries + 1):
        adapter = SafeMPPIAdapter(**cfg)
        state = torch.tensor([start[0], start[1], 0.0, 0.0], dtype=torch.float32, device=device)
        states = [state.detach().cpu().numpy().copy()]
        controls = []
        episode_seed = int(seed_base + pair_index + retry * 1_000_000)
        started = time.perf_counter()
        with torch.no_grad():
            for step in range(env.T):
                action, _info = adapter.plan(
                    state,
                    goal_t,
                    obs_plan,
                    gamma=float(gamma),
                    seed=episode_seed * 1000 + step,
                )
                state = _torch_di_step(state, action, float(env.dt))
                controls.append(action.detach().cpu().numpy().astype(np.float32))
                states.append(state.detach().cpu().numpy().astype(np.float32))
                if float(torch.linalg.norm(state[:2] - goal_t)) < reach:
                    break
        states_np = np.asarray(states, dtype=np.float32)
        controls_np = np.asarray(controls, dtype=np.float32)
        status = classify_path(states_np, env, goal, reach)
        last = (states_np, controls_np, status, retry + 1, time.perf_counter() - started)
        if status["success"]:
            break
    return last


def _empty_array(shape, dtype=np.float32):
    return np.empty((0, *shape), dtype=dtype)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    manifest = load_manifest(args.manifest)
    n_pairs = len(manifest["starts"])
    if args.start is None or args.stop is None:
        edges = np.linspace(0, n_pairs, args.num_shards + 1, dtype=int)
        start_i, stop_i = int(edges[args.shard_id]), int(edges[args.shard_id + 1])
    else:
        start_i, stop_i = int(args.start), int(args.stop)
    if not (0 <= start_i < stop_i <= n_pairs):
        raise ValueError(f"invalid worker range [{start_i}, {stop_i}) for {n_pairs} pairs")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA worker requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    env = SEEDS.make_walled_env(8)
    cfg = GS.mode1_config()

    grids: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_ok: list[np.ndarray] = []
    goals_ok: list[np.ndarray] = []
    pair_ok: list[int] = []
    window_counts: list[int] = []
    window_starts: list[np.ndarray] = []
    window_goals: list[np.ndarray] = []
    window_pairs: list[int] = []

    paths: list[np.ndarray] = []
    path_controls: list[np.ndarray] = []
    successes: list[bool] = []
    attempts: list[int] = []
    steps: list[int] = []
    min_clearances: list[float] = []
    endpoint_distances: list[float] = []
    episode_seconds: list[float] = []

    total_started = time.perf_counter()
    for local, pair_index in enumerate(range(start_i, stop_i), start=1):
        start = manifest["starts"][pair_index].astype(np.float32)
        goal = manifest["goals"][pair_index].astype(np.float32)
        states, controls, status, n_attempts, elapsed = rollout_pair(
            env,
            cfg,
            start,
            goal,
            args.gamma,
            pair_index,
            device=device,
            reach=args.reach,
            seed_base=args.seed_base,
            max_retries=args.max_retries,
        )
        paths.append(states)
        path_controls.append(controls)
        successes.append(bool(status["success"]))
        attempts.append(int(n_attempts))
        steps.append(len(controls))
        min_clearances.append(float(status["min_clearance"]))
        endpoint_distances.append(float(status["endpoint_distance"]))
        episode_seconds.append(float(elapsed))

        if status["success"] and len(controls) >= 2:
            g, l, h, u = goal_windows(states, controls, env, goal, args.gamma)
            n_windows = len(g)
            grids.extend(g)
            lows.extend(l)
            histories.extend(h)
            targets.extend(u)
            start4 = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float32)
            starts_ok.append(start4)
            goals_ok.append(goal)
            pair_ok.append(pair_index)
            window_counts.append(n_windows)
            window_starts.extend([start] * n_windows)
            window_goals.extend([goal] * n_windows)
            window_pairs.extend([pair_index] * n_windows)

        if local % args.progress_every == 0 or local == stop_i - start_i:
            rate = sum(successes) / local
            per_pair = (time.perf_counter() - total_started) / local
            print(
                f"gamma={args.gamma:g} shard={args.shard_id} {local}/{stop_i-start_i} "
                f"success={sum(successes)} ({rate:.1%}) windows={len(grids)} "
                f"{per_pair:.2f}s/pair",
                flush=True,
            )

    grid_np = np.asarray(grids, dtype=np.float32) if grids else _empty_array((3, GF.N_THETA, GF.N_R))
    low_np = np.asarray(lows, dtype=np.float32) if lows else _empty_array((5,))
    hist_np = np.asarray(histories, dtype=np.float32) if histories else _empty_array((GF.K_HIST, 2))
    target_np = np.asarray(targets, dtype=np.float32) if targets else _empty_array((GF.H_PRED, 2))
    worker_payload = {
        "grid": torch.from_numpy(grid_np),
        "low5": torch.from_numpy(low_np),
        "hist": torch.from_numpy(hist_np),
        "U": torch.from_numpy(target_np),
        "starts": torch.from_numpy(np.asarray(starts_ok, dtype=np.float32).reshape(-1, 4)),
        "goals": torch.from_numpy(np.asarray(goals_ok, dtype=np.float32).reshape(-1, 2)),
        "pair_indices": torch.tensor(pair_ok, dtype=torch.int32),
        "window_counts": torch.tensor(window_counts, dtype=torch.int32),
        "window_starts": torch.from_numpy(np.asarray(window_starts, dtype=np.float32).reshape(-1, 2)),
        "window_goals": torch.from_numpy(np.asarray(window_goals, dtype=np.float32).reshape(-1, 2)),
        "window_pair_indices": torch.tensor(window_pairs, dtype=torch.int32),
        "gamma": float(args.gamma),
        "pair_start": start_i,
        "pair_stop": stop_i,
        "n_pairs": stop_i - start_i,
        "n_traj": len(pair_ok),
        "n_windows": len(grids),
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.paths_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(worker_payload, args.out)
    np.savez_compressed(
        args.paths_out,
        paths=_object_array(paths),
        controls=_object_array(path_controls),
        pair_indices=np.arange(start_i, stop_i, dtype=np.int32),
        starts=manifest["starts"][start_i:stop_i].astype(np.float32),
        goals=manifest["goals"][start_i:stop_i].astype(np.float32),
        success=np.asarray(successes, dtype=bool),
        attempts=np.asarray(attempts, dtype=np.int16),
        steps=np.asarray(steps, dtype=np.int16),
        min_clearance=np.asarray(min_clearances, dtype=np.float32),
        endpoint_distance=np.asarray(endpoint_distances, dtype=np.float32),
        episode_seconds=np.asarray(episode_seconds, dtype=np.float32),
        gamma=np.float32(args.gamma),
    )
    summary = {
        "gamma": float(args.gamma),
        "pair_start": start_i,
        "pair_stop": stop_i,
        "pairs": stop_i - start_i,
        "successes": int(sum(successes)),
        "windows": len(grids),
        "wall_seconds": time.perf_counter() - total_started,
        "mean_episode_seconds": float(np.mean(episode_seconds)),
        "output": str(args.out.resolve()),
        "paths_output": str(args.paths_out.resolve()),
    }
    print("WORKER_DONE " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _object_array(items) -> np.ndarray:
    output = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        output[i] = item
    return output


def merge_workers(args: argparse.Namespace) -> dict[str, Any]:
    shards = []
    for path in args.shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if abs(float(payload["gamma"]) - args.gamma) > 1e-7:
            raise ValueError(f"gamma mismatch in {path}")
        shards.append((Path(path), payload))
    shards.sort(key=lambda item: int(item[1]["pair_start"]))
    if not shards:
        raise ValueError("no worker shards")
    cursor = 0
    for path, shard in shards:
        if int(shard["pair_start"]) != cursor:
            raise ValueError(f"gap/overlap before {path}: expected {cursor}, got {shard['pair_start']}")
        cursor = int(shard["pair_stop"])
    manifest = load_manifest(args.manifest)
    if cursor != len(manifest["starts"]):
        raise ValueError(f"worker coverage ends at {cursor}, manifest has {len(manifest['starts'])}")

    tensor_keys = (
        "grid", "low5", "hist", "U", "starts", "goals", "pair_indices", "window_counts",
        "window_starts", "window_goals", "window_pair_indices",
    )
    merged = {key: torch.cat([shard[key] for _, shard in shards], dim=0) for key in tensor_keys}
    merged.update(
        {
            "gamma": float(args.gamma),
            "n_pairs": len(manifest["starts"]),
            "n_seeds": len(manifest["starts"]),
            "n_traj": len(merged["starts"]),
            "n_windows": len(merged["grid"]),
            "wall_plugs": 8,
            "reach": float(args.reach),
            "manifest": str(args.manifest.resolve()),
            "schema_version": "w8sg-v1-goal-aware",
        }
    )
    if len(merged["grid"]) != int(merged["window_counts"].sum()):
        raise RuntimeError("window count mismatch after merge")
    if len(merged["window_goals"]) != len(merged["grid"]):
        raise RuntimeError("per-window goal count mismatch")
    if len(merged["low5"]) and not torch.allclose(
        merged["low5"][:, 4], torch.full_like(merged["low5"][:, 4], float(args.gamma))
    ):
        raise RuntimeError("low5 gamma channel mismatch")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)

    path_parts = []
    for shard_path, _payload in shards:
        path_file = shard_path.with_name(shard_path.stem + "_paths.npz")
        if not path_file.exists():
            raise FileNotFoundError(path_file)
        with np.load(path_file, allow_pickle=True) as saved:
            path_parts.append({key: np.asarray(saved[key]) for key in saved.files})
    path_keys = (
        "paths", "controls", "pair_indices", "starts", "goals", "success", "attempts", "steps",
        "min_clearance", "endpoint_distance", "episode_seconds",
    )
    combined_paths = {key: np.concatenate([part[key] for part in path_parts], axis=0) for key in path_keys}
    np.savez_compressed(args.paths_out, **combined_paths, gamma=np.float32(args.gamma))

    success = combined_paths["success"].astype(bool)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gamma": float(args.gamma),
        "pairs": len(success),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "windows": len(merged["grid"]),
        "mean_steps_success": float(combined_paths["steps"][success].mean()) if success.any() else None,
        "min_clearance_success": float(combined_paths["min_clearance"][success].min()) if success.any() else None,
        "dataset": str(args.out.resolve()),
        "paths": str(args.paths_out.resolve()),
        "worker_shards": [str(path.resolve()) for path, _ in shards],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("MERGE_DONE " + json.dumps(summary, sort_keys=True), flush=True)

    if args.cleanup:
        for shard_path, _ in shards:
            shard_path.unlink()
            shard_path.with_name(shard_path.stem + "_paths.npz").unlink()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--out", type=Path, default=DATA_DIR / "random_pairs_300.npz")
    manifest.add_argument("--pairs", type=int, default=300)
    manifest.add_argument("--seed", type=int, default=20260714)
    manifest.add_argument("--overwrite", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", type=Path, default=DATA_DIR / "random_pairs_300.npz")
    worker.add_argument("--gamma", type=float, required=True)
    worker.add_argument("--shard-id", type=int, default=0)
    worker.add_argument("--num-shards", type=int, default=1)
    worker.add_argument("--start", type=int)
    worker.add_argument("--stop", type=int)
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--reach", type=float, default=0.2)
    worker.add_argument("--seed-base", type=int, default=41000)
    worker.add_argument("--max-retries", type=int, default=1)
    worker.add_argument("--progress-every", type=int, default=10)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--paths-out", type=Path, required=True)

    merge = sub.add_parser("merge")
    merge.add_argument("--manifest", type=Path, default=DATA_DIR / "random_pairs_300.npz")
    merge.add_argument("--gamma", type=float, required=True)
    merge.add_argument("--reach", type=float, default=0.2)
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--paths-out", type=Path, required=True)
    merge.add_argument("--summary", type=Path, required=True)
    merge.add_argument("--cleanup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "manifest":
        result = create_manifest(args.out, args.pairs, args.seed, args.overwrite)
        print("MANIFEST " + json.dumps({**result, "output": str(args.out.resolve())}, sort_keys=True))
    elif args.command == "worker":
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError("shard-id must be in [0, num-shards)")
        run_worker(args)
    elif args.command == "merge":
        merge_workers(args)


if __name__ == "__main__":
    main()
