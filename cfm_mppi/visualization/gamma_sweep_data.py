from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from cfm_mppi.evaluation.eval_benchmark import DEFAULTS, BenchmarkPolicies, _dynamics_step, _make_episode, _set_seed
from cfm_mppi.evaluation.metrics import compute_episode_metrics
from cfm_mppi.safegpc_adapter import SafeMPPIAdapter


def build_gamma_grid(values: Optional[Sequence[float]] = None, count: int = 21) -> List[float]:
    raw = [float(v) for v in values] if values else np.linspace(0.0, 1.0, max(2, int(count))).tolist()
    grid = sorted({round(float(np.clip(v, 0.0, 1.0)), 10) for v in raw})
    if 0.0 not in grid:
        grid.insert(0, 0.0)
    if 1.0 not in grid:
        grid.append(1.0)
    return grid


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mizuta_checkpoint=args.mizuta_checkpoint,
        safe_cfm_checkpoint=args.safe_cfm_checkpoint,
        drifting_checkpoint=args.drifting_checkpoint,
        smoke=args.smoke,
        seed=args.seed,
    )


def _finish(args: argparse.Namespace, method: str, episode: int, gamma: Optional[float], states, controls, obstacles, goal, times, infos):
    states = np.asarray(states, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32) if controls else np.zeros((0, 2), dtype=np.float32)
    min_h = None
    violations = 0
    for info in infos:
        if info.get("min_barrier_h") is not None:
            h = float(info["min_barrier_h"])
            min_h = h if min_h is None else min(min_h, h)
        violations += int(info.get("num_barrier_violations", 0) or 0)
    rec = compute_episode_metrics(
        states,
        controls,
        obstacles,
        goal,
        safety_margin=args.safety_margin,
        success_threshold=args.success_threshold,
        planning_times=times,
        min_barrier_h=min_h,
        num_barrier_violations=violations,
    )
    calls = [int(i.get("model_calls_per_step", 0) or 0) for i in infos]
    nfes = [int(i.get("nfe", 0) or 0) for i in infos]
    rec.update(
        method=method,
        episode=int(episode),
        seed=int(args.seed),
        dataset=args.dataset,
        dynamics=args.dynamics,
        gamma=gamma,
        safety_margin=float(args.safety_margin),
        safety_guarantee_scope="linear_system_theorem_relevant" if args.dynamics == "doubleintegrator" else "empirical_only_unicycle",
        model_calls_per_step=float(np.mean(calls)) if calls else 0.0,
        nfe=float(np.mean(nfes)) if nfes else 0.0,
        checkpoint_path=args.mizuta_checkpoint if method == "mizuta_cfm_mppi" else None,
        states=states.astype(float).tolist(),
        controls=controls.astype(float).tolist(),
        obstacles=obstacles.astype(float).tolist(),
        goal=goal.astype(float).tolist(),
        planning_times=[float(t) for t in times],
        step_infos=jsonable(infos),
    )
    if method == "safemppi_gamma":
        rec["safemppi_num_samples"] = int(args.safemppi_num_samples)
        rec["safemppi_horizon"] = int(args.safemppi_horizon)
    return jsonable(rec)


def rollout_mizuta(args: argparse.Namespace, episode: int) -> Dict[str, Any]:
    policy = BenchmarkPolicies(policy_args(args), torch.device(args.device))
    state, goal, obstacles = _make_episode(args.seed + episode, args.dynamics, args.dataset)
    states, controls, times, infos = [state.copy()], [], [], []
    for step in range(args.horizon):
        action, info = policy.action("mizuta_cfm_mppi", state, goal, obstacles, controls, args.dynamics, 0.5, args.horizon)
        times.append(float(info.get("planning_wall_time", 0.0)))
        infos.append({"step": step, **jsonable(info)})
        controls.append(action.copy())
        state = _dynamics_step(state, action, args.dynamics, args.dt)
        states.append(state.copy())
        if np.linalg.norm(state[:2] - goal[:2]) <= args.success_threshold:
            break
    return _finish(args, "mizuta_cfm_mppi", episode, None, states, controls, obstacles, goal, times, infos)


def rollout_safemppi(args: argparse.Namespace, episode: int, gamma: float) -> Dict[str, Any]:
    device = torch.device(args.device)
    state, goal, obstacles = _make_episode(args.seed + episode, args.dynamics, args.dataset)
    planner = SafeMPPIAdapter(
        horizon=min(args.horizon, args.safemppi_horizon),
        dt=args.dt,
        num_samples=args.safemppi_num_samples,
        gamma=gamma,
        dynamics_type=args.dynamics,
        noise_sigma=args.safemppi_noise_sigma,
        temperature=args.safemppi_temperature,
        u_min=tuple(args.u_min),
        u_max=tuple(args.u_max),
        check_first_control_only=args.check_first_control_only,
    )
    states, controls, times, infos = [state.copy()], [], [], []
    for step in range(args.horizon):
        t0 = time.perf_counter()
        action_t, info = planner.plan(
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(goal, dtype=torch.float32, device=device),
            torch.tensor(obstacles, dtype=torch.float32, device=device),
            gamma=gamma,
            seed=args.seed + 100000 * episode + step,
        )
        action = action_t.detach().cpu().numpy().astype(np.float32)
        times.append(float(info.get("solve_time", time.perf_counter() - t0)))
        infos.append({"step": step, **jsonable(info)})
        controls.append(action.copy())
        state = _dynamics_step(state, action, args.dynamics, args.dt)
        states.append(state.copy())
        if np.linalg.norm(state[:2] - goal[:2]) <= args.success_threshold:
            break
    return _finish(args, "safemppi_gamma", episode, gamma, states, controls, obstacles, goal, times, infos)
