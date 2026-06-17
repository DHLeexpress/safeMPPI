from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from cfm_mppi.evaluation.metrics import compute_episode_metrics
from cfm_mppi.evaluation.result_writer import JSONLResultWriter, write_summary
from cfm_mppi.evaluation.eval_utils import CFMConfig, synthesize_control
from cfm_mppi.evaluation.run_drifting import load_drifting, sample_drifting_controls
from cfm_mppi.evaluation.run_safe_cfm import load_safe_cfm, sample_safe_cfm_controls
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import doubleintegrator_dynamics, stage_cost, terminal_cost, unicycle_dynamics
from cfm_mppi.safegpc_adapter import SafeMPPIAdapter, resolve_gamma_schedule


DEFAULTS = {
    "horizon": 80,
    "dt": 0.1,
    "safety_margin": 0.5,
    "success_threshold": 0.5,
    "u_min": (-2.0, -2.0),
    "u_max": (2.0, 2.0),
}


class _AgentHistory:
    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        self.data = None

    def update(self, new_data: torch.Tensor):
        new_data = new_data.unsqueeze(-1)
        if self.data is None:
            self.data = new_data
        elif len(self) < self.max_length:
            self.data = torch.cat([self.data, new_data], dim=-1)
        else:
            self.data = torch.cat([self.data[..., 1:], new_data], dim=-1)

    def get(self):
        return self.data

    def __len__(self):
        return self.data.shape[-1] if self.data is not None else 0


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dynamics_step(state: np.ndarray, action: np.ndarray, dynamics: str, dt: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if dynamics == "doubleintegrator":
        x = state.copy()
        x[0] = state[0] + dt * state[2] + 0.5 * dt * dt * action[0]
        x[1] = state[1] + dt * state[3] + 0.5 * dt * dt * action[1]
        x[2] = state[2] + dt * action[0]
        x[3] = state[3] + dt * action[1]
        return x.astype(np.float32)
    if dynamics == "unicycle":
        x = state.copy()
        x[0] = state[0] + dt * action[0] * np.cos(state[2])
        x[1] = state[1] + dt * action[0] * np.sin(state[2])
        x[2] = np.arctan2(np.sin(state[2] + dt * action[1]), np.cos(state[2] + dt * action[1]))
        return x.astype(np.float32)
    x = state.copy()
    x[:2] = state[:2] + dt * action
    return x.astype(np.float32)


def _goal_action(state: np.ndarray, goal: np.ndarray, dynamics: str) -> np.ndarray:
    direction = goal[:2] - state[:2]
    if dynamics == "unicycle":
        desired = np.arctan2(direction[1], direction[0])
        angle_err = np.arctan2(np.sin(desired - state[2]), np.cos(desired - state[2]))
        return np.array([np.clip(np.linalg.norm(direction), -2.0, 2.0), np.clip(2.0 * angle_err, -2.0, 2.0)], np.float32)
    if dynamics == "doubleintegrator":
        vel_err = -state[2:4] if state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
        return np.clip(0.6 * direction + 0.8 * vel_err, -2.0, 2.0).astype(np.float32)
    return np.clip(direction, -2.0, 2.0).astype(np.float32)


def _make_episode(seed: int, dynamics: str, dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    if dynamics == "unicycle":
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([6.0, 6.0], dtype=np.float32)
    n_obs = 5 if dataset == "sfm" else 8
    obs = []
    attempts = 0
    while len(obs) < n_obs and attempts < 500:
        attempts += 1
        c = rng.uniform(1.0, 5.5, size=2)
        r = rng.uniform(0.25, 0.55)
        if np.linalg.norm(c - start[:2]) < 1.2 or np.linalg.norm(c - goal) < 1.2:
            continue
        obs.append([c[0], c[1], r])
    return start, goal, np.asarray(obs, dtype=np.float32)


def _context_batch(state: np.ndarray, goal: np.ndarray, controls: List[np.ndarray], obstacles: np.ndarray, gamma: float, safety_margin: float, horizon: int) -> Dict[str, torch.Tensor]:
    state4 = np.zeros(4, dtype=np.float32)
    state4[: min(len(state), 4)] = state[: min(len(state), 4)]
    action_hist = np.zeros((10, 2), dtype=np.float32)
    if controls:
        recent = np.asarray(controls[-10:], dtype=np.float32)
        action_hist[-len(recent) :] = recent
    ego_hist = np.tile(state4[None, :], (10, 1)).astype(np.float32)
    rel = np.zeros(4, dtype=np.float32)
    if obstacles.size:
        d = np.linalg.norm(obstacles[:, :2] - state[:2][None, :], axis=1) - obstacles[:, 2]
        idx = int(np.argmin(d))
        rel[:2] = obstacles[idx, :2] - state[:2]
    obs_hist = np.tile(rel[None, :], (10, 1)).astype(np.float32)
    states = np.tile(state4[None, None, :], (1, horizon + 1, 1)).astype(np.float32)
    return {
        "states": torch.from_numpy(states),
        "controls_si": torch.zeros(1, horizon, 2),
        "controls_dyn": torch.zeros(1, horizon, 2),
        "start": torch.from_numpy(state[:2].reshape(1, 2).astype(np.float32)),
        "goal": torch.from_numpy(goal.reshape(1, 2).astype(np.float32)),
        "ego_history": torch.from_numpy(ego_hist.reshape(1, 10, 4)),
        "action_history": torch.from_numpy(action_hist.reshape(1, 10, 2)),
        "nearest_obstacle_history": torch.from_numpy(obs_hist.reshape(1, 10, 4)),
        "gamma": torch.tensor([gamma], dtype=torch.float32),
        "safety_margin": torch.tensor([safety_margin], dtype=torch.float32),
    }


class BenchmarkPolicies:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self._mizuta = None
        self._mizuta_episode = None
        self._safe_cfm = None
        self._drifting = None

    def _load_mizuta(self):
        if self._mizuta is not None:
            return self._mizuta
        ckpt_path = Path(self.args.mizuta_checkpoint)
        if not ckpt_path.exists():
            if self.args.smoke:
                self._mizuta = False
                return self._mizuta
            raise FileNotFoundError(f"Missing Mizuta checkpoint: {ckpt_path}")
        model = TransformerModel().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._mizuta = model
        return model

    def _reset_mizuta_episode(self, state: np.ndarray, goal: np.ndarray, dynamics: str, horizon: int):
        n_sample = 20 if self.args.smoke else 200
        if n_sample % 5 != 0:
            n_sample = 5 * max(1, n_sample // 5)
        dim_state = 3 if dynamics == "unicycle" else 4
        dyn_fn = unicycle_dynamics if dynamics == "unicycle" else doubleintegrator_dynamics
        sigma = torch.tensor([0.3, 0.6] if dynamics == "unicycle" else [0.4, 0.4], dtype=torch.float32)
        solver = FlowMPPI(
            num_samples=n_sample,
            dim_state=dim_state,
            dim_control=2,
            dynamics=dyn_fn,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            u_min=torch.tensor(DEFAULTS["u_min"], dtype=torch.float32),
            u_max=torch.tensor(DEFAULTS["u_max"], dtype=torch.float32),
            sigmas=sigma,
            lambda_=0.1,
            goal=torch.tensor(goal, dtype=torch.float32),
            horizon=horizon,
            dt=DEFAULTS["dt"],
            device=self.device,
            dynamics_type=dynamics,
        )
        self._mizuta_episode = {
            "solver": solver,
            "x_t": torch.randn(n_sample, 2, horizon, dtype=torch.float32, device=self.device),
            "histories": {
                "ego_state": _AgentHistory(max_length=10),
                "ego_control_sin": _AgentHistory(max_length=10),
                "obs_state": _AgentHistory(max_length=10),
                "obs_control": _AgentHistory(max_length=10),
            },
            "last_controls_sin": None,
            "n_sample": n_sample,
            "horizon": horizon,
            "step": 0,
        }

    def _mizuta_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        controls: List[np.ndarray],
        dynamics: str,
        horizon: int,
    ) -> Tuple[np.ndarray, Dict]:
        model = self._load_mizuta()
        if model is False:
            return _goal_action(state, goal, dynamics), {
                "checkpoint": None,
                "model_calls_per_step": 0,
                "nfe": 0,
                "smoke_fallback": "goal_controller_missing_checkpoint",
            }
        if not controls or self._mizuta_episode is None:
            self._reset_mizuta_episode(state, goal, dynamics, horizon)
        ep = self._mizuta_episode
        histories = ep["histories"]
        n_sample = ep["n_sample"]
        pos_obs = torch.tensor(obstacles[:, :2], dtype=torch.float32, device=self.device).view(1, -1, 2)
        vel_obs = torch.zeros_like(pos_obs)

        if ep["last_controls_sin"] is not None:
            control_history_len = len(histories["ego_control_sin"])
            noise_level = torch.tensor([0.8], device=self.device)
            noise = torch.randn(n_sample, ep["last_controls_sin"].shape[1], ep["last_controls_sin"].shape[2], device=self.device)
            x_t = noise_level * ep["last_controls_sin"] / 10.0 + (1.0 - noise_level) * noise
            x_t = x_t[:, :, (control_history_len + 1) :]
            histories["ego_control_sin"].update(ep["last_controls_sin"][:, :, control_history_len])
            histories["ego_state"].update(torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1))
            histories["obs_state"].update(pos_obs)
            histories["obs_control"].update(vel_obs)
            control_history_sin = histories["ego_control_sin"].get()
            ep["x_t"] = torch.cat([control_history_sin.expand(n_sample, -1, -1) / 10.0, x_t], dim=-1)

        first_step = ep["step"] == 0
        ode_times = [0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0] if first_step else [0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        t_curr = torch.tensor([0.0], device=self.device) if first_step else torch.tensor([0.8], device=self.device)
        config = CFMConfig(
            ode_times=ode_times,
            dt=DEFAULTS["dt"],
            agent_radius=DEFAULTS["safety_margin"],
            space_scale=10.0,
            safe_margin_coefs=[0.1, 0.3, 0.5, 0.7, 0.9],
            goal_margin_coef=0.1,
            device=str(self.device),
        )
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=self.device).view(1, 2)
        with torch.no_grad():
            controls_dyn, controls_sin = synthesize_control(
                model,
                ep["solver"],
                config,
                state_t,
                goal_t,
                ep["x_t"],
                t_curr,
                pos_obs,
                vel_obs,
                ep["x_t"].shape[-1],
                histories=histories,
                d=0.1,
                k_p=3.0,
            )
        ep["last_controls_sin"] = controls_sin.detach()
        ep["step"] += 1
        return controls_dyn[0, :, 0].detach().cpu().numpy(), {
            "checkpoint": str(self.args.mizuta_checkpoint),
            "model_calls_per_step": len(ode_times),
            "nfe": len(ode_times),
        }

    def action(self, method: str, state: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, controls: List[np.ndarray], dynamics: str, gamma: float, horizon: int) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        info: Dict = {"model_calls_per_step": 0, "nfe": 0}
        if method == "mizuta_cfm_mppi":
            action, step_info = self._mizuta_action(state, goal, obstacles, controls, dynamics, horizon)
            info.update(step_info)
        elif method == "safemppi_gamma":
            adapter = SafeMPPIAdapter(
                horizon=min(20, horizon),
                dt=DEFAULTS["dt"],
                num_samples=32 if self.args.smoke else 128,
                gamma=gamma,
                dynamics_type=dynamics,
                u_min=DEFAULTS["u_min"],
                u_max=DEFAULTS["u_max"],
            )
            action_t, step_info = adapter.plan(
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(goal, dtype=torch.float32),
                torch.tensor(obstacles, dtype=torch.float32),
                gamma=gamma,
                seed=self.args.seed + len(controls),
            )
            action = action_t.detach().cpu().numpy()
            info.update(step_info)
            info.update({"model_calls_per_step": 0, "nfe": 0})
        elif method == "safe_cfm":
            ckpt_path = Path(self.args.safe_cfm_checkpoint)
            if not ckpt_path.exists() and not self.args.smoke:
                raise FileNotFoundError(f"Missing safe CFM checkpoint: {ckpt_path}")
            if ckpt_path.exists():
                if self._safe_cfm is None:
                    self._safe_cfm = load_safe_cfm(ckpt_path, self.device)
                batch = _context_batch(state, goal, controls, obstacles, gamma, DEFAULTS["safety_margin"], horizon)
                seq = sample_safe_cfm_controls(self._safe_cfm, batch, horizon=horizon, nfe=8, device=self.device)
                action = seq[0, :, 0].detach().cpu().numpy()
                info.update({"model_calls_per_step": 8, "nfe": 8, "checkpoint": str(ckpt_path)})
            else:
                action = _goal_action(state, goal, dynamics)
                info.update({"smoke_fallback": "goal_controller_missing_checkpoint", "checkpoint": None})
        elif method == "drifting":
            ckpt_path = Path(self.args.drifting_checkpoint)
            if not ckpt_path.exists() and not self.args.smoke:
                raise FileNotFoundError(f"Missing Drifting checkpoint: {ckpt_path}")
            if ckpt_path.exists():
                if self._drifting is None:
                    self._drifting = load_drifting(ckpt_path, self.device)
                batch = _context_batch(state, goal, controls, obstacles, gamma, DEFAULTS["safety_margin"], horizon)
                seq = sample_drifting_controls(self._drifting, batch, horizon=horizon, device=self.device)
                action = seq[0, :, 0].detach().cpu().numpy()
                info.update({"model_calls_per_step": 1, "nfe": 1, "checkpoint": str(ckpt_path)})
            else:
                action = _goal_action(state, goal, dynamics)
                info.update({"smoke_fallback": "goal_controller_missing_checkpoint", "checkpoint": None, "nfe": 1, "model_calls_per_step": 1})
        else:
            raise ValueError(f"Unknown method: {method}")
        action = np.clip(action, np.asarray(DEFAULTS["u_min"]), np.asarray(DEFAULTS["u_max"]))
        info["planning_wall_time"] = time.perf_counter() - t0
        return action.astype(np.float32), info


def _run_episode(args, policies, method: str, episode_idx: int, gamma: float | None) -> Dict:
    horizon = 20 if args.smoke else DEFAULTS["horizon"]
    dt = DEFAULTS["dt"]
    state, goal, obstacles = _make_episode(args.seed + episode_idx, args.dynamics, args.dataset)
    states = [state.copy()]
    controls: List[np.ndarray] = []
    planning_times = []
    min_barrier_h = None
    num_barrier_violations = 0
    checkpoint_path = None
    model_calls_per_step = 0
    nfe = 0
    gamma_value = float(gamma if gamma is not None else (args.gamma_grid[0] if args.gamma_grid else 0.5))
    for _ in range(horizon):
        action, info = policies.action(method, state, goal, obstacles, controls, args.dynamics, gamma_value, horizon)
        planning_times.append(info.get("planning_wall_time", 0.0))
        checkpoint_path = info.get("checkpoint", checkpoint_path)
        model_calls_per_step = int(info.get("model_calls_per_step", model_calls_per_step))
        nfe = int(info.get("nfe", nfe))
        if info.get("min_barrier_h") is not None:
            min_barrier_h = info["min_barrier_h"] if min_barrier_h is None else min(min_barrier_h, info["min_barrier_h"])
        num_barrier_violations += int(info.get("num_barrier_violations", 0))
        state = _dynamics_step(state, action, args.dynamics, dt)
        states.append(state.copy())
        controls.append(action.copy())
        if np.linalg.norm(state[:2] - goal[:2]) <= DEFAULTS["success_threshold"]:
            break
    states_arr = np.asarray(states, dtype=np.float32)
    controls_arr = np.asarray(controls, dtype=np.float32) if controls else np.zeros((0, 2), dtype=np.float32)
    metrics = compute_episode_metrics(
        states_arr,
        controls_arr,
        obstacles,
        goal,
        safety_margin=DEFAULTS["safety_margin"],
        success_threshold=DEFAULTS["success_threshold"],
        planning_times=planning_times,
        min_barrier_h=min_barrier_h,
        num_barrier_violations=num_barrier_violations,
    )
    scope = "linear_system_theorem_relevant" if args.dynamics == "doubleintegrator" else "empirical_only_unicycle"
    metrics.update(
        {
            "episode": episode_idx,
            "seed": args.seed,
            "dataset": args.dataset,
            "dynamics": args.dynamics,
            "method": method,
            "gamma": gamma_value if method == "safemppi_gamma" else None,
            "safe_coef": None,
            "safety_margin": DEFAULTS["safety_margin"],
            "safety_guarantee_scope": scope,
            "checkpoint_path": checkpoint_path,
            "config_path": None,
            "model_calls_per_step": model_calls_per_step,
            "nfe": nfe,
        }
    )
    return metrics


def get_parser():
    p = argparse.ArgumentParser(description="Unified CFM-MPPI / safeGPC benchmark harness.")
    p.add_argument("--dataset", default="sfm", choices=["sfm", "ucy", "sdd"])
    p.add_argument("--dynamics", default="doubleintegrator", choices=["doubleintegrator", "unicycle"])
    p.add_argument("--methods", nargs="+", default=["mizuta_cfm_mppi"])
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/benchmark")
    p.add_argument("--gamma-grid", nargs="*", type=float, default=None)
    p.add_argument("--gamma-schedule", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mizuta-checkpoint", default="output_dir/cfm_transformer/checkpoint.pth")
    p.add_argument("--safe-cfm-checkpoint", default="output_dir/safe_contextual_cfm/checkpoint_best.pth")
    p.add_argument("--drifting-checkpoint", default="output_dir/drifting_generator/checkpoint_best.pth")
    return p


def main() -> None:
    args = get_parser().parse_args()
    if args.gamma_grid is None:
        args.gamma_grid = []
    _set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_root = Path(args.output_root) / timestamp
    run_root = timestamp_root / args.dataset / args.dynamics
    device = torch.device(args.device)
    policies = BenchmarkPolicies(args, device)
    all_records: List[Dict] = []
    gamma_values = resolve_gamma_schedule(args.gamma_grid, args.gamma_schedule)
    writers: Dict[str, JSONLResultWriter] = {}
    try:
        for method in args.methods:
            writers[method] = JSONLResultWriter(run_root / f"{method}.jsonl")
            variants = gamma_values if method == "safemppi_gamma" else [None]
            for gamma in variants:
                for ep in range(args.num_episodes):
                    rec = _run_episode(args, policies, method, ep, gamma)
                    writers[method].write(rec)
                    all_records.append(rec)
                    gtxt = "" if gamma is None else f" gamma={float(gamma):.3g}"
                    print(
                        f"{method}{gtxt} ep={ep+1}/{args.num_episodes} "
                        f"success={int(rec['success'])} collision={int(rec['collision'])} "
                        f"min_clearance={rec['min_clearance']:.3f}",
                        flush=True,
                    )
    finally:
        for writer in writers.values():
            writer.close()
    summary_paths = write_summary(timestamp_root, all_records)
    print(json.dumps({k: str(v) for k, v in summary_paths.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
