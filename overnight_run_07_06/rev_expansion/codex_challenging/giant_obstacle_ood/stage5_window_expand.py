#!/usr/bin/env python3
"""Stage 5 window-native expansion on the approved radius-1.2 giant scene.

This is the scene-specific driver around ``reference/window_expand_hardtail``.
It deliberately keeps the generic collector separate from the benchmark and
records every recipe choice needed by the final curriculum/internals figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
REF = ROOT / "reference"
for path in (WORK, ROOT.parent, ROOT, REF):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import window_expand_hardtail as W  # noqa: E402
from giant_obstacle_ood.stage1_geometry_sweep import make_scene  # noqa: E402
from giant_obstacle_ood.stage1b_smooth_expert import GOAL, RADIUS, START  # noqa: E402
from giant_obstacle_ood.stage4_frozen_ood import (  # noqa: E402
    CHECKPOINT,
    load_records,
    rollout_policy,
    save_records,
    summarize_method,
)
from viz_style import GAMMAS  # noqa: E402


STAGE = HERE / "stage_results/05_window_expand"
DEMO_SOURCE = HERE / "stage_results/04_frozen_ood/data/expert_m6.npz"
DEMO_PATH = STAGE / "data/expert_demo_balanced_h10.pt"
TEMPERATURES = (0.1, 0.5, 1.0)
ARM_NAMES = ("full", "no_socp", "no_progress", "no_curriculum")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_env(max_steps: int = 300):
    env = make_scene(RADIUS, START, GOAL)
    env.T = int(max_steps)
    W.GM2.GOAL_XY = np.asarray(GOAL, dtype=float)
    return env


def uniform_indices(n_controls: int, count: int) -> np.ndarray:
    n_full = n_controls - W.GF.H_PRED + 1
    if n_full < count:
        raise ValueError(f"trajectory has {n_full} complete windows, requested {count}")
    indices = np.rint(np.linspace(0, n_controls - W.GF.H_PRED, count)).astype(np.int32)
    if len(np.unique(indices)) != count:
        raise RuntimeError("uniform expert slicing created duplicate indices")
    return indices


def reconstruct_states(path: np.ndarray, controls: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    """Recover velocities from the exact controls while retaining recorded positions."""
    path = np.asarray(path, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    if len(path) != len(controls) + 1:
        raise ValueError(f"path/control length mismatch {len(path)} != {len(controls)}+1")
    states = np.zeros((len(path), 4), dtype=np.float32)
    states[:, :2] = path
    predicted = path[0].copy()
    velocity = np.zeros(2, dtype=np.float32)
    max_position_residual = 0.0
    for index, action in enumerate(controls):
        predicted = predicted + dt * velocity + 0.5 * dt * dt * action
        velocity = velocity + dt * action
        max_position_residual = max(
            max_position_residual, float(np.linalg.norm(predicted - path[index + 1]))
        )
        states[index + 1, 2:] = velocity
    return states, max_position_residual


def prepare_demo(windows_per_trajectory: int = 64) -> dict:
    """Build equal trajectory/gamma mass from the actual approved OOD expert."""
    STAGE.joinpath("data").mkdir(parents=True, exist_ok=True)
    records = load_records(DEMO_SOURCE)
    if len(records) != 6 * len(GAMMAS):
        raise RuntimeError(f"expected 42 expert trajectories, found {len(records)}")
    if not all(record["success"] for record in records):
        raise RuntimeError("expert demo source contains a non-successful trajectory")
    env = make_env(800)
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    dt = float(env.dt)

    grids: list[np.ndarray] = []
    low5: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    gamma_rows: list[float] = []
    trajectory_rows: list[int] = []
    step_rows: list[int] = []
    source_seed_rows: list[int] = []
    residuals: list[float] = []
    per_gamma = {str(float(gamma)): 0 for gamma in GAMMAS}

    for trajectory_id, record in enumerate(records):
        path = np.asarray(record["path"], dtype=np.float32)
        controls = np.asarray(record["controls"], dtype=np.float32)
        states, residual = reconstruct_states(path, controls, dt)
        residuals.append(residual)
        gamma = float(record["gamma"])
        for step in uniform_indices(len(controls), windows_per_trajectory):
            state = states[int(step)]
            grids.append(W.GF.axis_grid(state[:2], obstacles, rr))
            low5.append(W.GF.low5(state, GOAL, gamma))
            histories.append(W.GF.hist_pad(controls[max(0, int(step) - W.GF.K_HIST):int(step)]))
            targets.append(controls[int(step):int(step) + W.GF.H_PRED])
            gamma_rows.append(gamma)
            trajectory_rows.append(trajectory_id)
            step_rows.append(int(step))
            source_seed_rows.append(int(record["seed"]))
            per_gamma[str(gamma)] += 1

    payload = {
        "grid": torch.from_numpy(np.asarray(grids, dtype=np.float32)),
        "low5": torch.from_numpy(np.asarray(low5, dtype=np.float32)),
        "hist": torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        "U": torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        "gamma": torch.tensor(gamma_rows, dtype=torch.float32),
        "trajectory_id": torch.tensor(trajectory_rows, dtype=torch.int32),
        "window_step": torch.tensor(step_rows, dtype=torch.int32),
        "source_seed": torch.tensor(source_seed_rows, dtype=torch.int64),
        "metadata": {
            "source": str(DEMO_SOURCE.resolve()),
            "source_sha256": sha256(DEMO_SOURCE),
            "actual_ood_expert": True,
            "start": START.tolist(),
            "goal": GOAL.tolist(),
            "giant_radius": float(RADIUS),
            "horizon": int(W.GF.H_PRED),
            "windows_per_trajectory": int(windows_per_trajectory),
            "equal_loss_mass_per_trajectory": True,
            "terminal_padding": False,
            "per_gamma_windows": per_gamma,
            "max_physics_position_residual": float(max(residuals)),
        },
    }
    torch.save(payload, DEMO_PATH)
    audit = {
        "status": "PASS",
        **payload["metadata"],
        "output": str(DEMO_PATH.resolve()),
        "output_sha256": sha256(DEMO_PATH),
        "rows": int(payload["U"].shape[0]),
        "tensor_shapes": {key: list(payload[key].shape) for key in ("grid", "low5", "hist", "U")},
    }
    STAGE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    (STAGE / "logs/demo_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def make_config(*, gather_temperature: float, evaluation_temperature: float,
                iters: int, start_iter: int, probe_only: bool,
                demo_switch_iter: int = 11, beta: float = 0.2,
                alpha: float = 0.0, success_replay_quota: int = 0,
                inner_steps: int = 2, early_inner_steps: int = 2,
                learning_rate: float = 5e-6,
                encoder_lr_mult: float = 0.3,
                early_rollout_scale: float = 1.0,
                demo_stage_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
                cfm_eta: float = 1.0,
                resume_reset_optimizer: bool = False,
                lwf_eta: float = 0.0,
                lwf_x0_scale: float = 1.0,
                eval_temp_cfm_weight: float = 0.0,
                eval_endpoint_eta: float = 0.0,
                eval_endpoint_quota: int = 0,
                eval_endpoint_scope: str = "all",
                eval_endpoint_gammas: tuple[float, ...] = (),
                eval_goal_eta: float = 0.0,
                eval_goal_velocity_weight: float = 0.2,
                eval_mode_noise_coupling: bool = False,
                allow_recipe_drift: bool = False,
                train_gammas: tuple[float, ...] = tuple(GAMMAS),
                accepted_budget: dict[int, int] | None = None) -> W.CurConfig:
    cfg = W.CurConfig()
    cfg.iters = int(iters)
    cfg.start_iter = int(start_iter)
    cfg.N = 64
    cfg.temp = float(gather_temperature)
    cfg.eval_temp = float(evaluation_temperature)
    cfg.s = 0.9
    cfg.churn = 0.05
    cfg.nfe_explore = 8
    cfg.safe_filter = True
    cfg.targeted_frac = 0.0
    cfg.n_target = 0
    cfg.min_modes_per_gamma = 0
    cfg.min_modes_schedule = ()
    cfg.mode_hit_gate = False
    cfg.gp_buf = 200
    cfg.qbuf_cap = 200
    cfg.rollouts_per_iter = 14
    cfg.gather_attempt_cap = 28
    cfg.valid_prog_floor = 0.10
    cfg.min_rollouts = 1
    cfg.traj_prog_min = 0.0
    cfg.batch_cap = 16
    # The 1e-5 calibration repeatedly crossed the protected functional-step
    # bound.  This is the previously stable sanity rate.
    cfg.lr = float(learning_rate)
    cfg.quantile_schedule = ((0, 0.30),)
    cfg.mix_start = (0.4, 0.6)
    cfg.mix_end = (0.4, 0.6)
    cfg.beta = float(beta)
    cfg.alpha = float(alpha)
    cfg.neg_batch_cap = 16
    cfg.neg_windows_per_rollout = 24
    cfg.success_replay_quota = int(success_replay_quota)
    cfg.success_replay_cap = 3000
    cfg.success_windows_per_rollout = 32
    # The stadium and centered giant scene are exactly x/y symmetric.  Stage 3
    # required this paired objective to retain both modes; expansion must not
    # silently drop that inductive bias.
    cfg.symmetry_augment = True
    cfg.equivariance_weight = 1.0
    cfg.demo_stage_balanced = True
    cfg.demo_stage_weights = tuple(float(value) for value in demo_stage_weights)
    cfg.cfm_eta = float(cfm_eta)
    cfg.resume_reset_optimizer = bool(resume_reset_optimizer)
    cfg.lwf_eta = float(lwf_eta)
    cfg.lwf_x0_scale = float(lwf_x0_scale)
    cfg.eval_temp_cfm_weight = float(eval_temp_cfm_weight)
    cfg.eval_endpoint_eta = float(eval_endpoint_eta)
    cfg.eval_endpoint_quota = int(eval_endpoint_quota)
    cfg.eval_endpoint_scope = str(eval_endpoint_scope)
    cfg.eval_endpoint_gammas = tuple(float(gamma) for gamma in eval_endpoint_gammas)
    cfg.eval_goal_eta = float(eval_goal_eta)
    cfg.eval_goal_velocity_weight = float(eval_goal_velocity_weight)
    cfg.eval_mode_noise_coupling = bool(eval_mode_noise_coupling)
    cfg.early_until = 100
    cfg.early_rollout_scale = float(early_rollout_scale)
    cfg.cooldown_from = 400
    cfg.early_inner = int(early_inner_steps)
    cfg.inner_steps = int(inner_steps)
    cfg.cooldown_inner = 2
    cfg.resume_allow_recipe_drift = bool(allow_recipe_drift)
    cfg.field_grad_clip = 1.0
    cfg.enc_grad_clip = 5.0
    cfg.max_functional_step = 0.025
    cfg.max_anchor_drift = 0.016
    cfg.measure_every = max(1, min(5, int(iters)))
    cfg.M_measure = 2
    cfg.reach = 0.15
    cfg.T = 300
    cfg.demo_frac = 0.50
    # ``demo_switch_iter`` is absolute.  A negative value deliberately holds
    # the early expert anchor for the entire lineage.
    cfg.demo_frac_schedule = (
        ((0, 0.50),) if int(demo_switch_iter) < 0
        else ((0, 0.50), (int(demo_switch_iter), 0.25))
    )
    cfg.demo_override_path = str(DEMO_PATH.resolve())
    cfg.gammas = tuple(float(gamma) for gamma in train_gammas)
    cfg.ckpt_every = max(1, min(5, int(iters)))
    cfg.legacy_prime_iters = 1 if probe_only else 0
    cfg.recovery_frac = 0.0
    cfg.hard_quota = 0
    cfg.guard_quota = 0
    cfg.escape_quota = 0
    cfg.strip_probe_every = 0
    cfg.fresh_frac = 1.0
    cfg.warmup_gather = 0
    cfg.viz_db_every = 1
    cfg.log_comp_every = 1
    cfg.probe_escape = 0
    cfg.probe_cov = 0
    cfg.wall_plugs = 8
    cfg.accepted_window_budget = accepted_budget
    cfg.scene_name = "giant_obstacle_radius_1.2"
    cfg.giant_center = (2.5, 2.5)
    cfg.giant_radius = float(RADIUS)
    return cfg


def run_arm(arm: str, outdir: Path, *, gather_temperature: float,
            evaluation_temperature: float, iters: int,
            checkpoint_path: Path = CHECKPOINT, resume: bool = False,
            demo_switch_iter: int = 11, beta: float = 0.2,
            alpha: float = 0.0, success_replay_quota: int = 0,
            inner_steps: int = 2, early_inner_steps: int = 2,
            learning_rate: float = 5e-6,
            encoder_lr_mult: float = 0.3,
            early_rollout_scale: float = 1.0,
            demo_stage_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
            cfm_eta: float = 1.0,
            resume_reset_optimizer: bool = False,
            lwf_eta: float = 0.0,
            lwf_x0_scale: float = 1.0,
            eval_temp_cfm_weight: float = 0.0,
            eval_endpoint_eta: float = 0.0,
            eval_endpoint_quota: int = 0,
            eval_endpoint_scope: str = "all",
            eval_endpoint_gammas: tuple[float, ...] = (),
            eval_goal_eta: float = 0.0,
            eval_goal_velocity_weight: float = 0.2,
            eval_mode_noise_coupling: bool = False,
            allow_recipe_drift: bool = False,
            train_gammas: tuple[float, ...] = tuple(GAMMAS),
            probe_only: bool = False, budget_path: Path | None = None,
            overwrite: bool = False) -> dict:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm}")
    if not DEMO_PATH.exists():
        prepare_demo()
    if outdir.exists() and any(outdir.iterdir()) and not (overwrite or resume):
        raise FileExistsError(f"refusing to overwrite nonempty {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    accepted_budget = None
    if budget_path is not None:
        raw = json.loads(budget_path.read_text())
        accepted_budget = {int(key): int(value) for key, value in raw.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, checkpoint = W.HP.load_hp(checkpoint_path, device="cpu")
    checkpoint_iter = int(checkpoint.get("iter", 0))
    resume_state = checkpoint.get("train_state") if resume else None
    if resume and resume_state is None:
        raise ValueError(f"resume checkpoint has no complete train_state: {checkpoint_path}")
    if resume and int(resume_state.get("iter", -1)) != checkpoint_iter:
        raise ValueError(
            f"checkpoint/train_state iteration mismatch: {checkpoint_iter} vs "
            f"{resume_state.get('iter')}"
        )
    cfg = make_config(
        gather_temperature=gather_temperature,
        evaluation_temperature=evaluation_temperature,
        iters=iters, start_iter=checkpoint_iter, probe_only=probe_only,
        demo_switch_iter=demo_switch_iter, beta=beta, alpha=alpha,
        success_replay_quota=success_replay_quota,
        inner_steps=inner_steps, early_inner_steps=early_inner_steps,
        learning_rate=learning_rate,
        encoder_lr_mult=encoder_lr_mult,
        early_rollout_scale=early_rollout_scale,
        demo_stage_weights=demo_stage_weights,
        cfm_eta=cfm_eta,
        resume_reset_optimizer=resume_reset_optimizer,
        lwf_eta=lwf_eta,
        lwf_x0_scale=lwf_x0_scale,
        eval_temp_cfm_weight=eval_temp_cfm_weight,
        eval_endpoint_eta=eval_endpoint_eta,
        eval_endpoint_quota=eval_endpoint_quota,
        eval_endpoint_scope=eval_endpoint_scope,
        eval_endpoint_gammas=eval_endpoint_gammas,
        eval_goal_eta=eval_goal_eta,
        eval_goal_velocity_weight=eval_goal_velocity_weight,
        eval_mode_noise_coupling=eval_mode_noise_coupling,
        allow_recipe_drift=allow_recipe_drift,
        train_gammas=train_gammas,
        accepted_budget=accepted_budget,
    )
    cfg.ablate_socp = arm == "no_socp"
    cfg.ablate_progress = arm == "no_progress"
    cfg.ablate_curriculum = arm == "no_curriculum"
    if cfg.ablate_curriculum:
        cfg.mix_start = (1.0, 0.0)
        cfg.mix_end = (1.0, 0.0)
        if accepted_budget is None:
            raise ValueError("-Curriculum requires Full's accepted-window budget")

    random.seed(6010)
    np.random.seed(6010)
    torch.manual_seed(6010)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(6010)
    policy = policy.to(device)
    env = make_env(cfg.T)
    started = time.perf_counter()
    result = W.run_expand_cur(
        policy, env, cfg, device=str(device), outdir=str(outdir), log=print,
        freeze_enc=False, enc_lr_mult=float(encoder_lr_mult),
        tag=(f"giant_temp_probe_g{gather_temperature:g}_e{evaluation_temperature:g}"
             if probe_only else f"giant_{arm}_g{gather_temperature:g}_e{evaluation_temperature:g}"),
        resume_state=resume_state, teacher_ckpt=None, train_seed=6010,
    )
    final_iter = int(result["history"][-1]["iter"])
    manifest = {
        "status": "PASS",
        "arm": arm,
        "probe_only": bool(probe_only),
        "gather_temperature": float(gather_temperature),
        "evaluation_temperature": float(evaluation_temperature),
        "stateful_resume": bool(resume),
        "start_iter": int(checkpoint_iter),
        "final_iter": final_iter,
        "physical_gpu_requested": 2,
        "visible_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_raw_start_goal": bool(checkpoint["config"].get("raw_start_goal", False)),
        "demo": str(DEMO_PATH.resolve()),
        "demo_sha256": sha256(DEMO_PATH),
        "demo_frac_schedule": [list(row) for row in cfg.demo_frac_schedule],
        "beta": float(cfg.beta),
        "alpha": float(cfg.alpha),
        "negative_semantics": "coherent H=10 windows rejected by this arm's enabled predicates",
        "success_replay_quota": int(cfg.success_replay_quota),
        "success_replay_semantics": "accepted windows from reached rollouts; gamma/mode/rid balanced",
        "symmetry_augment": bool(cfg.symmetry_augment),
        "equivariance_weight": float(cfg.equivariance_weight),
        "demo_stage_balanced": bool(cfg.demo_stage_balanced),
        "demo_stage_weights": list(cfg.demo_stage_weights),
        "cfm_eta": float(cfg.cfm_eta),
        "resume_reset_optimizer": bool(cfg.resume_reset_optimizer),
        "lwf_eta": float(cfg.lwf_eta),
        "lwf_x0_scale": float(cfg.lwf_x0_scale),
        "eval_temp_cfm_weight": float(cfg.eval_temp_cfm_weight),
        "eval_endpoint_eta": float(cfg.eval_endpoint_eta),
        "eval_endpoint_quota": int(cfg.eval_endpoint_quota),
        "eval_endpoint_scope": str(cfg.eval_endpoint_scope),
        "eval_endpoint_gammas": list(cfg.eval_endpoint_gammas),
        "eval_goal_eta": float(cfg.eval_goal_eta),
        "eval_goal_velocity_weight": float(cfg.eval_goal_velocity_weight),
        "eval_mode_noise_coupling": bool(cfg.eval_mode_noise_coupling),
        "train_gammas": list(cfg.gammas),
        "inner_steps": int(cfg.inner_steps),
        "early_inner_steps": int(cfg.early_inner),
        "learning_rate": float(cfg.lr),
        "encoder_lr_mult": float(encoder_lr_mult),
        "early_rollout_scale": float(cfg.early_rollout_scale),
        "resume_allow_recipe_drift": bool(cfg.resume_allow_recipe_drift),
        "scene": {"start": START.tolist(), "goal": GOAL.tolist(), "giant_radius": float(RADIUS)},
        "iters": int(iters),
        "elapsed_seconds": time.perf_counter() - started,
        "result": result,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (outdir / f"chunk_{checkpoint_iter}_{final_iter}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def temperature_rollouts(repetitions: int = 6) -> dict:
    output = STAGE / "temperature_probe"
    output.joinpath("data").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = W.HP.load_hp(CHECKPOINT, device=device)
    summaries = {}
    for temperature in TEMPERATURES:
        records = rollout_policy(
            policy, repetitions=repetitions, temperature=temperature, nfe=8,
            T=300, seed0=80500, device=device,
            method=f"Pretrained T={temperature:g}",
        )
        path = output / f"data/pretrained_temp_{temperature:g}_m{repetitions}.npz"
        save_records(records, path, temperature=np.asarray(temperature), matched_seed0=np.asarray(80500))
        summaries[str(temperature)] = summarize_method(records)
    payload = {
        "status": "PASS",
        "temperatures": list(TEMPERATURES),
        "M_per_gamma": int(repetitions),
        "matched_seed0": 80500,
        "summaries": summaries,
    }
    (output / "rollout_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-demo")
    prep.add_argument("--windows-per-trajectory", type=int, default=64)
    temp = sub.add_parser("temperature-rollouts")
    temp.add_argument("--repetitions", type=int, default=6)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=ARM_NAMES, required=True)
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--temperature", type=float,
                     help="legacy alias: use the same temperature for gather and evaluation")
    run.add_argument("--gather-temperature", type=float)
    run.add_argument("--evaluation-temperature", type=float)
    run.add_argument("--iters", type=int, default=20)
    run.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--demo-switch-iter", type=int, default=11,
                     help="absolute 0.50->0.25 demo-fraction switch; negative holds 0.50")
    run.add_argument("--beta", type=float, default=0.2,
                     choices=(0.2, 0.3, 0.4, 0.5, 0.7, 1.0))
    run.add_argument("--alpha", type=float, default=0.0,
                     choices=(0.0, 0.001, 0.002, 0.005, 0.01))
    run.add_argument("--success-replay-quota", type=int, default=0,
                     choices=(0, 2, 4, 6, 8, 10))
    run.add_argument("--inner-steps", type=int, default=2,
                     choices=(2, 4, 6, 8, 12, 16, 32, 64, 100))
    run.add_argument("--early-inner-steps", type=int, default=2,
                     choices=(2, 4, 6, 8, 12, 16, 32, 64, 100))
    run.add_argument("--learning-rate", type=float, default=5e-6)
    run.add_argument("--encoder-lr-mult", type=float, default=0.3)
    run.add_argument("--early-rollout-scale", type=float, default=1.0)
    run.add_argument("--demo-stage-weights", type=float, nargs=4,
                     default=(0.25, 0.25, 0.25, 0.25),
                     metavar=("INITIAL", "APPROACH", "BOUNDARY", "POST"))
    run.add_argument("--eval-temp-cfm-weight", type=float, default=0.0)
    run.add_argument("--cfm-eta", type=float, default=1.0)
    run.add_argument("--resume-reset-optimizer", action="store_true")
    run.add_argument("--lwf-eta", type=float, default=0.0)
    run.add_argument("--lwf-x0-scale", type=float, default=1.0)
    run.add_argument("--allow-recipe-drift", action="store_true")
    run.add_argument("--train-gammas", type=float, nargs="+", default=tuple(GAMMAS))
    run.add_argument("--probe-only", action="store_true")
    run.add_argument("--budget-path", type=Path)
    run.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-demo":
        print(json.dumps(prepare_demo(args.windows_per_trajectory), indent=2))
    elif args.command == "temperature-rollouts":
        print(json.dumps(temperature_rollouts(args.repetitions), indent=2))
    else:
        gather_temperature = (args.gather_temperature if args.gather_temperature is not None
                              else args.temperature)
        if gather_temperature is None:
            parser.error("run requires --gather-temperature (or legacy --temperature)")
        evaluation_temperature = (
            args.evaluation_temperature if args.evaluation_temperature is not None
            else (args.temperature if args.temperature is not None else gather_temperature)
        )
        run_arm(
            args.arm, args.outdir.resolve(), gather_temperature=gather_temperature,
            evaluation_temperature=evaluation_temperature,
            iters=args.iters, probe_only=args.probe_only,
            checkpoint_path=args.checkpoint.resolve(), resume=args.resume,
            demo_switch_iter=args.demo_switch_iter, beta=args.beta, alpha=args.alpha,
            success_replay_quota=args.success_replay_quota,
            inner_steps=args.inner_steps, early_inner_steps=args.early_inner_steps,
            learning_rate=args.learning_rate,
            encoder_lr_mult=args.encoder_lr_mult,
            early_rollout_scale=args.early_rollout_scale,
            demo_stage_weights=tuple(args.demo_stage_weights),
            cfm_eta=args.cfm_eta,
            resume_reset_optimizer=args.resume_reset_optimizer,
            lwf_eta=args.lwf_eta,
            lwf_x0_scale=args.lwf_x0_scale,
            eval_temp_cfm_weight=args.eval_temp_cfm_weight,
            allow_recipe_drift=args.allow_recipe_drift,
            train_gammas=tuple(args.train_gammas),
            budget_path=args.budget_path.resolve() if args.budget_path else None,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
