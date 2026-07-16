#!/usr/bin/env python3
"""Batched all-gamma ID rollout gate for a Stage-3 checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT.parents[1], ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import grid_feats as GF  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
from giant_obstacle_ood.stage1_geometry_sweep import draw_scene  # noqa: E402
from giant_obstacle_ood.stage1b_smooth_expert import GOAL, START  # noqa: E402
from giant_obstacle_ood.stage2b_balanced_id_data import crossing_signature, make_id_env  # noqa: E402
from viz_style import GAMMAS, GAMMA_CMAP, GAMMA_COLORS, GAMMA_NORM, gamma_boundaries  # noqa: E402


STAGE = HERE / "stage_results/03_pretrain"


def object_array(items) -> np.ndarray:
    output = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        output[index] = item
    return output


def first_mode(path: np.ndarray) -> str:
    geometry = crossing_signature(path)
    if geometry["signature"]:
        return geometry["signature"][0]
    events = []
    for axis, letter in ((0, "R"), (1, "U")):
        values = path[:, axis]
        indices = np.where((values[:-1] < 1.0) & (values[1:] >= 1.0))[0]
        if len(indices):
            index = int(indices[0])
            denominator = values[index + 1] - values[index]
            fraction = (1.0 - values[index]) / denominator if abs(denominator) > 1e-12 else 1.0
            events.append((index + fraction, letter))
    return min(events)[1] if events else "?"


@torch.inference_mode()
def batched_rollouts(policy, *, repetitions: int, T: int, reach: float, nfe: int,
                     temperature: float, seed: int, device: torch.device,
                     h_exec: int = 1) -> list[dict]:
    env = make_id_env(T)
    obstacles = env.obstacles.detach().cpu().numpy()
    dt = float(env.dt)
    metadata = [(float(gamma), repetition, seed + gid * 10_000 + repetition)
                for gid, gamma in enumerate(GAMMAS) for repetition in range(repetitions)]
    count = len(metadata)
    states = np.zeros((count, 4), dtype=np.float32)
    states[:, :2] = START
    histories = np.zeros((count, GF.K_HIST, 2), dtype=np.float32)
    paths: list[list[np.ndarray]] = [[START.copy()] for _ in range(count)]
    controls: list[list[np.ndarray]] = [[] for _ in range(count)]
    active = np.ones(count, dtype=bool)
    dead_reason: list[str | None] = [None] * count
    if not 1 <= h_exec <= policy.T:
        raise ValueError(f"h_exec must be in [1,{policy.T}], got {h_exec}")
    cached_windows = np.zeros((count, policy.T, 2), dtype=np.float32)
    plan_age = np.full(count, h_exec, dtype=np.int16)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    np.random.seed(seed % (2**32))

    for step in range(T):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        replan = indices[plan_age[indices] >= h_exec]
        if len(replan):
            grid_np = np.stack([
                GF.axis_grid(states[index, :2], obstacles, float(env.r_robot)) for index in replan
            ])
            low_np = np.stack([
                GF.low5(states[index], GOAL, metadata[index][0]) for index in replan
            ])
            grid = torch.from_numpy(grid_np).to(device)
            low = torch.from_numpy(low_np).to(device)
            hist = torch.from_numpy(histories[replan]).to(device)
            context = policy.ctx_from(grid, low, hist)
            windows = policy.sample(len(replan), context, nfe=nfe, temp=temperature)
            cached_windows[replan] = windows.float().cpu().numpy()
            plan_age[replan] = 0
        actions = cached_windows[indices, plan_age[indices]]

        prior = states[indices].copy()
        states[indices, 0:2] = prior[:, 0:2] + dt * prior[:, 2:4] + 0.5 * dt * dt * actions
        states[indices, 2:4] = prior[:, 2:4] + dt * actions
        histories[indices, :-1] = histories[indices, 1:]
        histories[indices, -1] = actions
        plan_age[indices] += 1
        for local, index in enumerate(indices):
            action = actions[local].copy()
            position = states[index, :2].copy()
            controls[index].append(action)
            paths[index].append(position)
            clearance = float((np.linalg.norm(obstacles[:, :2] - position[None], axis=1)
                               - obstacles[:, 2] - float(env.r_robot)).min())
            collision = clearance < 0.0
            out_of_bounds = bool((position < 0.0).any() or (position > 5.0).any())
            reached = float(np.linalg.norm(position - GOAL)) < reach
            if collision or out_of_bounds or reached:
                active[index] = False
                dead_reason[index] = "collision" if collision else "out_of_bounds" if out_of_bounds else None
        if step == 0 or (step + 1) % 50 == 0:
            print(f"[rollout] step={step + 1}/{T} active={int(active.sum())}/{count}", flush=True)

    results = []
    for index, (gamma, repetition, rollout_seed) in enumerate(metadata):
        path = np.asarray(paths[index], dtype=np.float32)
        control = np.asarray(controls[index], dtype=np.float32)
        all_clearance = (np.linalg.norm(path[:, None] - obstacles[None, :, :2], axis=2)
                         - obstacles[None, :, 2] - float(env.r_robot))
        min_clearance = float(all_clearance.min())
        mean_clearance = float(all_clearance.min(axis=1).mean())
        collision = min_clearance < 0.0
        in_taskspace = bool(((path >= 0.0) & (path <= 5.0)).all())
        endpoint_distance = float(np.linalg.norm(path[-1] - GOAL))
        reached = endpoint_distance < reach
        reason = dead_reason[index]
        if reason is None and not reached:
            reason = "timeout"
        geometry = crossing_signature(path)
        mode = first_mode(path) if reached else "?"
        results.append({
            "gamma": gamma, "repetition": repetition, "seed": rollout_seed,
            "success": bool(reached and not collision and in_taskspace), "reached": reached,
            "collision": collision, "in_taskspace": in_taskspace, "dead_reason": reason,
            "steps": len(control), "time_s": len(control) * dt, "endpoint_distance": endpoint_distance,
            "min_clearance": min_clearance, "mean_clearance": mean_clearance,
            "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "first_mode": mode, "signature": geometry["signature"],
            "signature_eligible": geometry["signature_eligible"], "path": path, "controls": control,
        })
    return results


def summarize(results: list[dict], *, min_sr: float, max_cr: float,
              balance_low: float, balance_high: float) -> dict:
    per_gamma = {}
    all_successes = []
    gamma_gates = []
    for gamma in GAMMAS:
        selected = [row for row in results if row["gamma"] == gamma]
        successful = [row for row in selected if row["success"]]
        all_successes.extend(successful)
        modes = Counter(row["first_mode"] for row in successful)
        classified = modes["R"] + modes["U"]
        r_share = modes["R"] / classified if classified else None
        sr = len(successful) / len(selected)
        cr = sum(row["collision"] for row in selected) / len(selected)
        balance_ok = bool(classified < 5 or (balance_low <= r_share <= balance_high))
        gamma_gate = bool(sr >= min_sr and cr <= max_cr and balance_ok)
        gamma_gates.append(gamma_gate)
        per_gamma[str(float(gamma))] = {
            "rollouts": len(selected), "successes": len(successful), "success_rate": sr,
            "collisions": sum(row["collision"] for row in selected), "collision_rate": cr,
            "out_of_bounds": sum(not row["in_taskspace"] for row in selected),
            "timeouts": sum(row["dead_reason"] == "timeout" for row in selected),
            "r_first_successes": modes["R"], "u_first_successes": modes["U"],
            "unclassified_successes": modes["?"], "r_first_share": r_share,
            "unique_success_signatures": len(set(row["signature"] for row in successful if row["signature"])),
            "mean_time_s_success": float(np.mean([row["time_s"] for row in successful])) if successful else None,
            "mean_path_length_m_success": float(np.mean([row["path_length"] for row in successful])) if successful else None,
            "mean_min_clearance_m_success": float(np.mean([row["min_clearance"] for row in successful])) if successful else None,
            "minimum_clearance_m_all": min(row["min_clearance"] for row in selected),
            "best_endpoint_distance_m": min(row["endpoint_distance"] for row in selected),
            "gate_pass": gamma_gate,
        }
    global_modes = Counter(row["first_mode"] for row in all_successes)
    classified = global_modes["R"] + global_modes["U"]
    global_r_share = global_modes["R"] / classified if classified else None
    global_balance = bool(classified and balance_low <= global_r_share <= balance_high)
    return {
        "gate_pass": bool(all(gamma_gates) and global_balance),
        "criteria": {"minimum_success_rate_each_gamma": min_sr,
                     "maximum_collision_rate_each_gamma": max_cr,
                     "r_first_share_interval": [balance_low, balance_high]},
        "total_rollouts": len(results), "total_successes": len(all_successes),
        "global_success_rate": len(all_successes) / len(results),
        "global_collision_rate": sum(row["collision"] for row in results) / len(results),
        "global_r_first_successes": global_modes["R"], "global_u_first_successes": global_modes["U"],
        "global_unclassified_successes": global_modes["?"], "global_r_first_share": global_r_share,
        "per_gamma": per_gamma,
    }


def render_rollouts(results: list[dict], output: Path, reach: float, summary: dict) -> None:
    env = make_id_env(300)
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
    fig = plt.figure(figsize=(13.8, 15.0))
    layout = fig.add_gridspec(3, 4, height_ratios=(0.045, 1.55, 0.40), hspace=0.15, wspace=0.06)
    color_axis = fig.add_subplot(layout[0, 1:3])
    colorbar = mpl.colorbar.ColorbarBase(
        color_axis, cmap=GAMMA_CMAP, norm=GAMMA_NORM, boundaries=gamma_boundaries(),
        ticks=GAMMAS, spacing="uniform", orientation="horizontal", drawedges=True,
    )
    colorbar.ax.set_title(r"safety level $\gamma$", fontsize=13)
    colorbar.ax.tick_params(length=0)
    colorbar.dividers.set_color("white")

    main = fig.add_subplot(layout[1, :])
    draw_scene(main, env, START, GOAL, reach, None)
    for row in results:
        path = row["path"]
        linestyle = "-" if row["first_mode"] == "R" else "--" if row["first_mode"] == "U" else ":"
        main.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[row["gamma"]],
                  ls=linestyle, lw=1.15 if row["success"] else 0.55,
                  alpha=0.62 if row["success"] else 0.14, zorder=4)
        if not row["success"]:
            main.plot(path[-1, 0], path[-1, 1], marker="x", color="#c7351e", ms=3.5, alpha=0.45, zorder=7)
    main.set_xlabel(r"$x$ [m]")
    main.set_ylabel(r"$y$ [m]")
    main.set_xticks(np.arange(0, 6))
    main.set_yticks(np.arange(0, 6))
    main.set_title(
        f"Fresh endpoint-free pretrained policy — fixed-pair ID deployment\n"
        f"SR={100*summary['global_success_rate']:.1f}%, CR={100*summary['global_collision_rate']:.1f}%, "
        f"successful R/U={summary['global_r_first_successes']}/{summary['global_u_first_successes']}", fontsize=13,
    )
    main.legend(handles=(
        Line2D([], [], color="#333", ls="-", label="R-first (below diagonal)"),
        Line2D([], [], color="#333", ls="--", label="U-first (above diagonal)"),
        Line2D([], [], marker="x", ls="none", color="#c7351e", label="failed endpoint"),
    ), loc="lower right", fontsize=9, framealpha=0.92)

    mini = fig.add_gridspec(1, 7, top=0.215, bottom=0.052, left=0.035, right=0.985, wspace=0.06)
    for column, gamma in enumerate(GAMMAS):
        axis = fig.add_subplot(mini[0, column])
        draw_scene(axis, env, START, GOAL, reach, None)
        selected = [row for row in results if row["gamma"] == gamma]
        for row in selected:
            path = row["path"]
            linestyle = "-" if row["first_mode"] == "R" else "--" if row["first_mode"] == "U" else ":"
            axis.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], ls=linestyle,
                      lw=0.65 if row["success"] else 0.25,
                      alpha=0.70 if row["success"] else 0.13, zorder=4)
        metrics = summary["per_gamma"][str(float(gamma))]
        axis.set_title(
            rf"$\gamma$={gamma:g}" + f"\nSR {100*metrics['success_rate']:.0f}% · R/U "
            f"{metrics['r_first_successes']}/{metrics['u_first_successes']}",
            fontsize=8.5, color=GAMMA_COLORS[gamma],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_training(history_path: Path, output: Path, best_epoch: int) -> None:
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    epoch = values["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    axes[0].plot(epoch, values["train_cfm"], label="train", color="#1769aa")
    axes[0].plot(epoch, values["val_cfm"], label="stratified validation", color="#d32f2f")
    axes[0].axvline(best_epoch, ls="--", color="#555", lw=0.8)
    axes[0].set(title="CFM objective", xlabel="epoch", ylabel="loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(epoch, values["val_equivariance"], color="#8e44ad")
    axes[1].set(title="x↔y velocity-field mismatch", xlabel="epoch", ylabel="paired MSE")
    axes[2].plot(epoch, values["encoder_grad_norm"], color="#2a9d8f")
    axes[2].set_yscale("log")
    axes[2].set(title=r"Unfrozen $E(H_P)$", xlabel="epoch", ylabel="gradient norm")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_paths(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, paths=object_array([row["path"] for row in results]),
        controls=object_array([row["controls"] for row in results]),
        gammas=np.asarray([row["gamma"] for row in results], dtype=np.float32),
        seeds=np.asarray([row["seed"] for row in results], dtype=np.int64),
        success=np.asarray([row["success"] for row in results], dtype=bool),
        collision=np.asarray([row["collision"] for row in results], dtype=bool),
        in_taskspace=np.asarray([row["in_taskspace"] for row in results], dtype=bool),
        first_mode=np.asarray([row["first_mode"] for row in results]),
        signature=np.asarray([row["signature"] for row in results]),
        min_clearance=np.asarray([row["min_clearance"] for row in results], dtype=np.float32),
        endpoint_distance=np.asarray([row["endpoint_distance"] for row in results], dtype=np.float32),
        steps=np.asarray([row["steps"] for row in results], dtype=np.int16),
        start=START, goal=GOAL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--reach", type=float, default=0.2)
    parser.add_argument("--nfe", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--h-exec", type=int, default=1,
                        help="diagnostic open-loop commitment; authoritative evaluation uses 1")
    parser.add_argument("--seed", type=int, default=93000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-sr", type=float, default=0.70)
    parser.add_argument("--max-cr", type=float, default=0.15)
    parser.add_argument("--balance-low", type=float, default=0.20)
    parser.add_argument("--balance-high", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    policy, checkpoint = HP.load_hp(args.checkpoint, device=device)
    config = checkpoint["config"]
    # Legacy endpoint-free HP checkpoints predate the explicit flag; ctx_dim=37
    # is the authoritative architecture invariant for those files.
    if config.get("raw_start_goal", False) or config.get("ctx_dim") != 37:
        raise RuntimeError("rollout gate requires the original endpoint-free 37-D model")
    started = time.perf_counter()
    results = batched_rollouts(
        policy, repetitions=args.M, T=args.T, reach=args.reach, nfe=args.nfe,
        temperature=args.temperature, seed=args.seed, device=device, h_exec=args.h_exec,
    )
    summary = summarize(results, min_sr=args.min_sr, max_cr=args.max_cr,
                        balance_low=args.balance_low, balance_high=args.balance_high)
    paths_path = args.outdir / "id_rollouts.npz"
    figure_path = args.outdir / "id_rollouts_all_gamma.png"
    curves_path = args.outdir / "training_curves.png"
    metrics_path = args.outdir / "metrics.json"
    save_paths(paths_path, results)
    render_rollouts(results, figure_path, args.reach, summary)
    render_training(args.history, curves_path, int(checkpoint.get("best_epoch", -1)))
    payload = {
        "status": "PASS" if summary["gate_pass"] else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config": config,
        "fresh_from_scratch": checkpoint.get("stage3_pretrain_summary", {}).get("model", {}).get(
            "fresh_from_scratch", False),
        "scene": "ordinary symmetric 4x4 ID stadium", "start": START.tolist(), "goal": GOAL.tolist(),
        "plain_unguided": True,
        "settings": {"M_per_gamma": args.M, "T": args.T, "reach": args.reach,
                     "nfe": args.nfe, "temperature": args.temperature, "h_exec": args.h_exec,
                     "seed": args.seed},
        "summary": summary,
        "palette": {"gamma": "plasma_trunc", "sigma": "viridis"},
        "artifacts": {"paths": str(paths_path.resolve()), "rollout_figure": str(figure_path.resolve()),
                      "training_curves": str(curves_path.resolve())},
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for gamma in GAMMAS:
        row = summary["per_gamma"][str(float(gamma))]
        print(
            f"[gamma={gamma:g}] SR={100*row['success_rate']:.1f}% CR={100*row['collision_rate']:.1f}% "
            f"R/U={row['r_first_successes']}/{row['u_first_successes']} "
            f"clear={row['mean_min_clearance_m_success']} gate={row['gate_pass']}", flush=True,
        )
    print(
        f"[{payload['status']}] global SR={100*summary['global_success_rate']:.1f}% "
        f"CR={100*summary['global_collision_rate']:.1f}% "
        f"R/U={summary['global_r_first_successes']}/{summary['global_u_first_successes']} "
        f"wall={payload['wall_seconds']:.1f}s -> {metrics_path}", flush=True,
    )


if __name__ == "__main__":
    main()
