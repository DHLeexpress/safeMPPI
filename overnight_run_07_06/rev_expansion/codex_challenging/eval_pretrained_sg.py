#!/usr/bin/env python3
"""Evaluate the endpoint-free policy on unseen and expansion-target pairs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch

import grid_hp_expt as HP
import gen_uniform_data as SEEDS
from plot_sg_demo_overlay import draw_scene
from sg_rollout import rollout_sg
from viz_style import GAMMAS, GAMMA_CMAP, GAMMA_COLORS, GAMMA_NORM, gamma_boundaries


HERE = Path(__file__).resolve().parent
STAGE = HERE / "stage_results" / "03_pretrain"
if Path(HP.__file__).resolve().parent != HERE:
    raise ImportError(f"expected local grid_hp_expt.py, imported {HP.__file__}")

# Fixed before either training attempt. Both marginals are absent from the
# Stage 2 manifest, so this is stricter than withholding only the exact pair.
HELDOUT_START_INDEX = 7
HELDOUT_GOAL_INDEX = 65


def object_array(items) -> np.ndarray:
    output = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        output[index] = item
    return output


def load_training_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows], dtype=float) for key in rows[0]}


def render_training(history: dict[str, np.ndarray], output: Path, best_epoch: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    epoch = history["epoch"]
    axes[0].plot(epoch, history["train_cfm"], color="#1769aa", label="fine-tune")
    axes[0].plot(epoch, history["val_cfm"], color="#d32f2f", label="held-out pairs")
    axes[0].axvline(best_epoch, color="#555555", ls="--", lw=0.9, label=f"best epoch {best_epoch}")
    axes[0].set(xlabel="fine-tune epoch", ylabel="CFM loss", title="Endpoint-free policy fine-tuning")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(epoch, history["encoder_grad_norm"], color="#2a9d8f")
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="fine-tune epoch",
        ylabel="mean gradient norm",
        title=r"Unfrozen $E(H_P)$ gradient flow",
    )
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_rollouts(
    results: list[dict],
    start: np.ndarray,
    goal: np.ndarray,
    output: Path,
    *,
    title: str,
    reach: float,
    start_label: str,
    goal_label: str,
    diagonal_band: bool,
) -> None:
    env = SEEDS.make_walled_env(8)
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig = plt.figure(figsize=(13, 15.6))
    layout = fig.add_gridspec(3, 4, height_ratios=(0.045, 1.55, 0.42), hspace=0.14, wspace=0.06)

    color_ax = fig.add_subplot(layout[0, 1:3])
    colorbar = mpl.colorbar.ColorbarBase(
        color_ax,
        cmap=GAMMA_CMAP,
        norm=GAMMA_NORM,
        boundaries=gamma_boundaries(),
        ticks=GAMMAS,
        spacing="uniform",
        orientation="horizontal",
        drawedges=True,
    )
    colorbar.ax.set_title(r"$\gamma$", fontsize=15)
    colorbar.ax.tick_params(labelsize=10, length=0)
    colorbar.dividers.set_color("white")

    main_ax = fig.add_subplot(layout[1, :])
    draw_scene(main_ax, env, band=diagonal_band)
    for result in results:
        path = result["path"]
        main_ax.plot(
            path[:, 0],
            path[:, 1],
            color=GAMMA_COLORS[result["gamma"]],
            lw=1.35 if result["success"] else 0.65,
            alpha=0.78 if result["success"] else 0.20,
            zorder=4,
        )
    main_ax.add_patch(
        Circle(goal, reach, facecolor="none", edgecolor="#d32f2f", ls="--", lw=1.0, alpha=0.8, zorder=6)
    )
    main_ax.scatter(*start, marker="D", s=70, c="#1769aa", edgecolors="white", linewidths=0.8, zorder=7)
    main_ax.scatter(*goal, marker="*", s=190, c="#ffd000", edgecolors="#333333", linewidths=0.7, zorder=7)
    successes = sum(result["success"] for result in results)
    main_ax.set_xlabel(r"$x$ [m]")
    main_ax.set_ylabel(r"$y$ [m]")
    main_ax.set_title(f"{title} ({successes}/{len(results)} plain rollouts reached safely)")
    main_ax.legend(
        handles=(
            Line2D([], [], marker="D", ls="none", color="#1769aa", label=start_label),
            Line2D([], [], marker="*", ls="none", color="#ffd000", markeredgecolor="#333333", markersize=12,
                   label=goal_label),
            Line2D([], [], ls="--", color="#d32f2f", label=f"reach radius = {reach:.2f} m"),
        ),
        loc="center",
        framealpha=0.92,
        fontsize=9,
    )

    minis = fig.add_gridspec(1, 7, top=0.215, bottom=0.055, left=0.045, right=0.985, wspace=0.08)
    for column, gamma in enumerate(GAMMAS):
        axis = fig.add_subplot(minis[0, column])
        draw_scene(axis, env, band=False)
        gamma_results = [result for result in results if result["gamma"] == gamma]
        for result in gamma_results:
            path = result["path"]
            axis.plot(
                path[:, 0],
                path[:, 1],
                color=GAMMA_COLORS[gamma],
                lw=0.8 if result["success"] else 0.35,
                alpha=0.8 if result["success"] else 0.20,
                zorder=4,
            )
        axis.add_patch(Circle(goal, reach, facecolor="none", edgecolor="#d32f2f", ls="--", lw=0.45, zorder=5))
        axis.scatter(*start, marker="s", s=9, c="#1769aa", zorder=6)
        axis.scatter(*goal, marker="*", s=25, c="#ffd000", edgecolors="#333333", linewidths=0.25, zorder=6)
        axis.set_xticks([])
        axis.set_yticks([])
        gamma_success = sum(result["success"] for result in gamma_results)
        axis.set_title(
            rf"$\gamma$={gamma:g}" + f"\n{gamma_success}/{len(gamma_results)}",
            fontsize=10,
            color=GAMMA_COLORS[gamma],
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_case(policy, spec: dict, args: argparse.Namespace, case_offset: int) -> list[dict]:
    results = []
    for gamma_index, gamma in enumerate(GAMMAS):
        gamma_results = []
        for repetition in range(args.seeds_per_gamma):
            seed = args.seed + case_offset + gamma_index * 100 + repetition
            result = rollout_sg(
                policy,
                spec["start"],
                spec["goal"],
                gamma,
                seed=seed,
                T=args.T,
                reach=spec["reach"],
                nfe=args.nfe,
                temp=args.temp,
                device=args.device,
            )
            result["case"] = spec["name"]
            results.append(result)
            gamma_results.append(result)
        print(
            f"[{spec['name']}] gamma={gamma:g}: success={sum(r['success'] for r in gamma_results)}/"
            f"{len(gamma_results)} collision={sum(r['collision'] for r in gamma_results)} "
            f"oob={sum(not r['in_taskspace'] for r in gamma_results)} "
            f"best_distance={min(r['endpoint_distance'] for r in gamma_results):.3f}",
            flush=True,
        )
    return results


def summarize_case(results: list[dict]) -> dict:
    per_gamma = {}
    for gamma in GAMMAS:
        selected = [result for result in results if result["gamma"] == gamma]
        successful = [result for result in selected if result["success"]]
        per_gamma[str(gamma)] = {
            "rollouts": len(selected),
            "successes": len(successful),
            "success_rate": len(successful) / len(selected),
            "collisions": sum(result["collision"] for result in selected),
            "out_of_bounds": sum(not result["in_taskspace"] for result in selected),
            "timeouts": sum(result["dead_reason"] == "timeout" for result in selected),
            "minimum_clearance": min(result["min_clearance"] for result in selected),
            "best_endpoint_distance": min(result["endpoint_distance"] for result in selected),
            "mean_steps_success": (
                float(np.mean([result["steps"] for result in successful])) if successful else None
            ),
            "minimum_clearance_success": (
                min(result["min_clearance"] for result in successful) if successful else None
            ),
        }
    return {
        "total_rollouts": len(results),
        "total_successes": int(sum(result["success"] for result in results)),
        "per_gamma": per_gamma,
    }


def save_paths(path: Path, results: list[dict], start: np.ndarray, goal: np.ndarray, reach: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=object_array([result["path"] for result in results]),
        controls=object_array([result["controls"] for result in results]),
        gammas=np.asarray([result["gamma"] for result in results], dtype=np.float32),
        seeds=np.asarray([result["seed"] for result in results], dtype=np.int64),
        success=np.asarray([result["success"] for result in results], dtype=bool),
        collision=np.asarray([result["collision"] for result in results], dtype=bool),
        in_taskspace=np.asarray([result["in_taskspace"] for result in results], dtype=bool),
        endpoint_distance=np.asarray([result["endpoint_distance"] for result in results], dtype=np.float32),
        min_clearance=np.asarray([result["min_clearance"] for result in results], dtype=np.float32),
        steps=np.asarray([result["steps"] for result in results], dtype=np.int16),
        start=np.asarray(start, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        reach=np.float32(reach),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "pretrained_sg_walls8.pt")
    parser.add_argument(
        "--manifest", type=Path,
        default=HERE / "stage_results" / "02_demos" / "data" / "random_pairs_300.npz",
    )
    parser.add_argument("--history", type=Path, default=STAGE / "logs" / "pretrain_history.csv")
    parser.add_argument("--heldout-viz", type=Path, default=STAGE / "viz" / "heldout_pair_rollouts_all_gamma.png")
    parser.add_argument("--canonical-viz", type=Path, default=STAGE / "viz" / "canonical_target_rollouts_all_gamma.png")
    parser.add_argument("--curves", type=Path, default=STAGE / "viz" / "pretrain_curves.png")
    parser.add_argument("--heldout-paths", type=Path, default=STAGE / "data" / "heldout_pair_rollouts.npz")
    parser.add_argument("--canonical-paths", type=Path, default=STAGE / "data" / "canonical_target_rollouts.npz")
    parser.add_argument("--log", type=Path, default=STAGE / "logs" / "deployment_eval.json")
    parser.add_argument("--seeds-per-gamma", type=int, default=8)
    parser.add_argument("--seed", type=int, default=83000)
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--heldout-reach", type=float, default=0.2)
    parser.add_argument("--canonical-reach", type=float, default=0.15)
    parser.add_argument("--nfe", type=int, default=12)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    env = SEEDS.make_walled_env(8)
    blue, red = SEEDS.start_goal_pools(env)
    heldout_start = blue[HELDOUT_START_INDEX].astype(np.float32)
    heldout_goal = red[HELDOUT_GOAL_INDEX].astype(np.float32)
    with np.load(args.manifest) as manifest:
        start_seen = HELDOUT_START_INDEX in set(map(int, manifest["start_indices"]))
        goal_seen = HELDOUT_GOAL_INDEX in set(map(int, manifest["goal_indices"]))
        exact_pair_seen = bool(
            ((manifest["start_indices"] == HELDOUT_START_INDEX)
             & (manifest["goal_indices"] == HELDOUT_GOAL_INDEX)).any()
        )
    if start_seen or goal_seen or exact_pair_seen:
        raise RuntimeError("configured unseen pair leaked into the Stage 2 manifest")

    policy, checkpoint = HP.load_hp(args.checkpoint, device=args.device)
    if checkpoint["config"].get("raw_start_goal", True) or checkpoint["config"].get("ctx_dim") != 37:
        raise RuntimeError("evaluation checkpoint is not the endpoint-free 37-D model")

    specs = {
        "unseen_pair": {
            "name": "unseen_pair",
            "start": heldout_start,
            "goal": heldout_goal,
            "reach": args.heldout_reach,
        },
        "canonical_target": {
            "name": "canonical_target",
            "start": np.array([0.05, 0.05], dtype=np.float32),
            "goal": np.array([5.0, 5.0], dtype=np.float32),
            "reach": args.canonical_reach,
        },
    }
    unseen_results = evaluate_case(policy, specs["unseen_pair"], args, case_offset=0)
    canonical_results = evaluate_case(policy, specs["canonical_target"], args, case_offset=10000)

    save_paths(args.heldout_paths, unseen_results, heldout_start, heldout_goal, args.heldout_reach)
    save_paths(
        args.canonical_paths,
        canonical_results,
        specs["canonical_target"]["start"],
        specs["canonical_target"]["goal"],
        args.canonical_reach,
    )
    render_rollouts(
        unseen_results,
        heldout_start,
        heldout_goal,
        args.heldout_viz,
        title="Endpoint-free policy: fully unseen start→goal pair",
        reach=args.heldout_reach,
        start_label="unseen start marginal",
        goal_label="unseen goal marginal",
        diagonal_band=True,
    )
    render_rollouts(
        canonical_results,
        specs["canonical_target"]["start"],
        specs["canonical_target"]["goal"],
        args.canonical_viz,
        title="Endpoint-free policy: canonical expansion target",
        reach=args.canonical_reach,
        start_label="cleared start (0.05, 0.05)",
        goal_label="goal (5, 5)",
        diagonal_band=False,
    )
    history = load_training_history(args.history)
    render_training(history, args.curves, int(checkpoint["best_epoch"]))

    unseen_summary = summarize_case(unseen_results)
    canonical_summary = summarize_case(canonical_results)
    payload = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_schema": checkpoint["config"]["schema_version"],
        "raw_start_goal": False,
        "plain_unguided": True,
        "settings": {
            "seeds_per_gamma": args.seeds_per_gamma,
            "T": args.T,
            "nfe": args.nfe,
            "temperature": args.temp,
        },
        "unseen_pair": {
            "start_pool_index": HELDOUT_START_INDEX,
            "goal_pool_index": HELDOUT_GOAL_INDEX,
            "start": heldout_start.tolist(),
            "goal": heldout_goal.tolist(),
            "reach": args.heldout_reach,
            "start_marginal_seen_in_training_manifest": start_seen,
            "goal_marginal_seen_in_training_manifest": goal_seen,
            "exact_pair_seen_in_training_manifest": exact_pair_seen,
            **unseen_summary,
        },
        "canonical_target": {
            "start": specs["canonical_target"]["start"].tolist(),
            "goal": specs["canonical_target"]["goal"].tolist(),
            "reach": args.canonical_reach,
            "origin_plug_clearance_m": 0.0549509757,
            "protocol": "start-eps=0.05; goal remains (5,5); reach=0.15 stops before goal plugs",
            **canonical_summary,
        },
        "palette": {"gamma": "plasma_trunc", "sigma_uncertainty": "viridis"},
        "artifacts": {
            "unseen_rollout_figure": str(args.heldout_viz.resolve()),
            "canonical_rollout_figure": str(args.canonical_viz.resolve()),
            "training_curves": str(args.curves.resolve()),
            "unseen_paths": str(args.heldout_paths.resolve()),
            "canonical_paths": str(args.canonical_paths.resolve()),
        },
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"[PASS] unseen={unseen_summary['total_successes']}/{unseen_summary['total_rollouts']} "
        f"canonical={canonical_summary['total_successes']}/{canonical_summary['total_rollouts']} "
        f"-> {args.log}",
        flush=True,
    )


if __name__ == "__main__":
    main()
