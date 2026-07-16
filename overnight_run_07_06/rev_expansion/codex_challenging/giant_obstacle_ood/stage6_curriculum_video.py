#!/usr/bin/env python3
"""Giant-scene wrapper for the exact fixed-curriculum video grammar."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
REF = ROOT / "reference"
# Import the benchmark scene before loading the reference renderer.  ROOT and
# WORK must precede REF because both trees contain a ``gen_uniform_data.py``;
# only the benchmark copy defines the approved walled-stadium constructor.
for path in (REF, ROOT.parent, WORK, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from giant_obstacle_ood.stage1_geometry_sweep import GIANT_CENTER, make_scene  # noqa: E402
from giant_obstacle_ood.stage1b_smooth_expert import GOAL, RADIUS, START  # noqa: E402
from giant_obstacle_ood.stage4_frozen_ood import CHECKPOINT  # noqa: E402


SPEC = importlib.util.spec_from_file_location("exact_curriculum_v4", REF / "video_curriculum_fixed.py")
V = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V)

STAGE5 = HERE / "stage_results/05_window_expand"
RUN = STAGE5 / "runs/temp0.5_stable/full"
OUT = HERE / "stage_results/06_exact_reports"
VIDEO = OUT / "viz/curriculum_it20.mp4"
FRAMES = OUT / "viz/curriculum_frames"


def giant_scene(axis) -> None:
    axis.set_facecolor("#f7f6f4")
    rr = float(V.env.r_robot)
    for obstacle in V.OBS:
        is_giant = np.linalg.norm(obstacle[:2] - GIANT_CENTER) < 1e-6
        axis.add_patch(Circle(
            (obstacle[0], obstacle[1]), obstacle[2] + rr,
            facecolor="#686868" if is_giant else "#8a8a8a",
            ec="#b2182b" if is_giant else "none",
            lw=1.6 if is_giant else 0.0, zorder=2,
        ))
    axis.plot(START[0], START[1], "s", c="k", ms=7, zorder=8)
    axis.plot(GOAL[0], GOAL[1], "*", c="gold", mec="k", ms=15, zorder=8)
    axis.set_xlim(-.45, 5.45); axis.set_ylim(-.45, 5.45); axis.set_aspect("equal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=VIDEO)
    parser.add_argument("--fps", type=int, default=2)
    args_cli = parser.parse_args()
    run = args_cli.run.resolve(); output = args_cli.out.resolve()
    recipe = json.loads((run / "recipe.json").read_text())
    recs = [json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines() if line]
    if not recs:
        raise RuntimeError("curriculum run has no probe rows")

    V.env = make_scene(RADIUS, START, GOAL)
    V.env.T = 300
    V.OBS = V.env.obstacles.detach().cpu().numpy()
    V.EXTRA_OBS = []
    V.scene = giant_scene
    V.GX2.GM2.GOAL_XY = np.asarray(GOAL, dtype=float)

    config = SimpleNamespace(
        title="Safe Flow Expansion — giant-obstacle valid2-window curriculum",
        vpf=float(recipe["valid_prog_floor"]),
        batch_cap=int(recipe["batch"]),
        initial_demo_req=int(round(float(recipe["demo_frac"]) * int(recipe["batch"]))),
        n_max=max(int(row["iter"]) for row in recs),
    )
    r0 = {
        "iter": 0, "beta": recs[0].get("beta", .2),
        "n_easy": 0, "n_frontier": 0,
        "batch_e": 0, "batch_f": 0, "batch_d": 0,
        "demo_req": config.initial_demo_req,
        "mix_e": recs[0].get("mix_e", .4), "mix_f": recs[0].get("mix_f", .6),
        "lr": recs[0].get("lr", 5e-6),
    }
    recs = [r0] + recs
    by_iteration = {int(row["iter"]): row for row in recs}
    iterations = list(range(0, config.n_max + 1))
    db_by_iteration = {
        iteration: (V.load_db(str(run), iteration) if iteration > 0 else None)
        for iteration in iterations
    }
    sigma_values = [db["sig"] for db in db_by_iteration.values() if db is not None]
    sigma_limits = (
        0.0,
        float(np.percentile(np.concatenate(sigma_values), 98)) if sigma_values else 1.0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, _ = V.HP.load_hp(CHECKPOINT, device=device)
    torch.manual_seed(31337); np.random.seed(31337)
    pre_path = np.asarray(V.GR.fm_deploy(
        policy, V.env, .5, T=300, temp=.5, nfe=8, device=device
    )["path"])

    FRAMES.mkdir(parents=True, exist_ok=True)
    for frame_path in FRAMES.glob("*.png"):
        frame_path.unlink()
    figure = plt.figure(figsize=(26, 13)); figure.patch.set_facecolor("white")
    frame_index = 0
    for iteration in iterations:
        record = by_iteration.get(iteration)
        upto = [row for row in recs if int(row["iter"]) <= iteration]
        V.frame(
            figure, iteration, db_by_iteration[iteration], record, upto,
            pre_path, sigma_limits, config,
        )
        # Two copies at 2 fps retain the established slow one-second cadence.
        for _ in range(2):
            figure.savefig(
                FRAMES / f"f{frame_index:04d}.png", dpi=78,
                facecolor="white", transparent=False,
            )
            frame_index += 1
        print(f"frame it{iteration}", flush=True)
    plt.close(figure)

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(args_cli.fps),
        "-i", str(FRAMES / "f%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(output),
    ], check=True, capture_output=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(output),
    ], check=True, capture_output=True, text=True)
    manifest = {
        "status": "PASS",
        "run": str(run), "output": str(output),
        "iterations": iterations, "frames": frame_index,
        "temperature": .5, "gamma_shown_at_it0": .5,
        "sigma_colormap": "viridis", "gamma_colormap": "plasma_trunc in rollout/scatter companions",
        "ffprobe": json.loads(probe.stdout),
    }
    (output.parent / "curriculum_video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
