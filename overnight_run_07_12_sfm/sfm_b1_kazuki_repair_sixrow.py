"""Two-episode, six-row mechanism comparison for Kazuki-repaired gathering.

Rows are episode-major:

1. raw pretrained deployment with exact verifier geometry;
2. locked Kazuki generate--guide--refine with separate guidance arrows;
3. same-latent B4 repair gathering;

then the same three rows for the second fixed OOD episode.  Rendering consumes
an immutable trace bundle and never reruns a controller or verifier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_density_viz as DV
import sfm_b1_eval as BE
import sfm_b1_full_episode_viz as FV
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_viz as BV
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_scene as SS


STATUS = "SFM_B1_KAZUKI_REPAIR_SIXROW_TRACE_COMPLETE"
RENDER_STATUS = "SFM_B1_KAZUKI_REPAIR_SIXROW_RENDER_COMPLETE"
TRUE_BLUE = "#0057FF"
TRUE_RED = "#D62728"
MAGENTA = "#CC79A7"
CYAN = "#00A6D6"
GREEN = "#009E73"
DEFAULT_EPISODES = (250_001, 250_003)
DEFAULT_GAMMAS = (0.1, 0.5, 1.0)


def _write_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _save_torch(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _collect_raw(policy, episodes, gammas, environment, *, device, T, sample_seed):
    runs = {}
    pending = []
    for episode in episodes:
        for gamma in gammas:
            run = BE.raw_rollout(
                policy,
                episode,
                gamma,
                device=device,
                T=int(T),
                n_ped=int(environment["n_ped"]),
                temp=1.0,
                nfe=8,
                ped_speed_range=tuple(environment["ped_speed_range"]),
                sample_seed=int(sample_seed),
                collect_trace=True,
            )
            runs[(int(episode), float(gamma))] = run
            for trace_index, trace in enumerate(run["trace"]):
                pending.append((
                    len(pending),
                    0,
                    trace["state"],
                    trace["controls"],
                    trace["ped_xy"],
                    trace["ped_vel"],
                    float(gamma),
                    int(episode),
                    int(trace_index),
                ))

    worker_tasks = [row[:7] for row in pending]
    workers = min(32, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(SM.verify_in_worker, worker_tasks))
    for pending_row, result_row in zip(pending, results):
        _, _, result = result_row
        episode = pending_row[-2]
        trace_index = pending_row[-1]
        gamma = float(pending_row[6])
        runs[(episode, gamma)]["trace"][trace_index][
            "verifier_result"
        ] = result
    return runs


def _collect_kazuki(
    policy, episodes, gammas, environment, *, device, T, sample_seed,
):
    config = KZ.KazukiConfig(
        safe_coefs=(0.3,),
        goal_coef=0.5,
        safe_coef_gamma_span=0.0,
        goal_coef_gamma_span=0.0,
    ).validate()
    return {
        (int(episode), float(gamma)): KZ.kazuki_sfm_deploy(
            policy,
            episode=int(episode),
            gamma=float(gamma),
            cfg=config,
            n_ped=int(environment["n_ped"]),
            T=int(T),
            device=device,
            ped_speed_range=tuple(environment["ped_speed_range"]),
            sample_seed=int(sample_seed),
            collect_diagnostics=True,
        )
        for episode in episodes for gamma in gammas
    }


def collect(
    checkpoint,
    output_dir,
    *,
    episodes=DEFAULT_EPISODES,
    gammas=DEFAULT_GAMMAS,
    scene_profile="double_density_velocity_ood",
    selector="margin",
    device="cuda",
    verifier_workers=16,
    sample_seed=700_000,
    audit_seed=20260730,
    ell=RA.DEFAULT_ELL,
    T=180,
):
    episodes = tuple(map(int, episodes))
    gammas = tuple(map(float, gammas))
    if len(episodes) != 2 or len(set(episodes)) != 2:
        raise ValueError("six-row comparison requires two distinct episodes")
    if gammas != DEFAULT_GAMMAS:
        raise ValueError(f"six-row comparison requires gammas={DEFAULT_GAMMAS}")
    if int(T) != 180:
        raise ValueError("six-row comparison is scientifically pinned to T=180")
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    os.makedirs(output_dir)
    environment = SS.scene_profile(scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()

    raw = _collect_raw(
        policy,
        episodes,
        gammas,
        environment,
        device=device,
        T=T,
        sample_seed=sample_seed,
    )
    kazuki = _collect_kazuki(
        policy,
        episodes,
        gammas,
        environment,
        device=device,
        T=T,
        sample_seed=sample_seed,
    )
    repair_dir = os.path.join(output_dir, "repair")
    repair_path = RA.collect(
        checkpoint,
        scenarios=episodes,
        gammas=gammas,
        scene_profile=scene_profile,
        selector=selector,
        device=device,
        verifier_workers=verifier_workers,
        sample_seed=sample_seed,
        audit_seed=audit_seed,
        ell=ell,
        T=T,
        outdir=repair_dir,
    )
    repair = torch.load(
        repair_path, map_location="cpu", weights_only=False,
    )
    bundle = dict(
        version=1,
        status=STATUS,
        source=FA._source(),
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=RA.FA._sha256_file(checkpoint),
        episodes=list(episodes),
        gammas=list(gammas),
        scene_profile=scene_profile,
        environment=environment,
        selector=selector,
        sample_seed=int(sample_seed),
        audit_seed=int(audit_seed),
        raw=raw,
        kazuki=kazuki,
        repair=repair,
        method_semantics=dict(
            raw=(
                "pretrained temp=1,NFE=8 raw policy; exact verifier geometry "
                "is audit-only and never alters execution"
            ),
            kazuki=(
                "locked pretrained prior, safe_coef=.3, goal_coef=.5, full "
                "generate-guide-refine comparator"
            ),
            repair=(
                "K=16, RBF B=4; same-latent locked guidance only on NVP/trap; "
                "guided B is reverified; no raw fallback or privileged MPC"
            ),
        ),
    )
    trace_path = os.path.join(output_dir, "sixrow_trace.pt")
    _save_torch(trace_path, bundle)
    _write_json(os.path.join(output_dir, "TRACE_COMPLETE.json"), dict(
        status=STATUS,
        source=bundle["source"],
        trace_path=os.path.abspath(trace_path),
        trace_sha256=RA.FA._sha256_file(trace_path),
        checkpoint_sha256=bundle["checkpoint_sha256"],
        episodes=list(episodes),
        gammas=list(gammas),
        selector=selector,
        repair_complete=os.path.join(repair_dir, "COMPLETE.json"),
    ))
    return trace_path


def _repair_index(traces):
    output = {}
    for trace in traces:
        key = (int(trace["scenario_id"]), round(float(trace["gamma"]), 8))
        output.setdefault(key, {})[int(trace["step"])] = trace
    return output


def _clamped_trace(run, step):
    traces = list(run.get("trace") or ())
    if not traces:
        raise ValueError("run has no mechanism trace")
    return traces[min(max(int(step), 0), len(traces) - 1)]


def _draw_raw(axis, run, gamma, step):
    trace = _clamped_trace(run, step)
    DV.draw_method_panel(
        axis,
        "selected",
        run,
        gamma,
        int(step),
        verifier_result=trace["verifier_result"],
    )


def _draw_kazuki(axis, run, gamma, step):
    DV.draw_method_panel(
        axis,
        "kazuki",
        run,
        gamma,
        int(step),
        guidance_scale=3.0,
        guidance_cap=1.8,
    )


def _branch_path(trace):
    result = trace.get("executed_result")
    if not result:
        return None
    path = np.asarray(result.get("segment", ()), float)
    return path if path.shape == (11, 2) else None


def _draw_repair_history(axis, rows, step):
    states = []
    for value in sorted(key for key in rows if key <= int(step)):
        trace = rows[value]
        if trace.get("executed_result") is None:
            continue
        path = _branch_path(trace)
        if path is not None:
            branch_color = (
                MAGENTA
                if str(trace.get("execution_source", "")).startswith(
                    "kazuki_repair_"
                )
                else TRUE_BLUE
            )
            axis.plot(
                path[:, 0], path[:, 1],
                color=branch_color,
                lw=.7,
                marker=".",
                ms=1.1,
                alpha=.32,
                zorder=3,
            )
        states.append(np.asarray(trace["state"], float)[:2])
        states.append(np.asarray(trace["next_state"], float)[:2])
    if states:
        path = np.asarray(states)[::2]
        final = np.asarray(states[-1])[None]
        path = np.concatenate([path, final], axis=0)
        axis.plot(
            path[:, 0], path[:, 1],
            color="#111111",
            lw=1.25,
            marker=".",
            ms=1.4,
            zorder=10,
        )


def _draw_query_path(axis, trace, row, *, color, linewidth, alpha):
    path = np.asarray(
        row["result"].get(
            "segment",
            SM.rollout_positions(trace["state"], row["controls"]),
        ),
        float,
    )
    axis.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        lw=float(linewidth),
        marker=".",
        ms=1.35,
        alpha=float(alpha),
        zorder=6,
    )
    result = row["result"]
    if not (result.get("resolved") and int(result.get("y", 0)) == 1):
        axis.plot(
            path[-1, 0], path[-1, 1],
            "x", color=TRUE_RED, ms=4.4, mew=1.0, zorder=8,
        )


def _draw_repair(axis, rows, step):
    available = [value for value in rows if value <= int(step)]
    current_step = max(available) if available else min(rows)
    trace = rows[current_step]
    BV._draw_common(axis, trace, nominal_levels=False)
    _draw_repair_history(axis, rows, current_step)
    for row in trace["all_K"]:
        path = np.asarray(row["segment"], float)
        axis.plot(
            path[:, 0], path[:, 1],
            color=BV.GRAY, lw=.35, marker=".", ms=.9,
            alpha=.18, zorder=2,
        )
    for row in trace["query_rows"]:
        _draw_query_path(
            axis, trace, row, color=GREEN, linewidth=.95, alpha=.85,
        )
    for row in trace.get("guided_query_rows", ()):
        selected = (
            trace.get("repair_selected_id") is not None
            and int(row["candidate_id"]) == int(trace["repair_selected_id"])
        )
        _draw_query_path(
            axis,
            trace,
            row,
            color=MAGENTA,
            linewidth=2.0 if selected else 1.15,
            alpha=.98 if selected else .76,
        )
    if trace.get("executed_result") is not None:
        audit = DV.checked_verifier_levels(
            trace, dict(result=trace["executed_result"]), H=10,
        )
        DV._draw_verifier_geometry(axis, audit)
    else:
        position = np.asarray(trace["state"], float)[:2]
        axis.plot(
            position[0], position[1],
            marker="o", ms=9, mfc="none", mec=TRUE_RED,
            mew=1.5, zorder=12,
        )
    axis.plot(
        SS.GOAL[0], SS.GOAL[1], "*",
        color="#F0E442", mec="#333333", ms=8, zorder=15,
    )
    DV._set_clean_axis(axis)


def _layout(bundle):
    gammas = tuple(map(float, bundle["gammas"]))
    figure = plt.figure(figsize=(3.05 * len(gammas) + 4.2, 16.6))
    grid = figure.add_gridspec(
        6,
        len(gammas) + 1,
        width_ratios=[1.0] * len(gammas) + [1.22],
        left=.025,
        right=.985,
        bottom=.025,
        top=.985,
        wspace=.025,
        hspace=.035,
    )
    axes = np.empty((6, len(gammas)), dtype=object)
    sides = []
    for row in range(6):
        for column in range(len(gammas)):
            axes[row, column] = figure.add_subplot(grid[row, column])
        side = figure.add_subplot(grid[row, -1])
        side.set_axis_off()
        sides.append(side)
    return figure, axes, sides, gammas


def _side_text(
    bundle,
    row,
    frame_index,
    frame_count,
    simulator_step,
    repair_index,
):
    episode = int(bundle["episodes"][row // 3])
    method = row % 3
    labels = (
        (
            "Raw pretrained r0",
            "temperature 1 · NFE 8",
            "verifier geometry is audit-only",
        ),
        (
            "Locked Kazuki comparator",
            "safe=.3 · goal=.5",
            "cyan=goal · magenta=safety",
        ),
        (
            f"Active B4 repair · {bundle['selector']}",
            "base B=green · guided B=magenta",
            "no independent raw / privileged MPC",
        ),
    )
    lines = [
        f"episode {episode}",
        *labels[method],
    ]
    if method == 2:
        local_status = []
        for gamma in bundle["gammas"]:
            traces = repair_index[
                (episode, round(float(gamma), 8))
            ]
            final_step = max(traces)
            local_step = min(int(simulator_step), final_step)
            final_trace = traces[final_step]
            stopped = (
                int(simulator_step) >= final_step
                and final_trace.get("repair_trigger") is not None
                and final_trace.get("repair_selected_id") is None
            )
            suffix = " · repair NVP" if stopped else ""
            local_status.append(
                f"γ={float(gamma):g}: t={local_step}{suffix}"
            )
        lines.extend(("", *local_status))
    if row == 0:
        columns = ", ".join(f"{value:g}" for value in bundle["gammas"])
        lines.extend((
            "",
            f"columns γ: {columns}",
            f"frame {frame_index}/{frame_count - 1}",
            f"simulator step {simulator_step}",
        ))
    return "\n".join(lines)


def _maximum_step(bundle, repair_index):
    maximum = 0
    for group in ("raw", "kazuki"):
        for run in bundle[group].values():
            maximum = max(maximum, len(run.get("trace") or ()) - 1)
    for rows in repair_index.values():
        maximum = max(maximum, max(rows))
    return maximum


def draw_frame(bundle, frame_index, frames, *, layout=None):
    if bundle.get("status") != STATUS:
        raise ValueError("not a completed repair six-row bundle")
    if layout is None:
        layout = _layout(bundle)
    figure, axes, sides, gammas = layout
    step = int(frames[int(frame_index)])
    repair_index = _repair_index(bundle["repair"]["traces"])
    for row in range(6):
        sides[row].clear()
        sides[row].set_axis_off()
        sides[row].text(
            .02,
            .96,
            _side_text(
                bundle,
                row,
                frame_index,
                len(frames),
                step,
                repair_index,
            ),
            ha="left",
            va="top",
            fontsize=8.0,
            linespacing=1.42,
        )
    for episode_index, episode in enumerate(bundle["episodes"]):
        row0 = 3 * episode_index
        for column, gamma in enumerate(gammas):
            for row in range(row0, row0 + 3):
                axes[row, column].clear()
            key = (int(episode), float(gamma))
            _draw_raw(axes[row0, column], bundle["raw"][key], gamma, step)
            _draw_kazuki(
                axes[row0 + 1, column],
                bundle["kazuki"][key],
                gamma,
                step,
            )
            _draw_repair(
                axes[row0 + 2, column],
                repair_index[(int(episode), round(float(gamma), 8))],
                step,
            )
    return layout


def _legend():
    return [
        Line2D([], [], color=TRUE_BLUE, lw=1.4, label=r"base executed exact-positive $D^+$"),
        Line2D([], [], color="#111111", lw=1.2, label="executed first-action path"),
        Line2D([], [], color=GREEN, lw=1.1, label="base RBF B=4 query"),
        Line2D([], [], color=MAGENTA, lw=1.5, label="guided B repair / repaired executed branch"),
        Line2D([], [], color=TRUE_RED, marker="x", lw=0, label="exact rejected endpoint"),
        Line2D([], [], color=GREEN, lw=.7, label="executed verifier levels h=1..10"),
        Line2D([], [], color=CYAN, lw=2.2, label=r"Kazuki integrated $\nabla$ goal"),
        Line2D([], [], color=MAGENTA, lw=2.2, label=r"Kazuki integrated $\nabla$ safety"),
    ]


def render(trace_path, output_dir, *, fps=5, frame_stride=2, dpi=105):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    bundle = torch.load(
        trace_path, map_location="cpu", weights_only=False,
    )
    repair_index = _repair_index(bundle["repair"]["traces"])
    maximum = _maximum_step(bundle, repair_index)
    frames = list(range(0, maximum + 1, int(frame_stride)))
    if frames[-1] != maximum:
        frames.append(maximum)
    layout = _layout(bundle)
    figure = layout[0]
    figure.legend(
        handles=_legend(),
        loc="lower right",
        bbox_to_anchor=(.985, .012),
        frameon=False,
        fontsize=7.0,
    )
    draw_frame(bundle, len(frames) - 1, frames, layout=layout)
    last_frame = os.path.join(output_dir, "sixrow_last_frame.png")
    figure.savefig(last_frame, dpi=170)

    def update(frame_index):
        draw_frame(bundle, frame_index, frames, layout=layout)
        return []

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=range(len(frames)),
        interval=1000 / int(fps),
        blit=False,
    )
    mp4 = os.path.join(output_dir, "sixrow_comparison.mp4")
    movie.save(
        mp4,
        writer=animation.FFMpegWriter(fps=int(fps), bitrate=5200),
        dpi=int(dpi),
    )
    plt.close(figure)
    report = dict(
        status=RENDER_STATUS,
        source=bundle.get("source"),
        trace_path=os.path.abspath(trace_path),
        trace_sha256=RA.FA._sha256_file(trace_path),
        frames=frames,
        frame_count=len(frames),
        fps=int(fps),
        frame_stride=int(frame_stride),
        mp4=os.path.abspath(mp4),
        last_frame_png=os.path.abspath(last_frame),
        episodes=list(map(int, bundle["episodes"])),
        gammas=list(map(float, bundle["gammas"])),
        selector=bundle["selector"],
    )
    _write_json(os.path.join(output_dir, "RENDER_COMPLETE.json"), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--checkpoint", required=True)
    collect_parser.add_argument("--output-dir", required=True)
    collect_parser.add_argument(
        "--episodes", type=int, nargs=2, default=DEFAULT_EPISODES,
    )
    collect_parser.add_argument(
        "--gammas", type=float, nargs=3, default=DEFAULT_GAMMAS,
    )
    collect_parser.add_argument(
        "--scene-profile", default="double_density_velocity_ood",
    )
    collect_parser.add_argument(
        "--selector", choices=("margin", "safemppi_cost"), default="margin",
    )
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--verifier-workers", type=int, default=16)
    collect_parser.add_argument("--sample-seed", type=int, default=700_000)
    collect_parser.add_argument("--audit-seed", type=int, default=20260730)
    collect_parser.add_argument("--ell", type=float, default=RA.DEFAULT_ELL)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--trace", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument("--fps", type=int, default=5)
    render_parser.add_argument("--frame-stride", type=int, default=2)
    render_parser.add_argument("--dpi", type=int, default=105)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(
            args.checkpoint,
            args.output_dir,
            episodes=args.episodes,
            gammas=args.gammas,
            scene_profile=args.scene_profile,
            selector=args.selector,
            device=args.device,
            verifier_workers=args.verifier_workers,
            sample_seed=args.sample_seed,
            audit_seed=args.audit_seed,
            ell=args.ell,
            T=180,
        )
    else:
        render(
            args.trace,
            args.output_dir,
            fps=args.fps,
            frame_stride=args.frame_stride,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
