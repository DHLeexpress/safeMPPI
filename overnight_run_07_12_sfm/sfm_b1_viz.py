"""B1 query/certificate animation and four candidate-specific zoom panels."""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

import _paths  # noqa: F401
from polar_grid import polytope_HP
import sfm_scene as SS


def _nominal_contours(axis, state, ped_xy):
    obstacles = np.concatenate([
        np.asarray(ped_xy), np.full((len(ped_xy), 1), SS.R_PED)
    ], axis=1)
    hp, _ = polytope_HP(state[:2], obstacles, sensing=SS.R_SENSE, n_base=16)
    x = np.linspace(state[0] - 2.2, state[0] + 2.2, 90)
    y = np.linspace(state[1] - 2.2, state[1] + 2.2, 90)
    xx, yy = np.meshgrid(x, y)
    value = hp(np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)
    axis.contour(xx, yy, value, levels=[0.0, 0.5], colors=["#2166ac", "#67a9cf"], linewidths=[1.5, .8])


def _candidate_status(trace, candidate_id):
    for row in trace["query_rows"]:
        if int(row["candidate_id"]) == int(candidate_id):
            if not row["result"].get("resolved"):
                return "error", row
            return ("positive" if int(row["result"]["y"]) else "negative"), row
    return "unqueried", None


def _draw_common(axis, trace):
    state = np.asarray(trace["state"])
    ped_xy = np.asarray(trace["ped_xy"])
    ped_vel = np.asarray(trace["ped_vel"])
    _nominal_contours(axis, state, ped_xy)
    for position in ped_xy:
        axis.add_patch(Circle(position, SS.R_PED, color="#555555", alpha=.72, zorder=5))
    for index in range(len(ped_xy)):
        prediction = ped_xy[index][None] + np.arange(11)[:, None] * SS.DT * ped_vel[index][None]
        axis.plot(prediction[:, 0], prediction[:, 1], ".--", color="#777777", ms=2.2, lw=.65, alpha=.65)
    axis.plot(state[0], state[1], "o", color="#2166ac", ms=7, zorder=8)
    axis.set_aspect("equal")
    axis.set_xlim(max(SS.TASK_LO, state[0] - 2.35), min(SS.TASK_HI, state[0] + 2.35))
    axis.set_ylim(max(SS.TASK_LO, state[1] - 2.35), min(SS.TASK_HI, state[1] + 2.35))
    axis.grid(alpha=.15)


def draw_query_frame(axis, trace):
    _draw_common(axis, trace)
    selected = set(map(int, trace["selected_ids"]))
    executed = trace.get("executed_id")
    for row in trace["all_K"]:
        candidate = int(row["candidate_id"])
        path = np.asarray(row["segment"])
        color, width, alpha = "#999999", .65, .35
        if candidate in selected:
            status, _ = _candidate_status(trace, candidate)
            color = "#2ca02c" if status == "positive" else "#d62728" if status == "negative" else "#ff8c00"
            width, alpha = 1.35, .88
        axis.plot(path[:, 0], path[:, 1], color=color, lw=width, alpha=alpha, zorder=4)
    if executed is not None:
        row = trace["all_K"][int(executed)]
        path = np.asarray(row["segment"])
        axis.plot(path[:, 0], path[:, 1], color="#08519c", lw=3.0, zorder=9)
        axis.annotate("", xy=path[1], xytext=path[0],
                      arrowprops=dict(arrowstyle="->", color="#08519c", lw=2.8))
    axis.set_title(
        f"r{trace['round']} s{trace['scenario_id']} gamma={trace['gamma']} t={trace['step']}"
    )


def _draw_time_indexed_sets(axis, trace, query_row):
    result = query_row["result"]
    segment = np.asarray(result["segment"])
    center = segment[0]
    gamma = float(trace["gamma"])
    faces = [face for face in result.get("faces", []) if face.kind == "real-moving" and face.feasible]
    colors = plt.cm.Greens(np.linspace(.35, .95, len(segment)))
    extent = 3.2
    for horizon, position in enumerate(segment):
        beta = 1.0 - (1.0 - gamma) ** horizon
        for face in faces:
            normal = np.asarray(face.a, float)
            tangent = np.array([-normal[1], normal[0]])
            point = center + normal * (beta * float(face.m))
            ends = np.stack([point - extent * tangent, point + extent * tangent])
            axis.plot(ends[:, 0], ends[:, 1], color=colors[horizon], lw=.45, alpha=.28)
        axis.plot(position[0], position[1], "o", ms=2.8, color=colors[horizon], zorder=7)


def render_zoom_panels(trace, output):
    selected = list(map(int, trace["selected_ids"]))[:4]
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, candidate in zip(axes.flat, selected):
        _draw_common(axis, trace)
        status, query = _candidate_status(trace, candidate)
        path = np.asarray(trace["all_K"][candidate]["segment"])
        color = "#2ca02c" if status == "positive" else "#d62728" if status == "negative" else "#ff8c00"
        axis.plot(path[:, 0], path[:, 1], color=color, lw=2.4, zorder=8)
        if query is not None and query["result"].get("resolved"):
            _draw_time_indexed_sets(axis, trace, query)
        axis.set_title(f"queried candidate {candidate}: {status}; each h uses its h-set")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def render_query_animation(trace_path, output_mp4, output_panels, *, fps=5, max_frames=120):
    started = time.perf_counter()
    traces = torch.load(trace_path, map_location="cpu", weights_only=False)
    if not traces:
        raise ValueError("query trace is empty")
    render_zoom_panels(next(trace for trace in traces if len(trace["selected_ids"]) >= 4), output_panels)
    frames = traces[:int(max_frames)]
    figure, axis = plt.subplots(figsize=(7.5, 7.0))
    def update(index):
        axis.clear()
        draw_query_frame(axis, frames[index])
        return []
    movie = animation.FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    movie.save(output_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=130)
    plt.close(figure)
    return dict(
        trace=os.path.abspath(trace_path), mp4=os.path.abspath(output_mp4),
        panels=os.path.abspath(output_panels), frames=len(frames),
        rendering_wall_seconds=time.perf_counter() - started,
    )


def render_raw_gallery(r0_checkpoint, selected_checkpoint, output_png, output_mp4, *, device="cuda"):
    """Fixed raw rows: Hp10 r0, selected raw expansion, default Kazuki generate-refine."""
    import grid_policy_sfm as GPS
    import sfm_b1_eval as BE
    import sfm_kazuki as KZ
    import sfm_protocol as SP
    r0, _ = GPS.load_sfm_policy(r0_checkpoint, device=device)
    selected, _ = GPS.load_sfm_policy(selected_checkpoint, device=device)
    methods = ("Hp10 r0 raw", "selected raw", "default Kazuki generate-refine")
    rows = {name: [] for name in methods}
    config = KZ.KazukiConfig(safe_coefs=(0.3,), goal_coef=0.5).validate()
    for index, gamma in enumerate(SP.GAMMAS):
        episode = SP.CONFIRM_EP0 + index
        rows[methods[0]].append(BE.raw_rollout(r0, episode, gamma, device=device, collect_trace=True))
        rows[methods[1]].append(BE.raw_rollout(selected, episode, gamma, device=device, collect_trace=True))
        value = KZ.kazuki_sfm_deploy(
            r0, episode, gamma, cfg=config, T=SP.T, device=device,
            ped_speed_range=SS.OOD_PED_SPEED_RANGE, collect_diagnostics=False,
        )
        rows[methods[2]].append(value)
    figure, axes = plt.subplots(3, len(SP.GAMMAS), figsize=(19, 8.2), constrained_layout=True)
    colors = ("#666666", "#0072B2", "#D55E00")
    for row_index, method in enumerate(methods):
        for gamma_index, gamma in enumerate(SP.GAMMAS):
            axis = axes[row_index, gamma_index]
            value = rows[method][gamma_index]
            states = np.asarray(value["states"])
            peds = np.asarray(value["peds"])
            if len(peds):
                axis.scatter(peds[0, :, 0], peds[0, :, 1], s=5, color="#777777", alpha=.5)
            axis.plot(states[:, 0], states[:, 1], color=colors[row_index], lw=1.5)
            axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#009E73", ms=8)
            axis.set_xlim(SS.TASK_LO, SS.TASK_HI); axis.set_ylim(SS.TASK_LO, SS.TASK_HI)
            axis.set_aspect("equal"); axis.grid(alpha=.12)
            if row_index == 0:
                axis.set_title(f"gamma={gamma}")
            if gamma_index == 0:
                axis.set_ylabel(method)
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    figure.savefig(output_png, dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(3, len(SP.GAMMAS), figsize=(19, 8.2), constrained_layout=True)
    max_steps = max(len(np.asarray(value["states"])) for values in rows.values() for value in values)
    def update(frame):
        for row_index, method in enumerate(methods):
            for gamma_index, gamma in enumerate(SP.GAMMAS):
                axis = axes[row_index, gamma_index]; axis.clear()
                value = rows[method][gamma_index]
                states = np.asarray(value["states"]); peds = np.asarray(value["peds"])
                stop = min(frame + 1, len(states))
                axis.plot(states[:stop, 0], states[:stop, 1], color=colors[row_index], lw=1.6)
                if len(peds):
                    ped_index = min(frame, len(peds) - 1)
                    axis.scatter(peds[ped_index, :, 0], peds[ped_index, :, 1], s=5, color="#555555")
                axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#009E73", ms=8)
                axis.set_xlim(SS.TASK_LO, SS.TASK_HI); axis.set_ylim(SS.TASK_LO, SS.TASK_HI)
                axis.set_aspect("equal"); axis.grid(alpha=.12)
                if row_index == 0: axis.set_title(f"gamma={gamma}")
                if gamma_index == 0: axis.set_ylabel(method)
        return []
    movie = animation.FuncAnimation(figure, update, frames=max_steps, interval=100, blit=False)
    movie.save(output_mp4, writer=animation.FFMpegWriter(fps=10, bitrate=3000), dpi=95)
    plt.close(figure)
    return dict(png=os.path.abspath(output_png), mp4=os.path.abspath(output_mp4), methods=methods)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace")
    parser.add_argument("--mp4", required=True)
    parser.add_argument("--panels")
    parser.add_argument("--r0")
    parser.add_argument("--selected")
    parser.add_argument("--gallery")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.r0 or args.selected:
        if not (args.r0 and args.selected and args.gallery):
            parser.error("raw gallery requires --r0, --selected, and --gallery")
        report = render_raw_gallery(args.r0, args.selected, args.gallery, args.mp4, device=args.device)
    else:
        if not (args.trace and args.panels):
            parser.error("query animation requires --trace and --panels")
        report = render_query_animation(args.trace, args.mp4, args.panels)
    with open(args.report, "w") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
