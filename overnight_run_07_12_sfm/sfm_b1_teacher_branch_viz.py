"""Render ordinary executed D branches beside a separate MPC-teacher buffer.

The two inputs must be the exact matching macro-round shard and teacher
buffer.  Blue/red are the ordinary exact-verifier labels in D; purple is a
privileged Codex MPC teacher target and is deliberately never presented as a
certificate or as D+.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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
import sfm_b1_offline_store as OS
import sfm_b1_viz as BV
import sfm_metrics2 as SM
import sfm_scene as SS


TEACHER_STATUS = "SFM_UNVERIFIED_MPC_TEACHER_BUFFER_COMPLETE"
TEACHER_SOURCE = "codex_privileged_sfm_mpc"
D_POS_COLOR = "#0067C5"
D_NEG_COLOR = "#C62828"
TEACHER_COLOR = "#7B2CBF"
CONTEXT_FIELDS = ("scenario_id", "gamma", "step", "state", "ped_xy", "ped_vel")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _context_snapshot(bundle, record):
    for key in ("context_snapshot", "context"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    snapshots = bundle.get("context_snapshots")
    context_id = int(record["context_id"])
    if isinstance(snapshots, list) and 0 <= context_id < len(snapshots):
        return snapshots[context_id]
    if isinstance(snapshots, dict):
        value = snapshots.get(context_id, snapshots.get(str(context_id)))
        if isinstance(value, dict):
            return value
    if all(field in record for field in CONTEXT_FIELDS):
        return record
    raise ValueError(
        f"teacher {record.get('teacher_id')} has no auditable context snapshot"
    )


def _same_array(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return left.shape == right.shape and np.array_equal(left, right)


def _validate_snapshot(snapshot, context, teacher_id):
    scalar_match = (
        int(snapshot["scenario_id"]) == int(context["scenario_id"])
        and round(float(snapshot["gamma"]), 8)
        == round(float(context["gamma"]), 8)
        and int(snapshot["step"]) == int(context["step"])
    )
    array_match = all(
        _same_array(snapshot[field], context[field])
        for field in ("state", "ped_xy", "ped_vel")
    )
    if not scalar_match or not array_match:
        raise ValueError(
            f"teacher {teacher_id} context snapshot does not match round shard"
        )


def load_inputs(round_shard_path, teacher_buffer_path):
    """Load and fail closed unless every teacher row matches the exact shard."""
    round_shard_path = os.path.abspath(round_shard_path)
    teacher_buffer_path = os.path.abspath(teacher_buffer_path)
    shard_sha256 = _sha256(round_shard_path)
    shard = OS.ExecutedRoundShard.load(round_shard_path)
    bundle = torch.load(
        teacher_buffer_path, map_location="cpu", weights_only=False,
    )
    if bundle.get("status") != TEACHER_STATUS:
        raise ValueError("input is not a completed unverified MPC teacher buffer")
    if int(bundle.get("version", -1)) != 1:
        raise ValueError("unsupported unverified MPC teacher buffer version")
    if int(bundle.get("round", -1)) != int(shard.round_i):
        raise ValueError("teacher buffer round does not match D round shard")
    if bundle.get("round_shard_sha256") != shard_sha256:
        raise ValueError("teacher buffer does not authenticate the exact D round shard")

    records = list(bundle.get("records", ()))
    seen = set()
    by_context = defaultdict(list)
    family_counts = defaultdict(int)
    for index, record in enumerate(records):
        teacher_id = int(record.get("teacher_id", -1))
        if teacher_id < 0 or teacher_id in seen:
            raise ValueError("teacher IDs must be unique non-negative integers")
        seen.add(teacher_id)
        if record.get("source") != TEACHER_SOURCE:
            raise ValueError(f"teacher {teacher_id} has an undeclared source")
        forbidden = {"y", "query_id", "train_eligible", "x0"} & set(record)
        if forbidden:
            raise ValueError(
                f"teacher {teacher_id} illegally carries safety-label fields "
                f"{sorted(forbidden)}"
            )
        context_id = int(record["context_id"])
        if not 0 <= context_id < len(shard.contexts):
            raise ValueError(f"teacher {teacher_id} references missing context")
        context = shard.contexts[context_id]
        _validate_snapshot(
            _context_snapshot(bundle, record), context, teacher_id,
        )
        controls = np.asarray(record["controls"], np.float32)
        if (
            controls.shape != (10, 2)
            or not np.isfinite(controls).all()
            or float(np.max(np.abs(controls))) > float(SS.U_MAX) + 1.0e-6
        ):
            raise ValueError(
                f"teacher {teacher_id} controls violate the H10/input-limit contract"
            )
        normalized = dict(record)
        normalized["teacher_id"] = teacher_id
        normalized["context_id"] = context_id
        normalized["controls"] = controls
        by_context[context_id].append(normalized)
        candidate_source = record.get("candidate_source")
        family = (
            candidate_source.get("family", "unknown")
            if isinstance(candidate_source, dict)
            else record.get(
                "candidate_family", record.get("family", "unknown"),
            )
        )
        family_counts[str(family)] += 1
    return shard, bundle, dict(by_context), dict(
        round_shard_path=round_shard_path,
        round_shard_sha256=shard_sha256,
        teacher_buffer_path=teacher_buffer_path,
        teacher_buffer_sha256=_sha256(teacher_buffer_path),
        teacher_records=len(records),
        teacher_contexts=len(by_context),
        teacher_families=dict(sorted(family_counts.items())),
    )


def _lineages(shard):
    output = defaultdict(list)
    for window in shard.windows:
        context = shard.contexts[int(window["context_id"])]
        key = (
            int(context["scenario_id"]),
            round(float(context["gamma"]), 8),
        )
        output[key].append((int(context["step"]), context, window))
    for key, rows in output.items():
        rows.sort(key=lambda item: item[0])
        if len({step for step, _, _ in rows}) != len(rows):
            raise ValueError(f"duplicate D step in lineage {key}")
    return dict(output)


def _complete_scenarios(lineages):
    expected = {round(float(gamma), 8) for gamma in SS.GAMMAS}
    found = defaultdict(set)
    for scenario, gamma in lineages:
        found[int(scenario)].add(gamma)
    return tuple(
        scenario for scenario in sorted(found)
        if found[scenario] == expected
    )


def _next_state(state, controls):
    state = np.asarray(state, np.float32)
    action = np.asarray(controls, np.float32)[0]
    next_state = state.copy()
    next_state[:2] = (
        state[:2] + SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
    )
    next_state[2:4] = state[2:4] + SS.DT * action
    return next_state


def draw_cell(axis, rows, teachers_by_context, through_step):
    available = [row for row in rows if row[0] <= int(through_step)]
    if not available:
        available = [rows[0]]
    current_context = available[-1][1]
    BV._draw_common(axis, current_context, nominal_levels=False)
    axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#009E73", ms=8, zorder=12)

    trajectory = [np.asarray(context["state"], float)[:2]
                  for _, context, _ in available]
    last = available[-1]
    trajectory.append(
        _next_state(last[1]["state"], last[2]["controls"])[:2],
    )
    trajectory = np.asarray(trajectory)

    for _, context, window in available:
        path = SM.rollout_positions(context["state"], window["controls"])
        color = D_POS_COLOR if int(window["y"]) == 1 else D_NEG_COLOR
        axis.plot(
            path[:, 0], path[:, 1], color=color, lw=.62,
            marker=".", ms=.85, alpha=.36, zorder=3,
        )
        for teacher in teachers_by_context.get(int(context["context_id"]), ()):
            teacher_path = SM.rollout_positions(
                context["state"], teacher["controls"],
            )
            axis.plot(
                teacher_path[:, 0], teacher_path[:, 1],
                color=TEACHER_COLOR, ls="--", lw=1.35,
                marker=".", ms=1.25, alpha=.82, zorder=6,
            )

    axis.plot(
        trajectory[:, 0], trajectory[:, 1], color="#111111", lw=2.2,
        marker=".", ms=1.7, alpha=.97, zorder=10,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    return available[-1][0]


def _legend():
    return [
        Line2D([], [], color=D_POS_COLOR, lw=1.2,
               label=r"ordinary $D^+$ · exact full-H positive"),
        Line2D([], [], color=D_NEG_COLOR, lw=1.2,
               label=r"ordinary $D^-$ · exact full-H negative"),
        Line2D([], [], color="#111111", lw=2.2,
               label="ordinary executed first-action trajectory"),
        Line2D([], [], color=TEACHER_COLOR, ls="--", lw=1.8,
               label=r"$D_{\rm MPC}$ privileged dodge teacher · unverified"),
    ]


def render(
    round_shard_path, teacher_buffer_path, output_png, output_json, *,
    scenarios=None, output_mp4=None, fps=5, frame_stride=2,
):
    if int(fps) <= 0 or int(frame_stride) <= 0:
        raise ValueError("fps and frame_stride must be positive")
    shard, teacher_bundle, teachers, provenance = load_inputs(
        round_shard_path, teacher_buffer_path,
    )
    lineages = _lineages(shard)
    scenario_teacher_counts = defaultdict(int)
    for context_id, rows in teachers.items():
        scenario = int(shard.contexts[int(context_id)]["scenario_id"])
        scenario_teacher_counts[scenario] += len(rows)
    scenarios_were_explicit = scenarios is not None
    if scenarios is None:
        candidates = _complete_scenarios(lineages)
        if len(candidates) < 3:
            raise ValueError("round shard has fewer than three complete 7-gamma episodes")
        scenarios = tuple(sorted(
            candidates,
            key=lambda scenario: (
                -scenario_teacher_counts[int(scenario)], int(scenario),
            ),
        )[:3])
    scenarios = tuple(map(int, scenarios))
    if len(scenarios) != 3 or len(set(scenarios)) != 3:
        raise ValueError("renderer requires exactly three distinct scenarios")
    gammas = tuple(map(float, SS.GAMMAS))
    missing = [
        (scenario, gamma)
        for scenario in scenarios for gamma in gammas
        if (scenario, round(gamma, 8)) not in lineages
    ]
    if missing:
        raise ValueError(f"selected 3x7 grid is incomplete: {missing}")

    maximum = max(
        rows[-1][0] for key, rows in lineages.items()
        if key[0] in scenarios
    )
    frames = list(range(0, maximum + 1, int(frame_stride)))
    if frames[-1] != maximum:
        frames.append(maximum)
    figure, axes = plt.subplots(3, 7, figsize=(23.2, 10.1))
    figure.subplots_adjust(
        left=.035, right=.805, bottom=.025, top=.94,
        wspace=.025, hspace=.04,
    )
    for column, gamma in enumerate(gammas):
        figure.text(
            .035 + (.77 / 7) * (column + .5), .965,
            f"$\\gamma={gamma:g}$", ha="center", va="center", fontsize=10,
        )
    for row, scenario in enumerate(scenarios):
        figure.text(
            .012, .94 - (.915 / 3) * (row + .5),
            f"episode\n{scenario}", ha="center", va="center",
            rotation=90, fontsize=9,
        )
    figure.legend(
        handles=_legend(), loc="center left", bbox_to_anchor=(.815, .63),
        frameon=False, fontsize=8,
    )
    figure.text(
        .815, .41,
        "Purple branches are privileged MPC teacher targets.\n"
        "They are control-bounded but are not SOCP labels,\n"
        "not certificates, and never enter ordinary D+.\n\n"
        f"round: {shard.round_i}\n"
        f"ordinary D: {len(shard.D)}\n"
        f"ordinary D+: {len(shard.Dplus)}\n"
        f"ordinary D-: {len(shard.Dminus)}\n"
        f"teacher rows: {provenance['teacher_records']}\n"
        f"teacher contexts: {provenance['teacher_contexts']}\n"
        "teacher families: "
        + ", ".join(
            f"{name}={count}"
            for name, count in provenance["teacher_families"].items()
        ),
        ha="left", va="top", fontsize=8,
    )

    def update(step):
        for row, scenario in enumerate(scenarios):
            for column, gamma in enumerate(gammas):
                axis = axes[row, column]
                axis.clear()
                draw_cell(
                    axis,
                    lineages[(scenario, round(gamma, 8))],
                    teachers,
                    int(step),
                )
        return []

    for path in (output_png, output_json, output_mp4):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if output_mp4:
        movie = animation.FuncAnimation(
            figure, update, frames=frames,
            interval=1000 / int(fps), blit=False,
        )
        movie.save(
            output_mp4,
            writer=animation.FFMpegWriter(fps=int(fps), bitrate=4600),
            dpi=105,
        )
    update(maximum)
    figure.savefig(output_png, dpi=165, bbox_inches="tight")
    plt.close(figure)

    report = dict(
        status="SFM_B1_TEACHER_D_BRANCH_VIZ_COMPLETE",
        round=int(shard.round_i),
        scenarios=list(scenarios),
        gammas=list(gammas),
        ordinary_counts=dict(
            D=len(shard.D), Dplus=len(shard.Dplus), Dminus=len(shard.Dminus),
        ),
        teacher_counts=dict(
            records=provenance["teacher_records"],
            contexts=provenance["teacher_contexts"],
            families=provenance["teacher_families"],
        ),
        teacher_semantics=(
            "control-bounded privileged Codex SFM-MPC CFM targets; separate "
            "from ordinary verifier-labeled D/D+/D-; no safety label implied"
        ),
        context_match=(
            "round shard SHA-256 plus exact context_id, scenario, gamma, "
            "step, state, ped_xy, and ped_vel"
        ),
        scenario_selection=(
            "explicit CLI scenarios" if scenarios_were_explicit
            else "top three complete scenarios by teacher-row count"
        ),
        provenance=provenance,
        source_manifest=teacher_bundle.get("provenance"),
        frames=frames if output_mp4 else [maximum],
        png=os.path.abspath(output_png),
        mp4=None if not output_mp4 else os.path.abspath(output_mp4),
    )
    _write_json(output_json, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-shard", required=True)
    parser.add_argument("--teacher-buffer", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-mp4")
    parser.add_argument("--scenarios", nargs=3, type=int)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args(argv)
    render(
        args.round_shard, args.teacher_buffer,
        args.output_png, args.output_json,
        scenarios=args.scenarios, output_mp4=args.output_mp4,
        fps=args.fps, frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
