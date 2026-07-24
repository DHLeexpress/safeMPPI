"""Compare pretrained max-margin and native-SafeMPPI-cost data acquisition."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

import sfm_b1_branch_compare_viz as BC
import sfm_b1_d_branch_viz as DB
import sfm_b1_full_episode_viz as FV


STATUS = "SFM_B1_SELECTOR_COMPARISON_COMPLETE"
SELECTORS = ("margin", "safemppi_cost", "balanced_rank")


def _load(path, selector):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("status") != "SFM_B1_FULL_EPISODE_LABEL_AUDIT_COMPLETE":
        raise ValueError(f"not a completed branch audit: {path}")
    observed = bundle.get("protocol", {}).get("selector")
    if observed != selector:
        raise ValueError(
            f"expected selector={selector}, observed {observed} in {path}"
        )
    return bundle


def _validate_bundles(bundles):
    reference = bundles[0][1]
    for _, bundle in bundles[1:]:
        for key in (
            "scenarios", "gammas", "environment", "sample_seed", "audit_seed",
        ):
            if reference[key] != bundle[key]:
                raise ValueError(
                    f"selector comparison contract differs at {key}"
                )
        if (
            reference.get("checkpoint_sha256")
            != bundle.get("checkpoint_sha256")
        ):
            raise ValueError(
                "selector comparison requires one pretrained checkpoint"
            )
    if len(reference["scenarios"]) != 3 or len(reference["gammas"]) != 7:
        raise ValueError("selector comparison requires 3 episodes x 7 gammas")


def _layout(bundles):
    scenarios = tuple(map(int, bundles[0][1]["scenarios"]))
    gammas = tuple(map(float, bundles[0][1]["gammas"]))
    rows = len(bundles) * len(scenarios)
    figure, axes = plt.subplots(rows, 7, figsize=(23.5, 3.0 * rows))
    figure.subplots_adjust(
        left=.055, right=.82, bottom=.025, top=.96, wspace=.025, hspace=.04,
    )
    for column, gamma in enumerate(gammas):
        figure.text(
            .055 + (.765 / 7) * (column + .5), .975,
            f"$\\gamma={gamma:g}$", ha="center", va="center", fontsize=10,
        )
    indices = {}
    for selector_index, (label, bundle) in enumerate(bundles):
        index = FV._index(bundle["traces"])
        indices[label] = index
        for scenario_index, scenario in enumerate(scenarios):
            row = selector_index * len(scenarios) + scenario_index
            figure.text(
                .018, .96 - (.935 / rows) * (row + .5),
                f"{label}\nepisode {scenario}",
                ha="center", va="center", rotation=90, fontsize=8,
            )
    return figure, axes, scenarios, gammas, bundles, indices


def _draw(axes, scenarios, gammas, bundles, indices, step):
    for selector_index, (label, _) in enumerate(bundles):
        index = indices[label]
        for scenario_index, scenario in enumerate(scenarios):
            row = selector_index * len(scenarios) + scenario_index
            for column, gamma in enumerate(gammas):
                axis = axes[row, column]
                axis.clear()
                DB.draw_cell(
                    axis, index[(scenario, round(gamma, 8))], int(step),
                    branch_line_scale=2.7,
                    trajectory_linewidth=1.05,
                    trajectory_marker_size=1.25,
                    candidate_inset=True,
                )


def render(
        margin_trace, cost_trace, output_png, output_mp4, output_json,
        *, balanced_trace=None, fps=5, frame_stride=2,
):
    margin = _load(margin_trace, "margin")
    cost = _load(cost_trace, "safemppi_cost")
    bundles = [
        ("max one-step margin", margin),
        ("SafeMPPI cost", cost),
    ]
    if balanced_trace is not None:
        bundles.append((
            "balanced safety + performance rank",
            _load(balanced_trace, "balanced_rank"),
        ))
    _validate_bundles(bundles)
    (
        figure, axes, scenarios, gammas, bundles, indices,
    ) = _layout(tuple(bundles))

    reports = {label: BC.summarize(bundle) for label, bundle in bundles}
    maximum = max(
        max(max(rows) for rows in index.values())
        for index in indices.values()
    )
    frames = list(range(0, maximum + 1, int(frame_stride)))
    if frames[-1] != maximum:
        frames.append(maximum)

    figure.legend(
        handles=DB._legend(), loc="center left", bbox_to_anchor=(.835, .72),
        frameon=False, fontsize=8,
    )
    summary = []
    for label, report in reports.items():
        summary.extend([
            label,
            f"positive D: {report['executed_positive']}/{report['contexts']}",
            f"S/C/T: {report['success']}/"
            f"{report['collision']}/{report['timeout']}",
            "",
        ])
    summary.extend([
        "Same pretrained checkpoint, episodes,",
        "gammas, proposal-noise contract.",
        "Only the admissible-B execution ranking differs.",
        "Branches: planned H10 D samples.",
        "Thin black: executed first-action path.",
    ])
    figure.text(
        .835, .47, "\n".join(summary), ha="left", va="top", fontsize=8,
    )

    for path in (output_png, output_mp4, output_json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _draw(axes, scenarios, gammas, bundles, indices, maximum)
    figure.savefig(output_png, dpi=165, bbox_inches="tight")

    def update(step):
        _draw(axes, scenarios, gammas, bundles, indices, int(step))
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(
        output_mp4, writer=animation.FFMpegWriter(
            fps=int(fps), bitrate=5200,
        ), dpi=105,
    )
    plt.close(figure)

    report = {
        "status": STATUS,
        "margin_trace": os.path.abspath(margin_trace),
        "safemppi_cost_trace": os.path.abspath(cost_trace),
        "balanced_rank_trace": (
            None if balanced_trace is None
            else os.path.abspath(balanced_trace)
        ),
        "checkpoint_sha256": margin.get("checkpoint_sha256"),
        "scenarios": list(scenarios),
        "gammas": list(gammas),
        "comparison": reports,
        "controlled_difference": (
            "rank the same SOCP-positive and nominal-Hp-admissible B queries "
            "by max one-step Hp margin, minimum frozen native SafeMPPI cost, "
            "or the sum of ordinal safety/performance ranks with safety-first "
            "tie-breaking"
        ),
        "frames": frames,
        "png": os.path.abspath(output_png),
        "mp4": os.path.abspath(output_mp4),
    }
    temporary = os.path.abspath(output_json) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.replace(temporary, os.path.abspath(output_json))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-trace", required=True)
    parser.add_argument("--safemppi-cost-trace", required=True)
    parser.add_argument("--balanced-rank-trace")
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args(argv)
    render(
        args.margin_trace, args.safemppi_cost_trace,
        args.output_png, args.output_mp4, args.output_json,
        balanced_trace=args.balanced_rank_trace,
        fps=args.fps, frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
