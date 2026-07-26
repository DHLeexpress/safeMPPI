"""Compare two deterministic controller overlays on the same Hp10 SFM scene.

This module is diagnostic-only.  It neither edits the flow checkpoint nor
feeds controller trajectories into B1 replay.

``claude_v2_exact_socp`` is the dodge-then-cruise finite family from Claude's
offline augmentation, interpreted here as a pure receding-horizon controller.
It executes only a canonical full-H10 SOCP-certified plan and terminates NVP
when none exists.

``codex_privileged_sfm`` is the historical high-success deployment wrapper.
It augments the learned proposal with deterministic templates and simulates
the live reactive SFM crowd.  It is not an SOCP certificate and, when the
requested hard margin is unavailable, it still executes its best recoverable
candidate.  The two branch colors therefore have different semantics and are
kept explicitly separate in the output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch

import grid_policy_sfm as GPS
import sfm_claude_recovery_controller as CRC
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_scene as SS


STATUS = "SFM_ADHOC_CONTROLLER_COMPARISON_COMPLETE"
DEFAULT_SCENARIOS = (250_001, 250_003, 250_007)
DEFAULT_GAMMAS = (0.1, 0.5, 1.0)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def privileged_sfm_config():
    """The historical v3 wrapper recipe, without OOD retuning."""
    gammas = tuple(map(float, SS.GAMMAS))
    return KZ.KazukiConfig(
        safe_coefs=(0.3,),
        goal_coef=0.5,
        n_sample=200,
        n_elite=10,
        n_copy=200,
        exact_sfm_step_filter=True,
        step_filter_margin=0.22,
        step_filter_horizon=10,
        step_filter_goal_plans=12,
        step_filter_avoid_plans=18,
        step_filter_always_select=True,
        step_filter_min_progress=0.05,
        step_filter_goal_score_weight=1.0,
        step_filter_clearance_weight=0.05,
        step_filter_escape_patience=5,
        step_filter_escape_burst=3,
        step_filter_viability_lookahead=20,
        step_filter_viability_band=0.05,
        step_filter_viability_escalate=True,
        step_filter_viability_escalation_band=1.0,
        step_filter_viability_escalation_min_progress=2.0,
        step_filter_viability_escalation_entry_progress=5.0,
        step_filter_viability_escalation_burst=40,
        step_filter_stagnation_gamma_max=0.1,
        step_filter_stagnation_window=20,
        step_filter_stagnation_progress=0.1,
        step_filter_stagnation_horizon=20,
        step_filter_stagnation_burst=4,
        controller_gammas=gammas,
        safe_coef_by_gamma=(1.0, 0.3, 1.0, 0.3, 0.3, 0.3, 0.1),
        goal_coef_by_gamma=(2.0, 0.5, 2.0, 0.5, 0.5, 0.5, 3.0),
        step_filter_margin_by_gamma=(0.24, 0.22, 0.24, 0.22, 0.22, 0.22, 0.22),
        step_filter_goal_score_weight_by_gamma=(2.0, 1.0, 2.0, 1.0, 1.0, 6.0, 2.0),
        step_filter_clearance_weight_by_gamma=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0),
        step_filter_clearance_target_weight_by_gamma=(0.0,) * len(gammas),
    ).validate()


def run_comparison(
    checkpoint,
    *,
    scenarios=DEFAULT_SCENARIOS,
    gammas=DEFAULT_GAMMAS,
    scene_profile="double_density_velocity_ood",
    device="cuda",
    verifier_workers=32,
    T=180,
    reach=0.5,
    sample_seed=700_000,
):
    """Run both controllers from identical initial SFM scenarios."""
    scenarios = tuple(map(int, scenarios))
    gammas = tuple(map(float, gammas))
    if not scenarios or not gammas:
        raise ValueError("at least one scenario and gamma are required")
    environment = SS.scene_profile(scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    config = privileged_sfm_config()

    claude = []
    with ProcessPoolExecutor(max_workers=int(verifier_workers)) as executor:
        for scenario in scenarios:
            for gamma in gammas:
                claude.append(CRC.rollout_recovery_controller(
                    scenario,
                    gamma,
                    family="v2",
                    scene_profile=scene_profile,
                    T=int(T),
                    reach=float(reach),
                    executor=executor,
                ))

    codex = []
    for scenario in scenarios:
        for gamma in gammas:
            result = KZ.kazuki_sfm_deploy(
                policy,
                scenario,
                gamma,
                cfg=config,
                n_ped=int(environment["n_ped"]),
                T=int(T),
                reach=float(reach),
                device=device,
                ped_speed_range=tuple(environment["ped_speed_range"]),
                sample_seed=int(sample_seed),
                collect_diagnostics=True,
            )
            result["controller"] = "codex_privileged_sfm"
            result["scenario_id"] = int(scenario)
            result["status"] = (
                "success" if result["success"]
                else "collision" if result["collision"]
                else "timeout"
            )
            result["nvp"] = False
            result["timeout"] = result["status"] == "timeout"
            result["minimum_clearance"] = float(result["min_clear"])
            result["time_to_goal"] = (
                float(result["steps"]) * SS.DT if result["success"] else None
            )
            codex.append(result)

    return dict(
        status=STATUS,
        diagnostic_only=True,
        changes_checkpoint=False,
        enters_replay=False,
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
        scene=environment,
        scenarios=list(scenarios),
        gammas=list(gammas),
        T=int(T),
        reach=float(reach),
        sample_seed=int(sample_seed),
        controllers=dict(
            claude_v2_exact_socp=dict(
                semantics=CRC.controller_manifest(),
                rollouts=claude,
            ),
            codex_privileged_sfm=dict(
                semantics=dict(
                    controller=(
                        "learned flow/guidance/MPPI proposal plus deterministic "
                        "templates and privileged reactive-SFM look-ahead"
                    ),
                    certificate=False,
                    no_hard_margin_candidate=(
                        "executes best recoverable/inside/clearance candidate"
                    ),
                    original_evaluation_environment=dict(
                        n_ped=20, ped_speed_range=[1.0, 1.5],
                    ),
                    current_environment=environment,
                    config=config.to_dict(),
                ),
                rollouts=codex,
            ),
        ),
    )


def _summary(rollouts):
    count = len(rollouts)
    statuses = {
        key: sum(row["status"] == key for row in rollouts)
        for key in ("success", "collision", "nvp", "timeout")
    }
    successful_time = [
        float(row["time_to_goal"])
        for row in rollouts if row.get("time_to_goal") is not None
    ]
    return dict(
        episodes=count,
        **statuses,
        success_rate=float(statuses["success"] / count),
        collision_rate=float(statuses["collision"] / count),
        nvp_rate=float(statuses["nvp"] / count),
        timeout_rate=float(statuses["timeout"] / count),
        mean_minimum_clearance=float(np.mean([
            row["minimum_clearance"] for row in rollouts
        ])),
        mean_successful_time_to_goal=(
            float(np.mean(successful_time)) if successful_time else None
        ),
    )


def _closest_step(trace):
    """Declared snapshot: minimum current robot-to-pedestrian clearance."""
    if not trace:
        return None
    values = []
    for index, row in enumerate(trace):
        state = np.asarray(row["state"], float)
        ped_xy = np.asarray(row["ped_xy"], float)
        clearance = (
            float(np.linalg.norm(ped_xy - state[:2], axis=1).min() - SS.R_PED)
            if len(ped_xy) else float("inf")
        )
        values.append((clearance, index))
    return min(values)[1]


def _claude_branches(row):
    branches = []
    selected = row["pool"].get("best")
    selected_rank = None if selected is None else int(selected["query_rank"])
    for candidate in row["pool"]["queried"]:
        branches.append(dict(
            controls=np.asarray(candidate["controls"], np.float32),
            feasible=bool(candidate["certified"]),
            selected=(
                selected_rank is not None
                and int(candidate["query_rank"]) == selected_rank
            ),
        ))
    return branches


def _codex_branches(row):
    diagnostics = row.get("output_filter") or {}
    return [
        dict(
            controls=np.asarray(candidate["controls"], np.float32),
            feasible=bool(candidate["hard_margin_feasible"]),
            selected=bool(candidate["selected"]),
        )
        for candidate in diagnostics.get("candidate_pool", ())
    ]


def _draw_cell(axis, rollout, kind):
    path = np.asarray(rollout["path"], float)
    trace = list(rollout.get("trace") or ())
    snapshot = _closest_step(trace)
    if snapshot is None:
        axis.plot(path[:, 0], path[:, 1], color="#111111", lw=1.6)
        return
    row = trace[snapshot]
    state = np.asarray(row["state"], float)
    ped_xy = np.asarray(row["ped_xy"], float)
    ped_vel = np.asarray(row["ped_vel"], float)
    branches = (
        _claude_branches(row) if kind == "claude"
        else _codex_branches(row)
    )

    for branch in branches:
        segment = SM.rollout_positions(state, branch["controls"])
        if branch["selected"]:
            axis.plot(
                segment[:, 0], segment[:, 1],
                color="#0868d9", lw=2.6, alpha=1.0, zorder=8,
            )
        else:
            axis.plot(
                segment[:, 0], segment[:, 1],
                color="#159447" if branch["feasible"] else "#d62728",
                lw=0.65, alpha=0.38, zorder=4,
            )
    axis.plot(
        path[:, 0], path[:, 1], color="#111111", lw=1.35,
        marker=".", ms=1.2, zorder=7,
    )
    axis.plot(
        state[0], state[1], marker="o", ms=3.2,
        color="#111111", zorder=9,
    )
    for xy, velocity in zip(ped_xy, ped_vel):
        axis.add_patch(Circle(
            xy, SS.R_PED, facecolor="#b7b7b7",
            edgecolor="#666666", lw=0.3, alpha=0.85, zorder=5,
        ))
        future = xy + 10 * SS.DT * velocity
        axis.plot(
            [xy[0], future[0]], [xy[1], future[1]],
            color="#999999", lw=0.4, ls="--", alpha=0.55, zorder=3,
        )

    axis.plot(
        SS.GOAL[0], SS.GOAL[1], marker="*", ms=8,
        color="#ffd92f", mec="#111111", zorder=10,
    )
    axis.set_xlim(SS.TASK_LO, SS.TASK_HI)
    axis.set_ylim(SS.TASK_LO, SS.TASK_HI)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0.02, 0.02,
        f"{rollout['status']} · snapshot t={snapshot}",
        transform=axis.transAxes, fontsize=6.5,
        ha="left", va="bottom",
    )


def render(result, output_png):
    scenarios = tuple(map(int, result["scenarios"]))
    gammas = tuple(map(float, result["gammas"]))
    controllers = (
        ("Claude v2 · exact SOCP", "claude_v2_exact_socp", "claude"),
        ("Codex · privileged SFM", "codex_privileged_sfm", "codex"),
    )
    figure, axes = plt.subplots(
        len(controllers) * len(scenarios),
        len(gammas),
        figsize=(3.0 * len(gammas) + 3.3, 2.75 * len(controllers) * len(scenarios)),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.08, right=0.78, bottom=0.03, top=0.95,
        wspace=0.04, hspace=0.05,
    )
    lookup = {}
    reports = {}
    for label, key, kind in controllers:
        rollouts = result["controllers"][key]["rollouts"]
        reports[key] = _summary(rollouts)
        lookup[key] = {
            (int(row["scenario_id"]), round(float(row["gamma"]), 8)): row
            for row in rollouts
        }
    for column, gamma in enumerate(gammas):
        figure.text(
            0.08 + 0.70 / len(gammas) * (column + 0.5),
            0.975, f"$\\gamma={gamma:g}$",
            ha="center", va="center", fontsize=10,
        )
    for controller_index, (label, key, kind) in enumerate(controllers):
        for scenario_index, scenario in enumerate(scenarios):
            row_index = controller_index * len(scenarios) + scenario_index
            figure.text(
                0.035,
                0.95 - 0.92 / (len(controllers) * len(scenarios))
                * (row_index + 0.5),
                f"{label}\nepisode {scenario}",
                rotation=90, ha="center", va="center", fontsize=8,
            )
            for column, gamma in enumerate(gammas):
                _draw_cell(
                    axes[row_index, column],
                    lookup[key][(scenario, round(gamma, 8))],
                    kind,
                )
    legend = [
        Line2D([], [], color="#111111", lw=1.5, label="executed trajectory"),
        Line2D([], [], color="#0868d9", lw=2.6, label="selected H-step branch"),
        Line2D([], [], color="#159447", lw=1.0, label="passes controller gate"),
        Line2D([], [], color="#d62728", lw=1.0, label="rejected by controller gate"),
    ]
    figure.legend(
        handles=legend, loc="center left", bbox_to_anchor=(0.80, 0.76),
        frameon=False, fontsize=8,
    )
    claude = reports["claude_v2_exact_socp"]
    codex = reports["codex_privileged_sfm"]
    figure.text(
        0.80, 0.60,
        "Same initial OOD scenes\n"
        f"Claude S/C/NVP/T: {claude['success']}/{claude['collision']}/"
        f"{claude['nvp']}/{claude['timeout']}\n"
        f"Codex S/C/NVP/T: {codex['success']}/{codex['collision']}/"
        f"{codex['nvp']}/{codex['timeout']}\n\n"
        "Green has different semantics:\n"
        "Claude = exact full-H10 SOCP positive\n"
        "Codex = privileged SFM-lookahead hard-margin feasible\n\n"
        "Snapshot rule: minimum current\n"
        "robot–pedestrian clearance.\n"
        "No checkpoint update or replay.",
        ha="left", va="top", fontsize=8,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    figure.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return reports


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def deliver(result, *, outdir):
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse output directory: {outdir}")
    os.makedirs(outdir)
    trace_path = os.path.join(outdir, "controller_traces.pt")
    torch.save(result, trace_path)
    figure_path = os.path.join(outdir, "controller_branch_comparison.png")
    reports = render(result, figure_path)
    payload = {
        key: value for key, value in result.items() if key != "controllers"
    }
    payload["controllers"] = {
        key: {
            "semantics": row["semantics"],
            "metrics": reports[key],
        }
        for key, row in result["controllers"].items()
    }
    payload["artifacts"] = dict(
        trace=os.path.abspath(trace_path),
        figure=os.path.abspath(figure_path),
    )
    payload = _jsonable(payload)
    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path + ".tmp", "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(metrics_path + ".tmp", metrics_path)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    parser.add_argument(
        "--scene-profile", default="double_density_velocity_ood",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verifier-workers", type=int, default=32)
    parser.add_argument("--T", type=int, default=180)
    parser.add_argument("--reach", type=float, default=0.5)
    parser.add_argument("--sample-seed", type=int, default=700_000)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    result = run_comparison(
        args.checkpoint,
        scenarios=args.scenarios,
        gammas=args.gammas,
        scene_profile=args.scene_profile,
        device=args.device,
        verifier_workers=args.verifier_workers,
        T=args.T,
        reach=args.reach,
        sample_seed=args.sample_seed,
    )
    payload = deliver(result, outdir=args.outdir)
    print(json.dumps({
        "status": payload["status"],
        "metrics": {
            key: row["metrics"]
            for key, row in payload["controllers"].items()
        },
        "artifacts": payload["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
