#!/usr/bin/env python3
"""Exact v4-style rollouts, internals, and scatter for the giant benchmark."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from giant_obstacle_ood.stage1_geometry_sweep import GIANT_CENTER, make_scene  # noqa: E402
from giant_obstacle_ood.stage1b_smooth_expert import GOAL, RADIUS, START, route_mode  # noqa: E402
from giant_obstacle_ood.stage4_frozen_ood import load_records, summarize_method  # noqa: E402
from viz_style import GAMMAS  # noqa: E402


STAGE5 = HERE / "stage_results/05_window_expand"
RUN = STAGE5 / "runs/from_target3_it5_preserve/full"
RUNS = STAGE5 / "runs/from_target3_it5_preserve"
EVAL = STAGE5 / "evaluation"
OUT = HERE / "stage_results/06_exact_reports"
VIZ = OUT / "viz"
BASE = HERE / "stage_results/04_frozen_ood/data"
ID_PATHS = HERE / "stage_results/02b_balanced_id/data/balanced_id_paths_all_gamma.npz"
PRETRAINED = STAGE5 / "temperature_probe/data/pretrained_temp_0.5_m6.npz"
EXPERT = BASE / "expert_m6.npz"
MIZUTA = BASE / "mizuta_selected_m6.npz"
SELECTED_OURS_T05 = (
    STAGE5 / "gates/from_target3_it5_preserve/it0010/rollouts_temp0.5_m20.npz"
)
SELECTED_ABLATIONS_T05 = {
    arm: STAGE5 / (
        f"evaluation_target3/{arm}/temp0.5/"
        "rollouts_temp0.5_m20_nfe8_T300.npz"
    )
    for arm in ("no_socp", "no_progress", "no_curriculum")
}
GSEL = (0.1, 0.5, 1.0)
PLASMA = LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(np.linspace(0.04, 0.88, 256))
)
NORM = Normalize(vmin=0.1, vmax=1.0)
GAMMA_COLOR = {float(gamma): PLASMA(NORM(float(gamma))) for gamma in GAMMAS}


def eval_npz(arm: str, temperature: str = "0.5") -> Path:
    if arm == "full" and str(temperature) == "0.5":
        if not SELECTED_OURS_T05.exists():
            raise FileNotFoundError(f"selected rollout gate is missing: {SELECTED_OURS_T05}")
        return SELECTED_OURS_T05
    if arm in SELECTED_ABLATIONS_T05 and str(temperature) == "0.5":
        selected = SELECTED_ABLATIONS_T05[arm]
        if not selected.exists():
            raise FileNotFoundError(f"selected ablation rollout is missing: {selected}")
        return selected
    candidates = sorted((EVAL / arm / f"temp{temperature}").glob("rollouts_m*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no evaluation records for {arm} temp={temperature}")
    return candidates[-1]


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw_scene(axis, title: str, *, giant: bool, bold: bool = False) -> None:
    env = make_scene(RADIUS if giant else None, START, GOAL)
    axis.set_facecolor("#f8f7f4")
    rr = float(env.r_robot)
    for obstacle in env.obstacles.detach().cpu().numpy():
        is_giant = giant and np.linalg.norm(obstacle[:2] - GIANT_CENTER) < 1e-6
        axis.add_patch(plt.Circle(
            obstacle[:2], obstacle[2] + rr,
            color="#686868" if is_giant else "#cccccc",
            ec="#b2182b" if is_giant else "none",
            lw=1.5 if is_giant else 0.0, zorder=1,
        ))
    axis.plot(START[0], START[1], "ks", ms=5.5, zorder=8)
    axis.plot(GOAL[0], GOAL[1], "*", c="gold", mec="k", ms=13, zorder=8)
    axis.set_xlim(-0.42, 5.42); axis.set_ylim(-0.42, 5.42); axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_title(title, pad=6, fontsize=18, fontweight="bold" if bold else "normal")


def pick_records(records: list[dict], gamma: float, total: int = 3) -> list[dict]:
    subset = [record for record in records if np.isclose(record["gamma"], gamma)]
    good = [record for record in subset if record["success"]]
    bad = [record for record in subset if not record["success"] and len(record["path"]) > 5]
    chosen = good[:min(2, total)] + bad[:max(0, total - min(2, len(good)))]
    return (chosen or subset[:total])[:total]


def draw_record(axis, record: dict, *, dots: bool = True, lw: float = 1.4) -> None:
    path = np.asarray(record["path"], dtype=float)
    gamma = float(record["gamma"])
    if len(path) < 2:
        return
    axis.plot(path[:, 0], path[:, 1], c=GAMMA_COLOR[gamma], lw=lw, alpha=.92, zorder=3)
    if dots:
        axis.plot(path[::3, 0], path[::3, 1], ".", c="k", ms=1.6, alpha=.5, zorder=4)
    if not record["success"]:
        axis.plot(path[-1, 0], path[-1, 1], "x", c="#cc3311", ms=8, mew=2.2, zorder=7)


def add_zoom(axis, records: list[dict], box: tuple[float, float, float, float], loc: str) -> None:
    zoom = inset_axes(axis, width="42%", height="42%", loc=loc, borderpad=.4)
    env = make_scene(RADIUS, START, GOAL); rr = float(env.r_robot)
    for obstacle in env.obstacles.detach().cpu().numpy():
        is_giant = np.linalg.norm(obstacle[:2] - GIANT_CENTER) < 1e-6
        zoom.add_patch(plt.Circle(
            obstacle[:2], obstacle[2] + rr,
            color="#686868" if is_giant else "#cccccc", zorder=1,
        ))
    for record in records:
        draw_record(zoom, record, lw=1.5)
    zoom.set_xlim(box[0], box[1]); zoom.set_ylim(box[2], box[3]); zoom.set_aspect("equal")
    zoom.set_xticks([]); zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_color("#cc3311"); spine.set_linewidth(1.5)
    from matplotlib.patches import Rectangle
    axis.add_patch(Rectangle(
        (box[0], box[2]), box[1] - box[0], box[3] - box[2],
        fill=False, ec="#cc3311", lw=1.3, zorder=7,
    ))


def draw_method(axis, title: str, records: list[dict], *, bold: bool = False,
                zoom_box: tuple[float, float, float, float] | None = None) -> None:
    draw_scene(axis, title, giant=True, bold=bold)
    chosen = []
    for gamma in GSEL:
        group = pick_records(records, gamma)
        chosen.extend(group)
        for record in group:
            draw_record(axis, record)
    failures = [record for record in chosen if not record["success"]]
    if zoom_box is not None and failures:
        add_zoom(axis, failures[:4], zoom_box, "lower right")


def mirror_signature(word: str) -> str:
    return word.translate(str.maketrans({"R": "U", "U": "R"}))


def balanced_id_panel_records(archive) -> tuple[list[dict], dict]:
    """Return every real path from each displayed, exactly balanced stratum.

    The Stage-2B archive is sorted by reflection rank.  Taking a prefix therefore
    selects only one side of each R/U pair.  The paper panel must use complete
    strata, not an ordering-dependent preview.
    """
    gammas = np.asarray(archive["gammas"], dtype=float)
    signatures = np.asarray(archive["signatures"]).astype(str)
    pair_ranks = np.asarray(archive["pair_ranks"], dtype=int)
    seeds = np.asarray(archive["seeds"], dtype=np.int64)
    paths = archive["paths"]
    selected: list[dict] = []
    audit: dict[str, dict] = {}
    for gamma in GSEL:
        indices = np.flatnonzero(np.isclose(gammas, gamma))
        counts = Counter(signatures[indices].tolist())
        residuals = {
            word: int(count - counts.get(mirror_signature(word), 0))
            for word, count in sorted(counts.items())
        }
        if len(indices) != 24 or any(residuals.values()):
            raise RuntimeError(
                f"ID panel gamma={gamma:g} is not the approved 24-path mirror-balanced "
                f"stratum: n={len(indices)}, residuals={residuals}"
            )
        for index in indices:
            selected.append({
                "gamma": float(gamma),
                "seed": int(seeds[index]),
                "signature": str(signatures[index]),
                "pair_rank": int(pair_ranks[index]),
                "path": np.asarray(paths[index], dtype=float),
            })
        audit[f"{gamma:g}"] = {
            "paths": int(len(indices)),
            "signature_counts": dict(sorted(counts.items())),
            "mirror_count_residuals": residuals,
            "max_abs_mirror_count_residual": int(max(abs(value) for value in residuals.values())),
            "seeds": [int(seed) for seed in seeds[indices]],
        }
    return selected, audit


def rollout_figure() -> None:
    matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 12.5})
    methods = {
        "Expert": load_records(EXPERT),
        "Pretrained": load_records(PRETRAINED),
        r"CFM-MPPI$^{*}$": load_records(MIZUTA),
        "NO safety validity check": load_records(eval_npz("no_socp")),
        "NO progress check": load_records(eval_npz("no_progress")),
        "NO curriculum": load_records(eval_npz("no_curriculum")),
        "Ours": load_records(eval_npz("full")),
    }
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 10.0))
    data_axis = axes[0, 0]
    draw_scene(data_axis, "Pre-trained data", giant=False)
    with np.load(ID_PATHS, allow_pickle=True) as archive:
        panel_records, balance_audit = balanced_id_panel_records(archive)
        for record in panel_records:
            path = record["path"]
            gamma = record["gamma"]
            data_axis.plot(path[:, 0], path[:, 1], c=GAMMA_COLOR[gamma],
                           lw=.82, alpha=.24, zorder=3)
        # The distribution is fixed-pair; show both seed types explicitly.
        data_axis.plot(START[0], START[1], "o", c="k", ms=4.0, label="start seed", zorder=9)
        data_axis.plot(GOAL[0], GOAL[1], "*", c="gold", mec="k", ms=8.0,
                       label="goal seed", zorder=9)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs/pretrained_panel_balance.json").write_text(json.dumps({
        "status": "PASS",
        "source": str(ID_PATHS.resolve()),
        "selection": "all approved Stage-2B paths for each displayed gamma",
        "displayed_gammas": list(GSEL),
        "total_paths": int(len(panel_records)),
        "per_gamma": balance_audit,
    }, indent=2) + "\n")
    data_axis.legend(loc="lower left", fontsize=8, frameon=False, handletextpad=.3)

    draw_method(axes[0, 1], "Expert", methods["Expert"])
    draw_method(axes[0, 2], "Pretrained", methods["Pretrained"],
                zoom_box=(1.0, 4.0, 1.0, 4.0))
    draw_method(axes[0, 3], r"CFM-MPPI$^{*}$", methods[r"CFM-MPPI$^{*}$"],
                zoom_box=(.25, 2.5, .25, 2.5))
    draw_method(axes[1, 0], "NO safety validity check", methods["NO safety validity check"])
    draw_method(axes[1, 1], "NO progress check", methods["NO progress check"])
    draw_method(axes[1, 2], "NO curriculum", methods["NO curriculum"])
    draw_method(axes[1, 3], "Ours", methods["Ours"], bold=True)

    colors = [GAMMA_COLOR[gamma] for gamma in GSEL]
    cmap = ListedColormap(colors)
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm([0, 1, 2, 3], 3))
    cbar = fig.colorbar(scalar, ax=axes, location="right", fraction=.022, pad=.02,
                        ticks=[.5, 1.5, 2.5])
    cbar.ax.set_yticklabels(["0.1", "0.5", "1.0"])
    cbar.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(VIZ / f"rollouts_v4.{ext}", dpi=135, bbox_inches="tight")
    plt.close(fig)


def temperature_figure() -> None:
    """Matched visual record of the requested 0.1 / 0.5 / 1.0 test."""
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 9.3))
    for column, temperature in enumerate((.1, .5, 1.0)):
        ttag = f"{temperature:g}"
        pretrained_path = STAGE5 / f"temperature_probe/data/pretrained_temp_{ttag}_m6.npz"
        ours_path = eval_npz("full", ttag)
        for row, (label, path) in enumerate((("Pretrained", pretrained_path), ("Ours", ours_path))):
            records = load_records(path); summary = summarize_method(records)["overall"]
            title = (f"{label}, T={temperature:g}\nSR {summary['a_SR']:.2f} | "
                     f"CR {summary['b_CR']:.2f} | Δu {summary['mean_control_delta']:.2f}")
            draw_scene(axes[row, column], title, giant=True, bold=label == "Ours")
            axes[row, column].set_title(
                title, fontsize=13.5, pad=5,
                fontweight="bold" if label == "Ours" else "normal",
            )
            for gamma in GSEL:
                for record in pick_records(records, gamma, total=1):
                    draw_record(axes[row, column], record, dots=False, lw=1.55)
    colors = [GAMMA_COLOR[gamma] for gamma in GSEL]
    cmap = ListedColormap(colors)
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm([0, 1, 2, 3], 3))
    fig.subplots_adjust(hspace=.28, wspace=.18, right=.88)
    cbar_axis = fig.add_axes([.905, .20, .022, .60])
    cbar = fig.colorbar(scalar, cax=cbar_axis, ticks=[.5, 1.5, 2.5])
    cbar.ax.set_yticklabels(["0.1", "0.5", "1.0"])
    cbar.set_label(r"safety level $\gamma$")
    fig.suptitle("Matched deployment-temperature diagnostic", fontsize=16)
    for ext in ("png", "pdf"):
        fig.savefig(VIZ / f"temperature_sweep.{ext}", dpi=135, bbox_inches="tight")
    plt.close(fig)


def smooth(values, width: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values
    width = max(1, min(width, len(values)))
    return np.convolve(values, np.ones(width) / width, mode="same")


def forward_fill(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy(); last = np.nan
    for index, value in enumerate(result):
        if np.isfinite(value):
            last = value
        elif np.isfinite(last):
            result[index] = last
    return result


def internals_figure() -> None:
    matplotlib.rcParams.update({"font.size": 11.5, "axes.titlesize": 13})
    rows = json_lines(RUN / "probe.jsonl")
    history_rows = json.loads((RUN / "history.json").read_text())
    history = sorted(history_rows, key=lambda row: int(row["iter"]))
    iterations = np.asarray([row["iter"] for row in rows], dtype=float)

    def trace(key: str, default=np.nan) -> np.ndarray:
        return np.asarray([row.get(key) if row.get(key) is not None else default for row in rows], dtype=float)

    def held_history(key: str) -> np.ndarray:
        output = []
        for iteration in iterations:
            eligible = [row for row in history if int(row["iter"]) <= int(iteration)]
            output.append(eligible[-1].get(key, np.nan) if eligible else np.nan)
        return np.asarray(output, dtype=float)

    gathered_modes = set(); coverage = []
    for iteration in iterations.astype(int):
        db_path = RUN / f"viz_db/it{iteration}.pt"
        if db_path.exists():
            db = torch.load(db_path, map_location="cpu", weights_only=False)
            for path in db.get("paths", []):
                mode, _ = route_mode(np.asarray(path, dtype=float))
                if mode in ("upper-left", "lower-right"):
                    gathered_modes.add(mode)
        coverage.append(len(gathered_modes))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2))
    a = axes[0, 0]
    a.plot(iterations, held_history("SR"), "-o", c="#009944", lw=2, ms=4, label="SR (M2 probe)")
    a.plot(iterations, held_history("CR"), "-o", c="#cc3311", lw=2, ms=4, label="CR")
    a2 = a.twinx(); a2.plot(iterations, coverage, "--s", c="#4477aa", lw=1.4, ms=3,
                             label="gathered detour modes")
    a2.set_ylabel("gathered prefix-side modes", color="#4477aa"); a2.set_ylim(-.05, 2.1)
    a.set_ylim(-.02, 1.02); a.set_title("(A) probe SR / CR / gathered-prefix diversity")
    a.legend(loc="center right", fontsize=9); a.set_xlabel("iteration"); a.grid(alpha=.3)

    b = axes[0, 1]
    for index, gamma in enumerate(GAMMAS):
        shares = []
        for row in rows:
            counts = {round(float(key), 2): value for key, value in (row.get("gamma_counts") or {}).items()}
            total = max(1, sum(counts.values()))
            shares.append(100 * counts.get(round(float(gamma), 2), 0) / total)
        b.plot(iterations, smooth(shares), c=GAMMA_COLOR[float(gamma)], lw=2, label=f"γ{gamma:g}")
    b.set_title("(B) γ composition of accepted valid2 windows")
    b.set_xlabel("iteration"); b.set_ylabel("% of accepted windows (3-it smooth)")
    b.legend(fontsize=8, ncol=2); b.grid(alpha=.3)

    c = axes[0, 2]
    for key, style, color, label in (
        ("n_easy", "-", "#00b300", "pool easy"),
        ("n_frontier", "-", "#d62728", "pool frontier"),
        ("batch_e", "--", "#00b300", "batch e"),
        ("batch_f", "--", "#d62728", "batch f"),
        ("batch_d", "--", "#7f7f7f", "batch demo"),
    ):
        c.plot(iterations, np.maximum(trace(key, 0), .5), style, c=color,
               lw=1.8 if style == "-" else 1.2, label=label)
    c.set_yscale("log"); c.set_title("(C) pools (solid) vs batch (dashed)")
    c.set_xlabel("iteration"); c.legend(fontsize=8, ncol=2); c.grid(alpha=.3)

    d = axes[1, 0]
    d.plot(iterations, trace("functional_step"), c="#0072B2", lw=1.8, label="functional step")
    d.axhline(.025, c="k", ls=":", lw=1.1, label="rollback ceiling")
    d2 = d.twinx(); d2.plot(iterations, trace("loss"), c="#D55E00", lw=1.2, alpha=.7)
    d2.set_ylabel("FM loss", color="#D55E00"); d.set_title("(D) update magnitude / loss")
    d.set_xlabel("iteration"); d.grid(alpha=.3); d.legend(loc="upper right", fontsize=8)

    e = axes[1, 1]
    accepted_rate = []
    for row in rows:
        audit = row["gather_audit"]
        numerator = audit.get("accepted_windows_pre_cap", audit.get("accepted_windows", 0))
        denominator = max(1, audit.get("coherent_windows_total", 0))
        accepted_rate.append(100 * numerator / denominator)
    e.plot(iterations, smooth(accepted_rate), c="#009988", lw=1.8)
    e.set_title("(E) local valid2-window acceptance (3-it smooth)")
    e.set_xlabel("iteration"); e.set_ylabel("accepted coherent windows [%]")
    e.set_ylim(0, 100); e.grid(alpha=.3)

    f = axes[1, 2]
    f.plot(iterations, forward_fill(trace("sig_e")), c="#440154", lw=1.8, label=r"easy $\sigma$")
    f.plot(iterations, forward_fill(trace("sig_f")), c="#35b779", lw=1.8, label=r"frontier $\sigma$")
    f.plot(iterations, trace("sigma_plane"), ":", c="k", lw=1.4, label=r"$\sigma_q$ plane")
    f.set_title(r"(F) novelty $\sigma$ by class"); f.set_xlabel("iteration")
    f.legend(fontsize=9); f.grid(alpha=.3)
    for axis in axes.flat:
        axis.axvline(11, c="#7f7f7f", ls="--", lw=.9, alpha=.65)
    fig.suptitle("Safe Flow Expansion — training internals (giant OOD, temp 0.5, it0–20)", fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(VIZ / f"internals_v4.{ext}", dpi=135, bbox_inches="tight")
    plt.close(fig)


def summary_from(path: Path) -> dict:
    return summarize_method(load_records(path))


def scatter_figure() -> None:
    matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
    series = [
        ("Expert", summary_from(EXPERT), "o", 95, {}),
        ("Our approach", summary_from(eval_npz("full")), "*", 240,
         {"edgecolors": "k", "linewidths": .9}),
        ("Pretrained", summary_from(PRETRAINED), "s", 70, {"alpha": .55}),
        (r"CFM-MPPI$^{*}$", summary_from(MIZUTA), "v", 95, {}),
        ("NO safety validity check", summary_from(eval_npz("no_socp")), "X", 85, {}),
        ("NO progress check", summary_from(eval_npz("no_progress")), "P", 85, {}),
        ("NO curriculum", summary_from(eval_npz("no_curriculum")), "D", 70, {}),
    ]
    fig, (reliability, quality) = plt.subplots(1, 2, figsize=(16.2, 5.4))
    for zorder, (name, summary, marker, size, kwargs) in enumerate(series, start=3):
        for gamma in GAMMAS:
            row = summary["per_gamma"][str(float(gamma))]
            color = [GAMMA_COLOR[float(gamma)]]
            reliability.scatter(100 * row["a_SR"], 100 * row["b_CR"], c=color,
                                marker=marker, s=size, zorder=zorder, **kwargs)
            clearance = row.get("min_clearance_mean_success")
            time_s = row.get("d_time_s_mean_success")
            if clearance is not None and time_s is not None and np.isfinite(clearance) and np.isfinite(time_s):
                quality.scatter(time_s, clearance, c=color, marker=marker,
                                s=size, zorder=zorder, **kwargs)
    reliability.set_xlabel("success rate SR [%]"); reliability.set_ylabel("collision rate CR [%]")
    reliability.set_xlim(-5, 105); reliability.set_ylim(-3, 105); reliability.grid(alpha=.3)
    quality.set_xlabel("time to goal [s]"); quality.set_ylabel("min clearance (successes) [m]")
    quality.grid(alpha=.3)

    def label(name: str) -> str:
        return r"$\mathbf{Our\ approach}$" if name == "Our approach" else name

    handles = [Line2D([], [], c="#666666", marker=marker, ls="",
                      ms=11 if marker == "*" else 8, label=label(name))
               for name, _, marker, _, _ in series]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(.5, 1.025), fontsize=10.5)
    scalar = plt.cm.ScalarMappable(cmap=PLASMA, norm=NORM); scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=[reliability, quality], location="right",
                        fraction=.025, pad=.015, ticks=GAMMAS)
    cbar.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(VIZ / f"scatter_v4.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def report() -> None:
    health = json.loads((STAGE5 / "logs/health_audit.json").read_text())
    modes = json.loads((EVAL / "route_mode_audit.json").read_text())
    temperature = json.loads((EVAL / "temperature_sweep_metrics.json").read_text())
    ours = summary_from(eval_npz("full"))["overall"]
    lines = [
        "# Giant-obstacle bounded expansion report", "",
        "The promoted run uses temperature 0.5, unfrozen encoder (0.3× LR), beta 0.2, "
        "window-native H=10 validity, and an actual OOD expert demo schedule of 50% through "
        "iteration 10 then 25%.", "",
        f"- Ours: SR {ours['a_SR']:.3f}, CR {ours['b_CR']:.3f}, "
        f"coverage {ours['e_coverage']}, mean boundary arc {ours['mean_boundary_arc_rad']:.3f} rad.",
        f"- Learning health: {health['status']}; route-mode audit: {modes['status']}.",
        f"- Temperature sweep SR (0.1/0.5/1.0): " + "/".join(
            f"{temperature[str(value)]['overall']['a_SR']:.3f}" for value in (0.1, .5, 1.0)
        ) + ".", "",
        "## Requested artifacts", "",
        "- `viz/rollouts_v4.png` — pretraining data, Expert, Pretrained, CFM-MPPI*, all three No brothers, Ours.",
        "- `viz/internals_v4.png` — exact 2×3 training-internals grammar.",
        "- `viz/scatter_v4.png` — gamma-colored reliability and successful-trajectory quality planes.",
        "- `viz/curriculum_it20.mp4` — exact curriculum grammar, one second per iteration.", "",
        "No long-run claim is made from this bounded Stage-5 sanity.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    rollout_figure()
    temperature_figure()
    internals_figure()
    scatter_figure()
    report()
    print(f"wrote exact reports under {OUT}")


if __name__ == "__main__":
    main()
