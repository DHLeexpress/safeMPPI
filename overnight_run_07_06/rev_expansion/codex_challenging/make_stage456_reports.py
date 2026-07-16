#!/usr/bin/env python3
"""Build the bounded Stage 4--6 sanity figures, summaries, and reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch

import gen_uniform_data as SEEDS
from plot_sg_demo_overlay import draw_scene
from viz_style import GAMMAS, GAMMA_CMAP, GAMMA_COLORS, GAMMA_NORM, gamma_boundaries


HERE = Path(__file__).resolve().parent
S4 = HERE / "stage_results/04_canonical"
S5 = HERE / "stage_results/05_sanity"
S6 = HERE / "stage_results/06_baselines"


def rows(directory: Path):
    return [json.loads((directory / f"row_g{float(gamma)}.json").read_text()) for gamma in GAMMAS]


def aggregate(table):
    total = sum(int(row["M"]) for row in table)
    successes = sum(int(row["n_success"]) for row in table)
    collisions = sum(float(row["CR"]) * int(row["M"]) for row in table)
    return {
        "M": total,
        "n_success": successes,
        "SR": successes / total,
        "CR": collisions / total,
        "gammas_with_success": sum(int(row["n_success"] > 0) for row in table),
    }


def load_eval_paths(directory: Path):
    result = {}
    for gamma in GAMMAS:
        with np.load(directory / f"paths_g{float(gamma)}.npz", allow_pickle=True) as saved:
            result[gamma] = [np.asarray(path, dtype=float) for path in saved["paths"]]
    return result


def gamma_colorbar(fig, ax, label=r"safety level $\gamma$"):
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=GAMMA_NORM, cmap=GAMMA_CMAP),
        ax=ax,
        boundaries=gamma_boundaries(),
        ticks=GAMMAS,
        spacing="uniform",
        fraction=0.035,
        pad=0.025,
    )
    bar.set_label(label)
    return bar


def canonical_gamma(value):
    value = float(value)
    return min(GAMMAS, key=lambda gamma: abs(gamma - value))


def mark_task(ax):
    ax.scatter(0.05, 0.05, marker="s", s=32, c="black", zorder=8)
    ax.scatter(5.0, 5.0, marker="*", s=100, c="#ffd000", edgecolors="#333", linewidths=0.5, zorder=8)
    ax.add_patch(Circle((5, 5), 0.15, fill=False, ls="--", lw=0.8, ec="#d32f2f", zorder=7))


def stage4_figure(path_data, output: Path):
    env = SEEDS.make_walled_env(8)
    table = rows(S4 / "data/pretrained_m6")
    fig = plt.figure(figsize=(13.4, 8.4))
    overview = fig.add_axes([0.08, 0.42, 0.82, 0.52])
    draw_scene(overview, env, band=False)
    for gamma in GAMMAS:
        for path in path_data[gamma]:
            overview.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma], lw=0.7, alpha=0.24)
    mark_task(overview)
    overview.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]",
                 title="Stage 4 canonical deployment — all 42 faithful pretrained rollouts")
    minis = fig.add_gridspec(1, 7, left=0.035, right=0.965, bottom=0.055, top=0.33, wspace=0.08)
    for index, gamma in enumerate(GAMMAS):
        ax = fig.add_subplot(minis[0, index])
        draw_scene(ax, env, band=False)
        candidates = path_data[gamma]
        chosen = min(candidates, key=lambda p: float(np.linalg.norm(p[-1] - np.array([5.0, 5.0]))))
        ax.plot(chosen[:, 0], chosen[:, 1], color=GAMMA_COLORS[gamma], lw=1.6)
        ax.plot(chosen[-1, 0], chosen[-1, 1], "x", color="#c62828", ms=7, mew=1.8)
        mark_task(ax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$\gamma={gamma:g}$ · SR {table[index]['SR']:.0%} · CR {table[index]['CR']:.0%}", fontsize=9)
    gamma_colorbar(fig, overview)
    fig.text(0.50, 0.01, "One closest-endpoint rollout per γ is shown below; × marks its failure endpoint.",
             ha="center", fontsize=9, color="#555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def probe_record(name: str):
    path = S5 / "runs" / name / "probe.jsonl"
    return json.loads(path.read_text().splitlines()[-1])


def history(name: str):
    return json.loads((S5 / "runs" / name / "history.json").read_text())


def stage5_figure(output: Path):
    env = SEEDS.make_walled_env(8)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    ax = axes[0, 0]
    probe_names = [
        ("strict β=.05", "probe_beta_0.05"),
        ("strict β=.10", "probe_beta_0.1"),
        ("strict β=.20", "probe_unfrozen_b02_a0"),
        ("strict β=.40", "probe_beta_0.4"),
        ("−Progress", "probe_ablate_progress"),
        ("−Curriculum", "probe_ablate_curriculum"),
        ("−SOCP", "probe_ablate_socp"),
    ]
    values, annotations = [], []
    for label, name in probe_names:
        record = probe_record(name)
        values.append(record["vr"] / max(record["att"], 1))
        annotations.append(f"{record['vr']}/{record['att']}")
    colors = ["#8c8c8c"] * 6 + ["#d55e00"]
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.72)
    for bar, text in zip(bars, annotations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, text,
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(values)), [label for label, _ in probe_names], rotation=28, ha="right")
    ax.set_ylim(0, max(0.38, max(values) + 0.08))
    ax.set_ylabel("exact-valid rollout / query")
    ax.set_title("(A) Cold-start gate: certification, not β or labeling, is blocking")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[0, 1]
    draw_scene(ax, env, band=False)
    styles = (("hope_exact_b02_a0", "--", "α=0"),
              ("hope_exact_b02_a00005", "-", "α=5×10⁻⁴"))
    for arm, linestyle, _ in styles:
        for file in sorted((S5 / "runs" / arm / "viz_db").glob("it*.pt")):
            data = torch.load(file, map_location="cpu", weights_only=False)
            for gamma, path in zip(data["rollout_gamma"], data["paths"]):
                gamma = canonical_gamma(gamma)
                ax.plot(np.asarray(path)[:, 0], np.asarray(path)[:, 1], linestyle,
                        color=GAMMA_COLORS[gamma], lw=2.0 if linestyle == "-" else 1.35,
                        alpha=0.90 if linestyle == "-" else 0.72)
    mark_task(ax)
    ax.set_title("(B) Exact-certified trajectories opened by the one-time seed")
    ax.legend(handles=[Line2D([], [], ls=ls, color="#333", lw=2, label=label)
                       for _, ls, label in styles], loc="lower right", fontsize=9)
    gamma_colorbar(fig, ax)

    ax = axes[1, 0]
    data = torch.load(S5 / "runs/hope_exact_b02_a00005/viz_db/it3.pt",
                      map_location="cpu", weights_only=False)
    sigma = np.asarray(data["sigma"], dtype=float)
    gamma = np.asarray(data["gamma"], dtype=float)
    progress = np.asarray(data["prog"], dtype=float)
    margin = np.asarray(data["margin"], dtype=float)
    sigma_norm = mpl.colors.Normalize(vmin=float(sigma.min()), vmax=float(sigma.max()))
    face = mpl.colormaps["viridis"](sigma_norm(sigma))
    edge = np.asarray([GAMMA_COLORS[canonical_gamma(g)] for g in gamma])
    ax.scatter(progress, margin, s=38, c=face, edgecolors=edge, linewidths=1.0, alpha=0.88)
    ax.set(xlabel="10-step goal progress [m]", ylabel="SOCP real-face margin [m]",
           title="(C) Iteration 3 certified pool: σ fill vs γ edge")
    ax.grid(alpha=0.2)
    sigma_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=sigma_norm, cmap="viridis"), ax=ax,
                             fraction=0.045, pad=0.02)
    sigma_bar.set_label(r"GP uncertainty $\sigma$ (viridis)")
    gamma_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=GAMMA_NORM, cmap=GAMMA_CMAP), ax=ax,
                             boundaries=gamma_boundaries(), ticks=GAMMAS, spacing="uniform",
                             fraction=0.045, pad=0.11)
    gamma_bar.set_label(r"$\gamma$ edge (plasma)")

    ax = axes[1, 1]
    for arm, color, label in (("hope_exact_b02_a0", "#0072b2", "α=0"),
                              ("hope_exact_b02_a00005", "#d55e00", "α=5×10⁻⁴")):
        records = history(arm)[1:]
        x = [row["iter"] for row in records]
        valid = [row["valid_rollouts"] for row in records]
        online_cr = [row["online_CR"] for row in records]
        ax.plot(x, valid, "o-", color=color, lw=2.1, label=f"{label}: valid rollouts")
        ax.plot(x, online_cr, "s--", color=color, lw=1.4, alpha=0.75,
                label=f"{label}: online CR")
    ax.set_xticks([1, 2, 3]); ax.set_ylim(-0.05, 1.12)
    ax.set(xlabel="exact expansion iteration", ylabel="count / fraction",
           title="(D) Tiny α preserves query support for all 3 iterations")
    ax.grid(alpha=0.22); ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.text(0.99, 0.97, "Faithful post-run deployment: 0/42 success for both",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=.25", fc="#fff4e6", ec="#d55e00", alpha=.9))

    fig.suptitle("Stage 5 bounded sanity — the first exact-support signal (not a deployment win)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_kazuki_paths():
    result = {}
    for gamma in GAMMAS:
        path = S6 / f"results/kazuki_m3/g{gamma}/paths_g{float(gamma)}.npz"
        with np.load(path, allow_pickle=True) as saved:
            result[gamma] = [np.asarray(item, dtype=float) for item in saved["paths"]]
    return result


def stage6_figure(output: Path):
    env = SEEDS.make_walled_env(8)
    expert = load_eval_paths(S6 / "results/expert_m6")
    pretrained = load_eval_paths(S4 / "data/pretrained_m6")
    kazuki = load_kazuki_paths()
    methods = [
        ("SafeMPPI expert", expert, aggregate(rows(S6 / "results/expert_m6"))),
        ("Pretrained", pretrained, aggregate(rows(S4 / "data/pretrained_m6"))),
        ("Kazuki rough winner", kazuki, aggregate(rows(S6 / "results/kazuki_m3_rows"))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.25))
    for ax, (name, paths, agg) in zip(axes, methods):
        draw_scene(ax, env, band=False)
        for gamma in GAMMAS:
            for path in paths[gamma]:
                ax.plot(path[:, 0], path[:, 1], color=GAMMA_COLORS[gamma],
                        lw=1.05 if name == "SafeMPPI expert" else 0.72,
                        alpha=0.58 if name == "SafeMPPI expert" else 0.34)
        mark_task(ax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nSR {agg['n_success']}/{agg['M']} · CR {agg['CR']:.0%}")
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=GAMMA_NORM, cmap=GAMMA_CMAP),
        ax=axes,
        boundaries=gamma_boundaries(), ticks=GAMMAS, spacing="uniform",
        orientation="horizontal", fraction=0.055, pad=0.10, aspect=45,
    )
    bar.set_label(r"trajectory condition $\gamma$ (plasma)")
    fig.suptitle("Stage 6 sanity baselines on the same 8-plug canonical scene", fontsize=15)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.22, top=0.83, wspace=0.11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stage5_arm_summary():
    definitions = [
        ("diag_frozen_b02_a0", "−SOCP diagnostic", 0.2, 0.0, "frozen"),
        ("diag_unfrozen_b02_a0", "−SOCP diagnostic", 0.2, 0.0, "unfrozen×0.3"),
        ("diag_unfrozen_b01_a0", "−SOCP diagnostic", 0.1, 0.0, "unfrozen×0.3"),
        ("hope_exact_b02_a0", "exact verifier + seed", 0.2, 0.0, "unfrozen×0.3"),
        ("hope_exact_b02_a00005", "exact verifier + seed", 0.2, 5e-4, "unfrozen×0.3"),
    ]
    summary = []
    for name, verifier, beta, alpha, encoder in definitions:
        records = history(name)[1:]
        summary.append({
            "arm": name,
            "verifier": verifier,
            "beta": beta,
            "alpha": alpha,
            "encoder": encoder,
            "valid_rollouts": int(sum(row.get("valid_rollouts", 0) for row in records)),
            "certified_windows": int(sum(row.get("n_valid", 0) for row in records)),
            "mean_encoder_grad_rms": float(np.mean([row.get("enc_grad_rms", 0.0) for row in records])),
            "final_online_SR": records[-1]["online_SR"],
            "final_online_CR": records[-1]["online_CR"],
            "faithful_SR": records[-1]["SR"],
            "faithful_CR": records[-1]["CR"],
        })
    return summary


def write_reports():
    s4_rows = rows(S4 / "data/pretrained_m6"); s4_agg = aggregate(s4_rows)
    stage4 = {
        "status": "PASS",
        "protocol": {"start": [0.05, 0.05], "goal": [5.0, 5.0], "reach": 0.15,
                     "wall_plugs": 8, "faithful": True, "M_per_gamma": 6},
        "aggregate": s4_agg,
        "per_gamma": {str(row["gamma"]): row for row in s4_rows},
        "checkpoint": str((HERE / "pretrained_sg_walls8.pt").resolve()),
    }
    (S4 / "logs/stage4_summary.json").write_text(json.dumps(stage4, indent=2, allow_nan=True) + "\n")
    s4_lines = [
        "# Stage 4 — canonical pretrained baseline", "",
        "Protocol: endpoint-free policy, start `(0.05, 0.05)`, goal `(5, 5)`, 8 plugs, reach `0.15`, faithful unguided deployment.", "",
        f"Result: **{s4_agg['n_success']}/{s4_agg['M']} successes, SR {s4_agg['SR']:.1%}, CR {s4_agg['CR']:.1%}**. "
        "The closest-looking failures are not reclassified; collision remains disqualifying.", "",
        "See `viz/canonical_pretrained_m6.png` and `data/pretrained_m6/table.md`.", "",
        "Decision: retain this as the honest pre-expansion baseline.", "",
    ]
    (S4 / "REPORT.md").write_text("\n".join(s4_lines))

    arms = stage5_arm_summary()
    exact0 = next(row for row in arms if row["arm"] == "hope_exact_b02_a0")
    exacta = next(row for row in arms if row["arm"] == "hope_exact_b02_a00005")
    seed_summary = json.loads((S5 / "runs/canonical_seed_unfrozen/summary.json").read_text())
    stage5 = {
        "status": "SANITY_COMPLETE_NO_DEPLOYMENT_WIN",
        "trainer_gates": {"corrected": "20/20", "signed_negative": "PASS"},
        "cold_start": {"strict_queries": 44, "strict_valid_rollouts": 0,
                       "interpretation": "exact SOCP certification blocks the unseeded update"},
        "one_time_seed": {"exact_expert_windows": seed_summary["windows"],
                          "persistent_anchor": False,
                          "unfrozen_beta_0.2_probe": "1 exact-valid rollout / 5 queries",
                          "frozen_or_beta_0.1_probe": "0 / 12 each"},
        "arms": arms,
        "alpha_comparison": {
            "alpha0_valid_rollouts": exact0["valid_rollouts"],
            "alpha5e-4_valid_rollouts": exacta["valid_rollouts"],
            "alpha0_final_online_CR": exact0["final_online_CR"],
            "alpha5e-4_final_online_CR": exacta["final_online_CR"],
            "faithful_postrun": "both 0/42 success, 100% collision",
        },
        "hope": "unfrozen encoder + beta=0.2 + one-time exact seed opened exact query support; tiny alpha preserved it for 3/3 pilot iterations",
        "stop_reason": "bounded sanity completed; long run not authorized and faithful deployment remains 0%",
    }
    (S5 / "logs/stage5_sanity_summary.json").write_text(json.dumps(stage5, indent=2) + "\n")
    table = [
        "| Arm | Verifier | β | α | Encoder | valid rollouts | certified windows | mean encoder gRMS | final online SR/CR | faithful SR/CR |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        table.append(
            f"| {row['arm']} | {row['verifier']} | {row['beta']:.2f} | {row['alpha']:.4g} | {row['encoder']} | "
            f"{row['valid_rollouts']} | {row['certified_windows']} | {row['mean_encoder_grad_rms']:.4f} | "
            f"{row['final_online_SR']:.1%}/{row['final_online_CR']:.1%} | {row['faithful_SR']:.1%}/{row['faithful_CR']:.1%} |"
        )
    s5_lines = [
        "# Stage 5 — bounded sanity and the first hope signal", "",
        "The unseeded exact method cannot launch: **0 exact-valid rollouts in 44 strict queries** across β 0.05, 0.10, 0.20, and 0.40. "
        "−Progress and −Curriculum also remain at zero; only −SOCP supplies data. This identifies certification/support—not optimizer choice—as the cold-start blocker.", "",
        "A diagnostic one-time seed of 819 exact-valid canonical SafeMPPI windows was then used for 25 CFM steps and discarded. "
        "With the encoder unfrozen and β=0.2, it opened 1 exact-valid rollout in 5 queries. The frozen twin and β=0.1 did not.", "",
        "From that same checkpoint, α=0 kept exact support for 2/3 iterations (2 accepted rollouts total); α=5×10⁻⁴ kept it for 3/3 (3 accepted rollouts). "
        "The signed contribution is genuinely tiny, and the comparison is only a pilot—not enough seeds for a causal claim.", "",
        *table, "",
        "Crucially, **faithful deployment is still 0/42 with CR=100%** for every post-run checkpoint. The hope is exact-query support, not a claimed performance win.", "",
        "See `viz/stage5_sanity_dashboard.png`, where trajectory γ uses plasma and uncertainty σ uses viridis.", "",
        "Decision: pause before the big dive. The candidate recipe is one-time certified seed → unfrozen encoder (0.3× LR) → β=0.2 → α=5×10⁻⁴, with an early stop if exact support or faithful CR worsens.", "",
    ]
    (S5 / "SANITY_REPORT.md").write_text("\n".join(s5_lines))

    expert_rows = rows(S6 / "results/expert_m6"); expert = aggregate(expert_rows)
    kazuki_rows = rows(S6 / "results/kazuki_m3_rows"); kazuki = aggregate(kazuki_rows)
    lucky = json.loads((S6 / "results/kazuki_best_row/row_g0.5.json").read_text())
    stage6 = {
        "status": "SANITY_COMPLETE",
        "expert": {"aggregate": expert, "per_gamma": {str(r["gamma"]): r for r in expert_rows}},
        "kazuki_reduced_sweep": {
            "winner": {"w_safe": 0.3, "coll_w": 20, "goal_w": 2, "goal_coef": 0.5,
                       "n_sample": 100, "n_elite": 5, "n_copy": 50},
            "fixed_seed_panel": kazuki,
            "sensitivity_gamma_0.5": lucky,
            "note": "reduced sanity sweep; rerun faithful N=200/copy=200 in big dive",
        },
    }
    (S6 / "logs/stage6_summary.json").write_text(json.dumps(stage6, indent=2, allow_nan=True) + "\n")
    s6_lines = [
        "# Stage 6 — expert and Kazuki sanity baselines", "",
        f"SafeMPPI: **{expert['n_success']}/{expert['M']} success, CR {expert['CR']:.1%}** across all seven γ values. "
        "Mean successful clearance is 0.22–0.27 m, so the canonical scene is feasible and certifiable.", "",
        f"Reduced Kazuki rough-sweep winner (`w_safe=.3, coll_w=20, goal_w=2, goal_coef=.5`) gets {kazuki['n_success']}/{kazuki['M']} on the fixed M=3 panel. "
        f"A separate γ=.5 sensitivity panel gets {lucky['n_success']}/{lucky['M']} success with CR {lucky['CR']:.1%}; this meets the sanity requirement of finding nonzero behavior but is unstable.", "",
        "The Kazuki sweep used `n_sample=100, n_elite=5, n_copy=50`; the full 200/10/200 baseline is deferred to the big dive.", "",
        "See `viz/stage6_rollout_comparison.png` and `data/comparison_table.md`.", "",
        "Decision: expert is the target; Kazuki is beatable but not yet measured at final fidelity.", "",
    ]
    (S6 / "REPORT.md").write_text("\n".join(s6_lines))


def main():
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold"})
    stage4_paths = load_eval_paths(S4 / "data/pretrained_m6")
    stage4_figure(stage4_paths, S4 / "viz/canonical_pretrained_m6.png")
    stage5_figure(S5 / "viz/stage5_sanity_dashboard.png")
    stage6_figure(S6 / "viz/stage6_rollout_comparison.png")
    write_reports()
    print("wrote Stage 4--6 figures, summaries, and reports")


if __name__ == "__main__":
    main()
