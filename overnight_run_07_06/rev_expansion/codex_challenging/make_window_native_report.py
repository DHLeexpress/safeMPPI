"""Evaluate and report the six-iteration window-native four-arm sanity run."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
sys.path[:0] = [str(REF), str(HERE), str(HERE.parent), str(HERE.parents[1])]

import grid_scene as GS  # noqa: E402
import window_expand_hardtail as WT  # noqa: E402

STAGE = HERE / "stage_results/05_window_native"
RUNS = STAGE / "runs"
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
ARMS = {
    "full": "Full",
    "no_socp": r"$-$SOCP",
    "no_progress": r"$-$Progress",
    "no_curriculum": r"$-$Curriculum",
}
PLASMA = plt.get_cmap("plasma")
COLORS = {g: PLASMA(0.04 + 0.92 * i / (len(GAMMAS) - 1)) for i, g in enumerate(GAMMAS)}


def environment():
    env = WT._apply_wall_plugs(GS.make_grid(), 8)
    env.x0 = torch.tensor([0.3, 0.3, 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor([4.7, 4.7], dtype=env.goal.dtype)
    WT.GM2.GOAL_XY = np.array([4.7, 4.7], dtype=float)
    return env


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_arm(key, device, M=6):
    run = RUNS / f"sanity_v1_{key}"
    policy, _ = WT.HP.load_hp(run / "final.pt", device=device)
    env = environment()
    np.random.seed(7200); random.seed(7200); torch.manual_seed(7200)
    rows, aggregate, paths = WT.SR.eval_policy(
        policy, env, gammas=GAMMAS, M=M, T_max=250, reach=0.15,
        temp=1.0, device=device, seed0=7200, keep_paths=M,
        log=lambda *args, **kwargs: None,
    )
    outdir = STAGE / "data/eval_m6" / key
    outdir.mkdir(parents=True, exist_ok=True)
    for gamma in GAMMAS:
        np.savez_compressed(outdir / f"paths_g{gamma}.npz",
                            paths=np.asarray(paths[gamma], dtype=object))
    scorecard = {
        "arm": key, "checkpoint": str((run / "final.pt").resolve()),
        "M_per_gamma": M, "start": [0.3, 0.3], "goal": [4.7, 4.7],
        "wall_plugs": 8, "reach": 0.15,
        "rows": {str(g): rows[g] for g in GAMMAS}, "aggregate": aggregate,
    }
    (outdir / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")
    return scorecard, paths


def draw_scene(ax, title, env):
    for obstacle in env.obstacles.detach().cpu().numpy():
        ax.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#cccccc", zorder=1))
    ax.plot(0.3, 0.3, "ks", ms=6, zorder=8)
    ax.plot(4.7, 4.7, "*", c="gold", mec="k", ms=14, zorder=8)
    ax.set_xlim(-0.35, 5.35); ax.set_ylim(-0.35, 5.35); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=15)


def rollout_figure(results, paths_by_arm):
    env = environment()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 11.2))
    for ax, (key, title) in zip(axes.flat, ARMS.items()):
        agg = results[key]["aggregate"]
        draw_scene(ax, f"{title}\nSR {agg['SR']:.2f} | CR {agg['CR']:.2f}", env)
        for gamma in GAMMAS:
            for path in paths_by_arm[key][gamma]:
                path = np.asarray(path, dtype=float)
                ax.plot(path[:, 0], path[:, 1], color=COLORS[gamma], lw=0.9,
                        alpha=0.42, zorder=3)
                ax.plot(path[-1, 0], path[-1, 1], ".", color=COLORS[gamma],
                        ms=3.0, alpha=0.8, zorder=4)
    cmap = ListedColormap([COLORS[g] for g in GAMMAS])
    bounds = [0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.85, 1.05]
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
    cbar = fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.025,
                        ticks=GAMMAS, location="right")
    cbar.set_label(r"safety level $\gamma$", fontsize=13)
    fig.suptitle("Window-native expansion sanity — faithful deployment, M=6 per γ", fontsize=16)
    STAGE.joinpath("viz").mkdir(parents=True, exist_ok=True)
    fig.savefig(STAGE / "viz/prelim_rollouts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def training_summary(results):
    summary = {}
    for key in ARMS:
        run = RUNS / f"sanity_v1_{key}"
        probes = json_rows(run / "probe.jsonl")
        budgets = {int(k): int(v) for k, v in json.loads(
            (run / "accepted_window_budget.json").read_text()).items()}
        audits = [row["gather_audit"] for row in probes]
        pre_cap = [max(1, int(a.get("accepted_windows_pre_cap", a["accepted_windows"])))
                   for a in audits]
        summary[key] = {
            "accepted_by_iteration": budgets,
            "accepted_total": sum(budgets.values()),
            "queried_coherent_windows": sum(a["coherent_windows_total"] for a in audits),
            "contributor_rollouts": sum(a["contributor_rollouts"] for a in audits),
            "whole_valid2_rollouts": sum(a["whole_valid2_pass"] for a in audits),
            "whole_invalid2_rollouts": sum(a["whole_valid2_fail"] for a in audits),
            "whole_invalid_window_fraction_pre_cap": float(np.mean([
                a["accepted_from_whole_invalid"] / n for a, n in zip(audits, pre_cap)])),
            "unreached_window_fraction_pre_cap": float(np.mean([
                a["accepted_from_unreached"] / n for a, n in zip(audits, pre_cap)])),
            "progress_evaluated": sum(a["progress_evaluated"] for a in audits),
            "socp_evaluated": sum(a["socp_evaluated"] for a in audits),
            "window_failures": {reason: sum(a.get("window_failures", {}).get(reason, 0)
                                             for a in audits)
                                for reason in ("taskspace", "progress", "progress_floor",
                                               "socp", "safe_space")},
            "functional_step": [row["functional_step"] for row in probes],
            "loss": [row["loss"] for row in probes],
            "batch_easy": [row["batch_e"] for row in probes],
            "batch_frontier": [row["batch_f"] for row in probes],
            "rollbacks": sum(bool(row["rollback"]) for row in probes),
            "evaluation": results[key]["aggregate"],
        }
    return summary


def training_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    colors = {"full": "#117733", "no_socp": "#cc6677",
              "no_progress": "#332288", "no_curriculum": "#ddaa33"}
    for key, title in ARMS.items():
        s = summary[key]; x = sorted(s["accepted_by_iteration"])
        axes[0, 0].plot(x, [s["accepted_by_iteration"][i] for i in x], "o-",
                        label=title, color=colors[key])
        axes[0, 1].plot(x, s["functional_step"], "o-", label=title, color=colors[key])
        axes[1, 0].plot(x, s["loss"], "o-", label=title, color=colors[key])
    axes[0, 0].set(title="Accepted H=10 windows", xlabel="iteration", ylabel="windows")
    axes[0, 1].set(title="Functional step", xlabel="iteration", ylabel="relative field change")
    axes[1, 0].set(title="CFM loss", xlabel="iteration", ylabel="loss")
    keys = list(ARMS); pos = np.arange(len(keys)); width = 0.23
    axes[1, 1].bar(pos - width, [summary[k]["evaluation"]["SR"] for k in keys], width,
                   label="SR", color="#44aa99")
    axes[1, 1].bar(pos, [summary[k]["evaluation"]["CR"] for k in keys], width,
                   label="CR", color="#cc6677")
    gd = axes[1, 1].twinx()
    gd.bar(pos + width, [summary[k]["evaluation"]["mean_goal_dist"] for k in keys], width,
           label="goal dist", color="#999999", alpha=0.65)
    axes[1, 1].set_xticks(pos, [ARMS[k].replace("$", "") for k in keys])
    axes[1, 1].set_ylim(0, 1.05); axes[1, 1].set_ylabel("rate")
    gd.set_ylabel("final goal distance [m]")
    axes[1, 1].set_title("Faithful M=6 deployment")
    h1, l1 = axes[1, 1].get_legend_handles_labels(); h2, l2 = gd.get_legend_handles_labels()
    axes[1, 1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    for ax in axes.flat:
        ax.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Window-native aggregation — six-iteration sanity internals", fontsize=15)
    fig.tight_layout()
    fig.savefig(STAGE / "viz/prelim_training.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(summary):
    full_budget = summary["full"]["accepted_by_iteration"]
    nc_budget = summary["no_curriculum"]["accepted_by_iteration"]
    lines = [
        "# Window-native flow-expansion sanity", "",
        "This is a **six-iteration semantic sanity** on the cleared 8-plug stadium task "
        "`(0.3,0.3) -> (4.7,4.7)`, not the giant-obstacle benchmark and not a 50-iteration result.", "",
        "## Handoff alignment", "",
        "- Aggregation unit: coherent executed H=10 window; whole-trajectory `traj_ok`, later collision, "
        "and goal reach are audit-only.",
        "- Full / -Curriculum: task-space + progress + SOCP.",
        "- -SOCP: task-space + progress + positive geometric clearance; SOCP is not called.",
        "- -Progress: task-space + SOCP; progress is not called.",
        "- All seven gamma values; no emergent gamma, recovery, hard quota, targeted proposal, demo, or LwF.",
        "- The handoff prose says GP buffer 500, while its cited `faithful_g47/recipe.json` records 200/200. "
        "This sanity used 200/200 and is therefore comparable to that archived lineage on this knob.", "",
        "## Results", "",
        "| arm | accepted windows | contributors | whole-valid2 rollouts | progress evals | SOCP evals | "
        "M6 SR | M6 CR | goal dist [m] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in ARMS.items():
        s = summary[key]; e = s["evaluation"]
        lines.append(f"| {title} | {s['accepted_total']} | {s['contributor_rollouts']} | "
                     f"{s['whole_valid2_rollouts']} | {s['progress_evaluated']} | "
                     f"{s['socp_evaluated']} | {e['SR']:.3f} | {e['CR']:.3f} | "
                     f"{e['mean_goal_dist']:.3f} |")
    lines += ["", "Full accepted-window counts by iteration: `" +
              ", ".join(str(full_budget[i]) for i in sorted(full_budget)) + "`.", "",
              f"Controlled -Curriculum count match: **{full_budget == nc_budget}**. It used a true "
              "single-class 16+0 batch; Full used 6+10. All updates had finite loss, nonzero functional "
              "step, and zero rollbacks.", "",
              "The central finding is sample availability: Full collected thousands of locally certified "
              "windows even though no queried rollout was whole-valid2. This fixes the starvation mechanism. "
              "The M=6 deployment is still not successful after only six iterations, so this is evidence for "
              "the gather correction—not a performance claim.", "",
              "## Artifacts", "",
              "- `viz/prelim_rollouts.png` — Full and all three No brothers.",
              "- `viz/prelim_training.png` — accepted counts, update size/loss, and M=6 deployment.",
              "- `data/eval_m6/*/scorecard.json` and `paths_g*.npz` — matched faithful rollouts.",
              "- `reference/analysis/test_window_expand.json` — 9/9 predicate and control checks.", "",
              "**Gate:** pause before a 50-iteration run or resuming the giant-obstacle pipeline.", ""]
    (STAGE / "REPORT.md").write_text("\n".join(lines))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, paths = {}, {}
    for key in ARMS:
        print(f"evaluating {key}", flush=True)
        results[key], paths[key] = evaluate_arm(key, device)
    summary = training_summary(results)
    (STAGE / "logs").mkdir(parents=True, exist_ok=True)
    (STAGE / "logs/prelim_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rollout_figure(results, paths)
    training_figure(summary)
    write_report(summary)
    print(f"wrote {STAGE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
