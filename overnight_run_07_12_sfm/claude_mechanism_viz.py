"""Mechanism visualizations for the SFM recipe study (diagnostic-only).

``snapshot``: for one stored hard context (from an archived ExecutedRoundShard)
render, per checkpoint, the K=16 flow candidates regenerated with the EXACT
keyed gathering latents and labeled by the exact full-H10 verifier; overlay the
deterministic certified recovery escapes (family v1 and, when available, v2);
and show the frozen visual-encoder input (Hp10 polar stack) plus low5/history
conditioning for that context.

``episode``: closed-loop replay of one (scenario, gamma) gathering lineage
with a given checkpoint (margin-selector B1 semantics, round-1 acquisition
state), recording the trajectory, per-step K-positive counts and NVP flags;
``render-episodes`` overlays two replays (e.g. r0 vs a treated checkpoint).

These figures are explanatory evidence only; claims use fixed-bank metrics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import claude_offline_aug as AUG
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_offline_exec as OE
import sfm_b1_offline_store as OS
import sfm_b1_rbf as BR
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS

SEED = 20260724


def _find_context(shard, scenario, gamma, step):
    for context in shard.contexts:
        if (
            int(context["scenario_id"]) == int(scenario)
            and round(float(context["gamma"]), 8) == round(float(gamma), 8)
            and int(context["step"]) == int(step)
        ):
            return context
    raise KeyError(f"context not stored: s{scenario} g{gamma} step{step}")


def _executed_window(shard, context):
    for window in shard.windows:
        if int(window["context_id"]) == int(context["context_id"]):
            return window
    return None


@torch.no_grad()
def _k_candidates(policy, context, device):
    hp10 = torch.as_tensor(context["hp10"], device=device)[None]
    low = torch.as_tensor(context["low5"], device=device)[None]
    hist = torch.as_tensor(context["hist"], device=device)[None]
    ctx = policy.ctx_from(hp10.float(), low.float(), hist.float())
    generator = np.random.default_rng(OE._keyed_seed(
        SEED, 1, int(context["scenario_id"]),
        f"{float(context['gamma']):.8f}", int(context["step"]), "K",
    ))
    x0 = generator.standard_normal((16, int(policy.d)), dtype=np.float32)
    windows = BE.integrate_latents(
        policy,
        torch.as_tensor(x0, device=device),
        ctx.repeat_interleave(16, dim=0),
        nfe=8,
    ).reshape(16, 10, 2).cpu().numpy()
    return windows


def _verify_many(context, control_sets, workers=8):
    tasks = [
        (index, 0, context["state"], controls, context["ped_xy"],
         context["ped_vel"], context["gamma"])
        for index, controls in enumerate(control_sets)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(SM.verify_in_worker, tasks))
    ordered = [None] * len(control_sets)
    for index, _, result in results:
        ordered[index] = result
    return ordered


def _scene(axis, context, title):
    state = np.asarray(context["state"], np.float32)
    ped_xy = np.asarray(context["ped_xy"], np.float32)
    ped_vel = np.asarray(context["ped_vel"], np.float32)
    prediction = SM.predict_pedestrians(ped_xy, ped_vel, H=10)
    for j in range(len(ped_xy)):
        axis.add_patch(plt.Circle(
            ped_xy[j], SS.R_PED, color="#c2554f", alpha=.75, lw=0, zorder=3,
        ))
        axis.plot(
            prediction[:, j, 0], prediction[:, j, 1],
            color="#c2554f", lw=.7, ls=":", alpha=.55, zorder=2,
        )
    axis.plot(*SS.GOAL, marker="*", ms=17, color="#e6b422", mec="k",
              zorder=6)
    axis.plot(state[0], state[1], marker="o", ms=9, color="#1450a3",
              mec="k", zorder=6)
    axis.annotate(
        "", xy=state[:2] + 0.5 * state[2:4], xytext=state[:2],
        arrowprops=dict(arrowstyle="->", color="#1450a3", lw=2), zorder=6,
    )
    axis.add_patch(plt.Rectangle(
        (SS.TASK_LO, SS.TASK_LO), SS.TASK_HI - SS.TASK_LO,
        SS.TASK_HI - SS.TASK_LO, fill=False, ec="k", lw=.8, alpha=.6,
    ))
    axis.add_patch(plt.Circle(
        state[:2], SS.R_SENSE, fill=False, ec="#1450a3", lw=.6, ls="--",
        alpha=.5,
    ))
    pad = 2.35
    axis.set_xlim(state[0] - pad, state[0] + pad)
    axis.set_ylim(state[1] - pad, state[1] + pad)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=11)
    axis.grid(alpha=.2)


def _draw_windows(axis, context, windows, labels, *, executed=None):
    n_pos = 0
    for controls, result in zip(windows, labels):
        segment = SM.rollout_positions(context["state"], controls)
        positive = bool(result.get("resolved")) and int(result.get("y", 0)) == 1
        n_pos += int(positive)
        axis.plot(
            segment[:, 0], segment[:, 1],
            color="#1f8a4c" if positive else "#b22222",
            lw=1.7 if positive else 0.9,
            alpha=.95 if positive else .5,
            zorder=5 if positive else 4,
        )
    if executed is not None:
        segment = SM.rollout_positions(context["state"], executed)
        axis.plot(segment[:, 0], segment[:, 1], color="#550000", lw=3.2,
                  alpha=.95, zorder=5.5, label="executed (uncertified)")
    return n_pos


def snapshot(args):
    shard = OS.ExecutedRoundShard.load(args.shard)
    context = _find_context(shard, args.scenario, args.gamma, args.step)
    executed = _executed_window(shard, context)
    specs = [spec.split("=", 1) for spec in args.checkpoints]
    families = dict(v1="v1", v2="v2") if hasattr(AUG, "recovery_candidates_v2") \
        else dict(v1="v1")
    n_ckpt = len(specs)
    n_cols = n_ckpt + len(families)
    figure, axes = plt.subplots(
        2, max(n_cols, 3), figsize=(4.9 * max(n_cols, 3), 9.6),
    )
    summary = dict(
        scenario=int(args.scenario), gamma=float(args.gamma),
        step=int(args.step),
    )

    for column, (name, path) in enumerate(specs):
        policy, _ = GPS.load_sfm_policy(path, device=args.device)
        policy.eval()
        windows = _k_candidates(policy, context, args.device)
        results = _verify_many(context, list(windows), args.workers)
        axis = axes[0][column]
        _scene(axis, context, "")
        n_pos = _draw_windows(
            axis, context, windows, results,
            executed=None if executed is None or column else
            np.asarray(executed["controls"], np.float32),
        )
        axis.set_title(
            f"{name}: K=16 flow candidates\n"
            f"exact-verifier positives: {n_pos}/16", fontsize=11,
        )
        summary[f"K_positive_{name}"] = int(n_pos)
        del policy

    for offset, family in enumerate(sorted(families)):
        candidates = (
            AUG.recovery_candidates(context["state"]) if family == "v1"
            else AUG.recovery_candidates_v2(
                context["state"], context["ped_xy"], context["ped_vel"],
            )
        )
        scored = []
        for controls, provenance in candidates:
            objective = AUG._prefilter(context, controls)
            if objective is not None:
                scored.append((objective, controls, provenance))
        scored.sort(key=lambda row: (row[0], str(row[2])))
        pool = scored[:AUG.PREVERIFY_CAP]
        results = _verify_many(context, [row[1] for row in pool], args.workers)
        axis = axes[0][n_ckpt + offset]
        _scene(axis, context, "")
        certified = 0
        best_drawn = False
        for (objective, controls, provenance), result in zip(pool, results):
            segment = SM.rollout_positions(context["state"], controls)
            ok = bool(result.get("resolved")) and int(result.get("y", 0)) == 1
            if ok:
                certified += 1
                axis.plot(
                    segment[:, 0], segment[:, 1],
                    color="#0b6fa4" if family == "v1" else "#e07b00",
                    lw=3.0 if not best_drawn else 1.6,
                    alpha=.95 if not best_drawn else .7, zorder=5.4,
                )
                if not best_drawn:
                    summary[f"recovery_{family}_best"] = dict(
                        J=float(objective), generator=provenance,
                        slack=float(result["diagnostics"]["slack"]),
                        end_speed=float(np.linalg.norm(
                            BC.rollout_states(
                                context["state"], controls[None],
                            )[0, -1, 2:4].numpy()
                        )),
                    )
                best_drawn = True
            else:
                axis.plot(segment[:, 0], segment[:, 1], color="#888888",
                          lw=.7, alpha=.4, zorder=3.5)
        axis.set_title(
            f"certified deterministic recovery ({family})\n"
            f"{certified}/{len(pool)} exact-certified", fontsize=11,
        )
        summary[f"recovery_{family}_certified"] = int(certified)
        summary[f"recovery_{family}_pool"] = int(len(pool))

    hp10 = np.asarray(context["hp10"], np.float32)
    axis = axes[1][0]
    image = axis.imshow(
        hp10[-1].T, origin="lower", aspect="auto", cmap="RdBu",
        vmin=-1, vmax=1,
        extent=(-180, 180, 0, SS.R_SENSE),
    )
    axis.set_title("frozen encoder input: newest H_P frame\n"
                   "(clipped nominal polytope, polar)", fontsize=11)
    axis.set_xlabel("bearing [deg]")
    axis.set_ylabel("range [m]")
    plt.colorbar(image, ax=axis, fraction=.04)
    for column in range(1, min(3, axes.shape[1])):
        axis = axes[1][column]
        if column == 1:
            mosaic = np.concatenate([hp10[i].T for i in range(10)], axis=1)
            axis.imshow(
                mosaic, origin="lower", aspect="auto", cmap="RdBu",
                vmin=-1, vmax=1,
            )
            axis.set_title("Hp10 stack: 10 most recent H_P frames "
                           "(oldest left)", fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
        elif column == 2:
            axis.axis("off")
            low = np.asarray(context["low5"], np.float32)
            lines = [
                f"scenario {args.scenario}  gamma {args.gamma}  "
                f"step {args.step}",
                f"low5: relgoal=({low[0]:.2f},{low[1]:.2f}) "
                f"v=({low[2]:.2f},{low[3]:.2f}) gamma={low[4]:.2f}",
                "encoder enc_grid: FROZEN during expansion (SHA-checked)",
                "gradient flows into trunk/GRU/enc_low conditioned on",
                "these frozen grid features",
            ]
            if executed is not None:
                lines.append(
                    f"executed window: y={executed['y']} "
                    f"source={executed['execution_source']}"
                )
            for key in sorted(summary):
                if key.startswith(("K_positive", "recovery")):
                    value = summary[key]
                    if isinstance(value, dict):
                        value = {
                            k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in value.items() if k != "generator"
                        }
                    lines.append(f"{key}: {value}")
            axis.text(0.01, 0.98, "\n".join(str(l) for l in lines),
                      va="top", ha="left", fontsize=9, family="monospace",
                      transform=axis.transAxes, wrap=True)
    for column in range(n_cols, axes.shape[1]):
        axes[0][column].axis("off")
    for column in range(3, axes.shape[1]):
        axes[1][column].axis("off")
    figure.suptitle(
        f"Hard-context mechanism: s{args.scenario} γ={args.gamma} "
        f"step {args.step} (exact verifier everywhere)", fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(figure)
    OE._write_json(args.out + ".json", summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if not isinstance(v, dict)}, indent=1))


@torch.no_grad()
def episode(args):
    device = args.device
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=device)
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)
    cfg = OE.OfflineConfig(alpha=0.0, exposure_epochs=1, rounds=1, smoke=True)
    environment = SS.scene_profile(cfg.scene_profile)
    replica = BX.Replica(
        int(args.scenario), float(args.gamma),
        n_ped=environment["n_ped"],
        ped_speed_range=tuple(environment["ped_speed_range"]),
    )
    gp = BR.RBFGP(float(args.ell), float(cfg.gp_lam))
    beta, _ = OE._calibrate_beta(
        phi_policy, gp, [replica], cfg, device, round_i=1,
    )
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for step in range(int(cfg.T)):
            live, batch = BX._stack_prepared([replica], device)
            if not live:
                break
            windows, contexts, x0 = OE._keyed_windows(
                policy, live, batch, K=cfg.K, round_i=1, step=step,
                source="K", seed=cfg.seed, nfe=cfg.nfe, temp=cfg.temp,
            )
            raw_windows, _, _ = OE._keyed_windows(
                policy, live, batch, K=1, round_i=1, step=step,
                source="raw_continuation", seed=cfg.seed, nfe=cfg.nfe,
                temp=cfg.temp,
            )
            windows_np = windows[0].cpu().numpy()
            raw_np = raw_windows[0, 0].cpu().numpy()
            features = OE._features_from_x0(
                phi_policy, windows, contexts, x0, cfg.phi_s,
            )
            generator = torch.Generator(device=features.device)
            generator.manual_seed(OE._keyed_seed(
                cfg.seed, 1, replica.scenario_id,
                f"{replica.gamma:.8f}", step, "acquisition",
            ))
            selected, trace = gp.sequential_acquire(
                features[0], cfg.B, beta, generator=generator,
            )
            prepared = replica.prepared
            results = _verify_many(
                dict(state=prepared["state"], ped_xy=prepared["ped_xy"],
                     ped_vel=prepared["ped_vel"], gamma=replica.gamma),
                list(windows_np), args.workers,
            )
            query_rows = [
                dict(candidate_id=int(k), acquisition_step=j,
                     controls=windows_np[k], result=results[k], mode=None,
                     sigma=float(trace[j]["chosen_sigma"]))
                for j, k in enumerate(map(int, selected))
                if results[k].get("resolved")
            ]
            chosen = BC.select_admissible(
                query_rows, selector="margin", state=prepared["state"],
                ped_xy=prepared["ped_xy"], ped_vel=prepared["ped_vel"],
                gamma=replica.gamma,
            )
            controls = raw_np if chosen is None else np.asarray(
                chosen["controls"], np.float32,
            )
            frames.append(dict(
                step=int(step),
                state=prepared["state"].tolist(),
                ped_xy=prepared["ped_xy"].tolist(),
                K_positive=int(sum(
                    int(r.get("y", 0)) == 1 for r in results
                    if r.get("resolved")
                )),
                NVP=chosen is None,
            ))
            BX._advance(replica, controls[0])
            FA._post_action_terminal(replica)
    OE._finalize_alive([replica])
    payload = dict(
        checkpoint=os.path.abspath(args.checkpoint),
        scenario=int(args.scenario), gamma=float(args.gamma),
        status=replica.status, steps=len(replica.controls),
        min_clearance=float(replica.minimum_clearance),
        states=[s.tolist() for s in replica.states],
        frames=frames,
    )
    OE._write_json(args.out, payload)
    print(json.dumps(dict(status=replica.status,
                          steps=len(replica.controls)), indent=1))


def render_episodes(args):
    runs = []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        with open(path) as stream:
            runs.append((name, json.load(stream)))
    n = len(runs)
    figure, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.4))
    if n == 1:
        axes = [axes]
    for axis, (name, run) in zip(axes, runs):
        states = np.asarray(run["states"], np.float32)
        nvp_steps = {f["step"] for f in run["frames"] if f["NVP"]}
        final = run["frames"][-1]
        ped = np.asarray(final["ped_xy"], np.float32)
        for j in range(len(ped)):
            axis.add_patch(plt.Circle(
                ped[j], SS.R_PED, color="#c2554f", alpha=.5, lw=0,
            ))
        for t in range(len(states) - 1):
            color = "#d95f02" if t in nvp_steps else "#1450a3"
            axis.plot(states[t:t + 2, 0], states[t:t + 2, 1], color=color,
                      lw=2.6 if t in nvp_steps else 1.8, zorder=5)
        axis.plot(*SS.GOAL, marker="*", ms=17, color="#e6b422", mec="k")
        axis.plot(states[0, 0], states[0, 1], marker="s", ms=8,
                  color="#1450a3", mec="k")
        marker = dict(collision="X", success="*", timeout="P")[run["status"]]
        axis.plot(states[-1, 0], states[-1, 1], marker=marker, ms=14,
                  color={"collision": "#b22222", "success": "#1f8a4c",
                         "timeout": "#888888"}[run["status"]], mec="k",
                  zorder=7)
        axis.add_patch(plt.Rectangle(
            (SS.TASK_LO, SS.TASK_LO), SS.TASK_HI - SS.TASK_LO,
            SS.TASK_HI - SS.TASK_LO, fill=False, ec="k", lw=.8, alpha=.6,
        ))
        nvp_count = len(nvp_steps)
        axis.set_title(
            f"{name}: {run['status']} in {run['steps']} steps\n"
            f"NVP steps (orange): {nvp_count}; min clearance "
            f"{run['min_clearance']:.3f} m", fontsize=12,
        )
        axis.set_aspect("equal")
        axis.set_xlim(SS.TASK_LO - .2, SS.TASK_HI + .2)
        axis.set_ylim(SS.TASK_LO - .2, SS.TASK_HI + .2)
        axis.grid(alpha=.2)
    figure.suptitle(
        f"Closed-loop gathering lineage s{runs[0][1]['scenario']} "
        f"γ={runs[0][1]['gamma']} — final pedestrian frame shown",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot")
    s.add_argument("--shard", required=True)
    s.add_argument("--scenario", type=int, required=True)
    s.add_argument("--gamma", type=float, required=True)
    s.add_argument("--step", type=int, required=True)
    s.add_argument("--checkpoints", nargs="+", required=True,
                   help="NAME=PATH ...")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", required=True)

    e = sub.add_parser("episode")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--scenario", type=int, required=True)
    e.add_argument("--gamma", type=float, required=True)
    e.add_argument("--ell", type=float, required=True)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--out", required=True)

    r = sub.add_parser("render-episodes")
    r.add_argument("--runs", nargs="+", required=True, help="NAME=JSON ...")
    r.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    dict(snapshot=snapshot, episode=episode,
         render_episodes=render_episodes)[args.cmd.replace("-", "_")](args)


if __name__ == "__main__":
    main()
