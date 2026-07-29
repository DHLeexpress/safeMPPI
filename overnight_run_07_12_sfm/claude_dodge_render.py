"""Active-dodge branch comparison render (episode 250003, gamma=0.1, t~55).

Panels: privileged-controller candidate pool (feasible green / rejected red /
selected blue) with the executed trajectory; raw r1 branches at the SAME
reconstructed context; post-distillation raw branches with identical latents.
Raw branches carry audit-only exact-SOCP colors.  Explanatory evidence only.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import claude_mpc_pool as MP
import grid_feats as GF
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_hp_history as HH
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS

EPISODE, GAMMA = 250_003, 0.1
WINDOW = (40, 70)


def _deploy(policy, device):
    environment = SS.scene_profile("double_density_velocity_ood")
    return KZ.kazuki_sfm_deploy(
        policy, EPISODE, GAMMA, cfg=MP.privileged_sfm_config(),
        n_ped=environment["n_ped"], T=int(SP.T), reach=0.5, device=device,
        ped_speed_range=tuple(environment["ped_speed_range"]),
        sample_seed=700_000, collect_diagnostics=True,
    )


def _contexts(rollout):
    environment = SS.scene_profile("double_density_velocity_ood")
    humans = SS.make_humans(
        EPISODE, 0, environment["n_ped"],
        tuple(environment["ped_speed_range"]),
    )
    state = np.zeros(4, np.float32)
    history = HH.HpHistory()
    contexts = []
    controls = np.asarray(rollout["controls"], np.float32)
    for t in range(int(rollout["steps"])):
        ped_xy, ped_vel = SS.collect_humans(humans)
        obstacles = np.concatenate([
            ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32),
        ], axis=1)
        hp10 = history.append(torch.as_tensor(GF.axis_grid(
            state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE,
        )))
        contexts.append(dict(
            step=t, state=state.copy(),
            hp10=hp10.numpy().astype(np.float32),
            low5=np.asarray(GF.low5(state, SS.GOAL, GAMMA), np.float32),
            hist=np.asarray(GF.hist_pad(
                controls[max(0, t - 16):t] if t else np.zeros((0, 2)), 16,
            ), np.float32),
            ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
        ))
        action = controls[t]
        state = state.copy()
        state[:2] += SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
        state[2:4] += SS.DT * action
        SS.advance_humans(humans, state)
    return contexts


@torch.no_grad()
def _raw_branches(policy, context, device, n=16):
    hp10 = torch.as_tensor(context["hp10"], device=device)[None].float()
    low = torch.as_tensor(context["low5"], device=device)[None].float()
    hist = torch.as_tensor(context["hist"], device=device)[None].float()
    ctx = policy.ctx_from(hp10, low, hist)
    generator = np.random.default_rng(20260728 + context["step"])
    latents = generator.standard_normal((n, int(policy.d)), dtype=np.float32)
    windows = BE.integrate_latents(
        policy, torch.as_tensor(latents, device=device),
        ctx.repeat_interleave(n, dim=0), nfe=8,
    ).reshape(n, 10, 2).cpu().numpy()
    labels = []
    for k in range(n):
        audit = SM.verify_query(
            context["state"], windows[k], context["ped_xy"],
            context["ped_vel"], GAMMA,
        )
        labels.append(
            bool(audit.get("resolved")) and int(audit.get("y", 0)) == 1
        )
    return windows, labels


def _scene(axis, context, rollout, t):
    path = np.asarray(rollout["states"], np.float32)
    axis.plot(path[:t + 1, 0], path[:t + 1, 1], color="#111111", lw=1.6,
              zorder=7)
    axis.plot(path[t:, 0], path[t:, 1], color="#999999", lw=1.0, ls=":",
              zorder=6)
    for j in range(len(context["ped_xy"])):
        axis.add_patch(plt.Circle(
            context["ped_xy"][j], SS.R_PED, facecolor="#c2554f",
            alpha=.6, lw=0, zorder=3,
        ))
    axis.plot(*SS.GOAL, marker="*", ms=14, color="#e6b422", mec="k",
              zorder=9)
    state = context["state"]
    axis.plot(state[0], state[1], marker="o", ms=6, color="#1450a3",
              mec="k", zorder=9)
    pad = 2.1
    axis.set_xlim(state[0] - pad, state[0] + pad)
    axis.set_ylim(state[1] - pad, state[1] + pad)
    axis.set_aspect("equal")
    axis.grid(alpha=.2)


def _draw_controller(axis, context, rollout, t):
    _scene(axis, context, rollout, t)
    trace = list(rollout.get("trace") or ())
    pool = (trace[t].get("output_filter") or {}).get("candidate_pool", ()) \
        if t < len(trace) else ()
    feasible = rejected = 0
    for candidate in pool:
        segment = SM.rollout_positions(
            context["state"], np.asarray(candidate["controls"], np.float32),
        )
        if candidate.get("selected"):
            axis.plot(segment[:, 0], segment[:, 1], color="#0868d9", lw=2.8,
                      zorder=8)
        elif candidate.get("hard_margin_feasible"):
            feasible += 1
            axis.plot(segment[:, 0], segment[:, 1], color="#159447", lw=.7,
                      alpha=.4, zorder=4)
        else:
            rejected += 1
            axis.plot(segment[:, 0], segment[:, 1], color="#d62728", lw=.6,
                      alpha=.3, zorder=4)
    reason = (trace[t].get("output_filter") or {}).get("selection_reason") \
        if t < len(trace) else None
    axis.set_title(
        f"privileged controller t={t}\nfeasible {feasible} / rejected "
        f"{rejected} / {reason}", fontsize=10,
    )
    return len(pool)


def _draw_raw(axis, name, policy, context, rollout, t, device):
    _scene(axis, context, rollout, t)
    windows, labels = _raw_branches(policy, context, device)
    positive = sum(labels)
    for k in range(len(windows)):
        segment = SM.rollout_positions(context["state"], windows[k])
        axis.plot(
            segment[:, 0], segment[:, 1],
            color="#1f8a4c" if labels[k] else "#b22222",
            lw=1.6 if labels[k] else .8,
            alpha=.9 if labels[k] else .45, zorder=5,
        )
    axis.set_title(
        f"{name} raw branches t={t}\nexact-SOCP audit positive: "
        f"{positive}/16", fontsize=10,
    )
    return positive


def run(args):
    device = args.device
    os.makedirs(os.path.dirname(os.path.abspath(args.output_png)),
                exist_ok=True)
    r1_policy, _ = GPS.load_sfm_policy(args.r1_checkpoint, device=device)
    r1_policy.eval()
    post_policy, _ = GPS.load_sfm_policy(args.post_checkpoint, device=device)
    post_policy.eval()
    rollout = _deploy(r1_policy, device)
    contexts = _contexts(rollout)
    lo, hi = WINDOW
    hi = min(hi, len(contexts) - 1)
    candidates = []
    trace = list(rollout.get("trace") or ())
    for t in range(lo, hi + 1):
        clearance = float(np.linalg.norm(
            contexts[t]["ped_xy"] - contexts[t]["state"][:2][None], axis=1,
        ).min() - SS.R_PED)
        has_pool = t < len(trace) and bool(
            (trace[t].get("output_filter") or {}).get("candidate_pool"),
        )
        candidates.append((clearance, not has_pool, t))
    snapshot = min(candidates)[2]

    figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.8))
    pool_size = _draw_controller(axes[0], contexts[snapshot], rollout,
                                 snapshot)
    before = _draw_raw(axes[1], "immutable r1", r1_policy,
                       contexts[snapshot], rollout, snapshot, device)
    after = _draw_raw(axes[2], args.post_name, post_policy,
                      contexts[snapshot], rollout, snapshot, device)
    figure.suptitle(
        f"Active-dodge comparison — episode {EPISODE}, gamma={GAMMA}, "
        f"snapshot t={snapshot} (min-clearance rule in t∈[{lo},{hi}]); "
        f"controller status: {('success' if rollout['success'] else 'fail')}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(args.output_png, dpi=170, bbox_inches="tight")
    plt.close(figure)

    if args.output_mp4:
        figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.8))

        def _frame(t):
            for axis in axes:
                axis.clear()
            _draw_controller(axes[0], contexts[t], rollout, t)
            _draw_raw(axes[1], "immutable r1", r1_policy, contexts[t],
                      rollout, t, device)
            _draw_raw(axes[2], args.post_name, post_policy, contexts[t],
                      rollout, t, device)
            figure.suptitle(
                f"episode {EPISODE} gamma={GAMMA} t={t}", fontsize=12,
            )

        anim = animation.FuncAnimation(
            figure, _frame, frames=range(lo, hi + 1, 2), interval=400,
        )
        anim.save(args.output_mp4, writer="ffmpeg", fps=3, dpi=110)
        plt.close(figure)

    payload = dict(
        episode=EPISODE, gamma=GAMMA, snapshot_t=snapshot,
        controller_status=(
            "success" if rollout["success"] else
            "collision" if rollout["collision"] else "timeout"
        ),
        controller_steps=int(rollout["steps"]),
        pool_size=int(pool_size),
        raw_positive_before=int(before),
        raw_positive_after=int(after),
    )
    with open(args.output_png + ".json", "w") as stream:
        json.dump(payload, stream, indent=1)
    print(json.dumps(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--post-checkpoint", required=True)
    parser.add_argument("--post-name", default="post-distillation")
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-mp4")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
