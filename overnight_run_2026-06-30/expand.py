"""Safe Flow Expansion (Pillar 5) on the fixed cluttered scene, gated by the compact SOCP verifier.

Reuses ``overnight_run_today/src/safeflow.run_safeflow`` (ACTFLOW Algorithm 1) VERBATIM, swapping in
the SOCP gate and the clutter coverage metric by rebinding the two module globals
``safeflow.validity_label`` and ``safeflow.evaluate``.  gamma is single-sourced: it builds the
policy context AND sets the verifier ceiling ``cfg.gamma_max``.  ``alpha=0`` => rejected
trajectories are dropped (never train the policy) = "delete the whole invalid trajectory".  The
broad "surrounding" proposal is kept (rho0>0) — per FINDINGS it, not the sigma-tilt, discovers modes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import torch

import _paths
import env as E
from scene_encoder import SceneConditionedFlowPolicy
from safeflow import SFConfig, run_safeflow
import safeflow
from socp_gate import make_socp_validity_label
from metrics import build_omega_star_clutter, make_clutter_evaluate
import wandb_utils as W


def load_policy(ckpt, device="cpu"):
    pol = SceneConditionedFlowPolicy(
        T=ckpt["T"], token_dim=ckpt["token_dim"], width=ckpt["width"], depth=ckpt["depth"],
        n_max=ckpt["n_max"], S=ckpt["S"], R_enc=ckpt["R_enc"], r_robot=ckpt["r_robot"],
        u_max=ckpt["u_max"],
    ).to(device)
    pol.load_state_dict(ckpt["state_dict"])
    pol.eval()
    return pol


def fixed_env(ckpt, scene_idx, device="cpu"):
    sc = ckpt["scenes"][scene_idx % len(ckpt["scenes"])]
    return E.env_from_obstacles(sc["obstacles"], sc["start"], sc["goal"], T=ckpt["T"], dt=ckpt["dt"],
                                u_max=ckpt["u_max"], r_robot=ckpt["r_robot"], box=ckpt["box"], device=device)


def smoke_cfg(cfg: SFConfig):
    cfg.rounds, cfg.N, cfg.B, cfg.eval_K = 12, 192, 64, 400
    cfg.inner_steps, cfg.eval_every, cfg.warmup_pos = 60, 2, 16
    return cfg


def full_cfg(cfg: SFConfig):
    cfg.rounds, cfg.N, cfg.B, cfg.eval_K = 40, 256, 64, 1200
    cfg.inner_steps, cfg.eval_every, cfg.warmup_pos = 150, 4, 32
    return cfg


def run_expansion(ckpt, gamma, scene_idx, smoke, device, out_dir, omega_n, args=None, log=print):
    policy = load_policy(ckpt, device)
    policy.freeze_encoder()
    env = fixed_env(ckpt, scene_idx, device)
    ctx = policy.ctx_for_env(env, gamma).detach()

    cfg = SFConfig()
    cfg.alpha = 0.0                      # delete invalid trajectories (no unlearning)
    cfg.gamma_max = float(gamma)         # verifier ceiling == policy gamma (single source)
    cfg.rho0, cfg.rho_min = 0.7, 0.05    # keep the broad 'surrounding' proposal (mode discovery)
    cfg = smoke_cfg(cfg) if smoke else full_cfg(cfg)

    # ---- swap in the SOCP gate + clutter coverage via module-global rebind ----
    gate = make_socp_validity_label(env, R_ver=2.0, H_win=10, stride=2, reach_radius=0.6)
    safeflow.validity_label = gate
    star_bins, ranges, star_cells, star_modes, n_star, star_U = build_omega_star_clutter(
        env, cfg, gate, omega_n, device=device, log=log)
    safeflow.evaluate = make_clutter_evaluate(star_cells, star_modes)

    pre_state = copy.deepcopy(policy.state_dict())
    m0 = safeflow.evaluate(policy, env, ctx, star_bins, ranges, cfg)
    log(f"[pretrained g={gamma}] spatial_cov={m0['spatial_coverage']:.2f} val={m0['validity']:.2f} "
        f"modecov={m0['mode_coverage']:.2f} vendi={m0['vendi']:.2f} n_valid={m0['n_valid']}")

    run = W.init_run(args, name=f"expand-g{gamma}", group=f"expand-scene{scene_idx}", dir=out_dir,
                     config={"stage": "expand", "gamma": float(gamma), "scene_idx": scene_idx,
                             "omega_n": omega_n, "n_star": n_star, "alpha": cfg.alpha,
                             "rounds": cfg.rounds, "N": cfg.N, "B": cfg.B, "eval_K": cfg.eval_K,
                             "inner_steps": cfg.inner_steps, "rho0": cfg.rho0, "smoke": smoke}) if args else None
    # pretrained baseline at step 0 so the curve shows the pretrained -> expanded jump
    W.log(run, {f"expand/{k}": v for k, v in m0.items() if isinstance(v, (int, float))}, step=0)

    # Seed the expansion buffer D_0 with the verifier-certified surrounding set (paradigm's D_0).
    snap_rounds = set(range(0, cfg.rounds, cfg.eval_every)) | {cfg.rounds - 1}
    policy, history, snaps = run_safeflow(env, ctx, policy, star_bins, ranges, cfg,
                                          device=device, log=log, snapshot_rounds=snap_rounds,
                                          init_pos=star_U)
    for rec in history:                                     # per-round coverage/validity/vendi curves
        W.log(run, {f"expand/{k}": v for k, v in rec.items() if isinstance(v, (int, float))},
              step=int(rec["round"]) + 1)
    W.finish(run, summary={"pretrained_spatial_coverage": m0["spatial_coverage"],
                           "final_spatial_coverage": history[-1]["spatial_coverage"],
                           "final_validity": history[-1]["validity"],
                           "final_mode_coverage": history[-1]["mode_coverage"],
                           "final_vendi": history[-1]["vendi"], "n_star": n_star})

    out = os.path.join(out_dir, f"expand_g{gamma}.pt")
    torch.save({
        "gamma": gamma, "scene_idx": scene_idx, "history": history,
        "snapshots": snaps, "pretrained_state": pre_state, "m0": m0,
        "ctx": ctx.detach().cpu(), "n_star": n_star,
        "star_cells": list(star_cells), "star_modes": list(star_modes),
        "scene": ckpt["scenes"][scene_idx % len(ckpt["scenes"])],
        "policy_cfg": {k: ckpt[k] for k in ["T", "token_dim", "width", "depth", "n_max",
                                            "S", "R_enc", "r_robot", "u_max", "dt", "box"]},
        "sf_cfg": cfg.__dict__,
    }, out)
    log(f"saved {out}  (final spatial_cov={history[-1]['spatial_coverage']:.2f} "
        f"val={history[-1]['validity']:.2f} npos={history[-1]['n_pos']})")
    return history, m0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_paths.HERE, "results", "pretrained.pt"))
    ap.add_argument("--out-dir", default=os.path.join(_paths.HERE, "results"))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--scene-idx", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--omega-n", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    omega_n = args.omega_n if args.omega_n is not None else (2500 if args.smoke else 5000)
    summary = {}
    for g in args.gammas:
        print(f"\n=== Safe Flow Expansion  gamma={g}  scene={args.scene_idx}  smoke={args.smoke} ===", flush=True)
        history, m0 = run_expansion(ckpt, g, args.scene_idx, args.smoke, args.device, args.out_dir,
                                    omega_n, args=args)
        summary[str(g)] = {"pretrained": m0, "final": history[-1] if history else None}
    with open(os.path.join(args.out_dir, "expand_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nsaved expand_summary.json", flush=True)


if __name__ == "__main__":
    main()
