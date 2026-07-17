"""Portable TRUE evaluation runner (integration/afe2-terminal-dualscene-v1).

Scene-profile-aware, provenance-recorded, fail-closed bare-policy/baseline evaluation:

  * --scene-profile is REQUIRED; the rollout environment, SafeMPPI oracle, Kazuki baseline,
    metrics, and figures are all constructed from the same scene snapshot (afe2_scene_profiles).
  * --ckpt-dir must contain EXACTLY ckpt_0.pt .. ckpt_10.pt; any missing round aborts.
    There is no silent ckpt_n -> final.pt substitution and no default checkpoint path.
  * Existing cell outputs abort the run (no silent NPZ reuse); use a fresh --outdir.
  * RNG: named isolated Python/NumPy/Torch-CPU/CUDA streams per rollout, derived by SHA-256 from
    (metric_version, scene, source, gamma, rollout_index). Round curves use common random numbers
    keyed by (gamma, rollout_index) — NOT by round — so per-round comparisons are paired.
  * The bare-policy evaluation has no verifier, no NVP, no fallback: a rollout stops only at the
    first goal hit (reach), actual collision/OOB, or T=300.
  * Kazuki baseline: the existing definition, FIXED (gamma_ctx=0.5, w_safe .3, coll 5, goal 5,
    coef 1, beta 20, lambda .1, sigma .2, margin .05, 200/10/200), same pretrained checkpoint
    (ckpt_0); it is gamma-blind, so ONE M=100 batch is shared by every gamma column and each
    column certifies those trajectories at its own gamma.
  * Every cell writes paths_<cell>.npz + <cell>.provenance.json with: source git commit, scene
    sha256, checkpoint sha256, per-rollout seeds, gamma, round, stopping rule, metric version,
    and validated M.
Nothing here touches D, D+, or A; expansion state is never imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
for _p in (_WORK, _REV, _HERE):
    sys.path.insert(0, _p)

import numpy as np
import torch

import _paths  # noqa: F401

METRIC_VERSION = "true_eval_v2_terminal"
STOPPING_RULE = "first goal hit (reach) OR actual collision/OOB OR T cap; no verifier/NVP/fallback"
EVAL_GAMMAS = (0.1, 0.3, 0.5, 1.0)
KAZUKI_DEFINITION = dict(gamma_ctx=0.5, w_safe=0.3, coll_w=5.0, goal_w=5.0, goal_coef=1.0,
                         beta_mppi=20.0, mppi_lambda=0.1, mppi_sigma=0.2, r_margin=0.05,
                         n_sample=200, n_elite=10, n_copy=200)


def named_seed(*parts) -> int:
    """Deterministic 63-bit seed from a named key; the ONLY seed derivation in this evaluation."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2 ** 63 - 1)


def seed_all_streams(seed: int):
    """Named isolated streams: returns (python.Random, numpy Generator); torch CPU+CUDA global
    states are seeded (the policy/planner sampling paths draw from the torch global streams)."""
    import random as _random
    py = _random.Random(seed)
    npg = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    return py, npg


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=_HERE, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def require_round_checkpoints(ckpt_dir, rounds):
    """Exact ckpt_0..ckpt_R required; abort on ANY missing round (no final.pt substitution)."""
    paths = {}
    missing = []
    for n in range(rounds + 1):
        p = os.path.join(ckpt_dir, f"ckpt_{n}.pt")
        if os.path.isfile(p):
            paths[n] = p
        else:
            missing.append(f"ckpt_{n}.pt")
    if missing:
        raise FileNotFoundError(
            f"--ckpt-dir {ckpt_dir} is missing required round checkpoints: {missing}; "
            "true evaluation refuses substitutions")
    return paths


def cell_name(source, round_i, gamma):
    r = "NA" if round_i is None else str(round_i)
    return f"{source}_r{r}_g{gamma}"


def assert_fresh_cell(outdir, name):
    for p in (os.path.join(outdir, f"paths_{name}.npz"),
              os.path.join(outdir, f"{name}.provenance.json")):
        if os.path.exists(p):
            raise FileExistsError(
                f"stale output rejected: {p} already exists; use a fresh --outdir")


def save_cell(outdir, name, paths, seeds, provenance):
    if len(paths) != provenance["M"]:
        raise RuntimeError(f"cell {name}: rollout count {len(paths)} != declared M {provenance['M']}")
    pa = np.empty(len(paths), dtype=object)
    for i, p in enumerate(paths):
        pa[i] = np.asarray(p, np.float32)
    np.savez_compressed(os.path.join(outdir, f"paths_{name}.npz"), paths=pa)
    provenance = dict(provenance, seeds=[int(s) for s in seeds], n_paths=len(paths))
    with open(os.path.join(outdir, f"{name}.provenance.json"), "w") as f:
        json.dump(provenance, f, indent=1)


def bare_policy_cell(policy, env, gamma, cfg, prov, outdir, name, device):
    import grid_rollout as GR
    paths, seeds = [], []
    for m in range(cfg["M"]):
        seed = named_seed(METRIC_VERSION, prov["scene_sha256"], "policy", gamma, m)  # CRN: NOT round
        seeds.append(seed)
        seed_all_streams(seed)
        out = GR.fm_deploy(policy, env, float(gamma), T=cfg["T"], temp=1.0, nfe=8,
                           reach=cfg["reach"], device=device)
        paths.append(out["path"])
    save_cell(outdir, name, paths, seeds, prov)


def expert_cell(env, gamma, cfg, prov, outdir, name):
    """SafeMPPI evaluation-only oracle on THIS evaluation scene (planner sees this geometry)."""
    import grid_scene as GS
    from di_grid_viz import di_step
    from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
    goal_t = env.goal.detach().cpu().float()
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    obs_plan = GS.planner_obstacles(env)
    mcfg = GS.mode1_config()
    paths, seeds = [], []
    for m in range(cfg["M"]):
        seed = named_seed(METRIC_VERSION, prov["scene_sha256"], "expert", gamma, m)
        seeds.append(seed)
        seed_all_streams(seed)
        ad = SafeMPPIAdapter(**mcfg)
        st = env.x0.detach().cpu().numpy().astype(np.float32).copy()
        path = [st[:2].copy()]
        for t in range(cfg["T"]):
            a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_plan,
                           gamma=float(gamma), seed=seed % (2 ** 31 - 1) + t)
            st = di_step(st, a.detach().cpu().numpy().astype(np.float32), dt=env.dt)
            path.append(st[:2].copy())
            if np.linalg.norm(st[:2] - goal) < cfg["reach"]:
                break
            if (st[:2] < -0.05).any() or (st[:2] > 5.05).any():
                break
            if obs.size and (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1)
                             - obs[:, 2] - rr).min() < 0.0:
                break
        paths.append(np.asarray(path, np.float32))
    save_cell(outdir, name, paths, seeds, prov)


def kazuki_cell(policy, env, cfg, prov, outdir, name, device):
    """The FIXED Kazuki definition (gamma_ctx=0.5) — one gamma-blind batch shared by all columns."""
    import kazuki_baseline as KB
    KB.COLL_W = KAZUKI_DEFINITION["coll_w"]
    KB.GOAL_W = KAZUKI_DEFINITION["goal_w"]
    KB.GOAL_COEF = KAZUKI_DEFINITION["goal_coef"]
    KB.BETA_MPPI = KAZUKI_DEFINITION["beta_mppi"]
    KB.MPPI_LAMBDA = KAZUKI_DEFINITION["mppi_lambda"]
    KB.MPPI_SIGMA = KAZUKI_DEFINITION["mppi_sigma"]
    KB.R_MARGIN = KAZUKI_DEFINITION["r_margin"]
    KB.N_SAMPLE = KAZUKI_DEFINITION["n_sample"]
    KB.N_ELITE = KAZUKI_DEFINITION["n_elite"]
    KB.N_COPY = KAZUKI_DEFINITION["n_copy"]
    paths, seeds = [], []
    for m in range(cfg["M"]):
        seed = named_seed(METRIC_VERSION, prov["scene_sha256"], "kazuki", m)   # gamma-blind
        seeds.append(seed)
        seed_all_streams(seed)
        out = KB.kazuki_deploy(policy, env, [KAZUKI_DEFINITION["w_safe"]],
                               gamma_ctx=KAZUKI_DEFINITION["gamma_ctx"], T=cfg["T"],
                               reach=cfg["reach"], device=device, seed=seed % (2 ** 31 - 1))
        paths.append(out["path"])
    save_cell(outdir, name, paths, seeds, prov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-profile", required=True,
                    help="explicit scene profile (no default): claude_grid_v1 | codex_radius1_v1")
    ap.add_argument("--ckpt-dir", required=True,
                    help="run directory holding EXACT ckpt_0.pt..ckpt_<rounds>.pt (no fallback)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sources", nargs="+", default=["policy", "expert", "kazuki"],
                    choices=["policy", "expert", "kazuki"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--gammas", type=float, nargs="+", default=list(EVAL_GAMMAS))
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--reach", type=float, default=0.15)
    args = ap.parse_args()

    from afe2_scene_profiles import get_scene_profile, build_scene, scene_snapshot
    import grid_metrics2 as GM2
    import grid_hp_expt as HP

    profile = get_scene_profile(args.scene_profile)
    env = build_scene(profile)
    snapshot = scene_snapshot(env, profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = require_round_checkpoints(args.ckpt_dir, args.rounds)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "scene_snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=1)
    base_prov = dict(metric_version=METRIC_VERSION, stopping_rule=STOPPING_RULE,
                     scene_profile=profile.name, scene_sha256=snapshot["sha256"],
                     source_git_commit=git_commit(), M=int(args.M), T=int(args.T),
                     reach=float(args.reach))
    cfg = dict(M=args.M, T=args.T, reach=args.reach)

    if "policy" in args.sources:                            # every round, CRN across rounds
        for n in sorted(ckpts):
            pol, _ = HP.load_hp(ckpts[n], device="cpu")
            pol = pol.to(device)
            csha = sha256_file(ckpts[n])
            for g in args.gammas:
                name = cell_name("policy", n, g)
                assert_fresh_cell(args.outdir, name)
                prov = dict(base_prov, source="policy", round=n, gamma=float(g),
                            checkpoint=os.path.abspath(ckpts[n]), checkpoint_sha256=csha)
                bare_policy_cell(pol, env, g, cfg, prov, args.outdir, name, device)
                print(f"[policy r{n} g{g}] done", flush=True)
    if "expert" in args.sources:
        for g in args.gammas:
            name = cell_name("expert", None, g)
            assert_fresh_cell(args.outdir, name)
            prov = dict(base_prov, source="expert", round=None, gamma=float(g),
                        checkpoint=None, checkpoint_sha256=None,
                        oracle="SafeMPPI mode1 on this evaluation scene")
            expert_cell(env, g, cfg, prov, args.outdir, name)
            print(f"[expert g{g}] done", flush=True)
    if "kazuki" in args.sources:
        name = cell_name("kazuki", None, "blind")
        assert_fresh_cell(args.outdir, name)
        pol0, _ = HP.load_hp(ckpts[0], device="cpu")
        pol0 = pol0.to(device)
        prov = dict(base_prov, source="kazuki", round=None, gamma=None,
                    checkpoint=os.path.abspath(ckpts[0]),
                    checkpoint_sha256=sha256_file(ckpts[0]),
                    definition=KAZUKI_DEFINITION,
                    note="gamma-blind fixed baseline; every gamma column certifies these paths")
        kazuki_cell(pol0, env, cfg, prov, args.outdir, name, device)
        print("[kazuki blind] done", flush=True)
    with open(os.path.join(args.outdir, "RUN_COMPLETE.json"), "w") as f:
        json.dump(dict(base_prov, sources=args.sources, gammas=args.gammas,
                       rounds=sorted(ckpts)), f, indent=1)
    print("TRUE-EVAL RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
