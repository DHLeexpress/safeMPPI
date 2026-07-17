"""Portable TRUE evaluation runner for the canonical dual-scene AFE2 protocol.

Scene-profile-aware, provenance-recorded, fail-closed bare-policy/baseline evaluation:

  * --scene-profile is REQUIRED; the rollout environment, SafeMPPI oracle, Kazuki baseline,
    metrics, and figures are all constructed from the same scene snapshot (afe2_scene_profiles).
  * --pair-root must be a validated, delivered matched AFE2 pair. Evaluation is bound to its
    completed AFE arm and to the checkpoint hashes in the trainer-written completion inventory.
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

METRIC_VERSION = "true_eval_v4_sourcebound"
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
    """Seed the actual global Python/NumPy/Torch streams used by rollout implementations."""
    import random as _random
    _random.seed(seed)
    npg = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    return npg


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


def require_clean_source(expected_commit: str) -> str:
    """Require evaluation code from the exact clean trainer commit."""
    commit = git_commit()
    if commit != expected_commit:
        raise RuntimeError(
            f"true evaluation source {commit} != expansion source {expected_commit}"
        )
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=_HERE, text=True
    ).strip()
    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
        ).returncode != 0
    )
    untracked_runtime = [
        path
        for path in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
        ).splitlines()
        if path.startswith("overnight_run_07_06/") and path.endswith((".py", ".sh"))
    ]
    if tracked_dirty or untracked_runtime:
        raise RuntimeError(
            "true evaluation requires a clean frozen source tree; "
            f"untracked runtime sources={untracked_runtime}"
        )
    return commit


def load_json(path):
    with open(path) as f:
        return json.load(f)


def validate_afe_pair(pair_root, scene_profile, rounds):
    """Bind true evaluation to one validated/delivered pair's completed AFE arm.

    Absolute paths stored in the pair manifest are informational: a delivered directory may be
    relocated. Identity is established from the manifest/delivery hashes and the AFE trainer's
    complete artifact inventory under ``pair_root/afe_s910``.
    """
    root = os.path.abspath(pair_root)
    manifest_path = os.path.join(root, f"afe2_{scene_profile}_pair_manifest.json")
    delivery_path = os.path.join(root, "DELIVERY_COMPLETE.json")
    afe_root = os.path.join(root, "afe_s910")
    for path in (manifest_path, delivery_path, os.path.join(afe_root, "recipe.json"),
                 os.path.join(afe_root, "probe.jsonl"), os.path.join(afe_root, "COMPLETE.json")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"validated AFE pair artifact is missing: {path}")

    manifest = load_json(manifest_path)
    delivery = load_json(delivery_path)
    recipe = load_json(os.path.join(afe_root, "recipe.json"))
    complete = load_json(os.path.join(afe_root, "COMPLETE.json"))
    if manifest.get("status") != "VALIDATED_MATCHED_AFE2_PAIR":
        raise RuntimeError("pair manifest is not a validated matched AFE2 pair")
    if delivery.get("status") != "DELIVERY_COMPLETE":
        raise RuntimeError("pair root lacks a completed delivery contract")
    if manifest.get("scene_profile") != scene_profile:
        raise RuntimeError("pair manifest scene profile does not match --scene-profile")
    if recipe.get("arm") != "afe" or recipe.get("reference_recipe_locked") is not True:
        raise RuntimeError("true evaluation accepts only the locked AFE arm")
    if recipe.get("scene", {}).get("profile", {}).get("name") != scene_profile:
        raise RuntimeError("AFE recipe scene profile does not match --scene-profile")
    if recipe.get("scene", {}).get("sha256") != manifest.get("scene_sha256"):
        raise RuntimeError("AFE recipe scene hash disagrees with the pair manifest")
    if recipe.get("source_checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
        raise RuntimeError("AFE recipe source checkpoint disagrees with the pair manifest")
    if recipe.get("source_checkpoint_model_sha256") != manifest.get(
        "source_checkpoint_model_sha256"
    ):
        raise RuntimeError("AFE recipe model-state hash disagrees with the pair manifest")
    if recipe.get("source_checkpoint_contract_sha256") != manifest.get(
        "source_checkpoint_contract_sha256"
    ):
        raise RuntimeError("AFE recipe checkpoint contract disagrees with the pair manifest")
    if recipe.get("source_git_commit") != manifest.get("source_git_commit"):
        raise RuntimeError("AFE recipe source commit disagrees with the pair manifest")

    afe_entry = manifest.get("runs", {}).get("afe", {})
    fixed = {
        "recipe.json": afe_entry.get("recipe_sha256"),
        "probe.jsonl": afe_entry.get("probe_sha256"),
        "COMPLETE.json": afe_entry.get("complete_sha256"),
    }
    for relative, expected in fixed.items():
        path = os.path.join(afe_root, relative)
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"AFE pair-manifest artifact hash mismatch: {relative}")
    delivery_pair = delivery.get("artifacts", {}).get("pair_manifest", {})
    if delivery_pair.get("sha256") != sha256_file(manifest_path):
        raise RuntimeError("delivery contract does not authenticate the pair manifest")
    for key in (
        "scene_sha256",
        "source_checkpoint_sha256",
        "source_checkpoint_model_sha256",
        "source_checkpoint_contract_sha256",
        "source_git_commit",
    ):
        if delivery.get(key) != manifest.get(key):
            raise RuntimeError(f"delivery and pair manifest disagree on {key}")

    if complete.get("status") != "COMPLETE" or complete.get("completed_round") != rounds:
        raise RuntimeError(f"AFE arm is not a completed {rounds}-round run")
    if complete.get("scene_sha256") != manifest.get("scene_sha256"):
        raise RuntimeError("AFE completion scene hash disagrees with pair manifest")
    if complete.get("checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
        raise RuntimeError("AFE completion checkpoint hash disagrees with pair manifest")
    if complete.get("checkpoint_model_sha256") != manifest.get(
        "source_checkpoint_model_sha256"
    ):
        raise RuntimeError("AFE completion model-state hash disagrees with pair manifest")
    if complete.get("checkpoint_contract_sha256") != manifest.get(
        "source_checkpoint_contract_sha256"
    ):
        raise RuntimeError("AFE completion checkpoint contract disagrees with pair manifest")
    if complete.get("source_git_commit") != manifest.get("source_git_commit"):
        raise RuntimeError("AFE completion source commit disagrees with pair manifest")
    expected_rounds = list(range(rounds + 1))
    required = {
        "recipe.json", "probe.jsonl", "final.pt", "dstore.pt",
        *{f"ckpt_{n}.pt" for n in expected_rounds},
        *{f"viz_db/round{n}.pt" for n in expected_rounds[1:]},
    }
    inventory = complete.get("artifact_sha256", {})
    if set(inventory) != required:
        raise RuntimeError("AFE completion inventory is incomplete")
    for relative, expected in inventory.items():
        path = os.path.join(afe_root, relative)
        if not os.path.isfile(path) or sha256_file(path) != expected:
            raise RuntimeError(f"AFE completion artifact hash mismatch: {relative}")
    ckpts = require_round_checkpoints(afe_root, rounds)
    for n, path in ckpts.items():
        if sha256_file(path) != inventory[f"ckpt_{n}.pt"]:
            raise RuntimeError(f"AFE checkpoint {n} is not the completed artifact")
    contract = {
        "pair_root": root,
        "pair_manifest": os.path.abspath(manifest_path),
        "pair_manifest_sha256": sha256_file(manifest_path),
        "delivery_manifest": os.path.abspath(delivery_path),
        "delivery_manifest_sha256": sha256_file(delivery_path),
        "afe_root": os.path.abspath(afe_root),
        "afe_recipe_sha256": fixed["recipe.json"],
        "afe_complete_sha256": fixed["COMPLETE.json"],
        "afe_source_git_commit": manifest["source_git_commit"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "source_checkpoint_model_sha256": manifest[
            "source_checkpoint_model_sha256"
        ],
        "source_checkpoint_contract_sha256": manifest[
            "source_checkpoint_contract_sha256"
        ],
    }
    return ckpts, contract


def validate_afe_run(run_root, scene_profile, rounds):
    """Bind evaluation directly to one completed single-arm AFE-RBF run."""

    root = os.path.abspath(run_root)
    recipe_path = os.path.join(root, "recipe.json")
    complete_path = os.path.join(root, "COMPLETE.json")
    for path in (recipe_path, complete_path, os.path.join(root, "probe.jsonl")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"completed AFE-RBF artifact is missing: {path}")
    recipe = load_json(recipe_path)
    complete = load_json(complete_path)
    if recipe.get("algorithm") != "afe_rbf_previous_round_parallel_v1":
        raise RuntimeError("--run-root accepts only the declared single-arm AFE-RBF algorithm")
    if recipe.get("arm") != "afe" or recipe.get("single_arm") is not True:
        raise RuntimeError("--run-root is not a single AFE arm")
    if recipe.get("scene", {}).get("profile", {}).get("name") != scene_profile:
        raise RuntimeError("AFE-RBF recipe scene profile does not match --scene-profile")
    if complete.get("status") != "COMPLETE" or complete.get("completed_round") != rounds:
        raise RuntimeError(f"AFE-RBF run is not a completed {rounds}-round run")
    if complete.get("scene_sha256") != recipe.get("scene", {}).get("sha256"):
        raise RuntimeError("AFE-RBF completion scene hash disagrees with its recipe")
    checks = {
        "checkpoint_sha256": "source_checkpoint_sha256",
        "checkpoint_model_sha256": "source_checkpoint_model_sha256",
        "checkpoint_contract_sha256": "source_checkpoint_contract_sha256",
        "source_git_commit": "source_git_commit",
    }
    for complete_key, recipe_key in checks.items():
        if complete.get(complete_key) != recipe.get(recipe_key):
            raise RuntimeError(f"AFE-RBF completion disagrees with recipe: {complete_key}")
    expected_rounds = list(range(rounds + 1))
    required = {
        "recipe.json", "rbf_calibration.json", "probe.jsonl", "final.pt", "dstore.pt",
        *{f"ckpt_{round_i}.pt" for round_i in expected_rounds},
        *{f"viz_db/round{round_i}.pt" for round_i in expected_rounds[1:]},
    }
    inventory = complete.get("artifact_sha256", {})
    if set(inventory) != required:
        raise RuntimeError("AFE-RBF completion inventory is incomplete")
    for relative, expected in inventory.items():
        path = os.path.join(root, relative)
        if not os.path.isfile(path) or sha256_file(path) != expected:
            raise RuntimeError(f"AFE-RBF completion artifact hash mismatch: {relative}")
    checkpoints = require_round_checkpoints(root, rounds)
    contract = {
        "kind": "single_afe_rbf_run",
        "run_root": root,
        "recipe_sha256": sha256_file(recipe_path),
        "complete_sha256": sha256_file(complete_path),
        "afe_source_git_commit": recipe["source_git_commit"],
        "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
        "source_checkpoint_model_sha256": recipe["source_checkpoint_model_sha256"],
        "source_checkpoint_contract_sha256": recipe[
            "source_checkpoint_contract_sha256"
        ],
    }
    return checkpoints, contract


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
    npz_name = f"paths_{name}.npz"
    prov_name = f"{name}.provenance.json"
    npz_path = os.path.join(outdir, npz_name)
    prov_path = os.path.join(outdir, prov_name)
    np.savez_compressed(npz_path, paths=pa)
    provenance = dict(provenance, seeds=[int(s) for s in seeds], n_paths=len(paths),
                      paths_artifact=npz_name, paths_sha256=sha256_file(npz_path))
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=1)
    return {npz_name: sha256_file(npz_path), prov_name: sha256_file(prov_path)}


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
    return save_cell(outdir, name, paths, seeds, prov)


def expert_cell(env, gamma, cfg, prov, outdir, name):
    """SafeMPPI evaluation-only oracle on THIS evaluation scene (planner sees this geometry)."""
    import grid_scene as GS
    import grid_metrics as GM
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
            if not GM.in_taskspace(st[:2][None]):
                break
            if obs.size and (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1)
                             - obs[:, 2] - rr).min() < 0.0:
                break
        paths.append(np.asarray(path, np.float32))
    return save_cell(outdir, name, paths, seeds, prov)


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
    return save_cell(outdir, name, paths, seeds, prov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-profile", required=True,
                    help="explicit scene profile (no default): claude_grid_v1 | codex_radius1_v1")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--pair-root",
                        help="validated AFE2 pair root (legacy two-arm protocol)")
    source.add_argument("--run-root",
                        help="completed single-arm AFE-RBF run directory")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sources", nargs="+", default=["policy", "expert", "kazuki"],
                    choices=["policy", "expert", "kazuki"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--gammas", type=float, nargs="+", default=list(EVAL_GAMMAS))
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--pilot", action="store_true",
                    help="allow a labeled finite-M preflight instead of the locked M=100 paper run")
    args = ap.parse_args()
    shared_wrong = (
        args.T != 300
        or abs(args.reach - 0.15) > 1e-12
        or tuple(float(g) for g in args.gammas) != EVAL_GAMMAS
        or set(args.sources) != {"policy", "expert", "kazuki"}
    )
    if shared_wrong or (
        not args.pilot and (args.rounds != 10 or args.M != 100)
    ) or (args.pilot and (args.rounds < 1 or args.M < 10)):
        raise ValueError(
            "evaluation requires T=300, reach=0.15, gammas={0.1,0.3,0.5,1.0}, and all "
            "three sources; canonical mode also requires rounds=10/M=100, while --pilot "
            "requires rounds>=1/M>=10")

    from afe2_scene_profiles import get_scene_profile, build_scene, scene_snapshot
    import grid_metrics2 as GM2
    import grid_hp_expt as HP

    profile = get_scene_profile(args.scene_profile)
    env = build_scene(profile)
    snapshot = scene_snapshot(env, profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.run_root is not None:
        ckpts, expansion_contract = validate_afe_run(
            args.run_root, profile.name, args.rounds
        )
    else:
        ckpts, expansion_contract = validate_afe_pair(
            args.pair_root, profile.name, args.rounds
        )
    evaluation_source_commit = require_clean_source(
        expansion_contract["afe_source_git_commit"]
    )
    os.makedirs(args.outdir, exist_ok=True)
    scene_path = os.path.join(args.outdir, "scene_snapshot.json")
    with open(scene_path, "w") as f:
        json.dump(snapshot, f, indent=1)
    base_prov = dict(metric_version=METRIC_VERSION, stopping_rule=STOPPING_RULE,
                     scene_profile=profile.name, scene_sha256=snapshot["sha256"],
                     source_git_commit=evaluation_source_commit, M=int(args.M), T=int(args.T),
                     reach=float(args.reach), pilot=bool(args.pilot),
                     expansion_source=expansion_contract)
    cfg = dict(M=args.M, T=args.T, reach=args.reach)
    artifacts = {"scene_snapshot.json": sha256_file(scene_path)}
    required_cells = []

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
                artifacts.update(bare_policy_cell(pol, env, g, cfg, prov, args.outdir, name, device))
                required_cells.append(name)
                print(f"[policy r{n} g{g}] done", flush=True)
    if "expert" in args.sources:
        for g in args.gammas:
            name = cell_name("expert", None, g)
            assert_fresh_cell(args.outdir, name)
            prov = dict(base_prov, source="expert", round=None, gamma=float(g),
                        checkpoint=None, checkpoint_sha256=None,
                        oracle="SafeMPPI mode1 on this evaluation scene")
            artifacts.update(expert_cell(env, g, cfg, prov, args.outdir, name))
            required_cells.append(name)
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
        artifacts.update(kazuki_cell(pol0, env, cfg, prov, args.outdir, name, device))
        required_cells.append(name)
        print("[kazuki blind] done", flush=True)
    with open(os.path.join(args.outdir, "RUN_COMPLETE.json"), "w") as f:
        json.dump(dict(base_prov, status="TRUE_EVAL_RAW_COMPLETE", sources=args.sources,
                       gammas=args.gammas, rounds=sorted(ckpts), required_cells=required_cells,
                       artifact_sha256=artifacts), f, indent=1, sort_keys=True)
    print("TRUE-EVAL RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
