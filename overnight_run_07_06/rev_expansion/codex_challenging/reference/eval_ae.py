#!/usr/bin/env python3
"""Shared paper-table evaluator for SafeMPPI, flow policies, and saved baseline paths.

The success convention is the one fixed in GOAL.md: endpoint within 0.1 m,
collision-free, and no more than 250 controls.  Coverage is the empirical count
of distinct staircase IDs among those successful episodes; no denominator is
hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent
WORK = REV.parent
RUN = WORK.parent
OVERNIGHT = WORK / "codex_overnight"
sys.path[:0] = [str(HERE), str(REV), str(OVERNIGHT), str(RUN), str(WORK)]

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

import grid_hp_expt as HP
import grid_metrics as GM
import grid_scene as GS
import sr_cr_eval as SR

# Loading the endpoint-free policy intentionally restores codex_challenging/ to
# sys.path[0].  Import this SafeMPPI rollout helper by file identity so the local
# seed-visualization module with the same basename cannot shadow it.
_gud_spec = importlib.util.spec_from_file_location(
    "challenging_reference_gen_uniform_data", HERE / "gen_uniform_data.py")
GUD = importlib.util.module_from_spec(_gud_spec)
assert _gud_spec.loader is not None
_gud_spec.loader.exec_module(GUD)

GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
FIELDS = (
    "method", "gamma", "SR", "CR", "clearance_mean", "clearance_std",
    "per_obstacle_min_mean", "per_obstacle_min_std", "time_mean_s",
    "time_std_s", "coverage", "n_success", "M", "iterations_to_goal",
)



_WALL_STEP = 5.0 / 13.0
_WALL_PLUGS4 = [(_WALL_STEP, -0.2, 0.2), (5.0 - _WALL_STEP, 5.2, 0.2),
                (-0.2, _WALL_STEP, 0.2), (5.2, 5.0 - _WALL_STEP, 0.2)]
_WALL_PLUGS8 = _WALL_PLUGS4 + [
    (0.0, -0.2, 0.2), (-0.2, 0.0, 0.2),
    (5.2, 5.0, 0.2), (5.0, 5.2, 0.2)]


def _apply_wall_plugs_eval(env, n):
    if not n:
        return env
    import torch as _t
    plugs = _WALL_PLUGS4[:2] if n == 2 else _WALL_PLUGS8 if n == 8 else _WALL_PLUGS4
    env.obstacles = _t.cat([env.obstacles, _t.tensor(plugs, dtype=env.obstacles.dtype)], dim=0)
    return env

def _gstr(gamma: float) -> str:
    return str(float(gamma))


def _paths_array(paths):
    out = np.empty(len(paths), dtype=object)
    for i, path in enumerate(paths):
        out[i] = np.asarray(path, dtype=np.float32)
    return out


def save_paths(path: Path, paths, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, paths=_paths_array(paths), **metadata)


def load_paths(path: Path):
    with np.load(path, allow_pickle=True) as z:
        return [np.asarray(p, dtype=np.float32) for p in z["paths"]]


def summarize_paths(paths, env, gamma: float, method: str, reach: float = 0.1,
                    iterations_to_goal=None) -> dict:
    """Compute a--e from executed XY paths, with c/d/e on successful paths only."""
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    collisions, successes = [], []
    clearance_ep, per_obs_ep, times_ep, ids = [], [], [], set()

    for raw in paths:
        p = np.asarray(raw, dtype=float)
        if p.ndim != 2 or p.shape[1] < 2 or len(p) == 0:
            raise ValueError(f"invalid path shape {p.shape}")
        p = p[:, :2]
        d = np.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr
        coll = bool((d.min(axis=1) < 0.0).any())
        success = bool(np.linalg.norm(p[-1] - goal) < reach and not coll and len(p) - 1 <= env.T)
        collisions.append(coll)
        successes.append(success)
        if not success:
            continue
        # Primary c: at each time take the nearest-obstacle clearance, then average over time.
        clearance_ep.append(float(d.min(axis=1).mean()))
        # Footnote c: for every obstacle take its episode minimum, then average obstacles.
        per_obs_ep.append(float(d.min(axis=0).mean()))
        times_ep.append(float((len(p) - 1) * env.dt))
        sid = GM.staircase_id(p)
        if sid is not None:
            ids.add(sid)

    n = len(paths)
    ns = int(np.sum(successes))

    def stats(values):
        if not values:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values, ddof=0))

    cm, cs = stats(clearance_ep)
    om, osd = stats(per_obs_ep)
    tm, ts = stats(times_ep)
    return {
        "method": method,
        "gamma": float(gamma),
        "SR": ns / n if n else float("nan"),
        "CR": float(np.mean(collisions)) if n else float("nan"),
        "clearance_mean": cm,
        "clearance_std": cs,
        "per_obstacle_min_mean": om,
        "per_obstacle_min_std": osd,
        "time_mean_s": tm,
        "time_std_s": ts,
        "coverage": len(ids),
        "coverage_ids": sorted(ids),
        "n_success": ns,
        "M": n,
        "iterations_to_goal": iterations_to_goal,
    }


def save_row(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n")


def expert_worker(args) -> None:
    env = GS.make_grid()
    _apply_wall_plugs_eval(env, getattr(args, 'wall_plugs', 0))
    cfg = GS.mode1_config()
    # Walled scene: the exact origin sits ON the corner plugs; start at (eps,eps) to match the policy
    # eval. The nominal polytope is still TIGHT at the start (clearance ~ eps) and restricts the first move.
    _eps = float(getattr(args, 'start_eps', 0.0) or 0.0)
    paths = []
    t0 = time.time()
    for i in range(args.M):
        states, _ = GUD.rollout_from(env, cfg, args.gamma, (_eps, _eps),
                                     seed=args.seed0 + i, reach=args.reach)
        paths.append(states[:, :2])
        if (i + 1) % 10 == 0 or i + 1 == args.M:
            print(f"[expert g{args.gamma}] {i + 1}/{args.M} "
                  f"({(time.time() - t0) / (i + 1):.2f}s/rollout)", flush=True)
    outdir = Path(args.outdir)
    save_paths(outdir / f"paths_g{_gstr(args.gamma)}.npz", paths,
               gamma=float(args.gamma), seeds=np.arange(args.seed0, args.seed0 + args.M))
    row = summarize_paths(paths, env, args.gamma, args.method, args.reach)
    save_row(row, outdir / f"row_g{_gstr(args.gamma)}.json")
    print(json.dumps(row, indent=2), flush=True)


def policy_worker(args) -> None:
    dev = args.device
    pol, ck = HP.load_hp(args.ckpt, device=dev)
    env = GS.make_grid()
    _apply_wall_plugs_eval(env, getattr(args, 'wall_plugs', 0))
    # Start at +eps (free space) to MATCH training: the exact origin overlaps the corner plugs at
    # (0,-0.2)&(-0.2,0), so a policy trained with --start-eps must be evaluated the same way.
    _eps = float(getattr(args, 'start_eps', 0.0) or 0.0)
    if _eps > 0.0:
        env.x0 = torch.tensor([_eps, _eps, 0.0, 0.0], dtype=env.x0.dtype)
    rows, _, paths_by_g = SR.eval_policy(
        pol, env, gammas=[float(args.gamma)], M=args.M, T_max=args.T,
        reach=args.reach, temp=1.0, device=dev, seed0=args.seed0,
        keep_paths=args.M, log=print,
    )
    paths = paths_by_g[float(args.gamma)]
    outdir = Path(args.outdir)
    save_paths(outdir / f"paths_g{_gstr(args.gamma)}.npz", paths,
               gamma=float(args.gamma), ckpt=str(Path(args.ckpt).resolve()),
               seeds=np.arange(args.seed0, args.seed0 + args.M))
    row = summarize_paths(paths, env, args.gamma, args.method, args.reach,
                          args.iterations_to_goal)
    expected_sr = rows[float(args.gamma)]["SR"]
    if abs(row["SR"] - expected_sr) > 1e-12:
        raise RuntimeError(f"SR convention mismatch: evaluator={row['SR']} sr_cr_eval={expected_sr}")
    save_row(row, outdir / f"row_g{_gstr(args.gamma)}.json")
    print(json.dumps({"checkpoint_config": ck.get("config", {}), "row": row}, indent=2), flush=True)


def saved_worker(args) -> None:
    paths = load_paths(Path(args.paths))
    env = GS.make_grid()
    _apply_wall_plugs_eval(env, getattr(args, 'wall_plugs', 0))
    row = summarize_paths(paths, env, args.gamma, args.method, args.reach,
                          args.iterations_to_goal)
    outdir = Path(args.outdir)
    canonical = outdir / f"paths_g{_gstr(args.gamma)}.npz"
    if Path(args.paths).resolve() != canonical.resolve():
        save_paths(canonical, paths, gamma=float(args.gamma))
    save_row(row, outdir / f"row_g{_gstr(args.gamma)}.json")
    print(json.dumps(row, indent=2), flush=True)


def merge_paths(args) -> None:
    merged = []
    for source in args.inputs:
        merged.extend(load_paths(Path(source)))
    metadata = {"gamma": float(args.gamma), "merged_from": np.asarray(args.inputs)}
    save_paths(Path(args.out), merged, **metadata)
    print(f"merged {len(args.inputs)} files / {len(merged)} paths -> {args.out}", flush=True)


def _fmt_pm(row, mean_key, std_key, digits=3):
    a, b = row[mean_key], row[std_key]
    if not np.isfinite(a):
        return "—"
    return f"{a:.{digits}f} ± {b:.{digits}f}"


def write_tables(rows, prefix: Path, title: str) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    method_order = {"SafeMPPI": 0, "Flow-expanded": 1, "Kazuki-guidance": 2}
    rows = sorted(rows, key=lambda r: (method_order.get(r["method"], 99), r["method"], float(r["gamma"])))
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = [f"# {title}", "",
             "| Method | γ | SR | CR | Clearance (m) | Per-obstacle min (m) | Time (s) | Coverage | n success | M | Iters-to-goal |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        it = "—" if r.get("iterations_to_goal") is None else str(r["iterations_to_goal"])
        lines.append(
            f"| {r['method']} | {float(r['gamma']):.1f} | {r['SR']:.1%} | {r['CR']:.1%} | "
            f"{_fmt_pm(r, 'clearance_mean', 'clearance_std')} | "
            f"{_fmt_pm(r, 'per_obstacle_min_mean', 'per_obstacle_min_std')} | "
            f"{_fmt_pm(r, 'time_mean_s', 'time_std_s', 2)} | {r['coverage']} | "
            f"{r['n_success']} | {r['M']} | {it} |"
        )
    lines += ["", "Clearance is the successful-episode mean over time of the nearest-obstacle clearance; "
              "'Per-obstacle min' is the requested alternate interpretation (minimum over time for each obstacle, then mean over obstacles).",
              "Coverage is the empirical number of distinct staircase IDs among successful episodes; it is not normalized or hard-coded.", ""]
    prefix.with_suffix(".md").write_text("\n".join(lines))


def assemble(args) -> None:
    indir = Path(args.input_dir)
    rows = []
    for g in args.gammas:
        p = indir / f"row_g{_gstr(g)}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        rows.append(json.loads(p.read_text()))
    write_tables(rows, Path(args.table_prefix), args.title)
    print(f"wrote {args.table_prefix}.md and {args.table_prefix}.csv", flush=True)


def assemble_multi(args) -> None:
    rows = []
    for input_dir in args.input_dirs:
        indir = Path(input_dir)
        for g in args.gammas:
            p = indir / f"row_g{_gstr(g)}.json"
            if not p.exists():
                raise FileNotFoundError(p)
            rows.append(json.loads(p.read_text()))
    write_tables(rows, Path(args.table_prefix), args.title)
    print(f"wrote combined {args.table_prefix}.md and {args.table_prefix}.csv", flush=True)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gamma", type=float, required=True)
    common.add_argument("--M", type=int, default=100)
    common.add_argument("--reach", type=float, default=0.1)
    common.add_argument("--seed0", type=int, default=0)
    common.add_argument("--method", required=True)
    common.add_argument("--outdir", required=True)

    p = sub.add_parser("expert-worker", parents=[common])
    p.add_argument("--wall-plugs", type=int, choices=[0, 2, 4, 8], default=0)
    p.add_argument("--start-eps", type=float, default=0.0,
                   help="start at (eps,eps); required on the plugged scene (origin sits ON the corner plugs)")
    p.set_defaults(func=expert_worker)

    p = sub.add_parser("policy-worker", parents=[common])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--T", type=int, default=250)
    p.add_argument("--iterations-to-goal", type=int, default=None)
    p.add_argument("--wall-plugs", type=int, choices=[0, 2, 4, 8], default=0)
    p.add_argument("--start-eps", type=float, default=0.0,
                   help="start at (eps,eps) free-space offset; MUST match training on the plugged scene")
    p.set_defaults(func=policy_worker)

    p = sub.add_parser("saved-worker")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--paths", required=True)
    p.add_argument("--reach", type=float, default=0.1)
    p.add_argument("--method", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--iterations-to-goal", type=int, default=None)
    p.add_argument("--wall-plugs", type=int, choices=[0, 2, 4, 8], default=0)
    p.set_defaults(func=saved_worker)

    p = sub.add_parser("merge-paths")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=merge_paths)

    p = sub.add_parser("assemble")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--table-prefix", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    p.set_defaults(func=assemble)

    p = sub.add_parser("assemble-multi")
    p.add_argument("--input-dirs", nargs="+", required=True)
    p.add_argument("--table-prefix", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    p.set_defaults(func=assemble_multi)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
