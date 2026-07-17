"""Portable TRUE-evaluation metrics + figures (integration/afe2-terminal-dualscene-v1).

Consumes a true_eval_run.py output directory (one scene profile). Everything is computed from the
RAW stored paths with one uniform code path and full provenance checks:

  SR        final point within reach of the goal AND never collided/OOB
  CR_obs    min obstacle clearance < 0 anywhere            (reported SEPARATELY from)
  CR_oob    left the workspace box                          (out-of-bounds rate)
  clr       success-conditional minimum obstacle clearance  (bootstrap 95% CI)
  time      success-conditional time to goal [s]            (bootstrap 95% CI)
  V_safe    taskspace AND SOCP@gamma on the executed trajectory
  V_full    V_safe AND approach/progress
Binary metrics carry Wilson 95% intervals. These are finite-M, single-training-seed intervals,
not a probabilistic safety guarantee.

Caching is CONTENT-keyed (scene sha256, checkpoint sha256, seed-list sha256, M, metric version) —
never pathname-only; a mismatch forces recomputation. Gallery panels show a
"pre-specified outcome-stratified, ratio-matched random subset": k = round(10*SR) successes and
10-k non-successes drawn uniformly without replacement under a fixed named seed; the chosen
archive indices are persisted. The Kazuki row is one gamma-blind batch (gamma_ctx=0.5), certified
per column at that column's gamma.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
for _p in (_WORK, _REV, _HERE):
    sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

METRIC_VERSION = "true_eval_v2_terminal"
EVAL_GAMMAS = (0.1, 0.3, 0.5, 1.0)
PLA = plt.get_cmap("plasma")
GCOL = {0.1: PLA(0.08), 0.3: PLA(0.38), 0.5: PLA(0.58), 1.0: PLA(0.85)}
SUBSET_LABEL = "pre-specified outcome-stratified, ratio-matched random subset (10 of 100)"
Z95 = 1.959963984540054


def named_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2 ** 63 - 1)


def wilson95(p, n):
    if n <= 0:
        return (0.0, 0.0)
    z = Z95
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


def bootstrap95(values, key, n_boot=2000):
    v = np.asarray(values, float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(named_seed("bootstrap", key))
    stats = [float(np.mean(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


class SceneCtx:
    """Environment + metric helpers rebuilt from the run's scene snapshot (never hardcoded)."""

    def __init__(self, eval_dir, profile_name):
        from afe2_scene_profiles import get_scene_profile, build_scene, scene_snapshot, \
            assert_scene_snapshot
        import grid_metrics2 as GM2
        snap = json.load(open(os.path.join(eval_dir, "scene_snapshot.json")))
        assert_scene_snapshot(snap)
        if snap["profile"]["name"] != profile_name:
            raise ValueError(
                f"eval dir scene {snap['profile']['name']!r} != --scene-profile {profile_name!r}")
        self.profile = get_scene_profile(profile_name)
        self.env = build_scene(self.profile)
        rebuilt = scene_snapshot(self.env, self.profile)
        if rebuilt["sha256"] != snap["sha256"]:
            raise ValueError("rebuilt scene disagrees with the stored snapshot")
        self.snapshot = snap
        GM2.GOAL_XY = np.asarray(self.profile.goal, dtype=float)
        self.GM2 = GM2
        self.goal = np.asarray(self.profile.goal, float)
        self.obs = self.env.obstacles.detach().cpu().numpy()
        self.rr = float(self.env.r_robot)

    def traj_metrics(self, p, gamma, reach):
        import grid_metrics as GM
        p = np.asarray(p, float)
        if self.obs.size:
            d = (np.linalg.norm(p[:, None, :] - self.obs[None, :, :2], axis=2)
                 - self.obs[None, :, 2] - self.rr)
            clr = float(d.min())
        else:
            clr = float("inf")
        oob = bool((p < -GM.EPS_TASK).any() or (p > GM.GRID_M + GM.EPS_TASK).any())
        hit = bool(clr < 0.0)
        success = bool(np.linalg.norm(p[-1] - self.goal) < reach) and not hit and not oob
        v2, st = self.GM2.traj_breakdown(p, self.env, float(gamma))
        return dict(success=success, hit=hit, oob=oob, clr=clr, steps=len(p) - 1,
                    v_full=bool(v2), v_safe=bool(st["taskspace"] and st["socp"]),
                    approach=bool(st["approach"]))


def load_cell(eval_dir, name):
    prov = json.load(open(os.path.join(eval_dir, f"{name}.provenance.json")))
    z = np.load(os.path.join(eval_dir, f"paths_{name}.npz"), allow_pickle=True)
    paths = list(z["paths"])
    if len(paths) != int(prov["M"]):
        raise RuntimeError(f"cell {name}: {len(paths)} paths != declared M {prov['M']}")
    return paths, prov


def content_key(prov):
    seeds_sha = hashlib.sha256(json.dumps(prov.get("seeds", [])).encode()).hexdigest()
    return hashlib.sha256("|".join([
        str(prov.get("scene_sha256")), str(prov.get("checkpoint_sha256")),
        seeds_sha, str(prov.get("M")), METRIC_VERSION,
    ]).encode()).hexdigest()


def cell_metrics(scene: SceneCtx, eval_dir, name, gamma, reach, dt):
    """Content-cached full-metric computation for one cell at one certification gamma."""
    cache_path = os.path.join(eval_dir, f"metrics_{name}_g{gamma}.json")
    paths, prov = load_cell(eval_dir, name)
    key = content_key(prov)
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        if cached.get("content_key") == key:
            return cached
    ms = [scene.traj_metrics(p, gamma, reach) for p in paths]
    n = len(ms)
    suc = [m for m in ms if m["success"]]
    sr = float(np.mean([m["success"] for m in ms]))
    hit = float(np.mean([m["hit"] for m in ms]))
    oob = float(np.mean([m["oob"] for m in ms]))
    vs = float(np.mean([m["v_safe"] for m in ms]))
    vf = float(np.mean([m["v_full"] for m in ms]))
    clr_v = [m["clr"] for m in suc]
    time_v = [m["steps"] * dt for m in suc]
    out = dict(content_key=key, cell=name, gamma=float(gamma), M=n,
               SR=sr, SR_ci=wilson95(sr, n),
               CR_obs=hit, CR_obs_ci=wilson95(hit, n),
               CR_oob=oob, CR_oob_ci=wilson95(oob, n),
               V_safe=vs, V_safe_ci=wilson95(vs, n),
               V_full=vf, V_full_ci=wilson95(vf, n),
               clr=float(np.mean(clr_v)) if clr_v else float("nan"),
               clr_ci=bootstrap95(clr_v, (name, gamma, "clr")),
               time=float(np.mean(time_v)) if time_v else float("nan"),
               time_ci=bootstrap95(time_v, (name, gamma, "time")),
               success_mask=[bool(m["success"]) for m in ms],
               ci_note="finite-M single-training-seed intervals; not a safety guarantee")
    # pre-specified outcome-stratified, ratio-matched random subset for the gallery
    mask = np.asarray(out["success_mask"], bool)
    k = int(round(10 * sr))
    rng = np.random.default_rng(named_seed("gallery", prov["scene_sha256"], name, gamma))
    si, fi = np.where(mask)[0], np.where(~mask)[0]
    pick = (list(rng.choice(si, min(k, len(si)), replace=False)) +
            list(rng.choice(fi, min(10 - k, len(fi)), replace=False)))
    out["gallery_indices"] = [int(i) for i in pick]
    out["gallery_rule"] = SUBSET_LABEL
    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


def draw_panel(ax, scene: SceneCtx, eval_dir, name, gamma, met, title=None, row_label=None):
    for o in scene.obs:
        ax.add_patch(plt.Circle((o[0], o[1]), o[2], color="#cccccc", zorder=1))
    ax.plot(*scene.profile.start, marker="s", color="k", ms=5, ls="", zorder=6)
    ax.plot(*scene.goal, marker="*", c="gold", mec="k", ms=12, ls="", zorder=6)
    paths, _ = load_cell(eval_dir, name)
    mask = np.asarray(met["success_mask"], bool)
    for i in met["gallery_indices"]:
        p = np.asarray(paths[i], float)
        ax.plot(p[:, 0], p[:, 1], "-", color=GCOL[gamma], lw=1.2, alpha=0.85, zorder=3)
        ax.plot(p[::4, 0], p[::4, 1], ".", color="k", ms=1.3, alpha=0.5, zorder=4)
        if not mask[i]:
            ax.plot(p[-1, 0], p[-1, 1], "x", color="#cc3311", ms=8, mew=2.2, zorder=6)
    ax.set_xlim(-0.35, 5.35); ax.set_ylim(-0.35, 5.35); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=14)
    if row_label:
        ax.set_ylabel(row_label, fontsize=12)
    t = "-" if not np.isfinite(met["time"]) else f"{met['time']:.1f}s"
    c = "-" if not np.isfinite(met["clr"]) else f"{met['clr']:.2f}"
    ax.text(0.02, 0.02,
            f"SR {met['SR']:.2f} [{met['SR_ci'][0]:.2f},{met['SR_ci'][1]:.2f}]\n"
            f"CRobs {met['CR_obs']:.2f} OOB {met['CR_oob']:.2f}\n"
            f"clr {c}  t {t}\nVsafe {met['V_safe']:.2f} Vfull {met['V_full']:.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(fc="white", ec="0.6", alpha=0.88))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-profile", required=True)
    ap.add_argument("--eval-dir", required=True, help="true_eval_run.py output directory")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--gammas", type=float, nargs="+", default=list(EVAL_GAMMAS))
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    scene = SceneCtx(args.eval_dir, args.scene_profile)
    dt = float(scene.env.dt)
    ed = args.eval_dir

    rows = [
        ("SafeMPPI oracle", lambda g: (f"expert_rNA_g{g}", g)),
        ("Pretrained (bare, ckpt_0)", lambda g: (f"policy_r0_g{g}", g)),
        (f"AFE2 round {args.rounds} (bare)", lambda g: (f"policy_r{args.rounds}_g{g}", g)),
        ("CFM-MPPI (gamma-blind, same pretrained)", lambda g: ("kazuki_rNA_gblind", g)),
    ]
    fig, axes = plt.subplots(4, len(args.gammas), figsize=(4.1 * len(args.gammas), 16.8))
    for ri, (rlab, cellf) in enumerate(rows):
        for ci, g in enumerate(args.gammas):
            name, cert_g = cellf(g)
            ax = axes[ri, ci]
            try:
                met = cell_metrics(scene, ed, name, cert_g, args.reach, dt)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "missing cell", ha="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            draw_panel(ax, scene, ed, name, g, met,
                       title=(f"γ = {g}" if ri == 0 else None),
                       row_label=(rlab if ci == 0 else None))
    fig.suptitle(f"TRUE evaluation — {scene.profile.name}; M=100 random rollouts per cell; "
                 f"panels: {SUBSET_LABEL}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}_gallery.{ext}", dpi=130)
    plt.close(fig)

    # per-round curves (CRN-paired across rounds), six panels
    keys = [("SR", "success rate SR"), ("CR_obs", "obstacle-collision rate"),
            ("CR_oob", "out-of-bounds rate"), ("clr", "success min clearance [m]"),
            ("time", "time to success [s]"), ("V_safe", "V_safe (solid) / V_full (dashed)")]
    fig, axs = plt.subplots(1, 6, figsize=(29, 4.4))
    for g in args.gammas:
        R = list(range(0, args.rounds + 1))
        series = {k: [] for k, _ in keys}
        vf = []
        cis = {k: [] for k in ("clr", "time")}
        for n in R:
            met = cell_metrics(scene, ed, f"policy_r{n}_g{g}", g, args.reach, dt)
            for k, _ in keys:
                series[k].append(met[k])
            vf.append(met["V_full"])
            for k in cis:
                cis[k].append(met[f"{k}_ci"])
        for ax, (k, _) in zip(axs, keys):
            ax.plot(R, series[k], "-o", color=GCOL[g], lw=1.8, ms=4, label=f"γ={g}")
            if k in cis:
                lo = [c[0] for c in cis[k]]; hi = [c[1] for c in cis[k]]
                ax.fill_between(R, lo, hi, color=GCOL[g], alpha=0.12, linewidth=0)
        axs[5].plot(R, vf, "--", color=GCOL[g], lw=1.1, alpha=0.75)
    for ax, (k, lab) in zip(axs, keys):
        ax.set_xlabel("round"); ax.set_title(lab, fontsize=11); ax.grid(alpha=.3)
        if k in ("SR", "CR_obs", "CR_oob", "V_safe"):
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    fig.suptitle(f"Bare-policy metrics per AFE2 round — {scene.profile.name}; M=100/cell; "
                 "common random numbers keyed by (γ, rollout index); Wilson/bootstrap 95% CIs "
                 "(finite-M, single training seed)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}_curves.{ext}", dpi=140)
    print(f"wrote {args.out_prefix}_gallery/.curves (png+pdf)")


if __name__ == "__main__":
    main()
