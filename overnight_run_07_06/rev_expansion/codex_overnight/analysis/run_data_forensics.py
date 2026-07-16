#!/usr/bin/env python3
"""Reproduce the P2 run-data forensic tables without modifying training code.

Run from codex_overnight with:
  LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 \
    python analysis/run_data_forensics.py

Outputs are written beside this script.  The analysis intentionally reads every
P2 probe/history/checkpoint header, then performs the expensive vector-field and
late-context sampler checks only on the two full final-unit trajectories.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
P2 = ROOT / "results" / "p2"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent.parent))


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _gcount(mapping, gamma):
    return sum(int(v) for k, v in (mapping or {}).items()
               if abs(float(k) - gamma) < 1e-4)


def collect_run_summary():
    rows = []
    for p in sorted(P2.glob("*/probe.jsonl")):
        recs = _jsonl(p)
        if not recs:
            continue
        d = pd.DataFrame(recs).sort_values("iter")
        pr = d[d.get("sr50", pd.Series(index=d.index, dtype=float)).notna()]
        row = dict(run=p.parent.name, probe_rows=len(d), iter_first=int(d["iter"].min()),
                   iter_last=int(d["iter"].max()))
        if len(pr):
            row.update(sr50_first=float(pr.iloc[0].sr50), sr50_last=float(pr.iloc[-1].sr50),
                       sr50_min=float(pr.sr50.min()), sr50_max=float(pr.sr50.max()),
                       cr50_last=float(pr.iloc[-1].cr50), cov50_last=float(pr.iloc[-1].cov50),
                       cov50_best_safe=float(pr.loc[pr.cr50 == 0, "cov50"].max())
                       if (pr.cr50 == 0).any() else np.nan)
        for c in ("near0_e", "w2_e", "fld", "enc", "rid_n", "rid_dom", "vr", "att"):
            if c in d:
                row[c + "_mean"] = float(pd.to_numeric(d[c], errors="coerce").mean())
        for gamma in GAMMAS:
            row[f"g{gamma}_windows"] = sum(_gcount(r.get("gamma_counts"), gamma) for r in recs)
            row[f"g{gamma}_valid_rollouts"] = sum(
                _gcount(r.get("gamma_rollout_counts"), gamma) for r in recs)
            row[f"g{gamma}_attempts"] = sum(
                _gcount(r.get("gamma_attempt_counts"), gamma) for r in recs)
        row["gamma_ready_frac"] = np.mean([bool(r.get("gamma_ready", False)) for r in recs])
        hp = p.parent / "history.json"
        if hp.exists():
            try:
                h = json.loads(hp.read_text())
                row["history_rows"] = len(h)
                if h:
                    row.update(history_last_iter=h[-1].get("iter"), history_last_SR=h[-1].get("SR"),
                               history_last_CR=h[-1].get("CR"))
            except Exception:
                pass
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("run")
    df.to_csv(OUT / "forensics_run_summary.csv", index=False)
    return df


def _stair_id(path):
    import grid_metrics as GM
    return GM.staircase_id(path)


def collect_finalunit_viz():
    rows = []
    for seed in (15, 16):
        run = P2 / f"finalunit_q50_k14_s{seed}_from_it18"
        probes = {int(r["iter"]): r for r in _jsonl(run / "probe.jsonl")}
        for p in sorted((run / "viz_db").glob("it*.pt"),
                        key=lambda x: int(x.stem[2:])):
            d = torch.load(p, map_location="cpu", weights_only=False)
            labels = np.asarray(d["label"]); front = labels == "frontier"
            sigma = np.asarray(d["sigma"]); margin = np.asarray(d["margin"])
            prog = np.asarray(d["prog"]); low = d["low5"].numpy(); U = d["U"].numpy()
            pos = np.array([5.0, 5.0]) - 5.0 * low[:, :2]
            near = np.linalg.norm(pos, axis=1) < 1.0
            high_s = sigma >= float(d["sigma_plane"])
            low_m = margin <= float(d["margin_plane"])
            high_p = prog >= float(d["prog_plane"])
            ids = [_stair_id(path) for path in d.get("paths", [])]
            counts = collections.Counter(x for x in ids if x is not None)
            nids = sum(counts.values())
            late = np.linalg.norm(pos - np.array([5.0, 5.0]), axis=1) < 1.0
            udiff = U.sum(1)[:, 0] - U.sum(1)[:, 1]
            a0diff = U[:, 0, 0] - U[:, 0, 1]
            r = dict(run=run.name, seed=seed, iter=int(d["iter"]), n=len(front),
                     frontier_frac=float(front.mean()), sigma_pass=float(high_s.mean()),
                     margin_pass=float(low_m.mean()), progress_pass=float(high_p.mean()),
                     sigma_margin_pass=float((high_s & low_m).mean()),
                     sigma_progress_pass=float((high_s & high_p).mean()),
                     margin_progress_pass=float((low_m & high_p).mean()),
                     near_base=float(near.mean()), near_sigma=float(near[high_s].mean()),
                     near_progress=float(near[high_p].mean()), near_frontier=float(near[front].mean()),
                     sigma_near=float(sigma[near].mean()), sigma_far=float(sigma[~near].mean()),
                     frontier_widx=float(np.asarray(d["widx"])[front].mean()),
                     margin_nonpositive=float((margin <= 1e-12).mean()),
                     margin_negative=float((margin < -1e-8).mean()),
                     frontier_margin_nonpositive=float((margin[front] <= 1e-12).mean()),
                     frontier_margin_negative=float((margin[front] < -1e-8).mean()),
                     sigma_progress_corr=float(np.corrcoef(sigma, prog)[0, 1])
                     if np.std(sigma) and np.std(prog) else np.nan,
                     n_path_ids=len(counts), path_mode_dom=max(counts.values()) / nids if nids else np.nan,
                     all_U_x_minus_y=float(udiff.mean()),
                     frontier_U_x_minus_y=float(udiff[front].mean()),
                     late_a0_x_minus_y=float(a0diff[late].mean()),
                     late_U_x_minus_y=float(udiff[late].mean()),
                     gamma01_frac=float(np.isclose(d["gamma"].numpy(), 0.1).mean()))
            pr = probes.get(r["iter"], {})
            for k in ("sr50", "cr50", "cov50", "fld", "enc", "near0_e", "rid_n", "rid_dom"):
                r[k] = pr.get(k)
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["seed", "iter"])
    df.to_csv(OUT / "forensics_finalunit_viz.csv", index=False)
    return df


def collect_checkpoint_headers():
    rows = []
    for p in sorted(P2.glob("*/*.pt")):
        if p.parent.name == "viz_db":
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:
            rows.append(dict(path=str(p.relative_to(ROOT)), error=repr(exc)))
            continue
        if not isinstance(d, dict) or "state_dict" not in d:
            continue
        row = dict(path=str(p.relative_to(ROOT)), run=p.parent.name, file=p.name,
                   iter=d.get("iter"), SR=d.get("SR"), CR=d.get("CR"),
                   SR50=d.get("SR50"), CR50=d.get("CR50"), coverage50=d.get("coverage50"))
        sd = d["state_dict"]
        row["param_l2"] = math.sqrt(sum(float((x.float() ** 2).sum()) for x in sd.values()
                                         if torch.is_tensor(x)))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["run", "file"])
    df.to_csv(OUT / "forensics_checkpoint_headers.csv", index=False)
    return df


def _fixed_sample(policy, G, L, H, x0, K=128, nfe=8):
    with torch.no_grad():
        ctx = policy.ctx_from(G, L, H).repeat_interleave(K, 0)
        x = x0.clone()
        for i in range(nfe):
            tau = torch.full((len(x),), i / nfe)
            x = x + policy(x, tau, ctx) / nfe
    return x.reshape(len(G), K, 10, 2).clamp(-1, 1).numpy()


def collect_field_checks():
    import grid_hp_expt as HP
    db = torch.load(P2 / "finalunit_q50_k14_s15_from_it18/viz_db/it19.pt",
                    map_location="cpu", weights_only=False)
    low = db["low5"].numpy(); pos = np.array([5.0, 5.0]) - 5.0 * low[:, :2]
    late_idx = np.where(np.linalg.norm(pos - np.array([5.0, 5.0]), axis=1) < 1.0)[0]
    late_idx = late_idx[np.linspace(0, len(late_idx) - 1, min(64, len(late_idx))).astype(int)]
    G, L = db["grid"][late_idx], db["low5"][late_idx]
    H = torch.zeros(len(late_idx), 16, 2)
    K = 128
    x0 = torch.randn(len(late_idx) * K, 20, generator=torch.Generator().manual_seed(120))
    checkpoints = [("start18", P2 / "balanced_k14_s7_from_it15/probe_best.pt")]
    for seed in (15, 16):
        run = P2 / f"finalunit_q50_k14_s{seed}_from_it18"
        checkpoints += [(f"s{seed}_it{it}", run / f"ckpt_{it}.pt") for it in range(20, 101, 10)]
    rows = []
    for name, p in checkpoints:
        policy, _ = HP.load_hp(str(p), device="cpu"); policy.eval()
        U = _fixed_sample(policy, G, L, H, x0, K=K)
        a0 = U[:, :, 0]; full = U.sum(2)
        ad = a0[:, :, 0] - a0[:, :, 1]
        fd = full[:, :, 0] - full[:, :, 1]
        rows.append(dict(checkpoint=name, path=str(p.relative_to(ROOT)),
                         late_a0_x_minus_y=float(ad.mean()), late_full_x_minus_y=float(fd.mean()),
                         late_y_dominant_frac=float((ad < 0).mean()),
                         first_full_corr=float(np.corrcoef(ad.ravel(), fd.ravel())[0, 1]),
                         mean_a0_x=float(a0[:, :, 0].mean()), mean_a0_y=float(a0[:, :, 1].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "forensics_field_checks.csv", index=False)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    runs = collect_run_summary()
    viz = collect_finalunit_viz()
    ck = collect_checkpoint_headers()
    field = collect_field_checks()
    print(f"wrote {len(runs)} run, {len(viz)} finalunit-viz, {len(ck)} checkpoint, "
          f"and {len(field)} field rows to {OUT}")


if __name__ == "__main__":
    main()
