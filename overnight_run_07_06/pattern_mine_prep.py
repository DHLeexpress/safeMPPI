"""Consolidate ALL schedule-sweep arms into one analysis dataset for pattern mining.

Key framing (user): beta_fracs / early_frac / cooldown_frac are FRACTIONS of total iters, so arms with
different horizons (5k vs the 8k a32_unf_long) experienced different ABSOLUTE schedules at fixed batch=64.
This dataset therefore records, per arm: schedule metadata (incl. total T), the full measurement series
(SR/CR aggregate + per-gamma + oob + gdist + pools + n_pos + online SR/CR + covered), warm-up clear info,
and derived absolute-iter quantities (beta(t) events, frontier-portion crossings, lr-drop iter, cumulative
frontier-sample exposure). Output: results/pattern_mine/dataset.json (+ flat arms.csv).

Noise calibration built in:
  - a32_unf (helios) vs S1_BASE (nyx) = SAME recipe+seed, different machine -> nondeterminism band.
  - S1_M1 vs X_M1s1 = same recipe, seed 0 vs 1 -> seed band.
  - per-measure SR se ~ sqrt(p(1-p)/700) ~ 0.019 (7 gammas x M=100).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "pattern_mine")
os.makedirs(OUT, exist_ok=True)

# name -> (rel_dir, meta). meta: T, beta:(kind, fracs, steps, hi, lo), early, cooldown,
# mix_start, mix_end, lr0, inner, demo, eta, seed, note
D = lambda **kw: kw
REG = {
    "BASELINE":  ("results/sweep_overnight/a32_unf",      D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="winner recipe")),
    "S1_BASE":   ("results/stage1_nyx/S1_BASE",           D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="nyx", note="EXACT rerun of BASELINE on nyx -> nondeterminism probe")),
    "S1_BASEs1": ("results/stage1_nyx/S1_BASEs1",         D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=1, machine="nyx", note="baseline seed1 -> seed probe")),
    "A_hi":      ("results/sweep_overnight/a32_unf_hi",   D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.40, eta=.10, seed=0, machine="helios", note="heavier anchor d.4/e.1")),
    "A1":        ("results/sweep_ac/A1",                  D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=5e-5, inner=8, demo=.25, eta=.05, seed=0, machine="helios", note="lr5e-5 + inner8 jointly")),
    "B_long8k":  ("results/sweep_overnight/a32_unf_long", D(T=8000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="SAME fractional schedule stretched to 8k -> true-iter-vs-fraction witness (LOG-parsed, killed ~5.5k)")),
    "C1":        ("results/sweep_ac/C1",                  D(T=5000, beta=("exp",None,None), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="beta smooth exp 1->.1")),
    "C2":        ("results/sweep_ac/C2",                  D(T=5000, beta=("aggr",None,None), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="nyx", note="beta aggressive: .1 by frac .5")),
    "C3":        ("results/sweep_ac/C3",                  D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.55,.30,.15), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="frontier .15 from it0")),
    "S1_B1":     ("results/stage1/S1_B1",                 D(T=5000, beta=("step",(0,.15,.3,.5),(1,.5,.2,.1)),  early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="beta EARLY-greedy")),
    "S1_B2":     ("results/stage1/S1_B2",                 D(T=5000, beta=("step",(0,.35,.6,.85),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="beta protective/late")),
    "S1_B3":     ("results/stage1/S1_B3",                 D(T=5000, beta=("step",(0,.25,.5,.75),(1,.6,.35,.2)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="beta never-greedy floor .2")),
    "S1_M1":     ("results/stage1/S1_M1",                 D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.05, cooldown=.5, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="mix EARLY ramp (NB: cooldown .5 may also move lr drop!)")),
    "S1_M2":     ("results/stage1/S1_M2",                 D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.2, cooldown=.9, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="mix LATE ramp")),
    "S1_M3":     ("results/stage1/S1_M3",                 D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.1, cooldown=.75, mix_start=(.7,.3,0), mix_end=(.25,.30,.45), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="frontier-heavy END .45")),
    "X_B2M1":    ("results/stage2/X_B2M1",                D(T=5000, beta=("step",(0,.35,.6,.85),(1,.5,.2,.1)), early=.05, cooldown=.5, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="B2 x M1")),
    "X_B2M2":    ("results/stage2/X_B2M2",                D(T=5000, beta=("step",(0,.35,.6,.85),(1,.5,.2,.1)), early=.2, cooldown=.9, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="B2 x M2")),
    "X_B3M1":    ("results/stage2/X_B3M1",                D(T=5000, beta=("step",(0,.25,.5,.75),(1,.6,.35,.2)), early=.05, cooldown=.5, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="B3 x M1")),
    "X_B3M2":    ("results/stage2/X_B3M2",                D(T=5000, beta=("step",(0,.25,.5,.75),(1,.6,.35,.2)), early=.2, cooldown=.9, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="B3 x M2")),
    "X_M1s1":    ("results/stage2/X_M1s1",                D(T=5000, beta=("step",(0,.25,.5,.75),(1,.5,.2,.1)), early=.05, cooldown=.5, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=1, machine="helios", note="M1 seed1 -> seed probe")),
    "X_C2M1":    ("results/stage2/X_C2M1",                D(T=5000, beta=("aggr",None,None), early=.05, cooldown=.5, mix_start=(.7,.3,0), mix_end=(.34,.33,.33), lr0=1e-4, inner=4, demo=.25, eta=.05, seed=0, machine="helios", note="aggressive beta x M1")),
}

_MEAS = re.compile(r"^it(\d+) SR ([\d.]+) CR ([\d.]+) \| loss (nan|[\d.]+) gRMS\(fld ([\d.]+) enc ([\d.]+)\)"
                   r" \| \S+ ([\d.]+) mix ([\d.]+)/([\d.]+)/([\d.]+) pools (\d+)/(\d+)/(\d+) npos (\d+)"
                   r" \| on\(SR ([\d.]+) CR ([\d.]+)\)")
_WARM = re.compile(r"^it(\d+) WARMUP cleared after (\d+) deploys: online reached ([\d.]+) collided ([\d.]+).*?n_pos (\d+)")


def parse_log_series(path):
    recs = []
    for line in open(path):
        m = _MEAS.match(line)
        if m:
            recs.append(dict(iter=int(m[1]), SR=float(m[2]), CR=float(m[3]),
                             loss=None if m[4] == "nan" else float(m[4]),
                             field_grad_rms=float(m[5]), enc_grad_rms=float(m[6]), beta=float(m[7]),
                             mix=[float(m[8]), float(m[9]), float(m[10])],
                             n_easy=int(m[11]), n_mid=int(m[12]), n_frontier=int(m[13]), n_pos=int(m[14]),
                             online_SR=float(m[15]), online_CR=float(m[16]), lr=None, rows=None,
                             gdist=None, covered=None))
    return recs


def parse_warmup(logpath):
    if not os.path.exists(logpath):
        return None
    for line in open(logpath):
        m = _WARM.match(line)
        if m:
            return dict(clear_iter=int(m[1]), deploys=int(m[2]), online_SR=float(m[3]),
                        online_CR=float(m[4]), n_pos=int(m[5]))
    return None


def clean(r):
    out = dict(r)
    for k, v in list(out.items()):
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
    return out


def derive(series, meta):
    """Absolute-iter derived quantities."""
    T = meta["T"]
    it = [r["iter"] for r in series]
    sr = [r["SR"] for r in series]
    cr = [r["CR"] for r in series]
    fr = [(r["mix"][2] if r.get("mix") else None) for r in series]
    beta = [r.get("beta") for r in series]
    lr = [r.get("lr") for r in series]
    d = {}
    d["last3_SR"] = sum(sr[-3:]) / min(3, len(sr))
    d["last3_CR"] = sum(cr[-3:]) / min(3, len(cr))
    ib = max(range(len(sr)), key=lambda i: sr[i])
    d["best_SR"], d["best_iter"], d["CR_at_best"] = sr[ib], it[ib], cr[ib]
    d["final_SR"], d["final_CR"], d["last_iter"] = sr[-1], cr[-1], it[-1]
    # absolute beta events: iters where measured beta decreased
    d["beta_drops"] = [dict(iter=it[i], frm=beta[i - 1], to=beta[i]) for i in range(1, len(beta))
                       if beta[i] is not None and beta[i - 1] is not None and beta[i] < beta[i - 1] - 1e-9]
    d["iters_at_beta1"] = next((it[i] for i in range(len(beta)) if beta[i] is not None and beta[i] < 0.999),
                               it[-1])
    # frontier-portion crossings (absolute)
    for thr in (0.10, 0.15, 0.25, 0.32):
        d[f"iter_frontier_ge_{thr}"] = next((it[i] for i in range(len(fr)) if fr[i] is not None and fr[i] >= thr), None)
    # cumulative frontier exposure in SAMPLES: sum over intervals portion*di*batch*inner (approx trapezoid)
    batch, inner = 64, meta["inner"]
    cum, cums = 0.0, [0.0]
    for i in range(1, len(it)):
        f0 = fr[i - 1] or 0.0
        f1 = fr[i] or 0.0
        cum += 0.5 * (f0 + f1) * (it[i] - it[i - 1]) * batch * inner
        cums.append(cum)
    d["cum_frontier_samples"] = cums          # aligned with series
    d["total_frontier_samples"] = cum
    # lr drop (from measured lr when available)
    d["lr_drop_iter"] = next((it[i] for i in range(len(lr)) if lr[i] is not None and lr[i] < meta["lr0"] * 0.99), None)
    d["schedule_frac_axis"] = [x / T for x in it]
    return d


def main():
    ds = {"noise_probes": {
        "same_seed_diff_machine": ["BASELINE", "S1_BASE"],
        "diff_seed": [["S1_BASE", "S1_BASEs1"], ["S1_M1", "X_M1s1"]],
        "per_measure_SR_se": 0.019, "note": "7 gammas x M=100 per measure"},
        "arms": {}}
    for name, (rel, meta) in REG.items():
        base = os.path.join(HERE, rel)
        h = os.path.join(base, "history.json")
        lg = os.path.join(os.path.dirname(base), "logs", os.path.basename(base) + ".log")
        if os.path.exists(h):
            series = [clean(r) for r in json.load(open(h))]
            src = "history.json"
        elif os.path.exists(lg):
            series = parse_log_series(lg)
            src = "log-parse"
        else:
            print(f"[prep] MISSING {name} ({rel})")
            continue
        vdb = sorted(f for f in (os.listdir(os.path.join(base, "viz_db")) if os.path.isdir(os.path.join(base, "viz_db")) else []))
        ds["arms"][name] = dict(meta=meta, source=src, dir=rel,
                                warmup=parse_warmup(lg), series=series, derived=derive(series, meta),
                                viz_db=[os.path.join(rel, "viz_db", f) for f in vdb])
        print(f"[prep] {name:10} {src:12} rows {len(series):3} warmup {ds['arms'][name]['warmup']} "
              f"last3 SR {ds['arms'][name]['derived']['last3_SR']:.3f} CR {ds['arms'][name]['derived']['last3_CR']:.3f}")
    out = os.path.join(OUT, "dataset.json")
    json.dump(ds, open(out, "w"))
    # flat per-arm CSV for quick correlation work
    with open(os.path.join(OUT, "arms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "T", "beta_kind", "iters_at_beta1", "n_beta_drops", "early", "cooldown",
                    "mix_start_frontier", "mix_end_frontier", "lr0", "lr_drop_iter", "inner", "demo", "eta",
                    "seed", "machine", "warmup_clear_iter", "warmup_online_CR",
                    "iter_frontier_ge_0.15", "iter_frontier_ge_0.32", "total_frontier_samples",
                    "last3_SR", "last3_CR", "best_SR", "best_iter", "final_SR", "final_CR"])
        for n, a in ds["arms"].items():
            m, d, wu = a["meta"], a["derived"], a["warmup"] or {}
            w.writerow([n, m["T"], m["beta"][0], d["iters_at_beta1"], len(d["beta_drops"]), m["early"],
                        m["cooldown"], m["mix_start"][2], m["mix_end"][2], m["lr0"], d["lr_drop_iter"],
                        m["inner"], m["demo"], m["eta"], m["seed"], m["machine"],
                        wu.get("clear_iter"), wu.get("online_CR"),
                        d["iter_frontier_ge_0.15"], d["iter_frontier_ge_0.32"],
                        round(d["total_frontier_samples"]),
                        round(d["last3_SR"], 4), round(d["last3_CR"], 4), d["best_SR"], d["best_iter"],
                        d["final_SR"], d["final_CR"]])
    print(f"[prep] -> {out} + arms.csv ({len(ds['arms'])} arms)")


if __name__ == "__main__":
    main()
