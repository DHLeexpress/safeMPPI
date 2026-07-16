"""it10 walled reporter (2026-07-14): score a checkpoint against the demo expert on a-d + coverage.

For a checkpoint, eval all 7 gammas on the 8-plug WALLED scene (start-eps 0.05, reach 0.1) via
eval_ae.py policy-worker (parallel), then compare each gamma to the AUTHORITATIVE expert row
(results/expert_gt, kept as-is / open-scene) on the a-e requirements:

  a SR==1.0   b CR==0.0   c clearance>expert(safer)   d time<expert(faster)   e coverage>expert (optional)

Prints a compact per-gamma scorecard + the pooled 'four metrics + coverage' and an a-d pass count, and
writes <eval-dir>/scorecard.json.  This is the number that decides 'did the frontier curriculum beat the
demo for free'.

  python analysis/report_at.py --ckpt results/p2/fsw_b03/final.pt --tag fsw_b03_it10 --M 50
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _gstr(g):
    return str(float(g))


def eval_one(ckpt, g, M, outdir, wall_plugs, start_eps, reach, gpu):
    """Run eval_ae policy-worker for one gamma -> row_g{g}.json. Returns (g, ok, err)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4")
    ckpt = Path(ckpt).resolve()
    outdir = Path(outdir).resolve()
    cmd = [sys.executable, str(P2 / "eval_ae.py"), "policy-worker",
           "--ckpt", str(ckpt), "--gamma", str(g), "--M", str(M), "--reach", str(reach),
           "--seed0", "0", "--method", "Flow-expanded", "--outdir", str(outdir),
           "--wall-plugs", str(wall_plugs), "--start-eps", str(start_eps)]
    try:
        r = subprocess.run(cmd, env=env, cwd=str(P2), capture_output=True, text=True, timeout=1800)
        ok = (outdir / f"row_g{_gstr(g)}.json").exists() and r.returncode == 0
        return (g, ok, r.stderr[-400:] if not ok else "")
    except Exception as e:  # noqa: BLE001
        return (g, False, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    ap.add_argument("--wall-plugs", type=int, default=8)
    ap.add_argument("--start-eps", type=float, default=0.05)
    ap.add_argument("--reach", type=float, default=0.15,
                    help="eval reach; 0.15 (slightly > expert's 0.1) because the goal-corner plugs "
                         "(5.2,5.0)&(5.0,5.2) sit ON the goal surface, so the exact 0.1-zone is plug-blocked")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expert-dir", default=str(P2 / "results/expert_gt"))
    ap.add_argument("--outdir", default="")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="score already-written row_g*.json files without rerunning policy workers")
    args = ap.parse_args()

    outdir = (Path(args.outdir) if args.outdir else
              (P2 / "results/p2" / f"eval_{args.tag}")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    expert_dir = Path(args.expert_dir)

    # 1) parallel per-gamma walled eval
    if args.reuse_existing:
        results = [(g, (outdir / f"row_g{_gstr(g)}.json").exists(), "missing existing row")
                   for g in args.gammas]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(eval_one, args.ckpt, g, args.M, outdir, args.wall_plugs,
                              args.start_eps, args.reach, args.gpu) for g in args.gammas]
            results = [f.result() for f in futs]
    for g, ok, err in results:
        if not ok:
            print(f"[WARN] gamma {g} eval FAILED: {err}", file=sys.stderr)

    # 2) scorecard vs expert
    rows, exp = {}, {}
    for g in args.gammas:
        cp = outdir / f"row_g{_gstr(g)}.json"
        ep = expert_dir / f"row_g{_gstr(g)}.json"
        if cp.exists():
            rows[g] = json.loads(cp.read_text())
        if ep.exists():
            exp[g] = json.loads(ep.read_text())

    print(f"\n=== {args.tag}  walled(plug{args.wall_plugs},eps{args.start_eps},reach{args.reach}) "
          f"M{args.M}  vs expert_gt ===")
    hdr = f"{'g':>4} | {'SR':>5} {'CR':>5} | {'clr':>6} {'exp':>6} {'c✓':>3} | {'time':>6} {'exp':>6} {'d✓':>3} | {'cov':>4} {'exp':>4} | {'a-d':>3}"
    print(hdr); print("-" * len(hdr))
    per_g = {}
    tot_pass = 0
    for g in args.gammas:
        if g not in rows:
            print(f"{g:>4} |  eval MISSING"); continue
        c = rows[g]; e = exp.get(g, {})
        SR, CR = c["SR"], c["CR"]
        clr, tim, cov = c["clearance_mean"], c["time_mean_s"], c["coverage"]
        eclr = e.get("clearance_mean", float("nan")); etim = e.get("time_mean_s", float("nan"))
        ecov = e.get("coverage", -1)
        a = bool(SR == 1.0)
        b = bool(CR == 0.0)
        cc = bool(np.isfinite(clr) and np.isfinite(eclr) and clr > eclr)
        dd = bool(np.isfinite(tim) and np.isfinite(etim) and tim < etim)
        ee = bool(cov > ecov)
        npass = int(a) + int(b) + int(cc) + int(dd)
        tot_pass += npass
        per_g[_gstr(g)] = dict(SR=SR, CR=CR, clr=clr, exp_clr=eclr, time=tim, exp_time=etim,
                               cov=cov, exp_cov=ecov, a=a, b=b, c=cc, d=dd, e=ee, ad_pass=npass)

        def _f(x, w=6, p=3):
            return (f"{x:>{w}.{p}f}" if (x is not None and np.isfinite(x)) else f"{'—':>{w}}")
        print(f"{g:>4} | {SR:>5.2f} {CR:>5.2f} | {_f(clr)} {_f(eclr)} {('Y' if cc else '.'):>3} | "
              f"{_f(tim,6,2)} {_f(etim,6,2)} {('Y' if dd else '.'):>3} | {cov:>4} {ecov:>4} | {npass:>3}/4")

    n_g = len([g for g in args.gammas if g in rows])
    pooled = dict(
        SR=float(np.mean([rows[g]["SR"] for g in rows])),
        CR=float(np.mean([rows[g]["CR"] for g in rows])),
        clr=float(np.nanmean([rows[g]["clearance_mean"] for g in rows])),
        time=float(np.nanmean([rows[g]["time_mean_s"] for g in rows])),
        cov_sum=int(sum(rows[g]["coverage"] for g in rows)),
        ad_pass=tot_pass, ad_max=4 * n_g,
        gammas_all_ad=int(sum(1 for g in per_g.values() if g["ad_pass"] == 4)),
    )
    print("-" * len(hdr))
    print(f"POOLED SR {pooled['SR']:.2f}  CR {pooled['CR']:.2f}  clr {pooled['clr']:.3f}  "
          f"time {pooled['time']:.2f}  covΣ {pooled['cov_sum']}  |  a-d {tot_pass}/{pooled['ad_max']}  "
          f"| gammas fully-winning(a-d) {pooled['gammas_all_ad']}/{n_g}")

    summary = dict(tag=args.tag, ckpt=str(args.ckpt), M=args.M, wall_plugs=args.wall_plugs,
                   start_eps=args.start_eps, reach=args.reach, per_gamma=per_g, pooled=pooled)
    (outdir / "scorecard.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(f"wrote {outdir/'scorecard.json'}")


if __name__ == "__main__":
    main()
