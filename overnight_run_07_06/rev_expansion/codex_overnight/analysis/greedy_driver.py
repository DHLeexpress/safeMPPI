"""GREEDY per-iteration hill-climb from a checkpoint (user 2026-07-12): at each absolute iteration,
branch ONE training step under a grid of (beta, frontier-fraction), score each candidate's a-d on the
SAME fixed-seed episodes, and promote the config whose SR / CR / clearance / time ALL strictly improve.
If none strictly dominates, widen the beta sweep; if still none, take the best-effort step (max #metrics
improved, tie by composite) and record NO_STRICT so the user can inspect what changed.

Two swept knobs (user):
  beta      : tilt softmax temperature w=exp((sig-max)/beta) — HIGHER = flatter = exploit the policy.
  frontier% : fixed easy/frontier batch mix (phased curriculum OFF) — {12.5, 25, 50}% informative samples.

Each candidate is a full-state resume of the current checkpoint (identical optimizer + RNG), so the ONLY
difference between candidates is the two knobs (a clean paired A/B). Outputs a continuous canonical run
dir (ckpts + viz_db + probe.jsonl seeded from the base run's 0->40 history) + greedy_log.jsonl.

  python analysis/greedy_driver.py --start-ckpt results/p2/openscratch_base_s870/ckpt_40.pt \
      --base-run results/p2/openscratch_base_s870 --start-iter 40 --end-iter 50 --gpu 3 --M 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
PRE = os.path.join(P2, "..", "..", "results", "hp_repr", "pretrained_a32uni.pt")
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/dohyun/miniforge3/lib", OMP_NUM_THREADS="5")

BETAS_BASE = [0.3, 0.5]
BETAS_WIDE = [0.7, 1.0]
FRONTIER = [0.125, 0.25, 0.5]
N_CONC = 3


def trainer_cmd(cur_ckpt, outdir, beta, frontier, gpu):
    e = round(1.0 - frontier, 4)
    return ([
        "python", os.path.join(P2, "grid_expand_hardtail.py"),
        "--ckpt", cur_ckpt, "--outdir", outdir,
        "--iters", "1", "--seed", "870", "--lr", "2e-5", "--resume-allow-recipe-drift",
        "--rollouts-per-iter", "28", "--gather-attempt-cap", "600", "--batch", "64",
        "--valid-prog-floor", "0.15", "--min-rollouts", "1", "--traj-prog-min", "0",
        "--quantile-schedule", "0:0.50", "200:0.60", "400:0.70",
        "--mix-start", str(e), str(frontier), "--mix-end", str(e), str(frontier),
        "--beta", str(beta),
        "--early-until", "100", "--cooldown-from", "400",
        "--early-inner", "1", "--inner-steps", "1", "--cooldown-inner", "1",
        "--demo-frac", "0.125", "--lwf-eta", "0.05", "--teacher-ckpt", PRE,
        "--nfe-explore", "8", "--field-grad-clip", "1.0",
        "--max-functional-step", "999", "--max-anchor-drift", "999",
        "--targeted-frac", "0.5", "--n-target", "40", "--align-temp", "0.45",
        "--min-modes-per-gamma", "2", "--target-perp-brake",
        "--recovery-frac", "0.3",
        "--recovery-origin-band", "0.0", "1.0", "-0.05", "0.18", "0.0", "0.45", "-0.28", "0.05",
        "--recovery-goal-band", "4.3", "5.0", "4.6", "5.06", "-0.30", "0.30", "-0.05", "0.35",
        "--hard-quota", "12", "--hard-x0", "oob", "--hard-x0-cand", "64", "--strip-probe-every", "2",
        "--m-measure", "5", "--measure-every", "9999", "--probe-cov", "2", "--log-comp-every", "1",
        "--viz-db-every", "1", "--ckpt-every", "1", "--tag", os.path.basename(outdir),
    ], dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)))


def evaluate(ckpt, M, gpu):
    cmd = ["python", os.path.join(HERE, "greedy_eval.py"), "--ckpt", ckpt, "--M", str(M)]
    r = subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                       capture_output=True, text=True, timeout=600)
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    raise RuntimeError(f"eval failed for {ckpt}: {r.stderr[-500:]}")


def run_wave(cur_ckpt, step, combos, gpu, tmproot, log):
    """Train all combos (concurrent, capped) then eval each. Returns list of dicts."""
    out = []
    for i in range(0, len(combos), N_CONC):
        batch = combos[i:i + N_CONC]
        procs = []
        for (b, f) in batch:
            od = os.path.join(tmproot, f"cand_b{b}_f{f}")
            shutil.rmtree(od, ignore_errors=True)
            cmd, env = trainer_cmd(cur_ckpt, od, b, f, gpu)
            lf = open(os.path.join(tmproot, f"cand_b{b}_f{f}.log"), "w")
            procs.append((b, f, od, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)))
        for (b, f, od, p) in procs:
            p.wait()
        for (b, f, od, p) in procs:
            ck = os.path.join(od, f"ckpt_{step}.pt")
            if not os.path.exists(ck):
                log(f"  [warn] candidate b{b} f{f} produced no ckpt (skipped)")
                continue
            try:
                ad = evaluate(ck, M=EVAL_M, gpu=gpu)
            except Exception as ex:
                log(f"  [warn] eval failed b{b} f{f}: {ex}")
                continue
            ad.update(beta=b, frontier=f, ckpt=ck, outdir=od)
            out.append(ad)
            log(f"  cand b{b} f{f}: SR {ad['SR']:.3f} CR {ad['CR']:.3f} "
                f"clr {ad['clr']:.3f} time {ad['time']:.2f} cov {ad['cov']}")
    return out


def improved(c, base):
    """Per-metric strict-improvement booleans (a-d) on the 3-gamma POOL: SR up, CR down, clr up, time down."""
    return dict(SR=c["SR"] > base["SR"] + 1e-9, CR=c["CR"] < base["CR"] - 1e-9,
                clr=c["clr"] > base["clr"] + 1e-9, time=c["time"] < base["time"] - 1e-9)


def _cell_ok(cv, bv, up):
    """One (gamma,metric) cell improved? up=True -> higher better; NaN (no successes) never 'improves'."""
    try:
        if cv != cv or bv != bv:      # NaN guard (clr/time undefined when a gamma had no success)
            return 0, 0
    except Exception:
        return 0, 0
    if up:
        return (1 if cv > bv + 1e-9 else 0), (1 if cv < bv - 1e-9 else 0)
    return (1 if cv < bv - 1e-9 else 0), (1 if cv > bv + 1e-9 else 0)


def per_gamma_cells(c, base):
    """Net improvement across the 3 gamma x 4 metric grid (user: 'increase whole metrics of gamma
    0.1,0.5,1.0'). Returns (n_improved, n_regressed) over up to 12 cells."""
    cpg, bpg = c.get("per_gamma", {}), base.get("per_gamma", {})
    imp = reg = 0
    for g in ("0.1", "0.5", "1.0"):
        cg, bg = cpg.get(g), bpg.get(g)
        if not cg or not bg:
            continue
        for key, up in (("SR", True), ("CR", False), ("clr", True), ("time", False)):
            i, r = _cell_ok(cg.get(key), bg.get(key), up)
            imp += i; reg += r
    return imp, reg


def composite(c, base):
    """Scale-normalized net gain across the pooled a-d (final tiebreak)."""
    return ((c["SR"] - base["SR"]) - (c["CR"] - base["CR"])
            + (c["clr"] - base["clr"]) / 0.05 - (c["time"] - base["time"]) / 5.0)


def rank_key(c, base):
    """Whole-gamma ranking: (pooled-strict, net gamma-cells, composite). Higher is better."""
    pooled_strict = all(improved(c, base).values())
    imp, reg = per_gamma_cells(c, base)
    return (1 if pooled_strict else 0, imp - reg, composite(c, base))


def select(cands, base):
    """Winner = best rank_key; 'strict' = pooled a-d all improve AND net gamma-cells positive."""
    if not cands:
        return None, False
    w = max(cands, key=lambda c: rank_key(c, base))
    imp, reg = per_gamma_cells(w, base)
    strict = all(improved(w, base).values()) and (imp - reg) > 0
    return (w, True) if strict else (None, False)


def best_effort(cands, base):
    return max(cands, key=lambda c: rank_key(c, base))


def main():
    global EVAL_M
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-ckpt", required=True)
    ap.add_argument("--base-run", required=True, help="run dir to seed canonical history (viz_db, probe)")
    ap.add_argument("--start-iter", type=int, required=True)
    ap.add_argument("--end-iter", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--outdir", default=os.path.join(P2, "results/p2/greedy_gf_s870"))
    args = ap.parse_args()
    EVAL_M = args.M

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    tmproot = os.path.join(outdir, "_cand_tmp")
    os.makedirs(tmproot, exist_ok=True)
    logf = open(os.path.join(P2, "logs", "greedy_driver.log"), "a")

    def log(m):
        logf.write(m + "\n"); logf.flush(); print(m, flush=True)

    # seed canonical dir from the base run's 0->start_iter history (once)
    if not os.path.exists(os.path.join(outdir, "recipe.json")):
        for fn in ("recipe.json",):
            src = os.path.join(args.base_run, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(outdir, fn))
        os.makedirs(os.path.join(outdir, "viz_db"), exist_ok=True)
        vdb = os.path.join(args.base_run, "viz_db")
        if os.path.isdir(vdb):
            for f in os.listdir(vdb):
                if f.startswith("it") and f.endswith(".pt"):
                    try:
                        n = int(f[2:-3])
                    except ValueError:
                        continue
                    if n <= args.start_iter:
                        shutil.copy(os.path.join(vdb, f), os.path.join(outdir, "viz_db", f))
        pj = os.path.join(args.base_run, "probe.jsonl")
        if os.path.exists(pj):
            keep = [l for l in open(pj) if l.strip() and json.loads(l)["iter"] <= args.start_iter]
            open(os.path.join(outdir, "probe.jsonl"), "w").writelines(keep)
        shutil.copy(args.start_ckpt, os.path.join(outdir, f"ckpt_{args.start_iter}.pt"))
        log(f"[seed] canonical dir {outdir} seeded from {args.base_run} up to it{args.start_iter}")

    glog = open(os.path.join(outdir, "greedy_log.jsonl"), "a")
    cur = os.path.join(outdir, f"ckpt_{args.start_iter}.pt")
    if not os.path.exists(cur):
        shutil.copy(args.start_ckpt, cur)

    for step in range(args.start_iter + 1, args.end_iter + 1):
        t0 = time.time()
        base = evaluate(cur, M=EVAL_M, gpu=args.gpu)
        log(f"\n=== step it{step} | baseline SR {base['SR']:.3f} CR {base['CR']:.3f} "
            f"clr {base['clr']:.3f} time {base['time']:.2f} cov {base['cov']} ===")
        combos = list(itertools.product(BETAS_BASE, FRONTIER))
        cands = run_wave(cur, step, combos, args.gpu, tmproot, log)
        winner, strict = select(cands, base)
        widened = False
        if winner is None:
            log("  no strict domination in base grid -> widening beta sweep")
            widened = True
            cands += run_wave(cur, step, list(itertools.product(BETAS_WIDE, FRONTIER)),
                              args.gpu, tmproot, log)
            winner, strict = select(cands, base)
        if winner is None:
            winner = best_effort(cands, base)
            log(f"  NO STRICT improvement; best-effort b{winner['beta']} f{winner['frontier']} "
                f"(improved {sum(improved(winner, base).values())}/4)")
        # promote
        dst = os.path.join(outdir, f"ckpt_{step}.pt")
        shutil.copy(winner["ckpt"], dst)
        vsrc = os.path.join(winner["outdir"], "viz_db", f"it{step}.pt")
        if os.path.exists(vsrc):
            shutil.copy(vsrc, os.path.join(outdir, "viz_db", f"it{step}.pt"))
        psrc = os.path.join(winner["outdir"], "probe.jsonl")
        if os.path.exists(psrc):
            wl = [l for l in open(psrc) if l.strip() and json.loads(l)["iter"] == step]
            if wl:
                open(os.path.join(outdir, "probe.jsonl"), "a").write(wl[-1])
        _gimp, _greg = per_gamma_cells(winner, base)
        rec = dict(iter=step, chosen_beta=winner["beta"], chosen_frontier=winner["frontier"],
                   strict=strict, widened=widened, eval_M=EVAL_M, baseline=base,
                   winner={k: winner[k] for k in ("SR", "CR", "clr", "time", "cov", "per_gamma")},
                   improved=improved(winner, base),
                   gamma_cells=dict(improved=_gimp, regressed=_greg, net=_gimp - _greg),
                   delta=dict(SR=winner["SR"] - base["SR"], CR=winner["CR"] - base["CR"],
                              clr=winner["clr"] - base["clr"], time=winner["time"] - base["time"],
                              cov=winner["cov"] - base["cov"]),
                   all_candidates=[{k: c[k] for k in ("beta", "frontier", "SR", "CR", "clr", "time", "cov")}
                                   for c in cands],
                   secs=round(time.time() - t0, 1))
        glog.write(json.dumps(rec) + "\n"); glog.flush()
        tag = "STRICT" if strict else "best-effort"
        log(f"  -> it{step} promote b{winner['beta']} f{winner['frontier']} [{tag}] "
            f"SR {winner['SR']:.3f} CR {winner['CR']:.3f} clr {winner['clr']:.3f} "
            f"time {winner['time']:.2f} cov {winner['cov']} ({rec['secs']}s)")
        cur = dst

    log(f"\n[done] greedy hill-climb reached it{args.end_iter}")


if __name__ == "__main__":
    main()
