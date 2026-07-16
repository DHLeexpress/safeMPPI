"""Parallel off-diagonal DR data generation across GPU0+GPU1 (user 2026-07-06, "full GPU, full workers").

Shards each gamma's seed range into C chunks and runs every (gamma, chunk) shard as a concurrent
single-core (OMP=1) worker process, round-robined across the given GPUs. Each shard uses a DISJOINT
--s0 seed offset so trajectories never duplicate. When all workers finish, the chunk shards are merged
per gamma into dataset/<prefix>windows_g<g>.pt and the shard files are removed.

Usage (defaults = the 07/06 off-diagonal spec):
    python dr05_parallel.py                       # 7 gammas x 500 seeds, offdiag 0.5, margin 0.05
    python dr05_parallel.py --chunks 10 --gpus 0 1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
LOGD = os.path.join(HERE, "results", "dr05_par")


def _load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:      # older torch without the kwarg
        return torch.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--seeds", type=int, default=500, help="total seeds per gamma")
    ap.add_argument("--chunks", type=int, default=10, help="seed shards per gamma (workers = gammas*chunks)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--offdiag", type=float, default=0.5)
    ap.add_argument("--obs-margin", type=float, default=0.05)
    ap.add_argument("--prefix", default="dr05_")
    ap.add_argument("--max-concurrent", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(LOGD, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    # build tasks: (gamma, chunk, s0, cnt, gpu)
    base = args.seeds // args.chunks
    tasks, idx = [], 0
    for g in args.gammas:
        off = 0
        for c in range(args.chunks):
            cnt = base if c < args.chunks - 1 else args.seeds - base * (args.chunks - 1)
            tasks.append(dict(g=g, c=c, s0=off, cnt=cnt, gpu=args.gpus[idx % len(args.gpus)],
                              pfx=f"{args.prefix}s{c}_"))
            off += cnt; idx += 1
    print(f"[par] {len(tasks)} workers ({len(args.gammas)} gammas x {args.chunks} chunks), "
          f"gpus {args.gpus}, ~{base} seeds/shard, max_concurrent {args.max_concurrent}", flush=True)

    def launch(t):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(t["gpu"])
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = "1"
        env["LD_LIBRARY_PATH"] = "/home/dohyun/miniforge3/lib:" + env.get("LD_LIBRARY_PATH", "")
        lf = open(os.path.join(LOGD, f"g{t['g']}_s{t['c']}.log"), "w")
        cmd = [sys.executable, os.path.join(HERE, "gen_dr_data.py"),
               "--gammas", str(t["g"]), "--s0", str(t["s0"]), "--seeds", str(t["cnt"]),
               "--offdiag", str(args.offdiag), "--obs-margin", str(args.obs_margin),
               "--out-prefix", t["pfx"]]
        return (subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT), t, lf)

    running, done, t0, qi = [], [], time.time(), 0
    while qi < len(tasks) or running:
        while qi < len(tasks) and len(running) < args.max_concurrent:
            running.append(launch(tasks[qi])); qi += 1
        still = []
        for p, t, lf in running:
            rc = p.poll()
            if rc is None:
                still.append((p, t, lf))
            else:
                lf.close(); done.append((t, rc))
                print(f"[par] g{t['g']} s{t['c']} -> {'OK' if rc == 0 else f'FAIL(rc={rc})'}  "
                      f"({len(done)}/{len(tasks)}, {time.time()-t0:.0f}s)", flush=True)
        running = still
        if running:
            time.sleep(2)
    nfail = sum(1 for _, rc in done if rc != 0)
    print(f"[par] all workers finished in {time.time()-t0:.0f}s, {nfail} failed", flush=True)

    # merge shards per gamma -> dataset/<prefix>windows_g<g>.pt
    print("[merge] concatenating shards ...", flush=True)
    grand = 0
    for g in args.gammas:
        shards = [os.path.join(DATA, f"{args.prefix}s{c}_windows_g{g}.pt") for c in range(args.chunks)]
        shards = [s for s in shards if os.path.exists(s)]
        if not shards:
            print(f"[merge] g{g}: NO shards found!", flush=True); continue
        acc = None
        for s in shards:
            d = _load(s)
            if acc is None:
                acc = {k: d[k] for k in d}
            else:
                for k in ("grid", "low5", "hist", "U", "starts"):
                    if k in d and k in acc:
                        acc[k] = torch.cat([acc[k], d[k]], 0)
                acc["n_traj"] += d.get("n_traj", 0); acc["n_seeds"] += d.get("n_seeds", 0)
        out = os.path.join(DATA, f"{args.prefix}windows_g{g}.pt")
        torch.save(acc, out)
        grand += int(acc["n_traj"])
        print(f"[merge] g{g}: {acc['grid'].shape[0]} windows from {acc['n_traj']}/{acc['n_seeds']} "
              f"({100*acc['n_traj']/max(acc['n_seeds'],1):.1f}%) -> {out}", flush=True)
        for s in shards:
            os.remove(s)
    print(f"[merge] done; {grand} total trajectories across {len(args.gammas)} gammas. shards cleaned up.",
          flush=True)


if __name__ == "__main__":
    main()
