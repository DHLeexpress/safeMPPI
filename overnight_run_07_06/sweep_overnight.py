"""Autonomous overnight sweep (2026-07-07, user 07_06 full-stack directive). Self-contained: waits for the
running anchor matrix to clear, PRETRAINS the arch variants (repr × trunk-depth × encoder-depth), then runs
the EXPANSION sweep (freeze × anchor δ/η × easy-strict) across GPU 0,1,3. Each expansion: 5000 iters,
M=100 rollouts @ temp=1, collapse-termination, best.pt saved. Nothing here needs the main session alive.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYLIB = "/home/dohyun/miniforge3/lib"
GPUS = [0, 1, 3]
CKDIR = os.path.join(HERE, "results", "hp_repr")
EXPDIR = os.path.join(HERE, "results", "sweep_overnight"); os.makedirs(EXPDIR, exist_ok=True)
LOGDIR = os.path.join(EXPDIR, "logs"); os.makedirs(LOGDIR, exist_ok=True)


def _env(gpu):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["LD_LIBRARY_PATH"] = PYLIB + ":" + e.get("LD_LIBRARY_PATH", "")
    e["OMP_NUM_THREADS"] = e["MKL_NUM_THREADS"] = "2"
    return e


def run_pool(jobs, cap_per_gpu, phase):
    running, q, load, t0 = [], list(jobs), {g: 0 for g in GPUS}, time.time()
    while q or running:
        for g in GPUS:
            while load[g] < cap_per_gpu and q:
                name, argv = q.pop(0)
                lf = open(os.path.join(LOGDIR, f"{name}.log"), "w")
                p = subprocess.Popen([sys.executable] + argv, cwd=HERE, env=_env(g),
                                     stdout=lf, stderr=subprocess.STDOUT)
                running.append((p, name, g, lf)); load[g] += 1
                print(f"[{phase}] launch {name} gpu{g} ({len(running)} run / {len(q)} queued)", flush=True)
        still = []
        for p, name, g, lf in running:
            if p.poll() is None:
                still.append((p, name, g, lf))
            else:
                lf.close(); load[g] -= 1
                print(f"[{phase}] DONE {name} rc={p.returncode} ({time.time()-t0:.0f}s)", flush=True)
        running = still
        if running or q:
            time.sleep(10)
    print(f"[{phase}] all finished in {time.time()-t0:.0f}s", flush=True)


# ---- wait for the currently-running anchor matrix (results/expand_cur/a_*) to clear ----
print("[wait] waiting for the anchor matrix (expand_cur/a_) to finish ...", flush=True)
while subprocess.run(["pgrep", "-f", "expand_cur/a_"], capture_output=True).stdout.strip():
    time.sleep(60)
print("[wait] cleared; GPUs free for the sweep", flush=True)

# ---- Phase 1: pretrain arch variants (repr × trunk-depth × encoder-depth) ----
ARCHS = [  # (tag, repr, trunk_hidden, enc_depth)
    ("a15", 15, [128, 64], 2), ("a20", 20, [128, 64], 2), ("a20T", 20, [128, 128, 64], 2),
    ("a20E", 20, [128, 64], 3), ("a32", 32, [160, 96], 2), ("a48", 48, [192, 128], 2)]
pre = []
for tag, r, th, ed in ARCHS:
    if os.path.exists(os.path.join(CKDIR, f"pretrained_{tag}.pt")):
        continue
    pre.append((f"pre_{tag}", ["pretrain_repr.py", "--repr", str(r), "--enc-depth", str(ed),
                               "--trunk-hidden"] + [str(x) for x in th] + ["--epochs", "120", "--tag", tag]))
if pre:
    run_pool(pre, 2, "pretrain")

# ---- Phase 2: expansion sweep ----
A, HI = (0.25, 0.05), (0.40, 0.10)


def ex(tag, arch, freeze, anchor, strict, extra=None):
    argv = ["grid_expand_cur.py", "--ckpt", os.path.join(CKDIR, f"pretrained_{arch}.pt"),
            "--iters", "5000", "--measure-every", "500", "--m-measure", "100",
            "--outdir", os.path.join(EXPDIR, tag), "--tag", tag]
    argv += ["--freeze"] if freeze else ["--no-freeze", "--enc-lr-mult", "0.3"]
    if anchor:
        argv += ["--demo-frac", str(anchor[0]), "--lwf-eta", str(anchor[1])]
    if strict:
        argv += ["--easy-strict"]
    return (tag, argv + (extra or []))


EXP = [ex(f"{tag}_hero", tag, True, A, True) for tag, *_ in ARCHS]   # frozen + anchor + strict, every arch
EXP += [
    ex("a20_unf",   "a20",  False, A,    True),    # unfrozen (encoder grad flows + clipped)
    ex("a20_comp",  "a20",  True,  A,    False),   # composite easy (isolate strict vs composite)
    ex("a20_noanc", "a20",  True,  None, True),    # NO anchor (collapse control — expect early termination)
    ex("a20_hi",    "a20",  True,  HI,   True),    # stronger anchor δ0.4 η0.1
    ex("a32_unf",   "a32",  False, A,    True),
    ex("a32_noanc", "a32",  True,  None, True),
    ex("a15_unf",   "a15",  False, A,    True),
    ex("a20T_unf",  "a20T", False, A,    True),
]
run_pool(EXP, 2, "expand")   # 2 arms per GPU
print("SWEEP_OVERNIGHT_DONE", flush=True)
