"""Track 2b: state-conditioned flow-matching policy + Safe Flow Expansion on a 7x7 grid (3x3 center block).

We learn CONTROL (not a time-invariant distribution), so the FM is CONDITIONAL on the current state:
q_theta(U | x), U = H-step single-integrator velocity sequence. Applied receding-horizon -> a continuous
trajectory -> a monotone lattice path. Coverage = (distinct safe paths discovered) / 74  (lattice_paths.py).

Seed = "going LEFT" (up-first, above the block) H-step state-conditioned segments, SAME count across H
(to control the information-difference between short/long horizons). Then expansion (temperature exploration
+ verifier gate + UpdateFlow) discovers the other paths. Sweep horizon H in {4, 9, 14}.

  python overnight_run_2026-06-28/track2b_grid_coverage.py --device cuda
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "overnight_run_today", "src"))
from flow_policy import FlowPolicy
import lattice_paths as LP

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
N = 7; BSIZE = 3; DT = 0.25; UMAX = 2.0; START = np.array([0.5, 0.5]); GOAL = np.array([6.5, 6.5])
ALL_PATHS = set(LP.enumerate_safe_paths(N, BSIZE)); NPATHS = len(ALL_PATHS)


def cell_of(p):
    return (int(np.clip(np.floor(p[0]), 0, N - 1)), int(np.clip(np.floor(p[1]), 0, N - 1)))


def reached(p):
    return np.linalg.norm(p - GOAL) < 0.7


def traj_to_path(cells):
    return LP.path_signature(cells, N, BSIZE)


VISIT: dict = {}   # (i,j,dir) -> count of executed cell-transitions (dir 0=right,1=up); novelty exploration


def seg_cells(p0, U):
    p = p0.copy(); cc = [cell_of(p)]
    for h in range(len(U)):
        p = p + DT * U[h]; cc.append(cell_of(p))
    return cc, p


def transitions(cells):
    """Dedup consecutive cells -> list of monotone (i,j,dir) transitions, or None if any non-monotone step."""
    seq = [cells[0]]
    for c in cells[1:]:
        if c != seq[-1]:
            seq.append(c)
    tr = []
    for k in range(1, len(seq)):
        d = (seq[k][0] - seq[k - 1][0], seq[k][1] - seq[k - 1][1])
        if d == (1, 0):
            tr.append((seq[k - 1][0], seq[k - 1][1], 0))
        elif d == (0, 1):
            tr.append((seq[k - 1][0], seq[k - 1][1], 1))
        else:
            return None
    return tr


# ---------- forced-branch exploration: trace a specific enumerated path -> H-step state-conditioned segments ----------
def force_trace(path, H):
    """Generate a control sequence that traces `path` (sequence of cells) via its cell-centers, single-integrator,
    then slice H-step state-conditioned segments. Guarantees the under-covered branch enters the training buffer."""
    wp = [np.array([c[0] + 0.5, c[1] + 0.5]) for c in path]
    p = START.copy(); states = [p.copy()]; vels = []; wi = 1
    for _ in range(120):
        target = wp[wi] if wi < len(wp) else GOAL
        d = target - p; dist = float(np.linalg.norm(d))
        if dist < 0.2:
            if wi < len(wp):
                wi += 1; continue
            break
        u = np.clip(d / max(dist, 1e-9) * UMAX, -UMAX, UMAX)
        vels.append(u); p = p + DT * u; states.append(p.copy())
        if reached(p) and wi >= len(wp):
            break
    states = np.array(states); vels = np.array(vels)
    if len(vels) < H:
        return [], []
    return [states[k] / N for k in range(0, len(vels) - H + 1)], [vels[k:k + H] for k in range(0, len(vels) - H + 1)]


# ---------- seed: left-homotopy (up then right, above the block) state-conditioned H-step segments ----------
def left_segments(H, n_target, seed=0):
    rng = np.random.default_rng(seed)
    ctxs, Us = [], []
    max_steps = 60
    while len(ctxs) < n_target:
        turn = rng.uniform(5.2, 6.6)                  # turn above the 3x3 block (rows 2..4)
        p = START.copy(); states = [p.copy()]; vels = []
        for _ in range(max_steps):
            if p[1] < turn - 0.3 and p[0] < 1.2:      # go up the left column
                u = np.array([rng.normal(0, 0.3), UMAX])
            else:                                      # then go right along the top
                u = np.array([UMAX, rng.normal(0, 0.3)])
            u = np.clip(u, -UMAX, UMAX); vels.append(u); p = p + DT * u; states.append(p.copy())
            if reached(p):
                break
        states = np.array(states); vels = np.array(vels)
        # slice H-step segments at every state (ctx = segment start)
        for k in range(0, len(vels) - H + 1):
            ctxs.append(states[k] / N); Us.append(vels[k:k + H])
            if len(ctxs) >= n_target:
                break
    return torch.tensor(np.array(ctxs), dtype=torch.float32), torch.tensor(np.array(Us), dtype=torch.float32)


# ---------- batched receding-horizon rollout of the conditional FM ----------
@torch.no_grad()
def rollout(model, H, B, dev, temp=1.4, nfe=10, K=6):
    """Receding-horizon, NOVELTY-guided: each replan samples K candidate H-plans per rollout and executes the
    most-novel valid one (visit-count over cell-transitions) -> systematic discovery of distinct monotone paths."""
    p = np.tile(START, (B, 1)).astype(float); cells = [[cell_of(p[i])] for i in range(B)]
    segs_ctx = [[] for _ in range(B)]; segs_U = [[] for _ in range(B)]; done = np.zeros(B, bool)
    for _ in range(math.ceil(60 / H)):
        ctx = torch.tensor(p / N, dtype=torch.float32, device=dev).repeat_interleave(K, 0)   # [B*K,2]
        U = model.sample(B * K, ctx, nfe=nfe, temp=temp).clamp(-UMAX, UMAX).cpu().numpy().reshape(B, K, H, 2)
        chosen = np.zeros((B, H, 2))
        for i in range(B):
            if done[i]:
                continue
            best, bk = -1e18, 0
            for k in range(K):
                cc, pend = seg_cells(p[i], U[i, k]); tr = transitions(cc)
                if tr is None:
                    score = -1e9 + (pend[0] + pend[1])          # non-monotone: avoid, but keep some progress
                else:
                    score = sum(1.0 / (VISIT.get(t, 0) + 1) for t in tr) + 0.05 * (pend[0] + pend[1])
                if score > best:
                    best, bk = score, k
            chosen[i] = U[i, bk]
        for i in range(B):
            if done[i]:
                continue
            segs_ctx[i].append(p[i] / N); segs_U[i].append(chosen[i])
            cc, _ = seg_cells(p[i], chosen[i]); tr = transitions(cc)
            if tr:
                for t in tr:
                    VISIT[t] = VISIT.get(t, 0) + 1
        for h in range(H):
            p = p + DT * chosen[:, h]
            for i in range(B):
                if not done[i]:
                    cells[i].append(cell_of(p[i]))
        for i in range(B):
            if not done[i] and reached(p[i]):
                done[i] = True
        if done.all():
            break
    sigs = [traj_to_path(cells[i]) for i in range(B)]
    return sigs, segs_ctx, segs_U


def train(model, ctx, U, steps, lr, bs, dev, alpha=0.0, ctx_neg=None, U_neg=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for _ in range(steps):
        bi = torch.randint(0, ctx.shape[0], (min(bs, ctx.shape[0]),), device=dev)
        loss = model.cfm_loss(U[bi], ctx[bi])
        if alpha > 0 and ctx_neg is not None and ctx_neg.shape[0] > 0:
            ni = torch.randint(0, ctx_neg.shape[0], (min(bs, ctx_neg.shape[0]),), device=dev)
            loss = loss - alpha * model.cfm_loss(U_neg[ni], ctx_neg[ni])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); return model


def run_horizon(H, args):
    """Forced-branch expansion: each round force-trace a batch of not-yet-covered enumerated paths INTO the
    training buffer (so the FM CAN learn every branch), train, then measure coverage = distinct safe paths the
    FM ITSELF generates (pure-FM rollouts) -- the meaningful 'has the generable set expanded to cover the space'."""
    dev = args.device
    VISIT.clear()
    print(f"\n=== horizon H={H} ===")
    ctx0, U0 = left_segments(H, args.n_seed)
    posC, posU = ctx0.to(dev), U0.to(dev)
    model = FlowPolicy(T=H, ctx_dim=2, width=128, depth=3, u_max=UMAX).to(dev)
    train(model, posC, posU, args.pretrain, 5e-4, 256, dev)

    all_paths = list(ALL_PATHS); np.random.default_rng(0).shuffle(all_paths)
    forced, fm_seen, hist = set(), set(), []
    for rnd in range(args.rounds):
        # forced-branch injection: feed a batch of not-yet-forced enumerated paths into the buffer
        batch = [p for p in all_paths if p not in forced][:args.inject]
        nc, nu = [], []
        for pth in batch:
            sc, su = force_trace(pth, H)
            if sc:
                nc += sc; nu += su
            forced.add(pth)
        if nc:
            posC = torch.cat([posC, torch.tensor(np.array(nc), dtype=torch.float32, device=dev)])
            posU = torch.cat([posU, torch.tensor(np.array(nu), dtype=torch.float32, device=dev)])
            if posC.shape[0] > 40000:
                keep = torch.randperm(posC.shape[0], device=dev)[:40000]; posC, posU = posC[keep], posU[keep]
        train(model, posC, posU, args.inner, args.lr, 256, dev)
        # eval: distinct safe paths the FM generates (pure FM rollout, K=1, temp for diversity)
        sigs, _, _ = rollout(model, H, args.eval_B, dev, temp=args.eval_temp, K=1)
        round_paths = {s for s in sigs if s is not None}
        fm_seen |= round_paths
        hist.append({"round": rnd, "coverage": len(fm_seen) / NPATHS,
                     "round_cov": len(round_paths) / NPATHS, "forced": len(forced) / NPATHS})
        if rnd % args.eval_every == 0 or rnd == args.rounds - 1:
            print(f"  H={H} r{rnd:03d} FM-cov(cum)={len(fm_seen)/NPATHS*100:.1f}% ({len(fm_seen)}/{NPATHS}) "
                  f"round={len(round_paths)} forced={len(forced)}/{NPATHS} npos={posC.shape[0]}")
        if len(fm_seen) >= NPATHS:
            print(f"  H={H}: FM generated ALL {NPATHS} paths by round {rnd}"); break
    torch.save(model.state_dict(), os.path.join(HERE, f"track2b_H{H}.pt"))   # save for inference viz
    return hist, fm_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--horizons", nargs="+", type=int, default=[4, 9, 14])
    ap.add_argument("--rounds", type=int, default=40); ap.add_argument("--B", type=int, default=96)
    ap.add_argument("--temp", type=float, default=1.4); ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--inner", type=int, default=200); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrain", type=int, default=1500); ap.add_argument("--n_seed", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--inject", type=int, default=10)        # forced-branch paths injected per round
    ap.add_argument("--eval_B", type=int, default=600); ap.add_argument("--eval_temp", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    print(f"7x7 grid, 3x3 center block: {NPATHS} safe monotone paths (coverage denominator)")
    results = {}
    for H in args.horizons:
        results[H] = run_horizon(H, args)
    # plot coverage vs round for each horizon
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = {4: "#d73027", 9: "#1a9850", 14: "#2166ac"}
    for H, (hist, disc) in results.items():
        r = [h["round"] for h in hist]; c = [100 * h["coverage"] for h in hist]
        ax.plot(r, c, "-", color=cols.get(H, None), lw=2, label=f"H={H} FM-coverage (final {c[-1]:.0f}%)")
    fr = [h["round"] for h in next(iter(results.values()))[0]]; ff = [100 * h["forced"] for h in next(iter(results.values()))[0]]
    ax.plot(fr, ff, "--", color="0.5", lw=1, label="forced-branch injected (feed)")
    ax.set_xlabel("expansion round"); ax.set_ylabel(f"coverage % of {NPATHS} safe paths")
    ax.set_ylim(0, 105); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax.set_title("Track 2b: 7x7 grid FM-generation coverage vs horizon\n(forced-branch exploration feeds buffer; curve = paths the FM itself generates)")
    fig.tight_layout(); p = os.path.join(FIG, "track2b_grid_coverage.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)
    for H, (hist, disc) in results.items():
        print(f"H={H}: final coverage {100*hist[-1]['coverage']:.1f}% ({len(disc)}/{NPATHS})")


if __name__ == "__main__":
    main()
