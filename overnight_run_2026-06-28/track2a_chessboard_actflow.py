"""Track 2a: faithful reproduction of ACTFLOW Figure 2 (chessboard), safe/active expansion ONLY (no RecF).

Design = 2-D point x in [-3.5,3.5]^2 (UNCONDITIONAL flow, a time-invariant distribution). Valid set = 3x3
checkerboard (i+j) mod 2 == 0. Pretrain mis-specified on N((-1.1,0), 0.1^2) (one cell). Then run ACTFLOW:

  Eq.10  sigma_t(x) = sqrt( k(x,x) - k(x,X_t)(K+lam I)^-1 k(X_t,x) )   (GP, RBF lengthscale 0.08 on x)
  Eq.9   x_{t+1} ~ argmax_q E_q[sigma] - beta KL(q||p_theta)  ==  sample p_theta, weight by exp(sigma/beta), resample
  UpdateFlow  g = grad L+ (valid) - alpha_t grad L- (invalid)

Hyperparams from the paper appendix: T=500, B=64, beta=1/13, s=0.9, alpha_t=0.005, RBF l=0.08,
coverage = 100x100 hist tau, fine-tune 250 steps/round. Reported: coverage 1.16%->94.27%, validity 76%->95.89%.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "overnight_run_today", "src"))
from uncertainty import GPUncertainty   # Eq.10 GP posterior variance

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
LO, HI = -3.5, 3.5
CELLW = (HI - LO) / 3.0


def valid(x):  # checkerboard validity, x [...,2] -> bool
    i = np.clip(((x[..., 0] - LO) / CELLW).astype(int), 0, 2)
    j = np.clip(((x[..., 1] - LO) / CELLW).astype(int), 0, 2)
    inb = (x[..., 0] >= LO) & (x[..., 0] <= HI) & (x[..., 1] >= LO) & (x[..., 1] <= HI)
    return inb & ((i + j) % 2 == 0)


class FM2D(nn.Module):
    def __init__(self, width=128, depth=3, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        layers = [nn.Linear(2 + t_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers); self.head = nn.Linear(width, 2)

    def temb(self, t):
        f = torch.arange(1, self.t_dim // 2 + 1, device=t.device).float() * torch.pi
        a = t[:, None] * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t):
        return self.head(self.trunk(torch.cat([x, self.temb(t)], 1)))

    def cfm_loss(self, x1, w=None):
        x0 = torch.randn_like(x1); t = torch.rand(x1.shape[0], device=x1.device).clamp(1e-4, 1)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        per = ((self.forward(xt, t) - (x1 - x0)) ** 2).mean(1)
        return (per * w).mean() if w is not None else per.mean()

    @torch.no_grad()
    def sample(self, n, nfe=16, dev="cpu", temp=1.0):
        # temp>1 accesses the flow's tails -> the pool covers the whole board so the exp(sigma/beta) tilt
        # can acquire FAR valid cells (the practical realization of Eq.9's q* / the paper's noised latent).
        x = temp * torch.randn(n, 2, device=dev)
        for i in range(nfe):
            t = torch.full((n,), i / nfe, device=dev); x = x + (1.0 / nfe) * self.forward(x, t)
        return torch.nan_to_num(x, nan=0.0, posinf=4.2, neginf=-4.2).clamp(-4.2, 4.2)


def coverage_validity(model, dev, M=8000, nb=100, tau=1):
    x = model.sample(M, dev=dev).cpu().numpy()
    val = valid(x)
    # 100x100 histogram; a bin is "generable" if it has >= tau samples
    H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=nb, range=[[LO, HI], [LO, HI]])
    gen = H >= tau
    cx = (np.arange(nb) + 0.5) / nb * (HI - LO) + LO
    GX, GY = np.meshgrid(cx, cx, indexing="ij")
    binvalid = valid(np.stack([GX, GY], -1))
    cov = (gen & binvalid).sum() / max(binvalid.sum(), 1)
    return float(cov), float(val.mean()), x, val


def train(model, x1, steps, lr, bs, dev, w=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for _ in range(steps):
        idx = torch.randint(0, x1.shape[0], (min(bs, x1.shape[0]),), device=dev)
        loss = model.cfm_loss(x1[idx], None if w is None else w[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model


def actflow(args):
    dev = args.device
    # pretrain on the mis-specified single-cell Gaussian
    pre = (torch.randn(8000, 2, device=dev) * 0.1 + torch.tensor([-1.1, 0.0], device=dev))
    model = FM2D().to(dev)
    train(model, pre, args.pretrain, 1e-3, 256, dev)
    cov0, val0, x0, v0 = coverage_validity(model, dev)
    print(f"[pretrained] coverage={cov0*100:.2f}% validity={val0*100:.2f}%")
    torch.save(model.state_dict(), os.path.join(HERE, "track2a_scarce.pt"))   # scarce (left-figure) model

    unc = GPUncertainty(kernel="rbf", lengthscale=args.ell, lam=args.lam, normalize=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)   # PERSISTENT (continual fine-tune; keep momentum)
    Xq = torch.empty(0, 2, device=dev)   # all queried points (Eq.10 buffer)
    Dx = torch.empty(0, 2, device=dev); Dy = torch.empty(0, dtype=torch.bool, device=dev)
    hist = [{"round": -1, "coverage": cov0, "validity": val0}]; snaps = {0: (x0, v0)}
    for t in range(args.rounds):
        # Eq.10: fit sigma on queried points. Cap the GP buffer (subsample) so the Cholesky stays
        # tractable while still covering explored regions -- O(N^3) blows up otherwise. [coding tip]
        if Xq.shape[0] == 0:
            unc.set_buffer(None)
        elif Xq.shape[0] <= args.gp_max:
            unc.set_buffer(Xq)
        else:
            sub = Xq[torch.randperm(Xq.shape[0], device=dev)[:args.gp_max]]
            unc.set_buffer(sub)
        # Eq.9: sample pool from FM (with temperature -> covers the board), weight by exp(sigma/beta), resample B
        pool = model.sample(args.pool, dev=dev, temp=args.temp)
        sig = unc.sigma(pool)
        w = torch.exp(((sig - sig.max()) / max(args.beta, 1e-6)).clamp(-30, 30))
        cdf = torch.cumsum(w / w.sum().clamp_min(1e-12), 0)
        u = (torch.arange(args.B, device=dev) + torch.rand(1, device=dev)) / args.B
        idx = torch.searchsorted(cdf, u.clamp(max=1 - 1e-6)).clamp(max=pool.shape[0] - 1)
        q = pool[idx]
        # query verifier
        y = torch.tensor(valid(q.cpu().numpy()), device=dev)
        Xq = torch.cat([Xq, q]); Dx = torch.cat([Dx, q]); Dy = torch.cat([Dy, y])
        # UpdateFlow: CFM on valid (+) minus alpha * CFM on invalid (-)  [persistent optimizer]
        if Dy.any():
            model.train()
            pos = Dx[Dy]; neg = Dx[~Dy]
            for _ in range(args.inner):
                bi = torch.randint(0, pos.shape[0], (min(256, pos.shape[0]),), device=dev)
                loss = model.cfm_loss(pos[bi])
                if args.alpha > 0 and neg.shape[0] > 0:
                    ni = torch.randint(0, neg.shape[0], (min(256, neg.shape[0]),), device=dev)
                    loss = loss - args.alpha * model.cfm_loss(neg[ni])
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # prevent FM divergence -> NaN
                optim.step()
            model.eval()
        if t % args.eval_every == 0 or t == args.rounds - 1:
            cov, val, xs, vs = coverage_validity(model, dev)
            hist.append({"round": t, "coverage": cov, "validity": val})
            if t in (args.rounds // 2, args.rounds - 1):
                snaps[t] = (xs, vs)
            print(f"[r{t:03d}] coverage={cov*100:.2f}% validity={val*100:.2f}% "
                  f"npos={int(Dy.sum())} nneg={int((~Dy).sum())} ESSf={float((w.sum()**2)/(w**2).sum())/args.pool:.2f}")
    return model, hist, snaps


def plot(hist, snaps, path):
    fig, (axc, *axs) = plt.subplots(1, 1 + len(snaps), figsize=(4.6 + 3.6 * len(snaps), 4.2))
    r = [h["round"] for h in hist if h["round"] >= 0]
    axc.plot(r, [100 * h["coverage"] for h in hist if h["round"] >= 0], "-o", color="#2166ac", ms=3, label="coverage %")
    axc.plot(r, [100 * h["validity"] for h in hist if h["round"] >= 0], "-s", color="#1a9850", ms=3, label="validity %")
    axc.set_ylim(0, 105); axc.set_xlabel("round"); axc.grid(alpha=0.2); axc.legend(fontsize=8)
    axc.set_title("ACTFLOW chessboard (safe expansion)")
    for ax, (rr, (xs, vs)) in zip(axs, sorted(snaps.items())):
        for ii in range(3):
            for jj in range(3):
                if (ii + jj) % 2 == 0:
                    ax.add_patch(plt.Rectangle((LO + ii * CELLW, LO + jj * CELLW), CELLW, CELLW,
                                               facecolor="0.85", edgecolor="0.6", zorder=0))
        ax.scatter(xs[vs, 0], xs[vs, 1], s=2, c="#1a9850", alpha=0.4); ax.scatter(xs[~vs, 0], xs[~vs, 1], s=2, c="#d62728", alpha=0.2)
        ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"round {rr}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rounds", type=int, default=500); ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--pool", type=int, default=512); ap.add_argument("--beta", type=float, default=1 / 13)
    ap.add_argument("--ell", type=float, default=0.08); ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=0.005); ap.add_argument("--inner", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--pretrain", type=int, default=2500)
    ap.add_argument("--eval_every", type=int, default=20); ap.add_argument("--gp_max", type=int, default=1200)
    ap.add_argument("--temp", type=float, default=2.0)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    model, hist, snaps = actflow(args)
    torch.save(model.state_dict(), os.path.join(HERE, "track2a_expanded.pt"))   # expanded model
    plot(hist, snaps, os.path.join(FIG, "track2a_chessboard_actflow.png"))
    print("FINAL coverage=%.2f%% validity=%.2f%%" % (hist[-1]["coverage"] * 100, hist[-1]["validity"] * 100))


if __name__ == "__main__":
    main()
