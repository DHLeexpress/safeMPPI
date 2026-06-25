"""Train the reward-tilted conditional flow proposal q_θ(U|o,γ) by the
energy/reward-weighted CFM loss (EFM, arXiv:2503.04975).

Per training step we draw ONE control sequence from the stored top-K with
probability ∝ its MPPI weight w ∝ exp(-S/λ); regressing the rectified-flow target
on these reward-weighted draws makes the terminal marginal equal the tilt
p(U|o,γ) ∝ 1[U∈F] exp(-S/λ). Context is normalized and translation-invariant.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from cfm_mppi.models.tilted_flow import TiltedFlowProposal


def _load(paths):
    ctx, ctr, w, g = [], [], [], []
    H = None
    for p in paths:
        d = torch.load(p, weights_only=False)
        ctx.append(d["context"]); ctr.append(d["controls"]); w.append(d["weights"]); g.append(d["gamma"])
        H = d["meta"]["horizon"]
    return (torch.cat(ctx), torch.cat(ctr), torch.cat(w), torch.cat(g), H)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["dataset/tilted/tilt_a.pt", "dataset/tilted/tilt_b.pt"])
    p.add_argument("--output-dir", default="output_dir/tilted_flow")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()
    dev = torch.device(cli.device)

    ctx, ctr, w, g, H = _load(cli.data)
    S, K = w.shape
    mu = ctx.mean(0); sd = ctx.std(0).clamp_min(1e-3)
    ctx_n = ((ctx - mu) / sd).to(dev)
    ctr = ctr.to(dev); w = w.to(dev); g = g.to(dev)
    cond_dim = ctx.shape[1] + 1
    model = TiltedFlowProposal(horizon=H, cond_dim=cond_dim, hidden=cli.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cli.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cli.epochs)
    out = Path(cli.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"tilted data: {S} steps x {K} samples, H={H}, cond_dim={cond_dim}", flush=True)

    udim = H * 2
    nb = max(1, S // cli.batch_size)
    for epoch in range(cli.epochs):
        perm = torch.randperm(S, device=dev)
        tot = 0.0
        for bi in range(nb):
            idx = perm[bi * cli.batch_size:(bi + 1) * cli.batch_size]
            B = idx.numel()
            cond = torch.cat([ctx_n[idx], g[idx].view(B, 1)], dim=1)         # [B, cond_dim]
            j = torch.multinomial(w[idx].clamp_min(1e-8), 1).squeeze(1)       # ∝ tilt weight
            U = ctr[idx, j]                                                   # [B,H,2]
            x1 = U.reshape(B, udim)
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=dev)
            xt = t.view(B, 1) * x1 + (1 - t.view(B, 1)) * x0
            pred = model(xt, t, cond)
            loss = F.mse_loss(pred, x1 - x0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach())
        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == cli.epochs - 1:
            print(json.dumps({"epoch": epoch, "loss": tot / nb, "lr": sched.get_last_lr()[0]}), flush=True)
    torch.save({"model": model.state_dict(), "ctx_mu": mu, "ctx_sd": sd, "horizon": H,
                "cond_dim": cond_dim, "hidden": cli.hidden}, out / "checkpoint.pth")
    print(f"saved {out}/checkpoint.pth")


if __name__ == "__main__":
    main()
