"""GridGRUFlowPolicy2 — v2 model (2026-07-03): NO aux safety decoder (saves 164,672 params = 55% of v1),
trunk width parametrized for the lighter-model study (W256=132k / W192=92k / W128=59k total params).

Same conditioning as v1 (frame audit 1a: relgoal AND vel are both robot-attached, WORLD-axis quantities —
consistent; grid channel robot-centered axis-aligned): low21 = relgoal(2)+vel(2)+GRU(16)+γ(1) → E_l 48;
grid CNN → E_g 64; ctx=112. φ_s = trunk penultimate (dim = width). v1 files untouched.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import _paths  # noqa: F401
from flow_policy import FlowPolicy
import grid_feats as GF


class GridGRUFlowPolicy2(FlowPolicy):
    def __init__(self, H_pred=GF.H_PRED, grid_shape=(3, GF.N_THETA, GF.N_R), K_hist=GF.K_HIST,
                 gru_dim=16, low_token=48, grid_token=64, width=256, depth=2, u_max=GF.U_MAX):
        ctx_dim = low_token + grid_token                       # 48 + 64 = 112
        super().__init__(T=H_pred, ctx_dim=ctx_dim, width=width, depth=depth, u_max=u_max)
        self.H_pred = H_pred
        self.grid_shape = tuple(grid_shape)
        self.K_hist = K_hist
        self.gru_dim = gru_dim
        self.gru = nn.GRU(input_size=2, hidden_size=gru_dim, num_layers=1, batch_first=True)
        self.enc_low = nn.Sequential(nn.Linear(4 + gru_dim + 1, 64), nn.SiLU(), nn.Linear(64, low_token), nn.SiLU())
        self.enc_grid = nn.Sequential(
            nn.Conv2d(grid_shape[0], 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
            nn.Linear(16 * 4 * 3, grid_token), nn.SiLU())
        # NO safety_decoder in v2 (user 3a): loss = cfm only (+ signed negative term in expansion).

    # ---- context ---------------------------------------------------------
    def ctx_from(self, grid, low5, hist):
        """grid [B,3,16,12], low5 [B,5], hist [B,K,2] -> ctx [B,112] (GRU runs here, grads flow)."""
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if low5.dim() == 1:
            low5 = low5.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        _, h_n = self.gru(hist.float())
        h = h_n[-1]
        low21 = torch.cat([low5[:, :4], h, low5[:, 4:5]], dim=1)
        e_l = self.enc_low(low21)
        e_g = self.enc_grid(grid.float())
        return torch.cat([e_l, e_g], dim=1)

    def encoder_tokens(self, grid, low5, hist):
        """Diagnostics: (e_l [B,48], e_g [B,64], h_gru [B,16]) for collapse checks."""
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if low5.dim() == 1:
            low5 = low5.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        _, h_n = self.gru(hist.float())
        h = h_n[-1]
        low21 = torch.cat([low5[:, :4], h, low5[:, 4:5]], dim=1)
        return self.enc_low(low21), self.enc_grid(grid.float()), h

    # ---- convenience for rollout / expansion (same interface as v1) ------
    @torch.no_grad()
    def sample_window(self, grid, low5, hist, n=1, temp=1.0, nfe=12, churn=0.0):
        ctx = self.ctx_from(grid, low5, hist)
        if ctx.shape[0] == 1:
            ctx = ctx[0]
        return self.sample(n, ctx, nfe=nfe, temp=temp, churn=churn)

    def phi_s_at(self, U, grid, low5, hist, s=0.9):
        ctx = self.ctx_from(grid, low5, hist)
        if ctx.shape[0] == 1:
            ctx = ctx[0]
        return self.phi_s(U, ctx, s=s)

    def module_groups(self):
        """Named module groups for per-module gradient-flow diagnostics."""
        return dict(E_g=self.enc_grid, E_l=self.enc_low, GRU=self.gru, trunk=self.trunk, head=self.head)

    def config(self):
        return dict(arch="v2", H_pred=self.H_pred, grid_shape=self.grid_shape, K_hist=self.K_hist,
                    gru_dim=self.gru_dim, width=self.width,
                    depth=len([m for m in self.trunk if isinstance(m, nn.Linear)]), u_max=self.u_max)


def build_policy2(width=256, depth=2, gru_dim=16, K_hist=GF.K_HIST, u_max=GF.U_MAX, device="cpu"):
    return GridGRUFlowPolicy2(width=width, depth=depth, gru_dim=gru_dim, K_hist=K_hist, u_max=u_max).to(device)


def save_policy2(policy, path, extra=None):
    d = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_policy2(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    pol = GridGRUFlowPolicy2(H_pred=c["H_pred"], grid_shape=tuple(c["grid_shape"]), K_hist=c["K_hist"],
                             gru_dim=c["gru_dim"], width=c["width"], depth=c["depth"], u_max=c["u_max"])
    pol.load_state_dict(ck["state_dict"]); pol.to(device).eval()
    return pol, ck


def param_report(policy):
    groups = policy.module_groups()
    rep = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    rep["total"] = sum(p.numel() for p in policy.parameters())
    return rep


if __name__ == "__main__":
    torch.manual_seed(0)
    for w in (256, 192, 128):
        pol = build_policy2(width=w)
        rep = param_report(pol)
        print(f"W{w}: " + "  ".join(f"{k}={v:,}" for k, v in rep.items()))
    pol = build_policy2(width=128)
    B = 8
    grid = torch.rand(B, 3, 16, 12); low5 = torch.randn(B, 5); hist = torch.randn(B, GF.K_HIST, 2)
    U = torch.randn(B, GF.H_PRED, 2).clamp(-1, 1)
    ctx = pol.ctx_from(grid, low5, hist)
    loss = pol.cfm_loss(U, ctx)
    loss.backward()
    gn = {k: float(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None) ** 0.5)
          for k, m in pol.module_groups().items()}
    print("ctx", tuple(ctx.shape), "| phi_s", tuple(pol.phi_s(U, ctx, s=0.9).shape),
          "| cfm", round(float(loss), 4))
    print("grad norms (all must be >0):", {k: round(v, 5) for k, v in gn.items()})
    print("sample_window", tuple(pol.sample_window(grid[0], low5[0], hist[0], n=4).shape))
