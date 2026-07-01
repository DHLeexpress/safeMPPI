"""Scene conditioning for the FM policy (Pillar 4, the core design question).

How does the scene enter the FM policy's conditioning?  A permutation-invariant DeepSets
encoder over the obstacle SET (not a single nearest-obstacle vector, which jumps as the nearest
obstacle changes).  Per-obstacle feature relative to the reference (start, since generation is
one-shot end-to-end):

    f_j = [ (o_j - p_ref).x / S , (o_j - p_ref).y / S , ||o_j - p_ref|| / S ,
            (r_j + r_robot) / S , presence=1 ]

Shared MLP phi -> masked (mean || max) pool -> MLP rho -> a fixed-dim scene token.  The
"NO OBSTACLE" case is handled unambiguously: an empty sensed set returns a learned
``empty_token`` and an explicit ``n_sensed`` scalar is appended to the context — so "no obstacle"
(n_sensed=0) is distinct from "an obstacle sitting at the robot" (dist=0, presence=1, n_sensed>=1),
fixing the ``[0,0,0,0]`` ambiguity of the old nearest-obstacle token.

The full context fed to ``FlowPolicy`` is
    ctx = [ start/S (2), goal/S (2), gamma (1), scene_token (token_dim), n_sensed/n_max (1) ]
so ``ctx_dim = 6 + token_dim``.  ``SceneConditionedFlowPolicy`` subclasses ``FlowPolicy`` (which
already threads a per-sample ctx through cfm_loss/sample/phi_s) and OWNS the encoder so it trains
jointly.  For expansion we freeze the encoder and pass a constant precomputed ctx.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

import _paths  # noqa: F401
from flow_policy import FlowPolicy


class DeepSetsSceneEncoder(nn.Module):
    def __init__(self, hidden: int = 64, token_dim: int = 32, in_feat: int = 5):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_feat, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )
        self.empty_token = nn.Parameter(torch.zeros(token_dim))
        self.token_dim = token_dim

    def token(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats [B,M,5], mask [B,M] (1=present) -> token [B,token_dim]."""
        h = self.phi(feats)                                   # [B,M,hidden]
        m = mask[..., None]
        cnt = m.sum(dim=1).clamp_min(1.0)                     # [B,1]
        mean = (h * m).sum(dim=1) / cnt                       # [B,hidden]
        very_neg = torch.finfo(h.dtype).min
        mx = torch.where(m.bool(), h, torch.full_like(h, very_neg)).max(dim=1).values
        n_present = mask.sum(dim=1, keepdim=True)             # [B,1]
        mx = torch.where(n_present > 0, mx, torch.zeros_like(mx))
        tok = self.rho(torch.cat([mean, mx], dim=-1))         # [B,token_dim]
        empty = (mask.sum(dim=1) == 0)                        # [B]
        tok = torch.where(empty[:, None], self.empty_token[None].to(tok.dtype), tok)
        return tok


class SceneConditionedFlowPolicy(FlowPolicy):
    """FlowPolicy + DeepSets scene encoder. ctx = [start/S, goal/S, gamma, token, n_sensed/n_max]."""

    def __init__(self, T: int, token_dim: int = 32, width: int = 256, depth: int = 3,
                 hidden: int = 64, n_max: int = 16, S: float = 6.0, R_enc: float = 8.0,
                 r_robot: float = 0.2, u_max: float = 2.0):
        super().__init__(T, ctx_dim=6 + token_dim, width=width, depth=depth, u_max=u_max)
        self.encoder = DeepSetsSceneEncoder(hidden=hidden, token_dim=token_dim)
        self.token_dim = token_dim
        self.n_max = n_max
        self.S = float(S)
        self.R_enc = float(R_enc)
        self.r_robot = float(r_robot)
        # bank buffers (filled by attach_bank); shapes [n_scenes, ...]
        self.register_buffer("bank_obs", torch.zeros(0, n_max, 3), persistent=False)
        self.register_buffer("bank_cnt", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("bank_start", torch.zeros(0, 2), persistent=False)
        self.register_buffer("bank_goal", torch.zeros(0, 2), persistent=False)

    # ------------------------------------------------------------------ feature builder
    def _feats_from(self, obs: torch.Tensor, cnt: torch.Tensor, start: torch.Tensor):
        """obs [B,M,3], cnt [B], start [B,2] -> feats [B,M,5], mask [B,M]."""
        B, M, _ = obs.shape
        rel = obs[:, :, :2] - start[:, None, :]               # [B,M,2]
        dist = rel.norm(dim=2)                                # [B,M]
        slot = torch.arange(M, device=obs.device)[None].expand(B, M)
        valid = slot < cnt[:, None]
        within = (dist - obs[:, :, 2]) <= self.R_enc
        mask = (valid & within).float()
        feats = torch.stack([
            rel[:, :, 0] / self.S, rel[:, :, 1] / self.S, dist / self.S,
            (obs[:, :, 2] + self.r_robot) / self.S, torch.ones_like(dist),
        ], dim=2)                                             # [B,M,5]
        feats = feats * mask[..., None]
        return feats, mask

    def _assemble(self, start, goal, gamma, token, mask):
        n_sensed = mask.sum(dim=1, keepdim=True) / self.n_max
        return torch.cat([start / self.S, goal / self.S, gamma[:, None], token, n_sensed], dim=1)

    # ------------------------------------------------------------------ bank (training)
    def attach_bank(self, envs: List):
        """Register the pretraining scene bank so ctx_for(scene_ids, gammas) is cheap."""
        device = self.head.weight.device
        n = len(envs)
        obs = torch.zeros(n, self.n_max, 3, device=device)
        cnt = torch.zeros(n, dtype=torch.long, device=device)
        start = torch.zeros(n, 2, device=device)
        goal = torch.zeros(n, 2, device=device)
        for i, e in enumerate(envs):
            k = min(e.n_obs, self.n_max)
            obs[i, :k] = e.obstacles[:k].to(device)
            cnt[i] = k
            start[i] = e.x0[:2].to(device)
            goal[i] = e.goal.to(device)
        self.bank_obs = obs
        self.bank_cnt = cnt
        self.bank_start = start
        self.bank_goal = goal

    def ctx_for(self, scene_ids: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """Batched context for training: gather bank scenes by id, vary gamma. [B, ctx_dim]."""
        obs = self.bank_obs[scene_ids]
        cnt = self.bank_cnt[scene_ids]
        start = self.bank_start[scene_ids]
        goal = self.bank_goal[scene_ids]
        feats, mask = self._feats_from(obs, cnt, start)
        token = self.encoder.token(feats, mask)
        return self._assemble(start, goal, gammas.float(), token, mask)

    # ------------------------------------------------------------------ single env (viz/expand)
    def ctx_for_env(self, env, gamma: float) -> torch.Tensor:
        """Context for an arbitrary single env (bank or held-out fixed scene). [ctx_dim]."""
        device = self.head.weight.device
        k = min(env.n_obs, self.n_max)
        obs = torch.zeros(1, self.n_max, 3, device=device)
        obs[0, :k] = env.obstacles[:k].to(device)
        cnt = torch.tensor([k], device=device)
        start = env.x0[:2].to(device)[None]
        goal = env.goal.to(device)[None]
        feats, mask = self._feats_from(obs, cnt, start)
        token = self.encoder.token(feats, mask)
        g = torch.tensor([float(gamma)], device=device)
        return self._assemble(start, goal, g, token, mask)[0]

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)


if __name__ == "__main__":
    import env as E
    bank = E.scene_bank(4)
    pol = SceneConditionedFlowPolicy(T=48, token_dim=32)
    pol.attach_bank(bank)

    # (1) ctx dim
    ctx = pol.ctx_for(torch.tensor([0, 1, 2]), torch.tensor([0.1, 0.5, 0.9]))
    assert ctx.shape == (3, 6 + 32), ctx.shape
    print("ctx_for shape OK:", tuple(ctx.shape))

    # (2) permutation invariance: shuffle obstacle rows -> identical token
    e = bank[0]
    import copy
    e2 = copy.deepcopy(e)
    perm = torch.randperm(e2.n_obs)
    e2.obstacles = e2.obstacles[perm].clone()
    t1 = pol.ctx_for_env(e, 0.5)
    t2 = pol.ctx_for_env(e2, 0.5)
    print("permutation-invariance max|Δ|:", float((t1 - t2).abs().max()))
    assert torch.allclose(t1, t2, atol=1e-6)

    # (3) no-obstacle unambiguous: empty set -> empty_token & n_sensed=0, distinct from obstacle-at-start
    empty_env = copy.deepcopy(e)
    empty_env.obstacles = torch.zeros(0, 3)
    empty_env.obs_vel = torch.zeros(0, 2)
    # NOTE Env.n_obs reads obstacles.shape[0]; ctx_for_env handles k=0
    c_empty = pol.ctx_for_env(empty_env, 0.5)
    at_origin = copy.deepcopy(e)
    at_origin.obstacles = torch.tensor([[float(e.x0[0]), float(e.x0[1]), 0.3]])
    at_origin.obs_vel = torch.zeros(1, 2)
    c_origin = pol.ctx_for_env(at_origin, 0.5)
    n_sensed_empty = float(c_empty[-1])
    n_sensed_origin = float(c_origin[-1])
    tok_diff = float((c_empty[5:5 + 32] - c_origin[5:5 + 32]).abs().max())
    print(f"n_sensed empty={n_sensed_empty:.3f} (want 0) vs obstacle-at-start={n_sensed_origin:.3f}; "
          f"token max|Δ|={tok_diff:.3f} (want >0)")
    assert n_sensed_empty == 0.0 and tok_diff > 1e-6
    print("scene_encoder unit tests PASS")
