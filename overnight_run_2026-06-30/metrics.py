"""Coverage for a generative policy over a too-large control-sequence space (Pillar 5 metric).

The design U lives in R^{T x 2} (~160-D) — grid coverage there is intractable and meaningless
(measure-zero valid manifold; many controls -> one path).  We measure coverage in the low-dim
ROLLED-OUT BEHAVIOR space instead:

  * spatial_coverage (HEADLINE) — fraction of the reachable-safe free-space cells (Omega*) that the
    policy's VERIFIER-CERTIFIED trajectory tubes occupy.  Intuitive "covering the whole space".
  * coverage (secondary) — descriptor-bin coverage (lateral offset at each obstacle's slice = the
    homotopy descriptor), reusing overnight_run_today/src/descriptors.
  * mode_coverage (secondary) — fraction of Omega*'s distinct homotopy signatures (which side of
    each on-corridor obstacle) the policy reproduces.
  * vendi — diversity (effective number of distinct behaviors).

Omega* (the denominator) is the verifier-reachable-safe set, estimated by gating a broad
"surrounding" proposal through the SAME SOCP verifier.  You can only cover what is certifiable-safe.
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
from dynamics import clip_controls
import descriptors as D


# ------------------------------------------------------------------- spatial occupancy
def occupied_cells(paths_xy, env, cell: float = 0.25):
    """paths_xy [K,T+1,2] (np) -> set of (ix,iy) grid cells any path point lands in."""
    lo_x, lo_y = env.xlim[0], env.ylim[0]
    pts = np.asarray(paths_xy).reshape(-1, 2)
    ix = np.floor((pts[:, 0] - lo_x) / cell).astype(int)
    iy = np.floor((pts[:, 1] - lo_y) / cell).astype(int)
    return set(zip(ix.tolist(), iy.tolist()))


def spatial_coverage(theta_cells: set, star_cells: set) -> float:
    if not star_cells:
        return 0.0
    return len(theta_cells & star_cells) / len(star_cells)


# ------------------------------------------------------------------- homotopy signature
@torch.no_grad()
def homotopy_signatures(states, env, band: float = 1.2):
    """Signature = which side of each ON-CORRIDOR obstacle the path passes. -> list[tuple], keys."""
    p0 = env.x0[:2]
    g = env.goal
    d = (g - p0); d = d / d.norm().clamp_min(1e-9)
    e = torch.stack([-d[1], d[0]])
    p0 = p0.to(states.device); d = d.to(states.device); e = e.to(states.device)
    desc = D.descriptor(states, env)                       # [B,N] robot lateral at obstacle slice
    obs_lat = (env.obstacles[:, :2].to(states.device) - p0) @ e   # [N]
    on_corr = obs_lat.abs() < band                          # obstacles near the chord define homotopy
    side = torch.sign(desc - obs_lat[None])                 # [B,N] which side
    side = side[:, on_corr]                                 # [B,K]
    keys = []
    for row in side:
        keys.append(tuple(int(x) for x in row.tolist()))
    return keys


def mode_coverage(theta_keys, star_keys_set: set) -> float:
    if not star_keys_set:
        return 0.0
    return len(set(theta_keys) & star_keys_set) / len(star_keys_set)


# ------------------------------------------------------------------- Omega* (denominator)
def build_omega_star_clutter(env, cfg, validity_label, n, device="cpu", cell=0.25, band=1.2,
                             max_U=400, log=print):
    """Broad 'surrounding' proposal -> SOCP gate -> reachable-safe bins/cells/signatures + verified U."""
    from safeflow import surrounding_proposal
    U = surrounding_proposal(env, n, device=device, seed=7)
    valid, safe, states, _ = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
    sv = states[valid]
    Uv = U[valid]
    ranges = [env.ylim for _ in range(env.n_obs)]
    star_bins = D.build_star_bins(D.descriptor(sv, env), ranges, cfg.nbins, min_count=1) if sv.shape[0] else set()
    star_cells = occupied_cells(sv[:, :, :2].cpu().numpy(), env, cell) if sv.shape[0] else set()
    star_modes = set(homotopy_signatures(sv, env, band)) if sv.shape[0] else set()
    star_U = Uv[:max_U].detach()
    log(f"[omega*] valid {int(valid.sum())}/{n}  cells={len(star_cells)}  modes={len(star_modes)}  "
        f"bins={len(star_bins)}  seedU={star_U.shape[0]}")
    return star_bins, ranges, star_cells, star_modes, int(valid.sum()), star_U


# ------------------------------------------------------------------- evaluate (rebinds safeflow.evaluate)
def make_clutter_evaluate(star_cells, star_modes, cell=0.25, band=1.2):
    """Return an ``evaluate(policy, env, ctx, star_bins, ranges, cfg)`` for safeflow to rebind."""
    def evaluate(policy, env, ctx, star_bins, ranges, cfg):
        import safeflow
        U = clip_controls(policy.sample(cfg.eval_K, ctx, nfe=cfg.nfe), env)
        valid, safe, states, _ = safeflow.validity_label(U, env, cfg.gamma_max, cfg.n_angles)
        sv = states[valid]
        desc_valid = D.descriptor(sv, env) if sv.shape[0] else torch.zeros(0, env.n_obs)
        cov = D.coverage(desc_valid, star_bins, ranges, cfg.nbins, cfg.tau)
        theta_cells = occupied_cells(sv[:, :, :2].cpu().numpy(), env, cell) if sv.shape[0] else set()
        spatial = spatial_coverage(theta_cells, star_cells)
        modes = mode_coverage(homotopy_signatures(sv, env, band) if sv.shape[0] else [], star_modes)
        vendi = D.vendi_score(desc_valid)
        return {
            "validity": float(valid.float().mean()),
            "safe_rate": float(safe.float().mean()),
            "coverage": cov,
            "spatial_coverage": spatial,
            "mode_coverage": modes,
            "mode_probs": [],
            "vendi": vendi,
            "n_valid": int(valid.sum()),
        }
    return evaluate
