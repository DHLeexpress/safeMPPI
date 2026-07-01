"""Compact-SOCP polytope verifier gate (Pillar 3), applied as a sliding-window certificate.

The per-face SOCP core is copied verbatim from
``ieee_compact_polytope_verifier_package/src/demo_verifier_polytope.py`` (self-contained; the
sibling ``pillar3_m_bounds_6x3.py`` mkdirs ``/mnt/data`` at import, so we do NOT import it).

SOCP (one variable tangent face per sensed obstacle / artificial boundary anchor):
    max  sum_i w_i m_i
    s.t. a_i.(q_t - c) <= beta_t m_i         (level-set ruler, beta_t = 1-(1-gamma)^t)
         r_i ||a_i|| <= a_i.(o_i - c) - m_i   (disk separation, SOC)
         ||a_i|| <= 1                          (SOC)
         m_i >= m_min
Each independent 2-D circular block is solved EXACTLY by the feasible angular interval
(``feasible_theta_interval``); ``check_certificate`` re-verifies H_P(q_t) >= (1-gamma)^t on the
constructed faces -> a SOUND certificate.

The compact verifier is defined for one H=10 local rollout centered at c=q_0.  A full ~80-step
end-to-end FM trajectory is certified by SLIDING that window (``socp_certify_trajectory``):
re-center at each step, sense local obstacles + artificial boundary anchors, and require EVERY
window to certify.  ``make_socp_validity_label`` wraps this into the exact interface
``safeflow.validity_label`` expects, so it can be rebound into the reused ``run_safeflow`` loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
from dynamics import rollout
from safeflow import reaches_goal


# =============================================================================
# ---- compact-SOCP core (copied from demo_verifier_polytope.py, unchanged) ----
# =============================================================================
@dataclass
class Face:
    a: np.ndarray
    m: float
    kind: str
    label: str
    coefficient: float = 1.0
    feasible: bool = True
    interval: Optional[tuple] = None


def wrap_interval_intersection(current, center, halfwidth):
    lo, hi = current
    mid = 0.5 * (lo + hi)
    center0 = center + 2.0 * math.pi * round((mid - center) / (2.0 * math.pi))
    best = None
    for cc in (center0 - 2.0 * math.pi, center0, center0 + 2.0 * math.pi):
        a, b = cc - halfwidth, cc + halfwidth
        ilo, ihi = max(lo, a), min(hi, b)
        if ilo <= ihi + 1e-12:
            if best is None or (ihi - ilo) > (best[1] - best[0]):
                best = (ilo, ihi)
    return best


def feasible_theta_interval(d, radius, trajectory, beta, *, m_min=1e-6):
    """Exact feasible theta interval for one circular face. d=o-c, trajectory=q-c."""
    d = np.asarray(d, dtype=float).reshape(2)
    D = float(np.linalg.norm(d))
    if D <= radius + m_min + 1e-12:
        return None
    phi = math.atan2(d[1], d[0])
    half = math.acos(max(-1.0, min(1.0, (radius + m_min) / D)))
    current = (phi - half, phi + half)
    for p_t, beta_t in zip(trajectory[1:], beta[1:]):
        w = beta_t * d - p_t
        rho = beta_t * radius
        W = float(np.linalg.norm(w))
        if W <= 1e-12:
            if rho > 1e-12:
                return None
            continue
        ratio = rho / W
        if ratio > 1.0 + 1e-12:
            return None
        ratio = max(-1.0, min(1.0, ratio))
        center = math.atan2(w[1], w[0])
        half = math.acos(ratio)
        current = wrap_interval_intersection(current, center, half)
        if current is None:
            return None
    return current


def solve_face_interval(d, radius, trajectory, beta, *, coefficient, kind, label,
                        m_min=1e-6, signed_unit_diagnostic=False):
    d = np.asarray(d, dtype=float).reshape(2)
    interval = feasible_theta_interval(d, radius, trajectory, beta, m_min=m_min)
    if interval is None:
        return Face(np.array([1.0, 0.0]), 0.0, kind, label, coefficient, False, None)
    lo, hi = interval
    phi = math.atan2(d[1], d[0])
    mid = 0.5 * (lo + hi)
    phi = phi + 2.0 * math.pi * round((mid - phi) / (2.0 * math.pi))
    candidates = [lo, hi]
    if lo <= phi <= hi:
        candidates.append(phi)

    def margin_at(theta):
        return float(np.array([math.cos(theta), math.sin(theta)]) @ d - radius)

    if coefficient >= 0.0 or not signed_unit_diagnostic:
        theta = min(max(phi, lo), hi)
    else:
        theta = min(candidates, key=margin_at)
    a = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    m = float(a @ d - radius)
    feasible = bool(m >= m_min - 1e-9)
    return Face(a, m, kind, label, coefficient, feasible, interval)


def artificial_obstacles(R, K, rho_art):
    if K <= 0:
        return []
    M = R * math.cos(math.pi / K) if K >= 3 else R
    out = []
    for ell in range(K):
        th = 2.0 * math.pi * ell / K
        n = np.array([math.cos(th), math.sin(th)])
        center = (M + rho_art) * n
        out.append((float(center[0]), float(center[1]), float(rho_art)))
    return out


def make_variable_faces(real_obs, trajectory, beta, *, R, K_artificial, rho_art,
                        coeff_real=1.0, coeff_artificial=1.0, m_min=1e-6,
                        signed_unit_diagnostic=False):
    faces = []
    for j, (ox, oy, rr) in enumerate(real_obs):
        faces.append(solve_face_interval(
            np.array([ox, oy]), rr, trajectory, beta,
            coefficient=coeff_real, kind="real", label=f"real{j}",
            m_min=m_min, signed_unit_diagnostic=signed_unit_diagnostic))
    art = artificial_obstacles(R, K_artificial, rho_art)
    for ell, (ox, oy, rr) in enumerate(art):
        faces.append(solve_face_interval(
            np.array([ox, oy]), rr, trajectory, beta,
            coefficient=coeff_artificial, kind="artificial", label=f"art{ell}",
            m_min=m_min, signed_unit_diagnostic=signed_unit_diagnostic))
    return faces, art


def check_certificate(faces, trajectory, alpha, *, include_start=False):
    if any((not f.feasible) or f.m <= 1e-12 for f in faces):
        return False, -float("inf"), -1
    worst, worst_t = float("inf"), -1
    start = 0 if include_start else 1
    for t in range(start, len(trajectory)):
        h = min((f.m - float(f.a @ trajectory[t])) / f.m for f in faces)
        slack = h - float(alpha[t])
        if slack < worst:
            worst, worst_t = slack, t
    return bool(worst >= -1e-8), float(worst), int(worst_t)


# =============================================================================
# ---- sliding-window trajectory certificate (new) ----
# =============================================================================
def _sensed_obstacles(obstacles, c, r_robot, R_eff):
    """Obstacles whose (Minkowski-inflated) boundary is within R_eff of centre c, in c-frame."""
    out = []
    for (ox, oy, rr) in obstacles:
        dx, dy = float(ox - c[0]), float(oy - c[1])
        if math.hypot(dx, dy) - rr <= R_eff:
            out.append((dx, dy, float(rr + r_robot)))
    return out


def socp_certify_window(seg, obstacles, r_robot, gamma, *, R_ver=2.0, K_art=12,
                        rho_art=0.12, m_min=1e-4, r_ver_pad=1.3):
    """Certify one H-step forward segment (seg[0] = centre c). Returns (ok, faces_c)."""
    c = np.asarray(seg[0], dtype=float)
    traj_c = np.asarray(seg, dtype=float) - c
    Hs = traj_c.shape[0] - 1
    if Hs < 1:
        return True, []
    seg_reach = float(np.linalg.norm(traj_c, axis=1).max())
    R_eff = max(R_ver, r_ver_pad * seg_reach)
    real_obs_c = _sensed_obstacles(obstacles, c, r_robot, R_eff)
    alpha = (1.0 - gamma) ** np.arange(Hs + 1, dtype=float)
    beta = 1.0 - alpha
    faces, _ = make_variable_faces(
        real_obs_c, traj_c, beta, R=R_eff, K_artificial=K_art, rho_art=rho_art,
        coeff_real=1.0, coeff_artificial=1.0, m_min=m_min)
    ok, _, _ = check_certificate(faces, traj_c, alpha, include_start=False)
    return ok, faces


def socp_certify_trajectory(traj, obstacles, r_robot, gamma, *, R_ver=2.0, H_win=10,
                            stride=2, K_art=12, rho_art=0.12, m_min=1e-4, r_ver_pad=1.3):
    """Whole-trajectory certificate = every sliding H_win window certifies."""
    T = traj.shape[0] - 1
    for k in range(0, T, stride):
        Hs = min(H_win, T - k)
        if Hs < 1:
            break
        ok, _ = socp_certify_window(
            traj[k:k + Hs + 1], obstacles, r_robot, gamma,
            R_ver=R_ver, K_art=K_art, rho_art=rho_art, m_min=m_min, r_ver_pad=r_ver_pad)
        if not ok:
            return False
    return True


def window_faces_world(traj, obstacles, r_robot, gamma, k, *, R_ver=2.0, H_win=10,
                       K_art=12, rho_art=0.12, m_min=1e-4, r_ver_pad=1.3):
    """World-frame certifying faces (a, b_world) for the window starting at step k (for viz).

    Face is a.(x-c) <= m  ==>  a.x <= m + a.c.  Only real-obstacle faces are returned.
    """
    T = traj.shape[0] - 1
    Hs = min(H_win, T - k)
    if Hs < 1:
        return []
    c = np.asarray(traj[k], dtype=float)
    ok, faces = socp_certify_window(
        traj[k:k + Hs + 1], obstacles, r_robot, gamma,
        R_ver=R_ver, K_art=K_art, rho_art=rho_art, m_min=m_min, r_ver_pad=r_ver_pad)
    out = []
    for f in faces:
        if f.kind == "real" and f.feasible and f.m > 1e-9:
            out.append((f.a.copy(), float(f.m + f.a @ c), c.copy()))
    return out


# =============================================================================
# ---- collision prefilter + validity-label factory ----
# =============================================================================
def min_clearance(states, env):
    """states [B,T+1,4] -> [B] min over path & obstacles of (||p-o|| - r_obs - r_robot)."""
    p = states[:, :, :2]
    obs = env.obstacles.to(states.device)
    if obs.numel() == 0:
        return torch.full((states.shape[0],), float("inf"), device=states.device)
    d = torch.linalg.norm(p[:, :, None, :] - obs[None, None, :, :2], dim=-1)
    d = d - obs[None, None, :, 2] - env.r_robot
    return d.amin(dim=2).amin(dim=1)


def make_socp_validity_label(env, *, R_ver=2.0, H_win=10, stride=2, reach_radius=0.5,
                             K_art=12, rho_art=0.12, m_min=1e-4, r_ver_pad=1.3):
    """Return a ``validity_label(U, env, gamma_max, n_angles)`` compatible with safeflow."""
    def validity_label(U, env_, gamma_max, n_angles=None):
        states = rollout(U, env_)                                  # [B,T+1,4]
        B = states.shape[0]
        obs_np = env_.obstacles.detach().cpu().numpy()
        r_robot = float(env_.r_robot)
        collision_free = min_clearance(states, env_) >= 0.0        # [B] cheap prefilter
        sp = states[:, :, :2].detach().cpu().numpy()
        safe = torch.zeros(B, dtype=torch.bool, device=states.device)
        cf = collision_free.tolist()
        for b in range(B):
            if not cf[b]:
                continue
            if socp_certify_trajectory(sp[b], obs_np, r_robot, float(gamma_max),
                                       R_ver=R_ver, H_win=H_win, stride=stride,
                                       K_art=K_art, rho_art=rho_art, m_min=m_min,
                                       r_ver_pad=r_ver_pad):
                safe[b] = True
        valid = safe & reaches_goal(states, env_, radius=reach_radius)
        req = torch.zeros(B, device=states.device)                 # SOCP gate has no scalar req
        return valid, safe, states, req
    return validity_label


# =============================================================================
if __name__ == "__main__":
    # Realistic-length trajectories (T=48, matching the smoke horizon), sliding H=10 windows.
    obs = np.array([[1.3, 0.0, 0.35]])          # obstacle on the chord
    T = 48
    s = np.linspace(0, 1, T + 1)
    tline = np.stack([2.6 * s, np.zeros(T + 1)], axis=1)                 # straight through
    tdodge = np.stack([2.6 * s, 0.85 * np.sin(np.pi * s)], axis=1)       # gentle bulge around
    ttight = np.stack([2.6 * s, 0.60 * np.sin(np.pi * s)], axis=1)       # tighter bulge (hugs obstacle)
    for g in (0.3, 0.5, 0.7):
        cl = socp_certify_trajectory(tline, obs, 0.2, g, R_ver=2.0, stride=2)
        cd = socp_certify_trajectory(tdodge, obs, 0.2, g, R_ver=2.0, stride=2)
        ct = socp_certify_trajectory(ttight, obs, 0.2, g, R_ver=2.0, stride=2)
        print(f"gamma={g}: straight-through={cl} (want F) | wide-dodge={cd} (want T) | tight-dodge={ct}")
    # empty scene -> always certifiable
    print("empty-scene dodge:", socp_certify_trajectory(tdodge, np.zeros((0, 3)), 0.2, 0.5, stride=2))
