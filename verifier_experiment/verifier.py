"""Pillar-3 verifier (robot-centered level sets) — see VERIFIER_GEOMETRY.md for the derivation.

  (1) validate_polytope_v2 : the deterministic nominal polytope_v2 is a VALID safe polytope.
  (2) verify_trajectory    : does there EXIST a robot-centered polytope (slope set by gamma<=gamma_max) that
                             certifies the H-step trajectory via the recursive DTCBF?  Returns req_gamma, the
                             certifying offsets, or an INFEASIBILITY reason.  Closed form, GPU/np-batched.
  (3) demo                 : narrow-gap scene, certified vs infeasible cases, comparison to the nominal req_gamma.

Verifier horizon == MPPI horizon (the trajectory length). r_robot inflates obstacles (rho = r + margin + r_robot).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)


def norm_barrier_np(poly, pts):
    A = poly.A.numpy(); b = poly.b.numpy(); ref = poly.ref.numpy()
    mr = np.maximum(b - A @ ref, 1e-3)
    val = (b[None] - pts @ A.T) / mr[None]
    return val.min(1)


# ---------- (1) validate the deterministic nominal polytope_v2 ----------
def validate_polytope_v2(c, obstacles, sensing_range=3.5, n_base=16, margin=0.0, r_robot=0.2):
    c = np.asarray(c, float).reshape(2)
    poly, info = build_polytope_v2(c, obstacles, sensing_range=sensing_range, n_base=n_base, margin=margin)
    A = poly.A.numpy(); b = poly.b.numpy()
    robot_interior = bool(np.all(A @ c < b - 1e-9))                       # H_P(c) > 0
    obs = np.asarray(obstacles, float).reshape(-1, 3)
    sep = []                                                              # each in-range obstacle excluded by a face
    for o in obs:
        d = np.linalg.norm(o[:2] - c) - (o[2] + margin)
        if d > sensing_range:
            continue
        # obstacle disk (radius o2+margin+r_robot) outside P: some face has a_k.o >= b_k + rho
        rho = o[2] + margin + r_robot
        sep.append(bool(np.any(A @ o[:2] >= b + rho - 1e-6)) or np.any(A @ o[:2] >= b - 1e-6))
    all_separated = bool(all(sep)) if sep else True
    # no obstacle center strictly interior
    no_obs_inside = True
    for o in obs:
        if np.all(A @ o[:2] < b - 1e-6) and np.linalg.norm(o[:2] - c) - o[2] <= sensing_range:
            no_obs_inside = False
    valid = robot_interior and all_separated and no_obs_inside
    return dict(valid=valid, robot_interior=robot_interior, all_separated=all_separated,
                no_obstacle_inside=no_obs_inside, n_faces=int(info["n_faces"]), n_detected=int(info["n_detected"]))


# ---------- (2) robot-centered EXISTENCE verifier (closed form) ----------
def verify_trajectory(traj, obstacles, gamma_max=0.7, r_robot=0.2, margin=0.0, sensing_range=None, n_angles=360):
    """traj [H+1,2] (traj[0]=robot start=center c). For each obstacle, SWEEP the separating face normal and take
    the one minimizing req_gamma (the geometric optimization: find at least one certifying polytope). Returns the
    certificate / infeasibility. The chosen per-obstacle normals are the certifying polytope's faces."""
    traj = np.asarray(traj, float); c = traj[0]
    obs = np.asarray(obstacles, float).reshape(-1, 3)
    H = traj.shape[0] - 1
    disp = traj - c[None]                                                 # x_i - c, [H+1,2]
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    Nrm = np.stack([np.cos(thetas), np.sin(thetas)], 1)                   # candidate normals [A,2]
    proj = disp @ Nrm.T                                                   # n.(x_i-c)  [H+1,A]
    i_arr = np.arange(H + 1)[:, None].astype(float)                       # [H+1,1]
    per_obs = []; req_gamma = 0.0; infeasible = None; chosen = []
    for o in obs:
        rel = o[:2] - c; d = float(np.linalg.norm(rel)); rho = o[2] + margin + r_robot
        if sensing_range is not None and d - o[2] > sensing_range:
            continue
        if d - rho <= 0:
            infeasible = infeasible or f"start in collision with obstacle at {o[:2]}"
            per_obs.append(dict(o=o[:2].tolist(), reqg=float("inf"))); continue
        Cn = Nrm @ rel - rho                                              # face capacity per normal [A]
        valid = Cn > 1e-6                                                 # normals that separate the obstacle
        ratio = proj / np.where(np.abs(Cn) < 1e-9, 1e-9, Cn)[None, :]     # [H+1,A]
        with np.errstate(invalid="ignore"):
            base = np.clip(1.0 - ratio, 0.0, 1.0)
            g = 1.0 - np.power(base, 1.0 / np.maximum(i_arr, 1.0))        # min gamma per (step,normal)
        active = (proj > 1e-9) & (i_arr >= 1)
        g = np.where(active, g, 0.0)
        g = np.where(active & (ratio >= 1.0 - 1e-9), np.inf, g)           # projects past this face -> infeasible
        reqg_per_normal = np.where(valid, g.max(0), np.inf)               # [A]
        a_best = int(np.argmin(reqg_per_normal)); reqg_j = float(reqg_per_normal[a_best])
        per_obs.append(dict(o=o[:2].tolist(), reqg=reqg_j, normal=Nrm[a_best].tolist()))
        if np.isfinite(reqg_j):
            chosen.append((Nrm[a_best], float(Nrm[a_best] @ rel - rho)))  # (normal, offset cap) of this face
            req_gamma = max(req_gamma, reqg_j)
        else:
            infeasible = infeasible or f"no separating polytope for obstacle at {o[:2]} (trajectory passes it)"
    certified = bool(infeasible is None and req_gamma <= gamma_max)
    if infeasible is None and req_gamma > gamma_max:
        infeasible = f"req_gamma={req_gamma:.3f} > gamma_max={gamma_max:.2f} (approaches too fast)"
    return dict(certified=certified, req_gamma=float(req_gamma), infeasible=None if certified else infeasible,
                per_obstacle=per_obs, H=H, c=c.tolist(), faces=chosen)


def nominal_req_gamma(traj, obstacles, sensing_range=3.5, n_base=16, margin=0.0):
    """req_gamma the deterministic polytope_v2 (built at the start, WITH the sensing K-gon) needs for this traj."""
    traj = np.asarray(traj, float); c = traj[0]
    poly, _ = build_polytope_v2(c, obstacles, sensing_range=sensing_range, n_base=n_base, margin=margin)
    Hv = norm_barrier_np(poly, traj)                                      # H_{v2}(x_i)
    req = 0.0
    for i in range(1, traj.shape[0]):
        if Hv[i] <= 0:
            return float("inf"), poly                                     # trajectory exits the nominal polytope
        req = max(req, 1.0 - Hv[i] ** (1.0 / i))
    return float(req), poly


# ---------- (3) demo ----------
def _draw_levels(ax, poly, gamma, xlim, ylim, cmap="Blues"):
    gx = np.linspace(*xlim, 120); gy = np.linspace(*ylim, 120); GX, GY = np.meshgrid(gx, gy)
    Hh = norm_barrier_np(poly, np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GX.shape)
    lv = sorted({round((1 - gamma) ** i, 4) for i in range(8)} | {0.0})
    ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap=cmap, alpha=0.5, zorder=1)
    ax.contour(GX, GY, Hh, levels=[l for l in lv if l > 0], colors="#2166ac", linewidths=0.4, alpha=0.6, zorder=2)
    ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.4, zorder=3)


def demo(gamma_max=0.5, H=10, sensing=3.5):
    # narrow gap: two pedestrians at (3, +-0.9) r=0.5; robot at origin
    obstacles = np.array([[3.0, 0.9, 0.5], [3.0, -0.9, 0.5]])
    r_robot = 0.2
    c = np.array([0.0, 0.0])
    def line(end, H):
        return np.stack([np.linspace(0, end[0], H + 1), np.linspace(0, end[1], H + 1)], 1)
    cases = {
        "approach gap (slow)": line((1.8, 0.0), H),     # 10 steps toward the gap, stays well inside
        "thread into gap (far)": line((3.0, 0.0), H),   # reaches the gap mouth in 10 steps (faster approach)
        "veer into ped": line((2.6, 0.95), H),          # heads at the upper pedestrian -> should be infeasible
    }
    print(f"=== verifier demo (gamma_max={gamma_max}, H={H}) ===")
    val = validate_polytope_v2(c, obstacles, sensing_range=sensing, r_robot=r_robot)
    print("polytope_v2 validity at origin:", val)
    rows = []
    for name, tr in cases.items():
        v = verify_trajectory(tr, obstacles, gamma_max=gamma_max, r_robot=r_robot, sensing_range=sensing)
        nreq, poly = nominal_req_gamma(tr, obstacles, sensing_range=sensing)
        rows.append((name, tr, v, nreq, poly))
        print(f"  {name:22s}: verifier req_g={v['req_gamma']:.3f} certified={v['certified']} "
              f"| nominal req_g={nreq if nreq!=float('inf') else 'inf':<6} "
              f"| {'' if v['certified'] else 'INFEASIBLE: '+str(v['infeasible'])}")

    from cfm_mppi.safegpc_adapter.polytope import Polytope
    xlim = (-0.8, 4.2); ylim = (-2.2, 2.2)
    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 4.4))
    for ax, (name, tr, v, nreq, poly) in zip(axes, rows):
        # nominal polytope_v2 boundary (orange dashed) for contrast
        gx = np.linspace(*xlim, 120); gy = np.linspace(*ylim, 120); GX, GY = np.meshgrid(gx, gy)
        Hn = norm_barrier_np(poly, np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GX.shape)
        ax.contour(GX, GY, Hn, levels=[0.0], colors="#e6550d", linewidths=1.6, linestyles="--", zorder=3)
        # verifier certifying polytope (its chosen faces) + level sets, if certified
        if v["certified"] and v["faces"]:
            A = np.array([n for n, _ in v["faces"]]); cc = np.array(v["c"])
            b = np.array([cap + n @ cc for n, cap in v["faces"]])
            vpoly = Polytope(A=torch.tensor(A, dtype=torch.float32), b=torch.tensor(b, dtype=torch.float32),
                             ref=torch.tensor(cc, dtype=torch.float32))
            _draw_levels(ax, vpoly, gamma_max, xlim, ylim)                # certifying polytope level sets (blue)
        for (ox, oy, orr) in obstacles:
            ax.add_patch(Circle((ox, oy), orr, facecolor="#7b3294", alpha=0.5, edgecolor="#4d004b", zorder=4))
            ax.add_patch(Circle((ox, oy), orr + r_robot, facecolor="none", edgecolor="#7b3294", ls=":", lw=0.8, zorder=4))
        col = "#1a9850" if v["certified"] else "#d62728"
        ax.plot(tr[:, 0], tr[:, 1], "-o", color=col, ms=2.5, lw=1.4, zorder=6)
        ax.scatter([0], [0], s=50, c="#1a9850", edgecolor="k", zorder=9)
        verdict = f"CERTIFIED req_γ={v['req_gamma']:.2f}" if v["certified"] else "INFEASIBLE"
        ax.set_title(f"{name}\nverifier: {verdict}\nnominal(v2) req_γ={'inf' if nreq==float('inf') else round(nreq,2)}",
                     fontsize=9, color=col)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Pillar-3 verifier — robot-centered level sets, ∃ certifying polytope (γ_max={gamma_max}, H={H})\n"
                 "green=certified · red=infeasible · dotted=robot-inflated obstacle · blue=nominal polytope_v2 level sets",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); p = os.path.join(FIG, "verifier_demo.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    demo()
