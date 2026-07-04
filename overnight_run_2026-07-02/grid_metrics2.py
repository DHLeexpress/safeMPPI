"""Metrics v2 (2026-07-03) — window-level validity + coverage_cumulative / coverage_final.

validity2 (user 2b): the ONLY criteria are window-level, over every sliding 10-window (stride 2):
  (1) SOCP-certified (verifier polytope, unchanged);
  (2) in task space;
  (3) GOAL-APPROACH (replaces monotone-staircase + reaches-goal): with d_t = ||x_t - goal|| over the window,
      max_t d_t <= d_0 + SLACK   (never retreats away from the goal beyond slack)  AND
      d_H  <= d_0 - min(DELTA, 0.5*d_0)   (net progress; DELTA calibrated on demo windows below);
      a window starting inside the reach disk auto-passes (arrived).
coverage_cumulative = old coverage (distinct staircases ever realized by a VALID deploy / 252, monotone up).
coverage_final     = distinct staircases realized by VALID deploys within ONE measurement batch / 252
                     (non-cumulative, can decrease; ceiling = n_deploys/252).
Existing grid_metrics.py is untouched; staircase_id / in_taskspace / socp_ok are reused from it.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
import grid_metrics as GM
import grid_feats as GF

GOAL_XY = np.array([GM.GRID_M, GM.GRID_M], dtype=float)
# calibrated on demo windows (calibrate_approach): demo 1st-percentile net progress ~0.28 m / window;
# DELTA below it with margin so >=97% of expert windows pass while stalls/retreats fail.
DELTA_PROG = 0.12
SLACK_RETREAT = 0.15


def approach_ok(dists, delta=DELTA_PROG, slack=SLACK_RETREAT, reach=GM.REACH):
    """dists: ||x_t - goal|| for t = 0..H (window start included). Window-level goal-approach test."""
    d = np.asarray(dists, dtype=float)
    d0 = d[0]
    if d0 < reach:                                   # already arrived -> auto-pass
        return True
    if (d - d0).max() > slack:                       # retreated away from the goal beyond slack
        return False
    return d[-1] <= d0 - min(delta, 0.5 * d0)        # net progress over the window


def traj_valid2(path, env, gamma, H=10, stride=2, check_socp=True):
    """validity2 of an executed trajectory: every sliding window passes approach ∧ taskspace ∧ SOCP."""
    p = np.asarray(path, dtype=float)
    if len(p) < H + 1:
        return False
    if not GM.in_taskspace(p):
        return False
    D = np.linalg.norm(p - GOAL_XY[None], axis=1)
    for t in range(0, len(p) - H, stride):
        if not approach_ok(D[t:t + H + 1]):
            return False
    if check_socp and not GM.socp_ok(p, env, gamma):
        return False
    return True


def window_label_cheap(state, U, env, gamma):
    """Cheap per-window buffer label during exploration (NO SOCP — that runs once per trajectory):
    planned window in task space ∧ approaches the goal. Signature matches grid_rollout verify_fn."""
    import grid_rollout as GR
    seg = GR.window_positions(state, U, env.dt)
    if not GM.in_taskspace(seg):
        return False
    d = np.linalg.norm(np.vstack([np.asarray(state, float)[None, :2], seg]) - GOAL_XY[None], axis=1)
    return approach_ok(d)


def window_min_clearance(state, U, env):
    """Min obstacle clearance of the planned window (for the pos_margin data-hygiene gate)."""
    import grid_rollout as GR
    seg = GR.window_positions(state, U, env.dt)
    obs = env.obstacles.detach().cpu().numpy()
    if not obs.size:
        return np.inf
    d = np.linalg.norm(seg[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - float(env.r_robot)
    return float(d.min())


def measure2(paths, env, gamma, covered):
    """Batch measurement: validity2 %, coverage_cumulative (updates `covered` in place), coverage_final,
    reach-rate (transparency), avg steps of valid trajectories."""
    n_valid = 0
    final_set = set()
    steps = []
    n_reach = 0
    goal = env.goal.detach().cpu().numpy()
    for path in paths:
        reached = GM.reaches_goal(path, goal)
        n_reach += int(reached)
        if traj_valid2(path, env, gamma):
            n_valid += 1
            steps.append(len(path))
            if reached:
                sid = GM.staircase_id(path)
                if sid is not None:
                    covered.add(sid)
                    final_set.add(sid)
    n = max(len(paths), 1)
    return dict(validity=n_valid / n,
                coverage_cum=len(covered) / GM.N_STAIR,
                coverage_final=len(final_set) / GM.N_STAIR,
                n_final=len(final_set),
                reach_rate=n_reach / n,
                avg_steps=float(np.mean(steps)) if steps else 0.0)


def wilson_band(p, n, z=1.0):
    """+-z-sigma Wilson interval half-widths for a proportion (for validity plots)."""
    if n <= 0:
        return 0.0, 0.0
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, p - (ctr - hw)), max(0.0, (ctr + hw) - p)


def calibrate_approach(demo, dt=0.1, n_max=8000, seed=0):
    """Pass-rate of the approach test on expert demo windows, reconstructed goal-relative from low5:
    p = goal - relgoal*R_GOAL, v = low5[2:4]*V_SCALE, DI-roll U -> distances. Returns stats + suggested DELTA
    (1st percentile of demo net progress, floored at 0.05)."""
    import grid_rollout as GR
    rng = np.random.default_rng(seed)
    n = demo["U"].shape[0]
    idx = rng.permutation(n)[:min(n, n_max)]
    L = demo["low5"][idx].numpy()
    Uw = demo["U"][idx].numpy()
    prog, ok = [], 0
    for i in range(len(idx)):
        p = GOAL_XY - L[i, :2].astype(float) * GF.R_GOAL
        v = L[i, 2:4].astype(float) * GF.V_SCALE
        st = np.array([p[0], p[1], v[0], v[1]], np.float32)
        seg = GR.window_positions(st, Uw[i], dt)
        d = np.linalg.norm(np.vstack([p[None], seg]) - GOAL_XY[None], axis=1)
        prog.append(d[0] - d[-1])
        ok += int(approach_ok(d))
    prog = np.asarray(prog)
    return dict(pass_rate=ok / len(idx), n=len(idx),
                prog_p01=float(np.percentile(prog, 1)), prog_p05=float(np.percentile(prog, 5)),
                prog_med=float(np.median(prog)),
                suggested_delta=float(max(0.05, np.percentile(prog, 1))))


if __name__ == "__main__":
    import grid_scene as GS
    import grid_expand as GE
    env = GS.make_grid()
    # unit checks
    diag = np.array([[i * 0.1, i * 0.1] for i in range(51)])
    stall = np.vstack([np.array([[i * 0.1, i * 0.1] for i in range(20)]), np.tile([2.0, 2.0], (40, 1))])
    back = np.vstack([np.array([[i * 0.1, i * 0.1] for i in range(30)]),
                      np.array([[3.0 - i * 0.05, 3.0 - i * 0.05] for i in range(20)])])
    print("diag  valid2 (no socp):", traj_valid2(diag, env, 0.5, check_socp=False), "(expect True)")
    print("stall valid2 (no socp):", traj_valid2(stall, env, 0.5, check_socp=False), "(expect False)")
    print("back  valid2 (no socp):", traj_valid2(back, env, 0.5, check_socp=False), "(expect False)")
    cov = set()
    m = measure2([diag, stall], env, 0.5, cov)
    print("measure2 no-socp fields sanity:", {k: round(v, 3) for k, v in m.items() if k != "n_final"})
    # demo calibration per gamma
    for g in (0.1, 0.5, 1.0):
        demo = GE.load_demo(g)
        c = calibrate_approach(demo)
        print(f"γ{g}: demo approach pass {c['pass_rate']*100:.1f}% (n={c['n']})  "
              f"net-progress p01={c['prog_p01']:.3f} p05={c['prog_p05']:.3f} med={c['prog_med']:.3f}  "
              f"suggested DELTA={c['suggested_delta']:.3f}  (using {DELTA_PROG})")
