from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


def render_animation(root: Path, records: Sequence[Dict[str, Any]], summary: Dict[str, Any], gammas: Sequence[float], args) -> Dict[str, str]:
    if not args.show_live:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    miz = next((r for r in records if r["method"] == "mizuta_cfm_mppi" and r["episode"] == args.video_episode), None)
    safe_by_gamma = {f"{float(r['gamma']):.10g}": r for r in records if r["method"] == "safemppi_gamma" and r["episode"] == args.video_episode}
    g = np.asarray(gammas, dtype=float)
    s = summary["safemppi_gamma"]
    success = np.asarray([s[f"{x:.10g}"]["success_rate"] for x in g])
    collision = np.asarray([s[f"{x:.10g}"]["collision_rate"] for x in g])
    clearance = np.asarray([s[f"{x:.10g}"]["mean_min_clearance"] for x in g])
    final_dist = np.asarray([s[f"{x:.10g}"]["mean_final_goal_distance"] for x in g])
    ms = np.asarray([s[f"{x:.10g}"]["mean_planning_time_ms"] for x in g])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    def draw(frame: int):
        gamma = float(g[frame])
        rec = safe_by_gamma[f"{gamma:.10g}"]
        st = np.asarray(rec["states"], dtype=float)
        goal = np.asarray(rec["goal"], dtype=float)
        obs = np.asarray(rec["obstacles"], dtype=float)
        for ax in axes:
            ax.clear()
        ax0, ax1, ax2 = axes
        for o in obs:
            ax0.add_patch(Circle((o[0], o[1]), o[2] + args.safety_margin, fill=False, linestyle="--"))
            ax0.add_patch(Circle((o[0], o[1]), o[2], fill=False, alpha=0.5))
        if miz:
            m = np.asarray(miz["states"], dtype=float)
            ax0.plot(m[:, 0], m[:, 1], linestyle="--", label="Mizuta CFM-MPPI")
        ax0.plot(st[:, 0], st[:, 1], label=f"safeMPPI gamma={gamma:.2f}")
        ax0.scatter([st[0, 0]], [st[0, 1]], label="start")
        ax0.scatter([goal[0]], [goal[1]], marker="*", s=120, label="goal")
        ax0.axis("equal"); ax0.grid(True, alpha=0.3); ax0.set_title("1) trajectory"); ax0.legend(fontsize=8)
        ax0.text(0.02, 0.02, f"success={int(rec['success'])} collision={int(rec['collision'])}\nmin_clearance={rec['min_clearance']:.3f}", transform=ax0.transAxes, fontsize=8, va="bottom")
        ax1.plot(g, success, marker="o", label="safeMPPI success")
        ax1.plot(g, collision, marker="x", label="safeMPPI collision")
        ax1.axhline(summary["mizuta_cfm_mppi"].get("success_rate", 0.0), linestyle="--", label="Mizuta success")
        ax1.axvline(gamma); ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8); ax1.set_title("2) success/collision")
        ax2.plot(g, clearance, marker="o", label="clearance")
        ax2.plot(g, final_dist, marker="s", label="final distance")
        ax2.plot(g, ms / max(float(np.max(ms)), 1e-9), marker="^", linestyle=":", label="normalized ms")
        ax2.axhline(summary["mizuta_cfm_mppi"].get("mean_min_clearance", 0.0), linestyle="--", label="Mizuta clearance")
        ax2.axvline(gamma); ax2.set_xlim(-0.02, 1.02); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8); ax2.set_title("3) margin/performance/compute")
        fig.suptitle(f"{args.dataset}/{args.dynamics}: gamma={gamma:.2f}")
        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(g), interval=int(1000 / max(args.fps, 1)), repeat=args.repeat)
    draw(len(g) - 1)
    png = root / "live_gamma_sweep_last_frame.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    out = {"last_frame_png": str(png)}
    if not args.no_video:
        try:
            mp4 = root / "live_gamma_sweep.mp4"
            anim.save(mp4, writer="ffmpeg", fps=args.fps, dpi=args.dpi)
            out["mp4"] = str(mp4)
        except Exception as exc:
            gif = root / "live_gamma_sweep.gif"
            anim.save(gif, writer="pillow", fps=args.fps, dpi=args.dpi)
            out["gif"] = str(gif)
            out["render_note"] = str(exc)
    if args.show_live:
        plt.show()
    plt.close(fig)
    return out
