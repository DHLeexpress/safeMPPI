"""Improved curriculum-pool stack for the FRESH-ONLY 2-class method (rev_expansion, user 2026-07-09).
Per viz_db snapshot: (A) EVERY easy/frontier window drawn in the scene (NO subsampling),
(B) sigma histogram by pool, (C) 3D scatter of the frontier axes (sigma, SOCP-margin, net-progress).
Usage: python curriculum_stack_rev.py --dbs a/viz_db/it*.pt ... --out fig.png [--title ...]
"""
import sys, os
sys.path.insert(0, '/home/dohyun/projects/cfm_mppi/overnight_run_07_06')
import _paths  # noqa
import argparse
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa
import grid_scene as GS, grid_rollout as GR, grid_expand2 as GX2

COL = {'easy': '#2ca02c', 'frontier': '#d62728'}
env = GS.make_grid(); OBS = env.obstacles.detach().cpu().numpy()


def load_db(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    return dict(low5=d['low5'].numpy(), U=d['U'].numpy(), label=np.asarray(list(d['label']), dtype=object),
                sigma=np.asarray(d['sigma']), margin=np.asarray(d['margin']),
                prog=np.asarray(d.get('prog', np.zeros(len(d['sigma'])))), iter=int(d.get('iter', -1)),
                paths=[np.asarray(q) for q in d.get('paths', [])])


def scene(ax):
    for o in OBS:
        ax.add_patch(Circle((o[0], o[1]), o[2], facecolor='#dcdcdc', ec='#aaa', lw=.3, zorder=1))
    ax.plot([0, 5], [0, 5], 'k--', lw=.6, alpha=.4, zorder=2)
    ax.plot(0, 0, 's', c='k', ms=6, zorder=6); ax.plot(5, 5, '*', c='gold', mec='k', ms=13, zorder=6)
    ax.set_xlim(-.3, 5.3); ax.set_ylim(-.3, 5.3); ax.set_aspect('equal')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dbs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('--qsig', type=float, default=0.75, help='σ frontier quantile (hyperplane in panel C)')
    ap.add_argument('--qmarg', type=float, default=0.25, help='margin frontier quantile')
    ap.add_argument('--pfloor', type=float, default=0.35, help='progress frontier floor')
    args = ap.parse_args()
    Ds = [load_db(p) for p in sorted(args.dbs, key=lambda s: int(s.split('it')[-1].split('.')[0]))]
    # shared axes ranges for comparability
    alls = np.concatenate([D['sigma'] for D in Ds]); allm = np.concatenate([D['margin'] for D in Ds])
    allp = np.concatenate([D['prog'] for D in Ds])
    sbins = np.linspace(alls.min(), alls.max() + 1e-6, 24)
    mlim = (allm.min(), allm.max()); plim = (allp.min(), allp.max())
    R = len(Ds)
    fig = plt.figure(figsize=(15, 4.7 * R))
    gs = fig.add_gridspec(R, 3, width_ratios=[1.05, 0.85, 1.0], hspace=0.28, wspace=0.28)
    for i, D in enumerate(Ds):
        lab = D['label']
        ne = int((lab == 'easy').sum()); nf = int((lab == 'frontier').sum())
        # ---- A: EVERY window in the scene (no subsampling) + executed source trajectories ----
        axA = fig.add_subplot(gs[i, 0]); scene(axA)
        tcols = plt.cm.tab10(np.linspace(0, 1, max(len(D['paths']), 1)))
        for k, q in enumerate(D['paths']):                # the distinct gathered rollouts (prob #1 diversity)
            axA.plot(q[:, 0], q[:, 1], '-', color=tcols[k], lw=1.6, alpha=.55, zorder=2.5)
        for pool in ('easy', 'frontier'):
            idx = np.where(lab == pool)[0]
            for j in idx:
                st = GX2.state_from_low5(D['low5'][j]); seg = GR.window_positions(st, D['U'][j], env.dt)
                pts = np.vstack([np.asarray(st, float)[:2][None], seg])
                axA.plot(pts[:, 0], pts[:, 1], '-', color=COL[pool], lw=.7, alpha=.5, zorder=3)
                axA.plot(pts[0, 0], pts[0, 1], '.', color=COL[pool], ms=2.5, zorder=4)
        axA.set_title(f"it{D['iter']}  (A) ALL windows + {len(D['paths'])} source rollouts  ·  easy {ne} / frontier {nf}",
                      fontsize=10)
        axA.set_ylabel('y (m)')
        # ---- B: sigma histogram by pool ----
        axB = fig.add_subplot(gs[i, 1])
        for pool in ('frontier', 'easy'):
            m = lab == pool
            if m.any():
                axB.hist(D['sigma'][m], bins=sbins, color=COL[pool], alpha=0.6,
                         zorder=(3 if pool == 'easy' else 2), label=pool)
        axB.set_title('(B) $\\sigma$ by pool', fontsize=10); axB.set_xlabel('$\\sigma$ (GP novelty)')
        axB.set_ylabel('count')
        # ---- C: 3D frontier axes: sigma x margin x net-progress ----
        axC = fig.add_subplot(gs[i, 2], projection='3d')
        for pool in ('easy', 'frontier'):
            m = lab == pool
            if m.any():
                axC.scatter(D['sigma'][m], D['margin'][m], D['prog'][m], s=12, c=COL[pool], alpha=0.6,
                            depthshade=True, edgecolor='none')
        # the three frontier threshold HYPERPLANES (σ ≥ q_hi-quantile | margin ≤ q_lo-quantile | prog ≥ floor)
        s_thr = float(np.quantile(D['sigma'], args.qsig)); m_thr = float(np.quantile(D['margin'], args.qmarg))
        yy, zz = np.meshgrid(np.linspace(*mlim, 2), np.linspace(*plim, 2))
        axC.plot_surface(np.full_like(yy, s_thr), yy, zz, color='#9467bd', alpha=0.15, shade=False)
        xx, zz2 = np.meshgrid(np.linspace(sbins[0], sbins[-1], 2), np.linspace(*plim, 2))
        axC.plot_surface(xx, np.full_like(xx, m_thr), zz2, color='#1f77b4', alpha=0.15, shade=False)
        xx3, yy3 = np.meshgrid(np.linspace(sbins[0], sbins[-1], 2), np.linspace(*mlim, 2))
        axC.plot_surface(xx3, yy3, np.full_like(xx3, args.pfloor), color='#ff7f0e', alpha=0.15, shade=False)
        axC.set_xlabel('$\\sigma$', fontsize=8); axC.set_ylabel('SOCP margin', fontsize=8)
        axC.set_zlabel('net-progress', fontsize=8)
        axC.set_xlim(sbins[0], sbins[-1]); axC.set_ylim(*mlim); axC.set_zlim(*plim)
        axC.set_title(f'(C) frontier axes + thresholds ($\\sigma$≥{s_thr:.2f} | m≤{m_thr:.2f} | prog≥{args.pfloor})',
                      fontsize=9)
        axC.view_init(elev=20, azim=-60); axC.tick_params(labelsize=7)
    leg = [Line2D([], [], color=COL['easy'], lw=3, label='easy (gentle safe)'),
           Line2D([], [], color=COL['frontier'], lw=3, label='frontier (high-$\\sigma$ / low-margin / high-progress)')]
    fig.legend(handles=leg, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.005), fontsize=11)
    if args.title:
        fig.suptitle(args.title, fontsize=12.5, y=1.02)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches='tight'); print('SAVED', args.out)


if __name__ == '__main__':
    main()
