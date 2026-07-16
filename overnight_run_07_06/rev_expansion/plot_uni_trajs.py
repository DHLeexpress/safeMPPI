"""Figure: the ACTUAL 566 SafeMPPI expert trajectories PER GAMMA (druni dataset, reproduced with the same
seeds), styled after figures/image.png — discrete plasma gamma-colorbar on top, grey obstacles, |y-x|<1 band.
Top: all gammas overlaid (draw order interleaved). Bottom: one mini panel per gamma.
Output figures/uni_trajs_566_all_gamma.png"""
import sys, os
sys.path.insert(0, '/home/dohyun/projects/cfm_mppi/overnight_run_07_06')
import _paths  # noqa
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import BoundaryNorm, ListedColormap
import grid_scene as GS

mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10})
HERE = os.path.dirname(os.path.abspath(__file__))
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
CMAP = plt.cm.plasma(np.linspace(0.02, 0.9, len(GAMMAS)))
env = GS.make_grid(); OBS = env.obstacles.detach().cpu().numpy()


def scene(ax, band=True):
    ax.set_facecolor('#f7f6f4')
    if band:                                                    # |y-x|<1 exclusion band (as in image.png)
        ax.add_patch(Polygon([(-1, 0), (5, 6), (6, 6), (6, 5), (0, -1), (-1, -1)],
                             closed=True, facecolor='#e3e1dd', edgecolor='none', zorder=1))
    for o in OBS:
        ax.add_patch(Circle((o[0], o[1]), o[2], facecolor='#8a8a8a', edgecolor='none', zorder=3))
    ax.plot(0, 0, 's', c='k', ms=5, zorder=8)
    ax.plot(5, 5, '*', c='gold', mec='k', mew=0.5, ms=12, zorder=8)
    ax.set_xlim(-0.15, 5.15); ax.set_ylim(-0.15, 5.15); ax.set_aspect('equal')


def main():
    D = {}
    for g in GAMMAS:
        z = np.load(os.path.join(HERE, 'results', 'uni_trajs', f'paths_g{g}.npz'), allow_pickle=True)
        D[g] = [p for p, ok in zip(z['paths'], z['ok']) if ok]
        print(f'g{g}: {len(D[g])} trajs')

    fig = plt.figure(figsize=(13, 16.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.045, 1.55, 0.42], hspace=0.14, wspace=0.06)

    # ---- top: discrete gamma colorbar (image.png style) ----
    cax = fig.add_subplot(gs[0, 1:3])
    bounds = np.arange(len(GAMMAS) + 1)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=ListedColormap(CMAP), norm=BoundaryNorm(bounds, len(GAMMAS)),
                                   orientation='horizontal', ticks=bounds[:-1] + 0.5)
    cb.set_ticklabels([f'{g}' for g in GAMMAS]); cb.ax.tick_params(labelsize=11, length=0)
    cb.outline.set_linewidth(0.8)
    cax.set_title(r'$\gamma$', fontsize=15)

    # ---- middle: ALL 566 trajs x 7 gammas overlaid, interleaved draw order ----
    axM = fig.add_subplot(gs[1, :])
    scene(axM)
    order = [(gi, k) for gi, g in enumerate(GAMMAS) for k in range(len(D[g]))]
    rng = np.random.default_rng(0); rng.shuffle(order)
    for gi, k in order:
        p = D[GAMMAS[gi]][k]
        axM.plot(p[:, 0], p[:, 1], '-', color=CMAP[gi], lw=0.45, alpha=0.28, zorder=4,
                 solid_capstyle='round')
    st = np.load(os.path.join(HERE, 'results', 'uni_trajs', f'paths_g{GAMMAS[0]}.npz'),
                 allow_pickle=True)['starts']
    axM.plot(st[:, 0], st[:, 1], '.', c='k', ms=1.6, zorder=6)
    axM.set_xlabel(r'$x$ [m]', fontsize=13); axM.set_ylabel(r'$y$ [m]', fontsize=13)
    axM.set_title(f'SafeMPPI expert: {len(D[GAMMAS[0]])} uniform-grid starts $\\times$ 7 $\\gamma$ '
                  f'({sum(len(D[g]) for g in GAMMAS)} trajectories)', fontsize=13)

    # ---- bottom: one mini panel per gamma (first 4 + last 3 across the row grid) ----
    gs2 = fig.add_gridspec(1, 7, top=0.215, bottom=0.055, left=0.045, right=0.985, wspace=0.08)
    for gi, g in enumerate(GAMMAS):
        ax = fig.add_subplot(gs2[0, gi])
        scene(ax, band=False)
        for p in D[g]:
            ax.plot(p[:, 0], p[:, 1], '-', color=CMAP[gi], lw=0.3, alpha=0.22, zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'$\\gamma$={g}', fontsize=11, color=CMAP[gi] * np.array([0.85, 0.85, 0.85, 1]))
    out = os.path.join(HERE, 'figures', 'uni_trajs_566_all_gamma.png')
    fig.savefig(out, dpi=135, bbox_inches='tight'); print('SAVED', out)


if __name__ == '__main__':
    main()
