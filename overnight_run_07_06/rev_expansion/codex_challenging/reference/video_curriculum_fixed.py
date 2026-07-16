"""Curriculum video for the fixed three-axis AND-quantile expansion recipe.
  A (left)   scene: DASHED executed rollouts + every gathered window colored by sigma; easy = green ring
  B          sigma histogram by pool (easy/frontier)
  C          3D scatter margin x progress x SIGMA-on-Z + the three threshold planes
  D (right)  5 bins: [fresh easy, fresh frontier, buffer easy, buffer frontier, demo required];
             arrows above buffer/demo bins = samples actually USED in this iter's batch
  row 2      real-time traces up to t: beta | counts (batch used vs pool) | mix ratio incl demo (req vs measured) | lr
Iter 0 = pretrained faithful rollout (no samples). Data: <run>/viz_db/it*.pt + <run>/probe.jsonl.
Usage: python video_curriculum_fixed.py --run results/p2/final --out video/p2_final_curriculum.mp4
"""
import sys, os, json, glob, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import _paths  # noqa
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa
import grid_scene as GS, grid_rollout as GR, grid_hp_expt as HP, grid_expand2 as GX2
from eval_ae import _apply_wall_plugs_eval

mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 15,
                     'axes.titlesize': 19, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                     'ytick.labelsize': 15, 'legend.fontsize': 15})
env = GS.make_grid(); OBS = env.obstacles.detach().cpu().numpy()
C_E, C_F, C_D = '#00b300', '#d62728', '#7f7f7f'   # easy green / frontier red / demo grey


EXTRA_OBS = []                     # wall plugs etc., set from the run's recipe.json (drawn like obstacles)


def scene(ax):
    ax.set_facecolor('#f7f6f4')
    for o in list(OBS) + list(EXTRA_OBS):
        ax.add_patch(Circle((o[0], o[1]), o[2], facecolor='#8a8a8a', ec='none', zorder=2))
    ax.plot(0.05, 0.05, 's', c='k', ms=7, zorder=8); ax.plot(5, 5, '*', c='gold', mec='k', ms=15, zorder=8)
    ax.set_xlim(-.45, 5.45); ax.set_ylim(-.45, 5.45); ax.set_aspect('equal')


def load_db(run, it):
    p = os.path.join(run, 'viz_db', f'it{it}.pt')
    if not os.path.exists(p):
        return None
    z = torch.load(p, map_location='cpu', weights_only=False)
    pos = np.array([np.asarray(GX2.state_from_low5(l), float)[:2] for l in z['low5'].numpy()])
    return dict(pos=pos, sig=np.asarray(z['sigma']), marg=np.asarray(z['margin']),
                prog=np.asarray(z['prog']), lab=np.asarray(list(z['label']), dtype=object),
                quantile=float(z.get('quantile', .5)),
                sig_plane=float(z.get('sigma_plane', np.quantile(z['sigma'], .5))),
                marg_plane=float(z.get('margin_plane', np.quantile(z['margin'], .5))),
                prog_plane=float(z.get('prog_plane', np.quantile(z['prog'], .5))),
                paths=[np.asarray(q) for q in z.get('paths', [])])


def frame(fig, it, db, rec, recs_upto, pre_path, sig_lims, args):
    fig.clf()
    gs = fig.add_gridspec(2, 4, height_ratios=[2.1, 1.0], hspace=0.30, wspace=0.30,
                          left=0.045, right=0.985, top=0.90, bottom=0.075)
    # ---------- A: scene ----------
    axA = fig.add_subplot(gs[0, 0]); scene(axA)
    if it == 0:
        axA.plot(pre_path[:, 0], pre_path[:, 1], '--', c='#333333', lw=2.4, zorder=4)
        axA.set_title('(A) it0 — DEMO-DISTILLED warm start\n(no self-gathering yet)', fontsize=15)
    elif db is None:
        axA.set_title(f'(A) it{it}\nno valid windows gathered', fontsize=15)
    else:
        for q in db['paths']:
            axA.plot(q[:, 0], q[:, 1], '--', c='#333333', lw=1.9, alpha=.8, zorder=3)
        em = db['lab'] == 'easy'
        if em.any():                                   # easy: THIN green ring (trajectories stay visible)
            axA.scatter(db['pos'][em, 0], db['pos'][em, 1], c=db['sig'][em], cmap='viridis',
                        vmin=sig_lims[0], vmax=sig_lims[1], s=55, zorder=4,
                        edgecolors=C_E, linewidths=0.9)
        if (~em).any():                                # frontier: ring a bit THICKER than easy, on top
            axA.scatter(db['pos'][~em, 0], db['pos'][~em, 1], c=db['sig'][~em], cmap='viridis',
                        vmin=sig_lims[0], vmax=sig_lims[1], s=62, zorder=5,
                        edgecolors=C_F, linewidths=1.5)
        axA.set_title(f'(A) {len(db["lab"])} windows: {int(em.sum())} easy (thin ring) / '
                      f'{int((~em).sum())} frontier (thick ring)\n{len(db["paths"])} gathered rollouts (dashed)',
                      fontsize=15)
    sm = mpl.cm.ScalarMappable(cmap='viridis', norm=mpl.colors.Normalize(*sig_lims))
    cb = fig.colorbar(sm, ax=axA, fraction=0.042, pad=0.02)
    cb.set_label(r'$\sigma$', fontsize=15)
    # ---------- B: sigma histogram ----------
    axB = fig.add_subplot(gs[0, 1])
    if db is not None and it > 0:
        bins = np.linspace(sig_lims[0], sig_lims[1] + 1e-6, 22)
        em = db['lab'] == 'easy'
        if em.any():
            axB.hist(db['sig'][em], bins=bins, color=C_E, alpha=.7, label=f'easy {int(em.sum())}', zorder=2)
        if (~em).any():
            axB.hist(db['sig'][~em], bins=bins, color=C_F, alpha=.8,
                     label=f'frontier {int((~em).sum())}', zorder=3)
        axB.axvline(db['sig_plane'], color='k', ls=':', lw=2)
        axB.text(db['sig_plane'], axB.get_ylim()[1] * .95, r' $\sigma_q$', fontsize=13, va='top')
        axB.legend()
    else:
        axB.text(.5, .5, 'no samples', ha='center', va='center', transform=axB.transAxes, fontsize=15)
    axB.set_title(r'(B) gathered samples by $\sigma$'); axB.set_xlabel(r'$\sigma$ (GP novelty)')
    # ---------- C: 3D, sigma on Z ----------
    axC = fig.add_subplot(gs[0, 2], projection='3d')
    if db is not None and it > 0:
        em = db['lab'] == 'easy'
        for m, c, s_, al in ((em, C_E, 34, .28), ((~em), C_F, 20, .45)):
            if m.any():
                axC.scatter(db['marg'][m], db['prog'][m], db['sig'][m], c=c, s=s_, alpha=al, depthshade=True)
        ml = (min(-0.05, float(db['marg'].min()) - 0.02), max(.5, db['marg'].max()))
        pl = (0, max(.8, db['prog'].max())); zl = sig_lims
        m_thr = db['marg_plane']
        yy, zz = np.meshgrid(np.linspace(*pl, 2), np.linspace(*zl, 2))
        # frontier thresholds (blue/orange/purple)
        axC.plot_surface(np.full_like(yy, m_thr), yy, zz, color='#1f77b4', alpha=.15, shade=False)
        xx, zz2 = np.meshgrid(np.linspace(*ml, 2), np.linspace(*zl, 2))
        axC.plot_surface(xx, np.full_like(xx, db['prog_plane']), zz2, color='#ff7f0e', alpha=.15, shade=False)
        xx3, yy3 = np.meshgrid(np.linspace(*ml, 2), np.linspace(*pl, 2))
        axC.plot_surface(xx3, yy3, np.full_like(xx3, db['sig_plane']), color='#9467bd', alpha=.18, shade=False)
        # VALIDITY planes (red): SOCP safety boundary m=0, net-progress gate (valid2 0.10 / vpf 0.15)
        axC.plot_surface(np.full_like(yy, 0.0), yy, zz, color='#d62728', alpha=.10, shade=False)
        axC.plot_surface(xx, np.full_like(xx, args.vpf), zz2, color='#d62728', alpha=.10, shade=False)
        axC.set_xlim(*ml); axC.set_ylim(*pl); axC.set_zlim(*zl)
    axC.set_xlabel('SOCP margin', fontsize=17, labelpad=8); axC.set_ylabel('net-progress', fontsize=17, labelpad=8)
    axC.set_zlabel(r'$\sigma$', fontsize=18)
    qtxt = db['quantile'] if db is not None else .5
    axC.set_title(f'(C) $\\sigma$ on Z — RED: candidate cert boundary m=0, prog gate$\\geq${args.vpf}\n'
                  f'frontier AND-cell at q={qtxt:.2f}: $\\sigma\\geq\\sigma_q$ AND '
                  r'$m\leq m_{1-q}$ AND $p\geq p_q$', fontsize=13)
    axC.view_init(elev=18, azim=-55); axC.tick_params(labelsize=13)
    # ---------- D: 5 bins + arrows ----------
    axD = fig.add_subplot(gs[0, 3])
    g = lambda k, d=0: (rec.get(k, d) if rec else d) or 0
    ne, nf = g('n_easy'), g('n_frontier')
    be, bf, bd, dreq = g('batch_e'), g('batch_f'), g('batch_d'), g('demo_req', args.initial_demo_req)
    pe, pf = g('pile_e'), g('pile_f')                     # pile arms: buffer = the PILE
    bpe, bpf = g('batch_pe'), g('batch_pf')
    has_pile = rec is not None and 'pile_e' in rec
    if has_pile:
        vals = [ne, nf, pe, pf, dreq]
        arrows = [(0, be - bpe), (1, bf - bpf), (2, bpe), (3, bpf), (4, bd)]
        sub = 'buffer = PILE (FIFO, LRU draw)'
    else:
        vals = [ne, nf, ne, nf, dreq]                     # fresh-only: buffer == this iter's fresh pool
        arrows = [(2, be), (3, bf), (4, bd)]
        sub = 'fresh-only: buffer = this iter\'s pool'
    cols = [C_E, C_F, C_E, C_F, C_D]
    labs = ['fresh\neasy', 'fresh\nfrontier', 'buffer\neasy', 'buffer\nfrontier', 'demo\nrequired']
    x = np.arange(5)
    alphas = [.9, .9, .45, .45, .7]
    for xi, v, c, al in zip(x, vals, cols, alphas):
        axD.bar([xi], [v], color=c, alpha=al, edgecolor='k', linewidth=.8)
    top = max(max(vals), 1)
    for xi, v in zip(x, vals):
        axD.text(xi, v + top * .015, str(int(v)), ha='center', fontsize=14, fontweight='bold')
    for xi, used in arrows:                                # arrows: USED in this iter's batch
        axD.annotate(f'used {int(used)}', xy=(xi, vals[xi] + top * .10), xytext=(xi, vals[xi] + top * .30),
                     ha='center', fontsize=13, color='k',
                     arrowprops=dict(arrowstyle='<-', lw=2.2, color='k'))
    axD.set_xticks(x); axD.set_xticklabels(labs, fontsize=12)
    axD.set_ylim(0, top * 1.5)
    axD.set_title(f'(D) {sub}\nbatch used {int(be)}e+{int(bf)}f+{int(bd)}d = {int(be + bf + bd)}',
                  fontsize=15)
    # ---------- row 2: real-time traces up to t ----------
    T = np.array([r['iter'] for r in recs_upto], float)
    def tr(k):
        return np.array([r.get(k) if r.get(k) is not None else np.nan for r in recs_upto], float)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(T, tr('beta'), '-o', ms=4, c='#0072B2', lw=2.4)
    ax1.set_title(r'$\beta$ ($\sigma$-tilt; low = explore)'); ax1.set_ylim(0, 1.1)
    ax2 = fig.add_subplot(gs[1, 1])
    for k, c, lab, ls in (('batch_d', C_D, 'demo used', '-'), ('batch_e', C_E, 'fresh easy used', '-'),
                          ('batch_f', C_F, 'fresh frontier used', '-'),
                          ('n_easy', C_E, 'pool easy', '--'), ('n_frontier', C_F, 'pool frontier', '--')):
        ax2.plot(T, np.maximum(tr(k), 0.5), ls, marker='o', ms=3, c=c, lw=2.0, label=lab)
    ax2.set_yscale('log'); ax2.set_ylim(0.4, 900)
    ax2.legend(fontsize=10, ncol=2, loc='upper right'); ax2.set_title('samples: used (solid) vs pool (dashed)')
    ax3 = fig.add_subplot(gs[1, 2])
    tot = np.maximum(tr('batch_e') + tr('batch_f') + tr('batch_d'), 1)
    dfr = tr('demo_req') / float(args.batch_cap)
    for meas, req, c, lab in ((tr('batch_e') / tot, tr('mix_e') * (1 - dfr), C_E, 'easy'),
                              (tr('batch_f') / tot, tr('mix_f') * (1 - dfr), C_F, 'frontier'),
                              (tr('batch_d') / tot, dfr, C_D, 'demo')):
        ax3.plot(T, meas, '-', marker='o', ms=3, c=c, lw=2.4, label=f'{lab} measured')
        ax3.plot(T, req, '--', c=c, lw=1.4, alpha=.8)
    ax3.set_ylim(0, 1); ax3.legend(fontsize=10); ax3.set_title('batch mix ratio: measured (solid) vs required (dashed)')
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.plot(T, tr('lr'), '-o', ms=4, c='#D55E00', lw=2.4)
    ax4.set_yscale('log'); ax4.set_ylim(5e-6, 3e-4); ax4.set_title('learning rate')
    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(alpha=.3); ax.set_xlabel('iter'); ax.set_xlim(-1, args.n_max + 2)
        if it > 0:
            ax.axvline(it, color='k', lw=1.2, ls=':')
    wu = '   [WARM-UP: gather → pile only, NO update]' if (rec and rec.get('warmup')) else ''
    fig.suptitle(f'{args.title}   —   iteration {it}{wu}', fontsize=20, y=0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='../../results/hp_repr/pretrained_a32uni.pt')
    ap.add_argument('--title', default='Safe Flow Expansion — fixed AND-quantile curriculum')
    ap.add_argument('--vpf', type=float, default=0.15, help='validity net-progress gate plane (valid_prog_floor)')
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--iters', default='', help='comma list; default 0,1,3,5 then every 10')
    args = ap.parse_args()
    _rp = os.path.join(args.run, 'recipe.json')
    if os.path.exists(_rp):
        _recipe = json.load(open(_rp))
        _wp = _recipe.get('wall_plugs', 0)
        args.batch_cap = int(_recipe.get('batch', 16))
        args.initial_demo_req = int(round(float(_recipe.get('demo_frac', 0.0)) * args.batch_cap))
        if _wp:
            _step = 5.0 / 13.0
            _p4 = [(_step, -0.2, 0.2), (5.0 - _step, 5.2, 0.2), (-0.2, _step, 0.2), (5.2, 5.0 - _step, 0.2)]
            # 8-plug adds the ORIGIN-corner (crowds start; handled by start-eps) and GOAL-corner plugs
            # (5.2,5.0)&(5.0,5.2) — the latter sit ON the goal surface and cause the goal-overshoot
            # collisions the frontier curriculum must clean up. Draw them or the story is invisible.
            _p8 = _p4 + [(0.0, -0.2, 0.2), (-0.2, 0.0, 0.2), (5.2, 5.0, 0.2), (5.0, 5.2, 0.2)]
            globals()['EXTRA_OBS'] = _p4[:2] if _wp == 2 else (_p8 if _wp == 8 else _p4)
            globals()['env'] = _apply_wall_plugs_eval(env, _wp)
            env.x0 = torch.tensor([0.05, 0.05, 0.0, 0.0], dtype=env.x0.dtype)
    else:
        args.batch_cap = 16
        args.initial_demo_req = 0
    recs = [json.loads(l) for l in open(os.path.join(args.run, 'probe.jsonl'))]
    # Add the real pre-gather recipe state so the requested front-loaded demo
    # fraction and its immediate decay are visible in the bottom-row trace.
    r0 = dict(iter=0, beta=(recs[0].get('beta', .2) if recs else .2), n_easy=0, n_frontier=0,
              batch_e=0, batch_f=0, batch_d=0, demo_req=args.initial_demo_req,
              mix_e=(recs[0].get('mix_e', .4) if recs else .4),
              mix_f=(recs[0].get('mix_f', .6) if recs else .6),
              lr=(recs[0].get('lr', 5e-6) if recs else 5e-6))
    recs = [r0] + recs
    by_it = {r['iter']: r for r in recs}
    n_max = max(by_it)
    args.n_max = n_max
    its = ([int(x) for x in args.iters.split(',')] if args.iters else
           [0, 1, 3, 5] + list(range(10, n_max + 1, 10)))
    # sigma scale across selected iters
    allsig = [load_db(args.run, it)['sig'] for it in its if it > 0 and load_db(args.run, it) is not None]
    sig_lims = (0.0, float(np.percentile(np.concatenate(allsig), 98)) if allsig else 1.0)
    # pretrained rollout for it0
    dev = 'cpu'
    pol, _ = HP.load_hp(args.ckpt, device=dev)
    pre = np.asarray(GR.fm_deploy(pol, env, args.gamma, T=250, temp=1.0, nfe=8, device=dev)['path'])
    FR = os.path.join(HERE, 'video', '_frames')
    os.makedirs(FR, exist_ok=True)
    for f in glob.glob(FR + '/*.png'):
        os.remove(f)
    fig = plt.figure(figsize=(26, 13))
    fig.patch.set_facecolor('white')
    k = 0
    for it in its:
        db = load_db(args.run, it) if it > 0 else None
        rec = by_it.get(it)
        upto = [r for r in recs if r['iter'] <= it]
        frame(fig, it, db, rec, upto, pre, sig_lims, args)
        for _ in range(2):                                   # 2 copies @2fps -> 1 s per iteration
            fig.savefig(f'{FR}/f{k:04d}.png', dpi=78, facecolor='white', transparent=False); k += 1
        print(f'frame it{it}', flush=True)
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-framerate', '2', '-i', f'{FR}/f%04d.png',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', '-c:v', 'libx264',
                    args.out], check=True, capture_output=True)
    print('SAVED', args.out)


if __name__ == '__main__':
    main()
