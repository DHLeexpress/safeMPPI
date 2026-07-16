"""Per-gamma valid2 rate of a checkpoint on the 8-plug scene (gather-side signal).

The emergent-gamma curriculum trains each gamma on ITS OWN certified windows; the strict low gammas
(0.1/0.2) have none from the pretrained and can only join once the SHARED weights lift their clearance
enough to certify. This measures exactly that: the fraction of deploys whose whole trajectory is valid2
(taskspace AND net-progress AND SOCP) at each gamma. Rising 0.1/0.2 here == the low gammas are joining.

  python analysis/per_gamma_valid.py --ckpt results/p2/fsw_b03/final.pt --M 25
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import grid_hp_expt as HP, grid_scene as GS, grid_rollout as GR, grid_metrics2 as GM2, grid_expand_hardtail as HT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--M", type=int, default=25)
    ap.add_argument("--wall-plugs", type=int, default=8)
    ap.add_argument("--start-eps", type=float, default=0.05)
    ap.add_argument("--reach", type=float, default=0.2)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pol, _ = HP.load_hp(args.ckpt, device="cuda"); pol.eval()
    env = GS.make_grid(); HT._apply_wall_plugs(env, args.wall_plugs)
    if args.start_eps > 0:
        env.x0 = torch.tensor([args.start_eps, args.start_eps, 0., 0.], dtype=env.x0.dtype)
    out = {}
    print(f"per-gamma valid2 (8-plug, eps{args.start_eps}, reach{args.reach}, M{args.M}) {args.tag}")
    for g in args.gammas:
        ok = 0
        for s in range(args.M):
            torch.manual_seed(s)
            p = np.asarray(GR.fm_deploy(pol, env, float(g), T=250, temp=1.0, nfe=8, device="cuda",
                                        reach=args.reach)["path"], float)
            if GM2.traj_valid2(p, env, float(g)):
                ok += 1
        out[str(g)] = 100.0 * ok / args.M
        print(f"  g{g:.1f}: {out[str(g)]:5.1f}%  ({ok}/{args.M})")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
