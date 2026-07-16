"""Re-roll the ACTUAL druni SafeMPPI trajectories (same starts + seeds as gen_uniform_data -> identical paths)
and save them per gamma for the 566-traj figure. One process per gamma. Usage: --gamma 0.5 [--out-dir ...]"""
import sys, os
sys.path.insert(0, '/home/dohyun/projects/cfm_mppi/overnight_run_07_06')
import _paths  # noqa
import argparse, time
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '16')))
import grid_scene as GS
from gen_uniform_data import uniform_starts, rollout_from

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma', type=float, required=True)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results', 'uni_trajs'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    env = GS.make_grid(); cfg = GS.mode1_config()
    starts = uniform_starts()
    paths, ok_flags = [], []
    t0 = time.time()
    for i, s in enumerate(starts):
        states, _ = rollout_from(env, cfg, args.gamma, s, seed=i)     # same seed=i as the dataset gen
        ok, _ = GS.is_success(states[:, :2], env)
        paths.append(states[:, :2].astype(np.float32)); ok_flags.append(bool(ok))
        if (i + 1) % 100 == 0:
            print(f'g{args.gamma}: {i+1}/{len(starts)} ({(time.time()-t0)/(i+1):.2f}s/start)', flush=True)
    np.savez_compressed(os.path.join(args.out_dir, f'paths_g{args.gamma}.npz'),
                        paths=np.asarray(paths, dtype=object), ok=np.asarray(ok_flags),
                        starts=starts.astype(np.float32), allow_pickle=True)
    print(f'SAVED g{args.gamma}: {sum(ok_flags)}/{len(starts)} ok ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
