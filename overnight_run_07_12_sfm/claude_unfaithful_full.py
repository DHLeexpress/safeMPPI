"""UNFAITHFUL full launch: p=0.5, E=4, lr 1e-5, 10 rounds, M10 every round."""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
import torch
import _paths
import claude_afe_driver as AD
import claude_afe_fidelity as AF
import claude_continuation as CC
import claude_corrected_study as CS
import claude_unfaithful_pilot as UP
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_store as OS
import sfm_b1_store as BS

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True)
ap.add_argument("--workers", type=int, default=28)
ap.add_argument("--eval-workers", type=int, default=12)
a = ap.parse_args()
out = os.path.abspath(a.outdir); os.makedirs(out, exist_ok=True)
payload = torch.load(UP.TEACHER, map_location="cpu", weights_only=False)
_, teacher_records = CS._teacher_records(payload)
b9 = OS.ExecutedRoundShard.load(AD.B9_SHARD)
cache = AF.VerifierCache()
policy, _ = GPS.load_sfm_policy(AD.R1, device="cuda:0")
BS.configure_expansion_trainability(policy)
opt = torch.optim.Adam([q for q in policy.parameters() if q.requires_grad], lr=1e-5)
prev = b9
with ProcessPoolExecutor(max_workers=a.workers) as ex:
    for k in range(1, 11):
        ck = os.path.join(out, f"round_{k:02d}.pt")
        sp = os.path.join(out, f"shard_{k:02d}.pt")
        if os.path.isfile(ck):
            policy, _ = GPS.load_sfm_policy(ck, device="cuda:0")
            BS.configure_expansion_trainability(policy)
            opt = torch.optim.Adam([q for q in policy.parameters() if q.requires_grad], lr=1e-5)
            prev = AF.CertifiedQueryShard.load(sp); continue
        policy.eval()
        shard, ginfo = AD.gather_round(policy, prev, 1 + k, 0.5, "cuda:0", ex, cache, sp)
        records, w = UP.mixed_weights([prev, shard], teacher_records, 0.5)
        UP.replay(policy, opt, records, w, 4, 128, "cuda:0", AF.SEED + 77 * k)
        BX._save_checkpoint(policy, ck, dict(arm="p05_ep4_full", round=k, UNFAITHFUL=True))
        print(json.dumps(dict(round=k, Dplus=ginfo["Dplus"], outcomes=ginfo["outcomes"])), flush=True)
        prev = shard
specs = [(AD.R1, os.path.join(out, "eval_r0"))] + [
    (os.path.join(out, f"round_{k:02d}.pt"), os.path.join(out, f"eval_r{k}")) for k in range(1, 11)]
met = CC.evaluate_checkpoints(specs, cache_dir=os.path.join(out, "m10_cache"),
                              workers_each=a.eval_workers, wave=5, gpu=3)
with open(os.path.join(out, "FULL_M10.json"), "w") as s:
    json.dump({("r1" if p == AD.R1 else os.path.basename(p)[:-3]): met[p] for p, _ in specs},
              s, indent=1, allow_nan=False, default=float)
print("FULL_M10_DONE", flush=True)
