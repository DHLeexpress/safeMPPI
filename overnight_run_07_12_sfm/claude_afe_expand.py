"""Expansion stage + M50/M100 funnel for the two promoted AFE recipes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os

import torch

import _paths  # noqa: F401
import claude_afe_driver as AD
import claude_afe_fidelity as AF
import claude_continuation as CC
import claude_corrected_distill as CD
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_store as OS
import sfm_b1_store as BS

RECIPES = (dict(name="ess0p5_U_d1", ess=0.5), dict(name="ess0p25_U_d1", ess=0.25))
KEY = lambda m: (m["CR"], -m["Validity"], -(m["clearance"] or 0),
                 -m["SR"], m["timeout"], m["time"] or 1e9)


def run(args):
    out = os.path.abspath(args.outdir)
    device = "cuda:0"
    cache = AF.VerifierCache()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in RECIPES:
            rdir = os.path.join(out, f"expand_{rec['name']}")
            os.makedirs(rdir, exist_ok=True)
            policy, _ = GPS.load_sfm_policy(AD.R1, device=device)
            BS.configure_expansion_trainability(policy)
            opt = torch.optim.Adam(
                [p for p in policy.parameters() if p.requires_grad], lr=1e-5)
            prev = OS.ExecutedRoundShard.load(AD.B9_SHARD)
            for k in range(1, 11):
                ck = os.path.join(rdir, f"round_{k:02d}.pt")
                sp = os.path.join(rdir, f"shard_{k:02d}.pt")
                if os.path.isfile(ck):
                    policy, _ = GPS.load_sfm_policy(ck, device=device)
                    BS.configure_expansion_trainability(policy)
                    opt = torch.optim.Adam([p for p in policy.parameters()
                                            if p.requires_grad], lr=1e-5)
                    prev = AF.CertifiedQueryShard.load(sp)
                    continue
                policy.eval()
                shard, ginfo = AD.gather_round(
                    policy, prev, 1 + k, rec["ess"], device, ex, cache, sp)
                info = AF.certified_replay(
                    policy, opt, [prev, shard], objective="U", epochs=4,
                    batch=128, device=device, seed=AF.SEED + k)
                BX._save_checkpoint(policy, ck, dict(recipe=rec, round=k))
                print(json.dumps(dict(recipe=rec["name"], round=k,
                                      **{x: ginfo[x] for x in ("Dplus", "D")},
                                      loss=info["losses"][-1])), flush=True)
                prev = shard
    # M10 evolution + selection
    table = {}
    for rec in RECIPES:
        rdir = os.path.join(out, f"expand_{rec['name']}")
        specs = [(AD.R1, os.path.join(out, "eval_r1"))] + [
            (os.path.join(rdir, f"round_{k:02d}.pt"),
             os.path.join(rdir, f"eval_r{k}")) for k in (1, 2, 5, 10)]
        met = CC.evaluate_checkpoints(
            specs, cache_dir=os.path.join(out, "m10_cache"),
            workers_each=args.eval_workers, wave=5, gpu=args.gpu)
        rows = {("r0" if p == AD.R1 else p.split("_")[-1][:-3]): met[p]
                for p, _ in specs}
        table[rec["name"]] = rows
        best = min(
            [k for k in rows if k != "r0"], key=lambda k: KEY(rows[k]))
        table[rec["name"] + "_best"] = best
    with open(os.path.join(out, "EXPANSION_M10.json"), "w") as s:
        json.dump(table, s, indent=1, allow_nan=False, default=float)
    # M50 both, winner, M100
    finals = {}
    for rec in RECIPES:
        best = table[rec["name"] + "_best"]
        k = int(best[1:]) if best != "r0" else 1
        finals[rec["name"]] = os.path.join(
            out, f"expand_{rec['name']}", f"round_{k:02d}.pt")
    m50specs = [(AD.R1, os.path.join(out, "m50_r1"))] + [
        (p, os.path.join(out, f"m50_{n}")) for n, p in finals.items()]
    AD.M10_EP0, AD.M10_SEED = AD.M50_EP0, AD.M50_SEED
    CC.DEV_EP0, CC.DEV_NOISE_SEED, CC.DEV_M = AD.M50_EP0, AD.M50_SEED, 50
    m50 = CC.evaluate_checkpoints(
        m50specs, cache_dir=os.path.join(out, "m50_cache"),
        workers_each=args.eval_workers, wave=3, gpu=args.gpu)
    winner = min(finals, key=lambda n: KEY(m50[finals[n]]))
    CC.DEV_EP0, CC.DEV_NOISE_SEED, CC.DEV_M = AD.M100_EP0, AD.M100_SEED, 100
    m100 = CC.evaluate_checkpoints(
        [(AD.R1, os.path.join(out, "m100_r1")),
         (finals[winner], os.path.join(out, "m100_winner"))],
        cache_dir=os.path.join(out, "m100_cache"),
        workers_each=args.eval_workers * 2, wave=2, gpu=args.gpu)
    with open(os.path.join(out, "FUNNEL_FINAL.json"), "w") as s:
        json.dump(dict(m50={n: m50[p] for n, p in finals.items()},
                       m50_r1=m50[AD.R1], winner=winner,
                       winner_checkpoint=finals[winner],
                       m100_r1=m100[AD.R1],
                       m100_winner=m100[finals[winner]]),
                  s, indent=1, allow_nan=False, default=float)
    print(json.dumps(dict(winner=winner, m100_r1=m100[AD.R1],
                          m100_winner=m100[finals[winner]]),
          allow_nan=False, default=float), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--eval-workers", type=int, default=12)
    ap.add_argument("--gpu", type=int, default=3)
    run(ap.parse_args())
