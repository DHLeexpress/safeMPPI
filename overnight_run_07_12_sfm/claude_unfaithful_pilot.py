"""UNFAITHFUL pilot (explicitly directed): mix UNVERIFIED goal-directed
controller windows (no SOCP gate) into replay at portion p, corrected
whole-dataset low-dose steps, to keep goal-approach while gaining safety.
Arms: p in {0.3, 0.5} x epochs in {2, 4}, lr 1e-5, 3 rounds each from r1.
Labeled UNFAITHFUL everywhere; never mixed into certified stores.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_afe_driver as AD
import claude_afe_fidelity as AF
import claude_continuation as CC
import claude_corrected_study as CS
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_store as OS
import sfm_b1_store as BS

TEACHER = ("/data3/research1/claude_sfm_corrected_privileged_distill_"
           "736fa6c/phase12/D_MPC_corrected.pt")


def mixed_weights(shards, teacher_records, p):
    cert = [(s, r) for s in shards for r in s.Dplus]
    hm_c, _ = BS.hierarchy_mass(cert)
    hm_t, _ = BS.hierarchy_mass(teacher_records)
    w = {k: (1 - p) * v for k, v in hm_c.items()}
    w.update({k: p * v for k, v in hm_t.items()})
    return cert + teacher_records, w


def replay(policy, opt, records, w, epochs, batch, device, seed):
    policy.train()
    for e in range(epochs):
        opt.zero_grad(set_to_none=True)
        for st in range(0, len(records), batch):
            v = records[st:st + batch]
            g, l, h, c = BS._tensor_batch(v, device)
            ww = torch.as_tensor(
                [len(v) * w[(id(a), int(r["query_id"]))] for a, r in v],
                dtype=c.dtype, device=device)
            torch.manual_seed(seed + e * 999983 + st)
            policy.cfm_loss(c, policy.ctx_from(g, l, h),
                            weights=ww).backward()
        opt.step()
    policy.eval()


def run(args):
    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)
    device = "cuda:0"
    payload = torch.load(TEACHER, map_location="cpu", weights_only=False)
    holder, teacher_records = CS._teacher_records(payload)
    b9 = OS.ExecutedRoundShard.load(AD.B9_SHARD)
    cache = AF.VerifierCache()
    arms = [dict(p=p, ep=ep) for p in (0.3, 0.5) for ep in (2, 4)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for a in arms:
            name = f"p{a['p']}_ep{a['ep']}".replace(".", "")
            policy, _ = GPS.load_sfm_policy(AD.R1, device=device)
            BS.configure_expansion_trainability(policy)
            opt = torch.optim.Adam(
                [q for q in policy.parameters() if q.requires_grad],
                lr=1e-5)
            prev = b9
            for k in range(1, 4):
                sp = os.path.join(out, f"{name}_shard{k}.pt")
                shard, ginfo = AD.gather_round(
                    policy, prev, 1 + k, 0.5, device, ex, cache, sp)
                records, w = mixed_weights(
                    [prev, shard], teacher_records, a["p"])
                replay(policy, opt, records, w, a["ep"], 128, device,
                       AF.SEED + k)
                ck = os.path.join(out, f"{name}_r{k}.pt")
                BX._save_checkpoint(policy, ck, dict(arm=name, round=k,
                                                     UNFAITHFUL=True))
                print(json.dumps(dict(arm=name, round=k,
                                      Dplus=ginfo["Dplus"])), flush=True)
                prev = shard
    specs = [(AD.R1, os.path.join(out, "eval_r1"))] + [
        (os.path.join(out, f"p{p}_ep{e}".replace(".", "") + f"_r{k}.pt"),
         os.path.join(out, f"eval_p{p}_ep{e}_r{k}".replace(".", "")))
        for p in (0.3, 0.5) for e in (2, 4) for k in (1, 3)]
    met = CC.evaluate_checkpoints(
        specs, cache_dir=os.path.join(out, "m10_cache"),
        workers_each=args.eval_workers, wave=5, gpu=3)
    with open(os.path.join(out, "PILOT.json"), "w") as s:
        json.dump({os.path.basename(p): met[p] for p, _ in specs}, s,
                  indent=1, allow_nan=False, default=float)
    print(json.dumps({os.path.basename(p): {
        k: met[p][k] for k in ("SR", "CR", "Validity", "clearance", "time")}
        for p, _ in specs}, allow_nan=False, default=float), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--eval-workers", type=int, default=12)
    run(ap.parse_args())
