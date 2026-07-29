"""Staged AFE-fidelity funnel: phase0 -> screen -> expand -> M50 -> M100."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_afe_fidelity as AF
import claude_continuation as CC
import claude_corrected_distill as CD
import claude_mpc_pool as MP
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_eval as OEV
import sfm_b1_offline_store as OS
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_kazuki as KZ
import sfm_protocol as SP
import sfm_scene as SS

R1 = ("/data3/research1/claude_sfm_best_recipe_f06e8dd/stageB/"
      "B9_margin_origrec_e100/round_01.pt")
B9_SHARD = ("/data3/research1/claude_sfm_best_recipe_f06e8dd/stageB/"
            "B9_margin_origrec_e100/round_shards/round_01.pt")
PH0_EP0, M10_EP0, M10_SEED = 460_000, 390_000, 20_260_740
M50_EP0, M50_SEED, M100_EP0, M100_SEED = 470_000, 20_260_750, 480_000, 20_260_751
DOSES = (dict(name="d1", lr=1e-5, epochs=4), dict(name="d2", lr=3e-5, epochs=4))


def _phi(policy):
    phi = copy.deepcopy(policy).eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    return phi


def _beta(policy, gp, scenarios, gammas, device, ess, executor):
    vectors = []
    env = SS.scene_profile("double_density_velocity_ood")
    for s in scenarios[:4]:
        for g in gammas:
            humans = SS.make_humans(int(s), 0, env["n_ped"],
                                    tuple(env["ped_speed_range"]))
            ctx = _context0(s, g)
            pool = AF.build_pool(policy, ctx, humans, device)
            feats = AF.pool_features(policy, ctx, pool["plans"],
                                     pool["x0"], device)
            order = torch.as_tensor(AF.keyed_rng(
                AF.SEED, "beta", s, g,
            ).permutation(len(feats)))
            vectors.extend(gp.sequential_score_vectors(
                feats, order, AF.B_BUDGET,
            ))
    beta, achieved = BR.solve_beta(vectors, target=float(ess))
    return float(beta), float(achieved)


def _context0(scenario, gamma):
    import grid_feats as GF
    import sfm_hp_history as HH
    env = SS.scene_profile("double_density_velocity_ood")
    humans = SS.make_humans(int(scenario), 0, env["n_ped"],
                            tuple(env["ped_speed_range"]))
    ped_xy, ped_vel = SS.collect_humans(humans)
    state = np.zeros(4, np.float32)
    obstacles = np.concatenate([
        ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32)], axis=1)
    hp10 = HH.HpHistory().append(torch.as_tensor(GF.axis_grid(
        state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE)))
    return dict(scenario_id=int(scenario), gamma=float(gamma), step=0,
                state=state, hp10=hp10.numpy().astype(np.float32),
                low5=np.asarray(GF.low5(state, SS.GOAL, gamma), np.float32),
                hist=np.zeros((16, 2), np.float32),
                ped_xy=ped_xy, ped_vel=ped_vel)


def _summ(rows):
    n = len(rows)
    succ = [r for r in rows if r["status"] == "success"]
    cl = [r["min_clearance"] for r in succ]
    tt = [r["steps"] * SS.DT for r in succ]
    return dict(
        n=n, SR=len(succ) / n,
        CR=sum(r["status"] == "collision" for r in rows) / n,
        NVP=sum(r["status"] == "nvp" for r in rows) / n,
        timeout=sum(r["status"] == "timeout" for r in rows) / n,
        clearance=float(np.mean(cl)) if cl else None,
        time=float(np.mean(tt)) if tt else None,
    )


def phase0(out, policy, gp, beta, device, executor, cache):
    rows = {"F": [], "B8": []}
    support = []
    for gamma in SP.GAMMAS:
        for ep in range(PH0_EP0, PH0_EP0 + 10):
            for mode, key in (("full", "F"), ("b8", "B8")):
                r = AF.run_certified_episode(
                    policy, ep, gamma, mode=mode, gp=gp, beta=beta,
                    device=device, executor=executor, cache=cache)
                rows[key].append(dict(
                    episode=ep, gamma=gamma, status=r["status"],
                    steps=r["steps"], min_clearance=r["min_clearance"]))
                if mode == "full":
                    support.append(dict(
                        episode=ep, gamma=gamma,
                        pool_pos=[c for c in r["pool_positive_counts"]
                                  if c is not None]))
        print(json.dumps(dict(phase0_gamma=gamma,
                              F=_summ([x for x in rows["F"]
                                       if x["gamma"] == gamma]))), flush=True)
    pos_counts = [c for s in support for c in s["pool_pos"]]
    report = dict(
        F=_summ(rows["F"]), B8=_summ(rows["B8"]),
        p_pool_has_positive=float(np.mean([c > 0 for c in pos_counts])),
        mean_pool_positive_multiplicity=float(np.mean(pos_counts)),
        rows=rows, cache=dict(hits=cache.hits, misses=cache.misses),
    )
    with open(os.path.join(out, "phase0.json"), "w") as s:
        json.dump(report, s, indent=1, allow_nan=False, default=float)
    print(json.dumps(dict(F=report["F"], B8=report["B8"],
                          p_pos=report["p_pool_has_positive"])), flush=True)
    return report


def gather_round(policy, prev_shard, round_index, ess, device, executor,
                 cache, out_path):
    gp, _ = AF.build_round_gp(_phi(policy), prev_shard, device=device,
                              ess_target=ess, round_seed=round_index)
    scen = SP.expansion_scenarios(round_index)
    beta, ach = _beta(policy, gp, list(scen), list(SP.GAMMAS)[:2], device,
                      ess, executor)
    shard = AF.CertifiedQueryShard(round_index)
    outcomes = []
    for s in scen:
        for g in SP.GAMMAS:
            r = AF.run_certified_episode(
                policy, s, g, mode="b8", gp=gp, beta=beta, device=device,
                executor=executor, cache=cache, shard=shard)
            outcomes.append(r["status"])
    shard.save(out_path)
    return shard, dict(beta=beta, ess=ach,
                       outcomes={k: outcomes.count(k) for k in set(outcomes)},
                       Dplus=len(shard.Dplus), D=len(shard.windows))


def run(args):
    device = "cuda:0"
    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)
    policy, _ = GPS.load_sfm_policy(R1, device=device)
    policy.eval()
    b9 = OS.ExecutedRoundShard.load(B9_SHARD)
    cache = AF.VerifierCache()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        if not os.path.isfile(os.path.join(out, "phase0.json")):
            gp0, _ = AF.build_round_gp(_phi(policy), b9, device=device,
                                       ess_target=0.5, round_seed=0)
            beta0, _ = _beta(policy, gp0, [PH0_EP0], list(SP.GAMMAS),
                             device, 0.5, executor)
            phase0(out, policy, gp0, beta0, device, executor, cache)
        # ---- screening ----
        arms = []
        for ess in (0.25, 0.5):
            tag = str(ess).replace(".", "p")
            spath = os.path.join(out, f"screen_shard_ess{tag}.pt")
            if os.path.isfile(spath):
                shard = AF.CertifiedQueryShard.load(spath)
                ginfo = {}
            else:
                shard, ginfo = gather_round(
                    policy, b9, 2, ess, device, executor, cache, spath)
                print(json.dumps(dict(screen_gather=tag, **ginfo)),
                      flush=True)
            for obj in ("U", "G"):
                for dose in DOSES:
                    name = f"ess{tag}_{obj}_{dose['name']}"
                    ck = os.path.join(out, "arms", f"{name}.pt")
                    if not os.path.isfile(ck):
                        p2, _ = GPS.load_sfm_policy(R1, device=device)
                        BS.configure_expansion_trainability(p2)
                        opt = torch.optim.Adam(
                            [q for q in p2.parameters()
                             if q.requires_grad], lr=dose["lr"])
                        info = AF.certified_replay(
                            p2, opt, [b9, shard], objective=obj,
                            epochs=dose["epochs"], batch=128,
                            device=device, seed=AF.SEED)
                        os.makedirs(os.path.dirname(ck), exist_ok=True)
                        BX._save_checkpoint(p2, ck, dict(arm=name,
                                                         info=info))
                        del p2
                        torch.cuda.empty_cache()
                    arms.append(dict(name=name, checkpoint=ck))
    specs = [(R1, os.path.join(out, "eval_r1"))] + [
        (a["checkpoint"], os.path.join(out, f"eval_{a['name']}"))
        for a in arms]
    metrics = CC.evaluate_checkpoints(
        specs, cache_dir=os.path.join(out, "m10_cache"),
        workers_each=args.eval_workers, wave=args.eval_wave, gpu=args.gpu)
    r1m = metrics[R1]
    table = [dict(name="r1", m10=r1m)] + [
        dict(name=a["name"], m10=metrics[a["checkpoint"]],
             checkpoint=a["checkpoint"]) for a in arms]

    def dominated(row):
        m = row["m10"]
        for o in table:
            if o is row:
                continue
            q = o["m10"]
            ge = (q["CR"] <= m["CR"] and q["Validity"] >= m["Validity"]
                  and (q["clearance"] or 0) >= (m["clearance"] or 0))
            gt = (q["CR"] < m["CR"] or q["Validity"] > m["Validity"]
                  or (q["clearance"] or 0) > (m["clearance"] or 0))
            if ge and gt:
                return True
        return False

    order = sorted(
        [r for r in table if r["name"] != "r1" and not dominated(r)],
        key=lambda r: (r["m10"]["CR"], -r["m10"]["Validity"],
                       -(r["m10"]["clearance"] or 0), -r["m10"]["SR"],
                       r["m10"]["timeout"],
                       r["m10"]["time"] or 1e9))
    promoted = order[:2]
    with open(os.path.join(out, "SCREEN.json"), "w") as s:
        json.dump(dict(table=table,
                       promoted=[p["name"] for p in promoted]),
                  s, indent=1, allow_nan=False, default=float)
    print(json.dumps(dict(promoted=[p["name"] for p in promoted],
                          r1=r1m)), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--eval-workers", type=int, default=12)
    ap.add_argument("--eval-wave", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=3)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
