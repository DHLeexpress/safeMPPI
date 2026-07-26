"""Stage A local failure diagnosis for the SFM B1 offline recipe study.

Diagnostic-only tooling; trains nothing unless the ``update`` subcommand is
invoked, and never touches raw evaluation semantics.

Subcommands
-----------
``mine``    — classify failure contexts of an archived ExecutedRoundShard.
``branch``  — instrumented closed-loop branch trace of declared episodes:
              every one of the K=16 flow candidates is exact-verified (a
              diagnostic superset of the B=4 budget), B-selection replicates
              the round-1 acquisition (empty GP buffer, calibrated beta),
              both execution selectors are evaluated, and the episode
              advances with the B1 executed action (chosen selector,
              raw-continuation at NVP) exactly as the offline collector.
``update``  — apply exactly one replay update (declared knobs, opt-in data
              intervention) to the exact r0 checkpoint using an archived
              round-1 shard, and save the updated checkpoint.
``compare`` — before/after tables from two ``branch`` traces.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_offline_aug as AUG
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_expand as BX
import sfm_b1_eval as BE
import sfm_b1_full_episode_audit as FA
import sfm_b1_offline_exec as OE
import sfm_b1_offline_store as OS
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


def _write_json(path, payload):
    OE._write_json(path, payload)


# ---------------------------------------------------------------- mine ----

def mine(args):
    shard = OS.ExecutedRoundShard.load(args.shard)
    pop_a, pop_b, stats = AUG.tag_populations(shard)
    nvp = [w for w in shard.windows if w.get("nvp_context")]
    collisions = [w for w in shard.windows if w.get("collision_after_action")]
    traps = [w for w in shard.windows if w.get("trap_event")]
    by_gamma = Counter(
        str(shard.contexts[w["context_id"]]["gamma"]) for w in pop_b
    )
    payload = dict(
        shard=os.path.abspath(args.shard),
        stats=stats,
        NVP_contexts=len(nvp),
        collision_windows=len(collisions),
        trap_windows=len(traps),
        popB_by_gamma=dict(by_gamma),
        examples=dict(
            nvp=[_ctx_key(shard, w) for w in nvp[:20]],
            collision=[_ctx_key(shard, w) for w in collisions[:20]],
            trap=[_ctx_key(shard, w) for w in traps[:20]],
        ),
    )
    _write_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in (
        "NVP_contexts", "collision_windows", "trap_windows")}, indent=1))


def _ctx_key(shard, window):
    context = shard.contexts[int(window["context_id"])]
    return dict(
        scenario=int(context["scenario_id"]), gamma=float(context["gamma"]),
        step=int(context["step"]),
    )


# -------------------------------------------------------------- branch ----

@torch.no_grad()
def branch(args):
    device = args.device
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=device)
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)
    cfg = OE.OfflineConfig(alpha=0.0, exposure_epochs=1, rounds=1, smoke=True)
    environment = SS.scene_profile(cfg.scene_profile)
    pairs = [
        (int(s), float(g))
        for s in args.scenarios for g in args.gammas
    ]
    replicas = [
        BX.Replica(
            scenario_id, gamma, n_ped=environment["n_ped"],
            ped_speed_range=tuple(environment["ped_speed_range"]),
        )
        for scenario_id, gamma in pairs
    ]
    # Round-1 acquisition state: empty GP buffer + calibrated beta,
    # replicated exactly as the collector does at round 1.
    gp = BR.RBFGP(float(args.ell), float(cfg.gp_lam))
    beta, ess = OE._calibrate_beta(
        phi_policy, gp, replicas, cfg, device, round_i=1,
    )
    traces = []
    outcomes = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for step in range(int(cfg.T)):
            live, batch = BX._stack_prepared(
                [r for r in replicas if r.alive], device,
            )
            if not live:
                break
            windows, contexts, x0 = OE._keyed_windows(
                policy, live, batch, K=cfg.K, round_i=1, step=step,
                source="K", seed=cfg.seed, nfe=cfg.nfe, temp=cfg.temp,
            )
            raw_windows, _, raw_x0 = OE._keyed_windows(
                policy, live, batch, K=1, round_i=1, step=step,
                source="raw_continuation", seed=cfg.seed, nfe=cfg.nfe,
                temp=cfg.temp,
            )
            raw_windows = raw_windows[:, 0]
            windows_np = windows.detach().cpu().numpy()
            raw_np = raw_windows.detach().cpu().numpy()
            features = OE._features_from_x0(
                phi_policy, windows, contexts, x0, cfg.phi_s,
            )
            selected_by_context, sigmas = [], []
            for index, replica in enumerate(live):
                generator = torch.Generator(device=features.device)
                generator.manual_seed(OE._keyed_seed(
                    cfg.seed, 1, replica.scenario_id,
                    f"{replica.gamma:.8f}", step, "acquisition",
                ))
                selected, trace = gp.sequential_acquire(
                    features[index], cfg.B, beta, generator=generator,
                )
                selected_by_context.append(list(map(int, selected)))
                sigmas.append([float(r["chosen_sigma"]) for r in trace])
            # Diagnostic superset: verify ALL K candidates + the raw plan.
            tasks = []
            for index, replica in enumerate(live):
                prepared = replica.prepared
                for k in range(cfg.K):
                    tasks.append((
                        index, k, prepared["state"], windows_np[index, k],
                        prepared["ped_xy"], prepared["ped_vel"],
                        replica.gamma,
                    ))
                tasks.append((
                    index, -1, prepared["state"], raw_np[index],
                    prepared["ped_xy"], prepared["ped_vel"], replica.gamma,
                ))
            results = list(executor.map(SM.verify_in_worker, tasks))
            by_context = {}
            for index, k, result in results:
                by_context.setdefault(int(index), {})[int(k)] = result

            for index, replica in enumerate(live):
                prepared = replica.prepared
                rows = []
                for k in range(cfg.K):
                    result = by_context[index][k]
                    margin, _, _ = BC.nominal_hp_margin(
                        prepared["state"], windows_np[index, k][0],
                        prepared["ped_xy"], replica.gamma,
                    )
                    rows.append(dict(
                        candidate_id=k,
                        y=int(result.get("y", 0)) if result.get("resolved")
                        else None,
                        resolved=bool(result.get("resolved")),
                        hp_margin=float(margin),
                        in_B=k in selected_by_context[index],
                        controls=windows_np[index, k],
                        result=result,
                    ))
                # B1 execution semantics restricted to the B queried rows.
                query_rows = [
                    dict(
                        candidate_id=row["candidate_id"],
                        acquisition_step=selected_by_context[index].index(
                            row["candidate_id"],
                        ),
                        controls=row["controls"],
                        result=row["result"],
                        mode=None,
                        sigma=sigmas[index][
                            selected_by_context[index].index(
                                row["candidate_id"],
                            )
                        ],
                    )
                    for row in rows if row["in_B"] and row["resolved"]
                ]
                chosen = {}
                for selector in ("margin", "safemppi_cost"):
                    chosen[selector] = BC.select_admissible(
                        [dict(r) for r in query_rows], selector=selector,
                        state=prepared["state"], ped_xy=prepared["ped_xy"],
                        ped_vel=prepared["ped_vel"], gamma=replica.gamma,
                    )
                execute = chosen[args.selector]
                raw_result = by_context[index][-1]
                if execute is None:
                    controls = raw_np[index]
                    executed_y = (
                        int(raw_result.get("y", 0))
                        if raw_result.get("resolved") else None
                    )
                    source = "raw_continuation"
                else:
                    controls = np.asarray(execute["controls"], np.float32)
                    executed_y = int(execute["result"]["y"])
                    source = f"verified_{args.selector}"
                k_positive = sum(1 for r in rows if r["y"] == 1)
                b_positive = sum(
                    1 for r in rows if r["in_B"] and r["y"] == 1
                )
                b_admissible = sum(
                    1 for r in rows
                    if r["in_B"] and r["y"] == 1 and r["hp_margin"] >= -1e-9
                )
                clearance, displacement = AUG._window_geometry(
                    dict(
                        state=prepared["state"], ped_xy=prepared["ped_xy"],
                        ped_vel=prepared["ped_vel"],
                    ),
                    controls,
                )
                disagree = (
                    chosen["margin"] is not None
                    and chosen["safemppi_cost"] is not None
                    and int(chosen["margin"]["candidate_id"])
                    != int(chosen["safemppi_cost"]["candidate_id"])
                )
                traces.append(dict(
                    scenario=int(replica.scenario_id),
                    gamma=float(replica.gamma), step=int(step),
                    K_positive=int(k_positive),
                    B_positive=int(b_positive),
                    B_admissible=int(b_admissible),
                    NVP=execute is None,
                    K_pos_but_B_none=bool(k_positive > 0 and b_admissible == 0),
                    selector_disagreement=bool(disagree),
                    executed_source=source,
                    executed_y=executed_y,
                    executed_clearance=float(clearance),
                    executed_displacement=float(displacement),
                    sigma_selected=sigmas[index],
                ))
                BX._advance(replica, controls[0])
                FA._post_action_terminal(replica)
    OE._finalize_alive(replicas)
    for replica in replicas:
        outcomes.append(dict(
            scenario=int(replica.scenario_id), gamma=float(replica.gamma),
            status=replica.status, steps=len(replica.controls),
            min_clearance=float(replica.minimum_clearance),
        ))
    aggregate = dict(
        contexts=len(traces),
        NVP=sum(t["NVP"] for t in traces),
        K_pos_but_B_none=sum(t["K_pos_but_B_none"] for t in traces),
        selector_disagreement=sum(t["selector_disagreement"] for t in traces),
        mean_K_positive=float(np.mean([t["K_positive"] for t in traces])),
        mean_B_positive_fraction=float(np.mean([
            t["B_positive"] / cfg.B for t in traces
        ])),
        outcomes=Counter(o["status"] for o in outcomes),
        beta=float(beta), calibrated_ess=float(ess),
    )
    payload = dict(
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=OS.sha256_file(args.checkpoint),
        selector=args.selector, ell=float(args.ell),
        scenarios=list(map(int, args.scenarios)),
        gammas=list(map(float, args.gammas)),
        aggregate={
            **{k: v for k, v in aggregate.items() if k != "outcomes"},
            "outcomes": dict(aggregate["outcomes"]),
        },
        outcomes=outcomes,
        traces=traces,
    )
    torch.save(payload, args.out)
    _write_json(
        args.out + ".summary.json",
        {k: payload[k] for k in (
            "checkpoint", "checkpoint_sha256", "selector", "aggregate",
            "outcomes",
        )},
    )
    print(json.dumps(payload["aggregate"], indent=1))


# -------------------------------------------------------------- update ----

def update(args):
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=args.device)
    sha = OS.sha256_file(args.checkpoint)
    if sha != OE.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("update must start from the exact r0 checkpoint")
    BS.configure_expansion_trainability(policy)
    encoder_sha = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr,
    )
    shard = OS.ExecutedRoundShard.load(args.shard)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        replay = AUG.replay_with_mode(
            policy, optimizer, shard, mode=args.replay_mode,
            alpha=args.alpha, exposure_epochs=args.exposure_epochs,
            batch=128, device=args.device, seed=args.seed,
            executor=executor,
        )
    if BS.module_sha256(policy.enc_grid) != encoder_sha:
        raise RuntimeError("visual encoder changed")
    BX._save_checkpoint(policy, args.out, dict(
        role="stageA_single_update", source_sha256=sha,
        shard=os.path.abspath(args.shard), lr=args.lr, alpha=args.alpha,
        exposure_epochs=args.exposure_epochs, replay_mode=args.replay_mode,
        seed=args.seed,
    ))
    compact = {
        k: replay.get(k) for k in (
            "positive_eligible", "negative_eligible", "optimizer_steps",
            "module_relative_parameter_drift", "fixed_probe",
        )
    }
    compact["replay_intervention"] = {
        k: v for k, v in replay.get("replay_intervention", {}).items()
        if k != "recovery_audit"
    }
    audit = replay.get("replay_intervention", {}).get("recovery_audit")
    if audit is not None:
        compact["recovery_audit_counts"] = {
            k: v for k, v in audit.items() if k != "rows"
        }
        _write_json(args.out + ".recovery_audit.json", audit)
    _write_json(args.out + ".replay.json", dict(
        replay={k: v for k, v in replay.items() if k != "epochs"},
        compact=compact,
    ))
    print(json.dumps(compact, indent=1, default=str))


# ------------------------------------------------------------- compare ----

def compare(args):
    before = torch.load(args.before, map_location="cpu", weights_only=False)
    after = torch.load(args.after, map_location="cpu", weights_only=False)
    rows = []
    outcomes_b = {
        (o["scenario"], o["gamma"]): o for o in before["outcomes"]
    }
    outcomes_a = {
        (o["scenario"], o["gamma"]): o for o in after["outcomes"]
    }
    for key in sorted(outcomes_b):
        b, a = outcomes_b[key], outcomes_a.get(key)
        traces_b = [
            t for t in before["traces"]
            if (t["scenario"], t["gamma"]) == key
        ]
        traces_a = [
            t for t in after["traces"]
            if (t["scenario"], t["gamma"]) == key
        ]
        rows.append(dict(
            scenario=key[0], gamma=key[1],
            status_before=b["status"], status_after=a and a["status"],
            steps_before=b["steps"], steps_after=a and a["steps"],
            NVP_before=sum(t["NVP"] for t in traces_b),
            NVP_after=a and sum(t["NVP"] for t in traces_a),
            B_pos_frac_before=float(np.mean([
                t["B_positive"] / 4 for t in traces_b
            ])) if traces_b else None,
            B_pos_frac_after=float(np.mean([
                t["B_positive"] / 4 for t in traces_a
            ])) if traces_a else None,
        ))
    payload = dict(
        before=dict(
            checkpoint=before["checkpoint"],
            aggregate=before["aggregate"],
        ),
        after=dict(
            checkpoint=after["checkpoint"], aggregate=after["aggregate"],
        ),
        episodes=rows,
    )
    _write_json(args.out, payload)
    print(json.dumps(dict(
        before=before["aggregate"], after=after["aggregate"],
    ), indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine")
    m.add_argument("--shard", required=True)
    m.add_argument("--out", required=True)

    b = sub.add_parser("branch")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--scenarios", type=int, nargs="+", required=True)
    b.add_argument("--gammas", type=float, nargs="+", required=True)
    b.add_argument("--selector", default="margin",
                   choices=("margin", "safemppi_cost"))
    b.add_argument("--ell", type=float, required=True,
                   help="round-1 lengthscale from the control run manifest")
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--out", required=True)

    u = sub.add_parser("update")
    u.add_argument("--checkpoint", required=True)
    u.add_argument("--shard", required=True)
    u.add_argument("--lr", type=float, default=1e-4)
    u.add_argument("--alpha", type=float, default=0.01)
    u.add_argument("--exposure-epochs", type=int, default=10)
    u.add_argument("--replay-mode", default="original",
                   choices=AUG.REPLAY_MODES)
    u.add_argument("--seed", type=int, default=20260724 + 1_000_003)
    u.add_argument("--workers", type=int, default=16)
    u.add_argument("--device", default="cuda:0")
    u.add_argument("--out", required=True)

    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    dict(mine=mine, branch=branch, update=update, compare=compare)[args.cmd](
        args,
    )


if __name__ == "__main__":
    main()
