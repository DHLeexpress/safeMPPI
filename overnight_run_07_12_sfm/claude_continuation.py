"""Stable continuation from the confirmed r1 checkpoint (pre-registered).

Development mode: per continuation round, gather ONE shard from the currently
accepted checkpoint (unchanged margin-selector B1 protocol, next expansion
scenario block), build the round's exact-certified recovery positives, then
fork 16 candidates (E x lr x anchor_mass) from the accepted checkpoint.  All
candidates share the identical shard, recovery records, replay seed, and
deterministic batch ordering.  Positive replay mass composition (declared):

    anchor_mass * r1-self-anchor  +  0.05 * recovery  +
    (0.95 - anchor_mass) * new-shard D+

with the standard hierarchy mass inside each population and the unchanged
alpha=0.01 signed-gradient negative scheme on the new shard's D-.  Every
candidate is evaluated on the fixed raw M10/gamma development bank; the
pre-registered hard admissibility gate versus the immutable r1 baseline
applies, the lexicographic rule selects the accepted checkpoint, and the
procedure stops when nothing is admissible.

Confirmation mode: identical loop with one fixed (E, lr, anchor_mass) combo,
restarted from immutable r1, no round-dependent tuning.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import dataclass
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

import _paths  # noqa: F401
import claude_offline_aug as AUG
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_offline_exec as OE
import sfm_b1_offline_store as OS
import sfm_b1_r2_alpha_replay as R2
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS

E_GRID = (1, 4, 10, 25)
LR_GRID = (1e-5, 3e-5)
ANCHOR_GRID = (0.25, 0.50)
RECOVERY_SHARE = 0.05
DEV_EP0, DEV_NOISE_SEED, DEV_M = 390_000, 20_260_740, 10
BATCH = 128
ALPHA = 0.01
SEED = 20260724
SR_TOL = 0.05
TIMEOUT_TOL = 0.05
B9_ELL = 0.3259987743470518
HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class ContOpts:
    K: int = 16
    B: int = 4
    T: int = 180
    H: int = 10
    nfe: int = 8
    temp: float = 1.0
    phi_s: float = 0.9
    gp_lam: float = OE.GP_LAMBDA
    ess_target: float = 0.5
    seed: int = SEED
    selector: str = "margin"
    smoke: bool = False
    scene_profile: str = OE.SCENE_PROFILE


class AnchorShard:
    """Duck-typed frozen r1 self-anchor buffer for the mass machinery."""

    def __init__(self, payload):
        self.round_i = 0
        self.contexts, self.windows = [], []
        for index, record in enumerate(payload["records"]):
            context = record["context"]
            self.contexts.append(dict(
                context_id=index, round=0,
                scenario_id=int(record["episode"]),
                gamma=float(record["gamma"]), step=int(record["step"]),
                state=np.asarray(context["state"], np.float32),
                hp10=np.asarray(context["hp10"], np.float32),
                low5=np.asarray(context["low5"], np.float32),
                hist=np.asarray(context["hist"], np.float32),
                ped_xy=np.asarray(context["ped_xy"], np.float32),
                ped_vel=np.asarray(context["ped_vel"], np.float32),
            ))
            self.windows.append(dict(
                window_id=index, query_id=index, context_id=index,
                controls=np.asarray(record["controls"], np.float32), y=1,
            ))


def _interleave(streams):
    """Deterministic proportional interleave of any number of streams."""
    streams = [list(stream) for stream in streams if stream]
    cursors = [0] * len(streams)
    merged = []
    while any(c < len(s) for c, s in zip(cursors, streams)):
        progress = [
            (cursors[i] + 1) / len(streams[i]) if cursors[i] < len(streams[i])
            else float("inf")
            for i in range(len(streams))
        ]
        pick = int(np.argmin(progress))
        merged.append(streams[pick][cursors[pick]])
        cursors[pick] += 1
    return merged


def build_positive_populations(shard, recovery_records, anchor_shard,
                               anchor_mass):
    recovery_view = AUG.ShardView(shard, recovery_records) \
        if recovery_records else None
    populations = [
        ("new", shard, [(shard, row) for row in shard.Dplus],
         0.95 - float(anchor_mass)),
        ("anchor", anchor_shard,
         [(anchor_shard, row) for row in anchor_shard.windows],
         float(anchor_mass)),
    ]
    if recovery_view is not None:
        populations.insert(1, (
            "recovery", recovery_view,
            [(recovery_view, row) for row in recovery_view.windows],
            RECOVERY_SHARE,
        ))
    else:
        populations[0] = (
            "new", shard, populations[0][2],
            0.95 - float(anchor_mass) + RECOVERY_SHARE,
        )
    return populations


def build_weights(populations, negatives):
    total_positive = sum(len(records) for _, _, records, _ in populations)
    weights = {}
    membership = {}
    for name, _, records, share in populations:
        mass, _ = BS.hierarchy_mass(records)
        for holder, row in records:
            key = (id(holder), int(row["query_id"]))
            weights[key] = (
                float(share) * float(mass[key]) * float(total_positive)
            )
            membership[key] = name
    negative_mass, _ = BS.hierarchy_mass(negatives) if negatives else ({}, {})
    for holder, row in negatives:
        key = (id(holder), int(row["query_id"]))
        weights[key] = float(negative_mass[key]) * float(len(negatives))
        membership[key] = "negative"
    return weights, membership, total_positive


def epoch_batches(populations, negatives, *, batch, seed, epoch):
    streams = [
        BS.hierarchical_order(records, int(seed) + epoch * 1009 + 7 * index)
        for index, (_, _, records, _) in enumerate(populations)
    ]
    positive_stream = _interleave(streams)
    negative_stream = BS.hierarchical_order(
        negatives, int(seed) + epoch * 1009 + 997,
    ) if negatives else []
    total = len(positive_stream) + len(negative_stream)
    batch_count = math.ceil(total / int(batch)) if total else 0
    if positive_stream and len(positive_stream) < batch_count:
        raise RuntimeError("cannot seed every minibatch with a positive")
    batches = [[positive_stream[i]] for i in range(batch_count)]
    remaining = _interleave([positive_stream[batch_count:], negative_stream])
    index = 0
    for record in remaining:
        while len(batches[index]) >= int(batch):
            index = (index + 1) % batch_count
        batches[index].append(record)
        index = (index + 1) % batch_count
    return batches


def _weighted_cfm(policy, records, weights, device, generator_seed):
    grid, low, hist, controls = BS._tensor_batch(records, device)
    context = policy.ctx_from(grid, low, hist)
    values = torch.as_tensor([
        weights[(id(holder), int(row["query_id"]))]
        for holder, row in records
    ], dtype=controls.dtype, device=device)
    torch.manual_seed(int(generator_seed))
    return policy.cfm_loss(controls, context, weights=values)


def continuation_replay(policy, optimizer, populations, negatives, *,
                        epochs, batch, seed, device):
    weights, membership, total_positive = build_weights(
        populations, negatives,
    )
    policy.train()
    encoder_before = BS.module_sha256(policy.enc_grid)
    module_before = R2._module_snapshot(policy)
    per_pop_losses = {name: [] for name, *_ in populations}
    per_pop_losses["negative"] = []
    gradient_norms = []
    steps = 0
    for epoch in range(int(epochs)):
        batches = epoch_batches(
            populations, negatives, batch=batch, seed=seed, epoch=epoch,
        )
        for batch_index, values in enumerate(batches):
            positive = [
                r for r in values
                if membership[(id(r[0]), int(r[1]["query_id"]))] != "negative"
            ]
            negative = [
                r for r in values
                if membership[(id(r[0]), int(r[1]["query_id"]))] == "negative"
            ]
            if not positive:
                continue
            step_seed = int(seed) + epoch * 100_003 + batch_index
            optimizer.zero_grad(set_to_none=True)
            positive_loss = _weighted_cfm(
                policy, positive, weights, device, step_seed,
            )
            if not bool(torch.isfinite(positive_loss)):
                raise FloatingPointError("non-finite positive loss")
            positive_loss.backward()
            positive_gradient = BS._gradient_snapshot(policy)
            positive_norm = BS._gradient_norm(positive_gradient)
            rho = 0.0
            negative_gradient = {}
            if ALPHA > 0.0 and negative:
                optimizer.zero_grad(set_to_none=True)
                negative_loss = _weighted_cfm(
                    policy, negative, weights, device, step_seed + 51,
                )
                negative_loss.backward()
                negative_gradient = BS._gradient_snapshot(policy)
                negative_norm = BS._gradient_norm(negative_gradient)
                rho = ALPHA * positive_norm / (negative_norm + 1e-12)
                per_pop_losses["negative"].append(float(negative_loss))
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad:
                    continue
                pos = positive_gradient.get(name)
                neg = negative_gradient.get(name)
                if pos is None and neg is None:
                    parameter.grad = None
                elif pos is None:
                    parameter.grad = -rho * neg
                elif neg is None:
                    parameter.grad = pos
                else:
                    parameter.grad = pos - rho * neg
            optimizer.step()
            gradient_norms.append(float(positive_norm))
            steps += 1
            with torch.no_grad():
                for name, _, records, _ in populations:
                    subset = [
                        r for r in positive
                        if membership[(id(r[0]), int(r[1]["query_id"]))]
                        == name
                    ]
                    if subset:
                        per_pop_losses[name].append(float(_weighted_cfm(
                            policy, subset, weights, device, step_seed,
                        )))
    policy.eval()
    if BS.module_sha256(policy.enc_grid) != encoder_before:
        raise RuntimeError("visual encoder changed during continuation replay")
    drift = R2._module_relative_drift(
        module_before, R2._module_snapshot(policy),
    )
    def _summary(values):
        return None if not values else dict(
            first=values[0], last=values[-1], mean=float(np.mean(values)),
        )
    return dict(
        steps=steps, epochs=int(epochs),
        unique_positive=total_positive,
        unique_negative=len(negatives),
        gradient_norm=_summary(gradient_norms),
        losses={k: _summary(v) for k, v in per_pop_losses.items()},
        parameter_drift=drift,
    )


@torch.no_grad()
def probe_diagnostics(policy, reference_policy, anchor_shard, device):
    """Route diversity + representation drift on fixed probe contexts."""
    by_gamma = {}
    for context in anchor_shard.contexts:
        by_gamma.setdefault(round(context["gamma"], 8), []).append(context)
    probes = []
    for gamma in sorted(by_gamma):
        probes.extend(by_gamma[gamma][:5])
    probes = probes[:35]
    modes = {"yield": 0, "left": 0, "right": 0}
    spreads = []
    cosines = []
    for probe in probes:
        hp10 = torch.as_tensor(probe["hp10"], device=device)[None].float()
        low = torch.as_tensor(probe["low5"], device=device)[None].float()
        hist = torch.as_tensor(probe["hist"], device=device)[None].float()
        ctx = policy.ctx_from(hp10, low, hist)
        generator = np.random.default_rng(OE._keyed_seed(
            SEED, 0, int(probe["scenario_id"]), f"{probe['gamma']:.8f}",
            int(probe["step"]), "route_probe",
        ))
        x0 = generator.standard_normal((16, int(policy.d)), dtype=np.float32)
        windows = BE.integrate_latents(
            policy, torch.as_tensor(x0, device=device),
            ctx.repeat_interleave(16, dim=0), nfe=8,
        ).reshape(16, 10, 2)
        windows_np = windows.cpu().numpy()
        prediction = SM.predict_pedestrians(
            probe["ped_xy"], probe["ped_vel"], H=10,
        )
        for k in range(16):
            segment = SM.rollout_positions(probe["state"], windows_np[k])
            modes[BE.classify_candidate(segment, prediction)] += 1
        spreads.append(float(np.mean(np.linalg.norm(
            windows_np[:, None] - windows_np[None, :], axis=(2, 3),
        ))))
        features = policy.phi_s_from_x0(
            windows.reshape(16, 10, 2), ctx.repeat_interleave(16, dim=0),
            torch.as_tensor(x0, device=device), s=0.9,
        )
        if reference_policy is not None:
            ref_ctx = reference_policy.ctx_from(hp10, low, hist)
            ref_windows = BE.integrate_latents(
                reference_policy, torch.as_tensor(x0, device=device),
                ref_ctx.repeat_interleave(16, dim=0), nfe=8,
            ).reshape(16, 10, 2)
            ref_features = reference_policy.phi_s_from_x0(
                ref_windows, ref_ctx.repeat_interleave(16, dim=0),
                torch.as_tensor(x0, device=device), s=0.9,
            )
            cosine = torch.nn.functional.cosine_similarity(
                features, ref_features, dim=1,
            ).mean()
            cosines.append(float(cosine))
    total_modes = max(sum(modes.values()), 1)
    probabilities = [v / total_modes for v in modes.values() if v > 0]
    entropy = -sum(p * math.log(p) for p in probabilities)
    return dict(
        probe_contexts=len(probes),
        route_counts=modes,
        route_entropy=float(entropy),
        mean_pairwise_control_spread=float(np.mean(spreads)),
        representation_cosine_vs_accepted=(
            None if not cosines else float(np.mean(cosines))
        ),
    )


def m10_pooled(path):
    with open(path) as stream:
        payload = json.load(stream)
    record = payload["records"][0]
    cell = record["cell"]["summary"]
    p = cell["pooled"]
    def _cell(gamma):
        c = cell["per_gamma"][gamma]
        return dict(
            clearance=c["successful_clearance"]["mean"],
            time=c["successful_time_to_goal"]["mean"],
        )
    return dict(
        SR=float(p["SR"]), CR=float(p["CR"]), timeout=float(p["timeout"]),
        Validity=float(p["Validity"]["mean"]),
        clearance=p["successful_clearance"]["mean"],
        time=p["successful_time_to_goal"]["mean"],
        g01=_cell("0.1"), g10=_cell("1.0"),
    )


def admissible(candidate, r1):
    checks = dict(
        SR=candidate["SR"] >= r1["SR"] - SR_TOL,
        timeout=candidate["timeout"] <= r1["timeout"] + TIMEOUT_TOL,
        CR=candidate["CR"] <= r1["CR"],
        Validity=candidate["Validity"] >= r1["Validity"],
        clearance=(
            candidate["clearance"] is not None
            and r1["clearance"] is not None
            and candidate["clearance"] >= r1["clearance"]
        ),
        gamma_trend=(
            candidate["g01"]["clearance"] is not None
            and candidate["g10"]["clearance"] is not None
            and candidate["g01"]["time"] is not None
            and candidate["g10"]["time"] is not None
            and candidate["g01"]["clearance"] >= candidate["g10"]["clearance"]
            and candidate["g01"]["time"] >= candidate["g10"]["time"]
        ),
    )
    return all(checks.values()), checks


def lex_key(candidate):
    return (
        candidate["CR"], -candidate["Validity"],
        -(candidate["clearance"] if candidate["clearance"] is not None
          else -1.0),
        candidate["time"] if candidate["time"] is not None else 1e9,
    )


def gather_round(policy, previous_shard, round_index, opts, device,
                 executor):
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)
    environment = SS.scene_profile(opts.scene_profile)
    replicas = [
        BX.Replica(s, g, n_ped=environment["n_ped"],
                   ped_speed_range=tuple(environment["ped_speed_range"]))
        for s in SP.expansion_scenarios(round_index, smoke=opts.smoke)
        for g in SP.GAMMAS
    ]
    gp, _, _ = OE.gp_from_previous(
        phi_policy, previous_shard, round_i=round_index, ell=B9_ELL,
        cap=OE.CAP, lam=opts.gp_lam, phi_s=opts.phi_s, device=device,
        seed=opts.seed + round_index * 101,
    )
    beta, ess = OE._calibrate_beta(
        phi_policy, gp, replicas, opts, device, round_i=round_index,
    )
    shard = OS.ExecutedRoundShard(round_index)
    gather = OE.gather_offline_round(
        policy, phi_policy, gp, beta, replicas, opts, shard, device,
        executor, round_i=round_index,
    )
    _, pop_b, pop_stats = AUG.tag_populations(shard)
    recovery, recovery_audit = AUG.build_recovery_records(
        shard, pop_b, executor, family="v1",
    )
    return shard, recovery, dict(
        beta=float(beta), ess=float(ess),
        outcomes={s: sum(o["status"] == s for o in gather["outcomes"])
                  for s in ("success", "collision", "timeout")},
        counts=dict(gather["counts"]),
        populations=pop_stats,
        recovery_kept=recovery_audit["certified_kept"],
    )


def evaluate_checkpoints(specs, *, cache_dir, workers_each, wave, gpu):
    """specs: list of (checkpoint, outdir). Runs waves of parallel evals."""
    results = {}
    for start in range(0, len(specs), int(wave)):
        processes = []
        for checkpoint, outdir in specs[start:start + int(wave)]:
            if os.path.isfile(os.path.join(
                outdir, f"raw_m{DEV_M}_offline_metrics.json",
            )):
                continue
            env = dict(os.environ)
            env.update(
                CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=HERE,
            )
            processes.append(subprocess.Popen(
                [sys.executable,
                 os.path.join(HERE, "sfm_b1_offline_eval.py"),
                 "--checkpoints", checkpoint, "--labels", "r1",
                 "--scene-profile", "double_density_velocity_ood",
                 "--ep0", str(DEV_EP0), "--noise-seed", str(DEV_NOISE_SEED),
                 "--m-per-gamma", str(DEV_M), "--device", "cuda:0",
                 "--workers", str(int(workers_each)),
                 "--cache-dir", cache_dir, "--output-dir", outdir],
                env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, cwd=HERE,
            ))
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("candidate M10 evaluation failed")
    for checkpoint, outdir in specs:
        results[checkpoint] = m10_pooled(os.path.join(
            outdir, f"raw_m{DEV_M}_offline_metrics.json",
        ))
    return results


def run(args):
    device = args.device
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise FileExistsError(outdir)
    os.makedirs(outdir)
    anchor_payload = torch.load(
        args.anchor, map_location="cpu", weights_only=False,
    )
    anchor_shard = AnchorShard(anchor_payload)
    r1_baseline = m10_pooled(args.r1_dev_metrics)
    combos = (
        [dict(E=E, lr=lr, anchor=am)
         for E, lr, am in itertools.product(E_GRID, LR_GRID, ANCHOR_GRID)]
        if args.mode == "develop"
        else [dict(E=int(args.fixed_E), lr=float(args.fixed_lr),
                   anchor=float(args.fixed_anchor))]
    )
    opts = ContOpts()
    accepted_path = os.path.abspath(args.r1_checkpoint)
    accepted_sha = OS.sha256_file(accepted_path)
    previous_shard = OS.ExecutedRoundShard.load(args.previous_shard)
    template_policy, _ = GPS.load_sfm_policy(accepted_path, device=device)
    history = []
    with ProcessPoolExecutor(max_workers=args.verifier_workers) as executor:
        for round_k in range(1, int(args.rounds) + 1):
            start = time.perf_counter()
            round_index = 1 + round_k
            accepted_policy, _ = GPS.load_sfm_policy(
                accepted_path, device=device,
            )
            shard, recovery, gather_info = gather_round(
                accepted_policy, previous_shard, round_index, opts, device,
                executor,
            )
            shard.save(os.path.join(
                outdir, "round_shards", f"cont_{round_k:02d}.pt",
            ))
            candidates = []
            for combo in combos:
                name = (
                    f"E{combo['E']:02d}_lr{combo['lr']:.0e}"
                    f"_a{str(combo['anchor']).replace('.', 'p')}"
                ).replace("-", "m")
                policy = copy.deepcopy(template_policy)
                policy.load_state_dict(accepted_policy.state_dict())
                BS.configure_expansion_trainability(policy)
                optimizer = torch.optim.Adam(
                    [p for p in policy.parameters() if p.requires_grad],
                    lr=combo["lr"],
                )
                populations = build_positive_populations(
                    shard, recovery, anchor_shard, combo["anchor"],
                )
                negatives = [(shard, row) for row in shard.Dminus]
                replay_log = continuation_replay(
                    policy, optimizer, populations, negatives,
                    epochs=combo["E"], batch=BATCH,
                    seed=SEED + round_k * 1_000_003, device=device,
                )
                probe = probe_diagnostics(
                    policy, accepted_policy, anchor_shard, device,
                )
                ckpt = os.path.join(
                    outdir, f"round_{round_k:02d}", f"{name}.pt",
                )
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                BX._save_checkpoint(policy, ckpt, dict(
                    round=round_k, combo=combo, accepted_parent=accepted_path,
                    accepted_parent_sha256=accepted_sha,
                ))
                candidates.append(dict(
                    name=name, combo=combo, checkpoint=ckpt,
                    replay=replay_log, probe=probe,
                ))
                del policy, optimizer
            specs = [
                (c["checkpoint"], os.path.join(
                    outdir, f"round_{round_k:02d}", f"eval_{c['name']}",
                ))
                for c in candidates
            ]
            metrics = evaluate_checkpoints(
                specs, cache_dir=os.path.join(outdir, "dev_cache"),
                workers_each=args.eval_workers, wave=args.eval_wave,
                gpu=args.gpu_index,
            )
            for candidate in candidates:
                candidate["m10"] = metrics[candidate["checkpoint"]]
                ok, checks = admissible(candidate["m10"], r1_baseline)
                candidate["admissible"] = ok
                candidate["admissibility_checks"] = checks
            admissible_rows = [c for c in candidates if c["admissible"]]
            selected = (
                min(admissible_rows, key=lambda c: lex_key(c["m10"]))
                if admissible_rows else None
            )
            record = dict(
                continuation_round=round_k,
                gather_round_index=round_index,
                gather=gather_info,
                shard=dict(D=len(shard.D), Dplus=len(shard.Dplus),
                           Dminus=len(shard.Dminus)),
                r1_baseline=r1_baseline,
                candidates=[{
                    k: c[k] for k in (
                        "name", "combo", "checkpoint", "replay", "probe",
                        "m10", "admissible", "admissibility_checks",
                    )
                } for c in candidates],
                n_admissible=len(admissible_rows),
                selected=None if selected is None else dict(
                    name=selected["name"], combo=selected["combo"],
                    checkpoint=selected["checkpoint"],
                    m10=selected["m10"],
                ),
                wall_seconds=time.perf_counter() - start,
            )
            history.append(record)
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(dict(
                round=round_k,
                admissible=len(admissible_rows),
                selected=None if selected is None else selected["name"],
                selected_m10=None if selected is None else {
                    k: selected["m10"][k]
                    for k in ("SR", "CR", "Validity", "clearance", "time")
                },
                wall=record["wall_seconds"],
            )), flush=True)
            if selected is None:
                print("STOP: no admissible candidate", flush=True)
                break
            accepted_path = selected["checkpoint"]
            accepted_sha = OS.sha256_file(accepted_path)
            previous_shard = shard
    OE._write_json(os.path.join(outdir, "COMPLETE.json"), dict(
        status="R1_CONTINUATION_COMPLETE",
        mode=args.mode,
        rounds_run=len(history),
        rounds_accepted=sum(1 for r in history if r["selected"]),
        final_accepted_checkpoint=accepted_path,
        final_accepted_sha256=accepted_sha,
        r1_dev_baseline=r1_baseline,
        combos=combos,
        history=history,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("develop", "confirm"),
                        required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--previous-shard", required=True,
                        help="archived B9 round-1 shard (GP chain seed)")
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--r1-dev-metrics", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--fixed-E", type=int)
    parser.add_argument("--fixed-lr", type=float)
    parser.add_argument("--fixed-anchor", type=float)
    parser.add_argument("--verifier-workers", type=int, default=14)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument("--eval-wave", type=int, default=8)
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.mode == "confirm" and None in (
        args.fixed_E, args.fixed_lr, args.fixed_anchor,
    ):
        raise SystemExit("confirm mode requires --fixed-E/--fixed-lr/--fixed-anchor")
    run(args)


if __name__ == "__main__":
    main()
