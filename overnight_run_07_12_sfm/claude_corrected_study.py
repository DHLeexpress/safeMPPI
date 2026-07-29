"""Phase 4: bounded causal study with corrected replay and corrected teacher.

Arms (identical seeds, latents, ordering, evaluation bank):
  A. immutable r1, no update;
  B. corrected ordinary B1 only (alpha=.01, ONE accumulated epoch = ONE Adam
     step) on the fail-closed W=2 window {archived B9 round-1 shard, a new
     round-2 shard gathered from r1 with the unchanged protocol};
  C. corrected teacher only (applied to r1 BEFORE any ordinary continuation):
     lr in {1e-5, 3e-5} x epochs in {1, 4};
  D. best teacher dose (M10 lexicographic among liveness-preserving rows)
     followed by B's exact single corrected ordinary epoch.

Every candidate is evaluated on the fixed raw temperature-1 M10/gamma bank.
No hard-gate termination: the complete Pareto table is reported; promotion to
the disjoint M50 follows the pre-registered non-domination + catastrophic-
liveness rule.  All required diagnostics are logged per arm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_continuation as CC
import claude_corrected_distill as CD
import claude_teacher_rounds as TRND
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_store as OS
import sfm_b1_r2_alpha_replay as R2
import sfm_b1_store as BS

TEACHER_DOSES = tuple(
    dict(lr=lr, epochs=epochs)
    for lr, epochs in itertools.product((1e-5, 3e-5), (1, 4))
)
ORDINARY = dict(alpha=0.01, epochs=1, batch=128)
CATASTROPHIC_SR = 0.15
CATASTROPHIC_TIMEOUT = 0.15


def _teacher_records(buffer_payload):
    """Flatten the corrected D_MPC into (holder, row) records with contexts."""

    class _Holder:
        def __init__(self):
            self.round_i = 2
            self.contexts = []
            self.windows = []

    holder = _Holder()
    for lineage in buffer_payload["lineages"]:
        for window in lineage["windows"]:
            context = window["context"]
            context_id = len(holder.contexts)
            holder.contexts.append(dict(
                context_id=context_id, round=2,
                scenario_id=int(lineage["episode"]),
                gamma=float(lineage["gamma"]), step=int(window["start"]),
                state=context["state"], hp10=context["hp10"],
                low5=context["low5"], hist=context["hist"],
                ped_xy=context["ped_xy"], ped_vel=context["ped_vel"],
            ))
            holder.windows.append(dict(
                window_id=context_id, query_id=context_id,
                context_id=context_id,
                controls=np.asarray(window["controls"], np.float32), y=1,
            ))
    return holder, [(holder, row) for row in holder.windows]


def _fresh_policy(checkpoint, device):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    BS.configure_expansion_trainability(policy)
    return policy


def _probes(teacher_holder, per_gamma=5):
    by_gamma = {}
    for context in teacher_holder.contexts:
        by_gamma.setdefault(round(context["gamma"], 8), []).append(context)
    probes = []
    for gamma in sorted(by_gamma):
        probes.extend(by_gamma[gamma][:per_gamma])
    return probes


def _gradient_cosines(policy, recent, teacher_records, mass, *, device,
                      seed):
    policy.train()

    def _grad(records, weights_fn):
        policy.zero_grad(set_to_none=True)
        for start in range(0, len(records), 128):
            values = records[start:start + 128]
            grid, low, hist, controls = BS._tensor_batch(values, device)
            context = policy.ctx_from(grid, low, hist)
            torch.manual_seed(int(seed) + start)
            loss = policy.cfm_loss(
                controls, context, weights=weights_fn(values),
            )
            loss.backward()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }

    def _uniform(values):
        return None

    positives = recent.positive_records()[:256]
    negatives = recent.negative_records()[:256]
    teachers = teacher_records[:256]
    teacher_gradient = _grad(
        teachers,
        lambda values: torch.as_tensor(
            [len(values) * mass[(id(h), int(r["query_id"]))]
             for h, r in values], dtype=torch.float32, device=device),
    )
    positive_gradient = _grad(positives, _uniform)
    negative_gradient = _grad(negatives, _uniform)
    policy.zero_grad(set_to_none=True)
    policy.eval()

    def _cos(a, b):
        common = sorted(set(a) & set(b))
        num = sum(float((a[n].double() * b[n].double()).sum()) for n in common)
        na = sum(float(a[n].double().square().sum()) for n in common) ** 0.5
        nb = sum(float(b[n].double().square().sum()) for n in common) ** 0.5
        return num / (na * nb) if na and nb else None

    return dict(
        cos_teacher_positive=_cos(teacher_gradient, positive_gradient),
        cos_teacher_negative=_cos(teacher_gradient, negative_gradient),
        cos_positive_negative=_cos(positive_gradient, negative_gradient),
    )


def run(args):
    device = args.device
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise FileExistsError(outdir)
    os.makedirs(outdir)
    r1_path = os.path.abspath(args.r1_checkpoint)
    buffer_payload = torch.load(
        args.teacher_buffer, map_location="cpu", weights_only=False,
    )
    teacher_holder, teacher_records = _teacher_records(buffer_payload)
    mass, mass_accounting = CD.normalized_teacher_mass(teacher_records)
    probes = _probes(teacher_holder)
    shard_b9 = OS.ExecutedRoundShard.load(args.b9_shard)
    r1_policy = _fresh_policy(r1_path, device)

    # one new round-2 shard gathered from r1 with the unchanged protocol
    opts = CC.ContOpts()
    with ProcessPoolExecutor(max_workers=args.verifier_workers) as executor:
        new_shard, gather_info = TRND._gather(
            r1_policy, shard_b9, 2, opts, device, executor,
        )
    new_shard.save(os.path.join(outdir, "round_shards", "round_02.pt"))
    recent = CD.CorrectedRecent([shard_b9, new_shard])
    cosines_at_r1 = _gradient_cosines(
        _fresh_policy(r1_path, device), recent, teacher_records, mass,
        device=device, seed=20260728,
    )

    arms = []

    def _register(name, policy, training_log):
        checkpoint = os.path.join(outdir, "arms", f"{name}.pt")
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        BX._save_checkpoint(policy, checkpoint, dict(arm=name))
        drift = R2._module_relative_drift(
            R2._module_snapshot(r1_policy), R2._module_snapshot(policy),
        )
        rmse = CD.fixed_context_rmse(
            r1_policy, policy, probes, device=device, seed=20260729,
        )
        arms.append(dict(
            name=name, checkpoint=checkpoint, training=training_log,
            parameter_drift=drift, output_rmse=rmse,
        ))
        del policy
        torch.cuda.empty_cache()

    arms.append(dict(
        name="A_r1", checkpoint=r1_path,
        training=dict(adam_steps=0),
        parameter_drift={}, output_rmse=dict(
            first_action_rmse=0.0, h10_window_rmse=0.0,
        ),
    ))

    policy = _fresh_policy(r1_path, device)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4,
    )
    ordinary_log = CD.corrected_ordinary_epochs(
        policy, optimizer, recent, alpha=ORDINARY["alpha"],
        epochs=ORDINARY["epochs"], batch=ORDINARY["batch"], device=device,
        seed=20260730,
    )
    _register("B_ordinary_e1", policy, ordinary_log)

    for dose in TEACHER_DOSES:
        name = f"C_teacher_lr{dose['lr']:g}_ep{dose['epochs']}".replace(
            "-", "m",
        )
        policy = _fresh_policy(r1_path, device)
        optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad],
            lr=dose["lr"],
        )
        teacher_log = CD.corrected_teacher_epochs(
            policy, optimizer, teacher_records, epochs=dose["epochs"],
            batch=128, device=device, seed=20260731,
        )
        _register(name, policy, teacher_log)

    specs = [
        (arm["checkpoint"], os.path.join(outdir, f"eval_{arm['name']}"))
        for arm in arms
    ]
    metrics = CC.evaluate_checkpoints(
        specs, cache_dir=os.path.join(outdir, "m10_cache"),
        workers_each=args.eval_workers, wave=args.eval_wave,
        gpu=args.gpu_index,
    )
    for arm in arms:
        arm["m10"] = metrics[arm["checkpoint"]]
    r1_m10 = next(a for a in arms if a["name"] == "A_r1")["m10"]

    def _liveness_ok(m):
        return (m["SR"] >= r1_m10["SR"] - CATASTROPHIC_SR
                and m["timeout"] <= r1_m10["timeout"] + CATASTROPHIC_TIMEOUT)

    teacher_rows = [a for a in arms if a["name"].startswith("C_")]
    live_teachers = [a for a in teacher_rows if _liveness_ok(a["m10"])]
    best_pool = live_teachers if live_teachers else teacher_rows
    best_teacher = min(best_pool, key=lambda a: CC.lex_key(a["m10"]))

    policy = _fresh_policy(best_teacher["checkpoint"], device)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4,
    )
    d_log = CD.corrected_ordinary_epochs(
        policy, optimizer, recent, alpha=ORDINARY["alpha"],
        epochs=ORDINARY["epochs"], batch=ORDINARY["batch"], device=device,
        seed=20260730,
    )
    name = f"D_{best_teacher['name']}_then_ordinary"
    _register(name, policy, dict(
        teacher=best_teacher["training"], ordinary=d_log,
        adam_steps=best_teacher["training"]["adam_steps"]
        + d_log["adam_steps"],
    ))
    d_arm = arms[-1]
    d_metrics = CC.evaluate_checkpoints(
        [(d_arm["checkpoint"],
          os.path.join(outdir, f"eval_{d_arm['name']}"))],
        cache_dir=os.path.join(outdir, "m10_cache"),
        workers_each=args.eval_workers, wave=1, gpu=args.gpu_index,
    )
    d_arm["m10"] = d_metrics[d_arm["checkpoint"]]

    def _dominated(row, others):
        m = row["m10"]
        for other in others:
            if other is row:
                continue
            o = other["m10"]
            better_eq = (o["CR"] <= m["CR"]
                         and o["Validity"] >= m["Validity"]
                         and (o["clearance"] or 0) >= (m["clearance"] or 0))
            strictly = (o["CR"] < m["CR"]
                        or o["Validity"] > m["Validity"]
                        or (o["clearance"] or 0) > (m["clearance"] or 0))
            if better_eq and strictly:
                return True
        return False

    for arm in arms:
        m = arm["m10"]
        arm["liveness_ok"] = _liveness_ok(m)
        arm["improves_safety"] = (
            m["CR"] < r1_m10["CR"] or m["Validity"] > r1_m10["Validity"]
            or (m["clearance"] or 0) > (r1_m10["clearance"] or 0)
        )
    for arm in arms:
        arm["non_dominated"] = not _dominated(arm, arms)
        arm["promoted"] = bool(
            arm["name"] != "A_r1" and arm["non_dominated"]
            and arm["improves_safety"] and arm["liveness_ok"]
        )

    report = dict(
        status="CORRECTED_CAUSAL_STUDY_M10_COMPLETE",
        teacher_buffer=os.path.abspath(args.teacher_buffer),
        teacher_buffer_sha256=OS.sha256_file(args.teacher_buffer),
        teacher_records=len(teacher_records),
        teacher_mass_gamma=mass_accounting["gamma"],
        gather=gather_info,
        w2_rounds=[int(s.round_i) for s in recent.rounds],
        gradient_cosines_at_r1=cosines_at_r1,
        r1_m10=r1_m10,
        best_teacher=best_teacher["name"],
        arms=[{k: v for k, v in arm.items()} for arm in arms],
        promotion_rule=(
            "non-dominated on (CR,-Validity,-clearance), improves at least "
            "one safety metric vs r1, SR >= r1-0.15, timeout <= r1+0.15"
        ),
    )
    with open(os.path.join(outdir, "M10_STUDY.json"), "w") as stream:
        json.dump(report, stream, indent=1, allow_nan=False, default=float)
    print(json.dumps(dict(
        best_teacher=best_teacher["name"],
        promoted=[a["name"] for a in arms if a.get("promoted")],
        r1={k: r1_m10[k] for k in ("SR", "CR", "Validity", "clearance")},
    ), allow_nan=False, default=float), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--b9-shard", required=True)
    parser.add_argument("--teacher-buffer", required=True)
    parser.add_argument("--verifier-workers", type=int, default=14)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument("--eval-wave", type=int, default=6)
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
