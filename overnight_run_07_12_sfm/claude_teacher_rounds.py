"""Unverified-MPC-teacher continuation rounds (pre-registered driver).

Per round: (1) unchanged ordinary gathering + unchanged ordinary
``sfm_b1_offline_replay.replay`` (alpha .01, exposures 10, lr 1e-4, fresh
Adam per round — the accepted lineage may switch checkpoints between rounds)
producing ``theta_(n+1/2)`` and its exact round shard; (2) the pinned
``claude_apply_unverified_mpc_teacher.py`` on ``theta_(n+1/2)`` for the nine
dose arms (lr {1e-6,3e-6,1e-5} x epochs {1,2,4}); arm 1 harvests, arms 2-9
authenticate-and-reuse the identical deterministic ``D_MPC`` buffer, and the
driver asserts ``teacher_buffer_sha256`` equality across all nine arms;
(3) fixed-CRN raw M10 for the no-teacher control and all arms; (4) the
pre-registered gates and lexicographic rule pick the accepted checkpoint;
the loop stops when nothing is admissible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import subprocess
import sys
import time

import torch

import _paths  # noqa: F401
import claude_continuation as CC
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_exec as OE
import sfm_b1_offline_replay as OR
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_protocol as SP
import sfm_scene as SS

TEACHER_LRS = (1e-6, 3e-6, 1e-5)
TEACHER_EPOCHS = (1, 2, 4)
ORDINARY = dict(alpha=0.01, exposure_epochs=10, lr=1e-4, batch=128)
HERE = os.path.dirname(os.path.abspath(__file__))


def _gather(policy, previous_shard, round_index, opts, device, executor):
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
        phi_policy, previous_shard, round_i=round_index, ell=CC.B9_ELL,
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
    return shard, dict(
        beta=float(beta), ess=float(ess),
        outcomes={s: sum(o["status"] == s for o in gather["outcomes"])
                  for s in ("success", "collision", "timeout")},
        NVP=int(gather["counts"].get("NVP_contexts", 0)),
    )


def _run_teacher(checkpoint, checkpoint_sha, shard_path, outdir, lr, epochs,
                 *, reuse_buffer, gpu, workers, audit):
    env = dict(os.environ)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=HERE)
    command = [
        sys.executable,
        os.path.join(HERE, "claude_apply_unverified_mpc_teacher.py"),
        "--checkpoint", checkpoint,
        "--expected-checkpoint-sha256", checkpoint_sha,
        "--round-shard", shard_path,
        "--output-dir", outdir,
        "--teacher-lr", f"{lr:g}", "--teacher-epochs", str(int(epochs)),
        "--max-contexts", "400", "--seed", "20260728",
        "--verifier-workers", str(int(workers)), "--device", "cuda:0",
    ]
    if audit:
        command.append("--audit-socp")
    if reuse_buffer:
        command.extend(["--reuse-buffer", reuse_buffer])
    completed = subprocess.run(
        command, env=env, cwd=HERE, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher block failed ({outdir}):\n{completed.stderr[-2000:]}"
        )
    with open(os.path.join(outdir, "COMPLETE.json")) as stream:
        return json.load(stream)


def run(args):
    device = args.device
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise FileExistsError(outdir)
    os.makedirs(outdir)
    r1_baseline = CC.m10_pooled(args.r1_dev_metrics)
    opts = CC.ContOpts()
    accepted_path = os.path.abspath(args.r1_checkpoint)
    previous_shard = OS.ExecutedRoundShard.load(args.previous_shard)
    with ProcessPoolExecutor(max_workers=args.verifier_workers) as executor:
        for round_k in range(1, int(args.rounds) + 1):
            start = time.perf_counter()
            round_index = 1 + round_k
            round_dir = os.path.join(outdir, f"round_{round_k:02d}")
            os.makedirs(round_dir)
            policy, _ = GPS.load_sfm_policy(accepted_path, device=device)
            shard, gather_info = _gather(
                policy, previous_shard, round_index, opts, device, executor,
            )
            shard_path = os.path.join(
                outdir, "round_shards", f"round_{round_k:02d}.pt",
            )
            shard.save(shard_path)
            BS.configure_expansion_trainability(policy)
            optimizer = torch.optim.Adam(
                [p for p in policy.parameters() if p.requires_grad],
                lr=ORDINARY["lr"],
            )
            replay = OR.replay(
                policy, optimizer, shard, alpha=ORDINARY["alpha"],
                exposure_epochs=ORDINARY["exposure_epochs"],
                batch=ORDINARY["batch"], device=device,
                seed=opts.seed + round_index * 1_000_003,
            )
            half_path = os.path.join(round_dir, "theta_half.pt")
            BX._save_checkpoint(policy, half_path, dict(
                round=round_k, phase="post_ordinary_pre_teacher",
                accepted_parent=accepted_path,
            ))
            half_sha = OS.sha256_file(half_path)
            del policy, optimizer
            torch.cuda.empty_cache()

            arms = []
            reuse = None
            for lr in TEACHER_LRS:
                for epochs in TEACHER_EPOCHS:
                    name = f"lr{lr:g}_ep{epochs}".replace("-", "m")
                    arm_dir = os.path.join(round_dir, f"arm_{name}")
                    report = _run_teacher(
                        half_path, half_sha, shard_path, arm_dir, lr, epochs,
                        reuse_buffer=reuse, gpu=args.gpu_index,
                        workers=args.verifier_workers,
                        audit=(reuse is None),
                    )
                    if reuse is None:
                        reuse = os.path.join(arm_dir, "D_MPC.pt")
                    arms.append(dict(
                        name=name, lr=lr, epochs=int(epochs),
                        checkpoint=report["post_teacher_checkpoint"],
                        buffer_sha=report["teacher_buffer_sha256"],
                        harvest_counts={
                            k: v for k, v in report["harvest"].items()
                            if isinstance(v, (int, float, str))
                        },
                        update=report["update"],
                        probe_before=report["conflict_probe_before"],
                        probe_after=report["conflict_probe_after"],
                    ))
            buffer_shas = {arm["buffer_sha"] for arm in arms}
            if len(buffer_shas) != 1:
                raise RuntimeError(
                    f"D_MPC buffers diverged across arms: {buffer_shas}"
                )
            specs = [(half_path, os.path.join(round_dir, "eval_theta_half"))]
            specs += [
                (arm["checkpoint"],
                 os.path.join(round_dir, f"eval_{arm['name']}"))
                for arm in arms
            ]
            metrics = CC.evaluate_checkpoints(
                specs, cache_dir=os.path.join(outdir, "dev_cache"),
                workers_each=args.eval_workers, wave=args.eval_wave,
                gpu=args.gpu_index,
            )
            rows = [dict(
                name="no_teacher_control", lr=None, epochs=None,
                checkpoint=half_path, m10=metrics[half_path],
            )]
            for arm in arms:
                arm["m10"] = metrics[arm["checkpoint"]]
                rows.append(arm)
            for row in rows:
                ok, checks = CC.admissible(row["m10"], r1_baseline)
                row["admissible"] = ok
                row["admissibility_checks"] = checks
            admissible_rows = [r for r in rows if r["admissible"]]
            selected = (
                min(admissible_rows, key=lambda r: CC.lex_key(r["m10"]))
                if admissible_rows else None
            )
            record = dict(
                round=round_k, gather=gather_info,
                shard=dict(D=len(shard.D), Dplus=len(shard.Dplus),
                           Dminus=len(shard.Dminus)),
                ordinary_replay=dict(steps=replay["optimizer_steps"]),
                theta_half=half_path, theta_half_sha256=half_sha,
                buffer_sha256=next(iter(buffer_shas)),
                r1_baseline=r1_baseline,
                rows=[{k: v for k, v in row.items()} for row in rows],
                n_admissible=len(admissible_rows),
                selected=None if selected is None else dict(
                    name=selected["name"],
                    checkpoint=selected["checkpoint"],
                    m10=selected["m10"],
                ),
                wall_seconds=time.perf_counter() - start,
            )
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(dict(
                round=round_k, admissible=len(admissible_rows),
                selected=None if selected is None else selected["name"],
                selected_m10=None if selected is None else {
                    k: selected["m10"][k]
                    for k in ("SR", "CR", "Validity", "clearance", "time")
                },
                cos_teacher_pos_before=arms[0]["probe_before"][
                    "cos_teacher_positive"],
                wall=record["wall_seconds"],
            )), flush=True)
            if selected is None:
                print("STOP: no admissible candidate", flush=True)
                break
            accepted_path = selected["checkpoint"]
            previous_shard = shard
    OE._write_json(os.path.join(outdir, "COMPLETE.json"), dict(
        status="TEACHER_ROUNDS_COMPLETE",
        rounds_dir=outdir,
        final_accepted=accepted_path,
        final_accepted_sha256=OS.sha256_file(accepted_path),
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--previous-shard", required=True)
    parser.add_argument("--r1-dev-metrics", required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--verifier-workers", type=int, default=14)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument("--eval-wave", type=int, default=5)
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
