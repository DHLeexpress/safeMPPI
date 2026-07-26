"""Ordinary B1 expansion + dedicated per-round D_MPC+ distillation blocks.

Per macro-round:
1. ordinary gather (immutable ``sfm_b1_offline_exec.gather_offline_round``)
   and ordinary replay (untouched ``sfm_b1_offline_replay.replay``);
2. save the post-ordinary checkpoint (``round_XX_pre_block.pt``);
3. harvest this round's ``D_MPC+`` (``claude_mpc_pool``): privileged Codex
   pool ∧ exact full-H10 SOCP, ranked by native SafeMPPI cost, per-round
   fresh, gamma/context-balanced through the standard hierarchy mass;
4. audit BEFORE the block, run the dedicated distillation block (its own
   Adam, swept lr/epochs), audit AFTER, save ``round_XX.pt``.

Audits per block: (a) local MPC-context raw sampling — SOCP-positive rate,
mean predicted clearance and goal progress of 16 raw temperature-1 samples
per held audit context, and target recovery (min normalized L2 distance from
samples to the stored MPC target); (b) fixed raw temperature-1 M10/gamma
evaluation (CR, executed-window Validity, successful clearance and time) on
the declared MPC study bank.  D_MPC+ never touches D/D+, the GP, or
acquisition; the privileged controller is never used at evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass
import json
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
import claude_mpc_pool as MP
import claude_offline_aug as AUG
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_offline_exec as OE
import sfm_b1_offline_replay as OR
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS

M10_EP0 = 350_000
M10_NOISE_SEED = 20_260_733
AUDIT_CONTEXTS = 64
AUDIT_SAMPLES = 16
MAX_HARVEST_CONTEXTS = 400


@dataclass(frozen=True)
class MPCConfig:
    lr_dedicated: float
    epochs_dedicated: int
    selector: str = "margin"
    alpha: float = 0.01
    exposure_epochs: int = 10
    lr: float = 1.0e-4
    ess_target: float = 0.5
    rounds: int = 4
    K: int = 16
    B: int = 4
    T: int = 180
    H: int = 10
    batch: int = 128
    nfe: int = 8
    temp: float = 1.0
    phi_s: float = 0.9
    gp_lam: float = OE.GP_LAMBDA
    verifier_workers: int = 8
    seed: int = 20260724
    scene_profile: str = OE.SCENE_PROFILE
    smoke: bool = False
    tag: str = "mpc"

    def validate(self):
        if not 0.0 < float(self.lr_dedicated) <= 1.0e-3:
            raise ValueError("dedicated lr out of range")
        if not 1 <= int(self.epochs_dedicated) <= 32:
            raise ValueError("dedicated epochs out of range")
        if (
            int(self.K), int(self.B), int(self.T), int(self.H),
            int(self.batch), float(self.gp_lam), float(self.temp),
            self.scene_profile, int(self.nfe), float(self.phi_s),
        ) != (16, 4, 180, 10, 128, OE.GP_LAMBDA, 1.0, OE.SCENE_PROFILE, 8, 0.9):
            raise ValueError("immutable offline core changed")
        return self

    @property
    def arm_name(self):
        lr = f"{self.lr_dedicated:.0e}".replace("-", "m")
        return f"{self.tag}_lrd{lr}_epd{int(self.epochs_dedicated):02d}"


@torch.no_grad()
def local_mpc_audit(policy, shard, records, *, device, executor, seed_tag):
    """Raw-sampling audit at gamma-balanced held D_MPC+ contexts."""
    if not records:
        return dict(contexts=0)
    by_gamma = {}
    for record in records:
        gamma = round(float(shard.contexts[record["context_id"]]["gamma"]), 8)
        by_gamma.setdefault(gamma, []).append(record)
    chosen = []
    quota = max(1, AUDIT_CONTEXTS // max(len(by_gamma), 1))
    for gamma in sorted(by_gamma):
        rows = sorted(by_gamma[gamma], key=lambda r: (
            r["context_id"], r["rank"],
        ))
        seen = set()
        for row in rows:
            if row["context_id"] in seen:
                continue
            seen.add(row["context_id"])
            chosen.append(row)
            if len(seen) >= quota:
                break
    positive = clearance = progress = recovery = support = 0.0
    for row in chosen:
        context = shard.contexts[row["context_id"]]
        hp10 = torch.as_tensor(context["hp10"], device=device)[None].float()
        low = torch.as_tensor(context["low5"], device=device)[None].float()
        hist = torch.as_tensor(context["hist"], device=device)[None].float()
        ctx = policy.ctx_from(hp10, low, hist)
        generator = np.random.default_rng(OE._keyed_seed(
            20260724, 99, int(context["scenario_id"]),
            f"{float(context['gamma']):.8f}", int(context["step"]),
            f"mpc_audit_{seed_tag}",
        ))
        x0 = generator.standard_normal(
            (AUDIT_SAMPLES, int(policy.d)), dtype=np.float32,
        )
        windows = BE.integrate_latents(
            policy, torch.as_tensor(x0, device=device),
            ctx.repeat_interleave(AUDIT_SAMPLES, dim=0), nfe=8,
        ).reshape(AUDIT_SAMPLES, 10, 2).cpu().numpy()
        tasks = [
            (k, 0, context["state"], windows[k], context["ped_xy"],
             context["ped_vel"], context["gamma"])
            for k in range(AUDIT_SAMPLES)
        ]
        results = {k: r for k, _, r in executor.map(SM.verify_in_worker, tasks)}
        n_pos = sum(
            1 for r in results.values()
            if r.get("resolved") and int(r.get("y", 0)) == 1
        )
        positive += n_pos / AUDIT_SAMPLES
        support += float(n_pos > 0)
        geometry = [
            AUG._window_geometry(context, windows[k])
            for k in range(AUDIT_SAMPLES)
        ]
        clearance += float(np.mean([g[0] for g in geometry]))
        state = np.asarray(context["state"], np.float32)
        goal_now = float(np.linalg.norm(state[:2] - SS.GOAL))
        segs = [SM.rollout_positions(state, windows[k])[-1]
                for k in range(AUDIT_SAMPLES)]
        progress += float(np.mean([
            goal_now - float(np.linalg.norm(seg - SS.GOAL)) for seg in segs
        ]))
        target = np.asarray(row["controls"], np.float32)
        distances = [
            float(np.linalg.norm(windows[k] - target) / np.sqrt(target.size))
            for k in range(AUDIT_SAMPLES)
        ]
        recovery += min(distances)
    n = max(len(chosen), 1)
    return dict(
        contexts=len(chosen),
        socp_positive_rate=positive / n,
        support_fraction=support / n,
        mean_sample_clearance=clearance / n,
        mean_sample_progress=progress / n,
        target_recovery_rmse=recovery / n,
    )


def m10_eval(checkpoint, label, *, outdir, cache_dir, workers, device):
    import sfm_b1_offline_eval as EV
    args = argparse.Namespace(
        checkpoints=[checkpoint], labels=[label],
        scene_profile=OE.SCENE_PROFILE, ep0=M10_EP0,
        noise_seed=M10_NOISE_SEED, m_per_gamma=10, device=device,
        workers=int(workers), cache_dir=cache_dir, output_dir=outdir,
    )
    result = EV.run(args)
    pooled = result["records"][0]["cell"]["summary"]["pooled"]
    return dict(
        SR=float(pooled["SR"]), CR=float(pooled["CR"]),
        timeout=float(pooled["timeout"]),
        Validity=float(pooled["Validity"]["mean"]),
        clearance=pooled["successful_clearance"]["mean"],
        time=pooled["successful_time_to_goal"]["mean"],
        per_gamma={
            gamma: dict(
                CR=cell["CR"],
                Validity=float(cell["Validity"]["mean"]),
                clearance=cell["successful_clearance"]["mean"],
                time=cell["successful_time_to_goal"]["mean"],
            )
            for gamma, cell in
            result["records"][0]["cell"]["summary"]["per_gamma"].items()
        },
    )


def run(checkpoint, outdir, cfg, *, device):
    cfg.validate()
    checkpoint = os.path.abspath(checkpoint)
    outdir = os.path.abspath(outdir)
    checkpoint_sha = OS.sha256_file(checkpoint)
    if checkpoint_sha != OE.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("MPC study must start from the exact r0 checkpoint")
    if os.path.exists(outdir):
        raise FileExistsError(outdir)
    os.makedirs(outdir)
    environment = SS.scene_profile(cfg.scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    BS.configure_expansion_trainability(policy)
    encoder_sha = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr,
    )
    optimizer_dedicated = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr_dedicated,
    )
    BX._save_checkpoint(policy, os.path.join(outdir, "round_00.pt"), dict(
        round=0, experiment=cfg.arm_name, source_sha256=checkpoint_sha,
        recipe=asdict(cfg),
    ))
    preflight = [
        BX.Replica(s, g, n_ped=environment["n_ped"],
                   ped_speed_range=tuple(environment["ped_speed_range"]))
        for s in SP.expansion_scenarios(1, smoke=cfg.smoke)
        for g in SP.GAMMAS
    ]
    ell0, ell, _ = OE._initial_lengthscale(policy, preflight, cfg, device)
    history = []
    previous_shard = None
    eval_cache = os.path.join(outdir, "m10_cache")
    with ProcessPoolExecutor(max_workers=cfg.verifier_workers) as executor:
        for round_i in range(1, cfg.rounds + 1):
            start = time.perf_counter()
            replicas = [
                BX.Replica(s, g, n_ped=environment["n_ped"],
                           ped_speed_range=tuple(
                               environment["ped_speed_range"]))
                for s in SP.expansion_scenarios(round_i, smoke=cfg.smoke)
                for g in SP.GAMMAS
            ]
            policy.eval()
            phi_policy = copy.deepcopy(policy).eval()
            for parameter in phi_policy.parameters():
                parameter.requires_grad_(False)
            gp, gp_ids, gp_selection = OE.gp_from_previous(
                phi_policy, previous_shard, round_i=round_i, ell=ell,
                cap=OE.CAP, lam=cfg.gp_lam, phi_s=cfg.phi_s, device=device,
                seed=cfg.seed + round_i * 101,
            )
            beta, ess = OE._calibrate_beta(
                phi_policy, gp, replicas, cfg, device, round_i=round_i,
            )
            shard = OS.ExecutedRoundShard(round_i)
            gather = OE.gather_offline_round(
                policy, phi_policy, gp, beta, replicas, cfg, shard, device,
                executor, round_i=round_i,
            )
            shard.save(os.path.join(
                outdir, "round_shards", f"round_{round_i:02d}.pt",
            ))
            replay = OR.replay(
                policy, optimizer, shard, alpha=cfg.alpha,
                exposure_epochs=cfg.exposure_epochs, batch=cfg.batch,
                device=device, seed=cfg.seed + round_i * 1_000_003,
            )
            pre_path = os.path.join(
                outdir, f"round_{round_i:02d}_pre_block.pt",
            )
            BX._save_checkpoint(policy, pre_path, dict(
                round=round_i, phase="post_ordinary_pre_block",
                experiment=cfg.arm_name, recipe=asdict(cfg),
            ))

            # ---- D_MPC+ harvest (separate buffer; never enters D/GP) ----
            pop_a, pop_b, pop_stats = AUG.tag_populations(shard)
            hard = {int(w["window_id"]): w for w in pop_b}
            for window in shard.windows:
                if window.get("nvp_context"):
                    hard[int(window["window_id"])] = window
            policy.eval()
            records, harvest_audit = MP.harvest_round(
                policy, shard, list(hard.values()), executor,
                device=device, environment=environment,
                max_contexts=MAX_HARVEST_CONTEXTS,
            )
            torch.save(
                dict(round=round_i, records=records, audit=harvest_audit),
                os.path.join(outdir, f"d_mpc_plus_round_{round_i:02d}.pt"),
            )

            audit_before = dict(
                local=local_mpc_audit(
                    policy, shard, records, device=device,
                    executor=executor, seed_tag="fixed",
                ),
                m10=m10_eval(
                    pre_path, f"r{2 * round_i - 1}",
                    outdir=os.path.join(
                        outdir, "m10", f"round_{round_i:02d}_pre",
                    ),
                    cache_dir=eval_cache, workers=cfg.verifier_workers,
                    device=device,
                ),
            )
            block = MP.distill_block(
                policy, optimizer_dedicated, shard, records,
                epochs=cfg.epochs_dedicated, batch=cfg.batch,
                seed=cfg.seed + round_i * 7_000_003,
            )
            if BS.module_sha256(policy.enc_grid) != encoder_sha:
                raise RuntimeError("visual encoder changed")
            post_path = os.path.join(outdir, f"round_{round_i:02d}.pt")
            BX._save_checkpoint(policy, post_path, dict(
                round=round_i, phase="post_block",
                experiment=cfg.arm_name, recipe=asdict(cfg),
            ))
            audit_after = dict(
                local=local_mpc_audit(
                    policy, shard, records, device=device,
                    executor=executor, seed_tag="fixed",
                ),
                m10=m10_eval(
                    post_path, f"r{2 * round_i}",
                    outdir=os.path.join(
                        outdir, "m10", f"round_{round_i:02d}_post",
                    ),
                    cache_dir=eval_cache, workers=cfg.verifier_workers,
                    device=device,
                ),
            )
            record = dict(
                round=round_i, experiment=cfg.arm_name,
                beta=float(beta), calibrated_ess=float(ess),
                gather_counts=gather["counts"],
                outcomes=gather["outcomes"],
                replay=dict(
                    optimizer_steps=replay["optimizer_steps"],
                    positive=replay["positive_eligible"],
                    negative=replay["negative_eligible"],
                ),
                populations=pop_stats,
                harvest=dict(
                    counts=harvest_audit["counts"],
                    per_gamma=harvest_audit["per_gamma"],
                ),
                distill_block=block,
                audit_before=audit_before,
                audit_after=audit_after,
                checkpoints=dict(pre=pre_path, post=post_path),
                wall_seconds=time.perf_counter() - start,
            )
            history.append(record)
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(dict(
                round=round_i, arm=cfg.arm_name,
                kept=harvest_audit["counts"]["kept"],
                block_steps=block["steps"],
                m10_CR_before=audit_before["m10"]["CR"],
                m10_CR_after=audit_after["m10"]["CR"],
                m10_V_before=audit_before["m10"]["Validity"],
                m10_V_after=audit_after["m10"]["Validity"],
                socp_rate_before=audit_before["local"].get(
                    "socp_positive_rate"),
                socp_rate_after=audit_after["local"].get(
                    "socp_positive_rate"),
                wall=record["wall_seconds"],
            )), flush=True)
            previous_shard = shard

    OE._write_json(os.path.join(outdir, "COMPLETE.json"), dict(
        status="CLAUDE_MPC_DISTILL_COMPLETE",
        experiment=cfg.arm_name, recipe=asdict(cfg),
        source_checkpoint_sha256=checkpoint_sha,
        environment=environment,
        m10_bank=dict(ep0=M10_EP0, noise_seed=M10_NOISE_SEED, m_per_gamma=10),
        constants=dict(
            ell=ell, ell0=ell0, keep_per_context=MP.KEEP_PER_CONTEXT,
            max_harvest_contexts=MAX_HARVEST_CONTEXTS,
            separation=(
                "D_MPC+ is per-round, never enters D/D+/GP/acquisition; "
                "privileged controller never used at evaluation"
            ),
        ),
        history=history,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--lr-dedicated", type=float, required=True)
    parser.add_argument("--epochs-dedicated", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="mpc")
    args = parser.parse_args(argv)
    cfg = MPCConfig(
        lr_dedicated=args.lr_dedicated,
        epochs_dedicated=args.epochs_dedicated,
        rounds=args.rounds, verifier_workers=args.verifier_workers,
        smoke=args.smoke, tag=args.tag,
    )
    run(args.checkpoint, args.outdir, cfg, device=args.device)


if __name__ == "__main__":
    main()
