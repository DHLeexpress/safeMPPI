"""Extended offline executed-window expansion runner (opt-in knobs).

This is an additive experiment driver.  It reuses the immutable core of
``sfm_b1_offline_exec`` verbatim — ``gather_offline_round`` (56 lineages,
K=16, B=4 exact verifier queries, executed-window-only store, NVP
continuation), ``gp_from_previous``, ``_calibrate_beta``,
``_initial_lengthscale`` — and differs ONLY in the declared recipe knobs:

- ``lr`` (optimizer learning rate; control 1e-4),
- ``ess_target`` (acquisition ESS calibration target; control 0.5),
- ``rounds`` (number of macro-rounds),
- ``replay_mode`` in {original, hard, hard_recovery}
  (see ``claude_offline_aug``; ``original`` delegates verbatim to
  ``sfm_b1_offline_replay.replay``),
- ``alpha`` / ``exposure_epochs`` restricted to the original replay
  contract sets {0, 0.01, 0.1} and {1, 10, 100}.

With (lr=1e-4, ess_target=0.5, rounds=10, replay_mode="original") the run is
behaviourally identical to ``sfm_b1_offline_exec`` (same keyed seeds, same
calls); this equivalence is asserted by comparing round-1 shard digests
against the archived control run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass
import json
import os
import time

import torch

import _paths  # noqa: F401
import claude_offline_aug as AUG
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_offline_exec as OE
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


@dataclass(frozen=True)
class ExtConfig:
    alpha: float
    exposure_epochs: int
    selector: str = "margin"
    rounds: int = 10
    lr: float = 1.0e-4
    ess_target: float = 0.5
    replay_mode: str = "original"
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
    tag: str = "ext"

    def validate(self):
        if self.selector not in OE.EXECUTION_SELECTORS:
            raise ValueError(f"selector must be one of {OE.EXECUTION_SELECTORS}")
        if float(self.alpha) not in OE.ALPHAS:
            raise ValueError(f"alpha must be one of {OE.ALPHAS}")
        if int(self.exposure_epochs) not in OE.EXPOSURE_EPOCHS:
            raise ValueError(
                f"exposure_epochs must be one of {OE.EXPOSURE_EPOCHS}"
            )
        if self.replay_mode not in AUG.REPLAY_MODES:
            raise ValueError(f"replay_mode must be one of {AUG.REPLAY_MODES}")
        if not 0.05 <= float(self.ess_target) <= 0.95:
            raise ValueError("ess_target out of the studied range")
        if not 0.0 < float(self.lr) <= 1.0e-3:
            raise ValueError("lr out of the studied range")
        if not 1 <= int(self.rounds) <= 20:
            raise ValueError("rounds out of the studied range")
        # The immutable scientific core is pinned exactly as in the control.
        if (
            int(self.K), int(self.B), int(self.T), int(self.H),
            int(self.batch), float(self.gp_lam), float(self.temp),
            self.scene_profile, int(self.nfe), float(self.phi_s),
        ) != (16, 4, 180, 10, 128, OE.GP_LAMBDA, 1.0, OE.SCENE_PROFILE, 8, 0.9):
            raise ValueError("immutable offline executed-window core changed")
        if int(self.verifier_workers) < 1:
            raise ValueError("verifier_workers must be positive")
        return self

    @property
    def arm_name(self):
        alpha = str(float(self.alpha)).replace(".", "p")
        lr = f"{self.lr:.0e}".replace("-", "m")
        ess = str(float(self.ess_target)).replace(".", "p")
        return (
            f"{self.tag}_{self.selector}_a{alpha}_e{int(self.exposure_epochs):03d}"
            f"_lr{lr}_ess{ess}_{self.replay_mode}"
        )


def run(checkpoint, outdir, cfg, *, device):
    cfg.validate()
    checkpoint = os.path.abspath(checkpoint)
    outdir = os.path.abspath(outdir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = OS.sha256_file(checkpoint)
    if checkpoint_sha != OE.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA mismatch: expected "
            f"{OE.EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse output directory: {outdir}")
    os.makedirs(outdir)
    environment = SS.scene_profile(cfg.scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    frozen_parameters = BS.configure_expansion_trainability(policy)
    visual_encoder_sha = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr,
    )
    BX._save_checkpoint(policy, os.path.join(outdir, "round_00.pt"), dict(
        round=0, experiment=cfg.arm_name, source_checkpoint=checkpoint,
        source_sha256=checkpoint_sha, encoder_sha256=visual_encoder_sha,
        recipe=asdict(cfg),
    ))
    history = []
    previous_shard = None
    preflight_scenarios = SP.expansion_scenarios(1, smoke=cfg.smoke)
    preflight_replicas = [
        BX.Replica(
            scenario_id, gamma,
            n_ped=environment["n_ped"],
            ped_speed_range=tuple(environment["ped_speed_range"]),
        )
        for scenario_id in preflight_scenarios for gamma in SP.GAMMAS
    ]
    ell0, ell, ell_preflight = OE._initial_lengthscale(
        policy, preflight_replicas, cfg, device,
    )
    with ProcessPoolExecutor(max_workers=cfg.verifier_workers) as executor:
        for round_i in range(1, cfg.rounds + 1):
            round_start = time.perf_counter()
            scenarios = SP.expansion_scenarios(round_i, smoke=cfg.smoke)
            replicas = [
                BX.Replica(
                    scenario_id, gamma,
                    n_ped=environment["n_ped"],
                    ped_speed_range=tuple(environment["ped_speed_range"]),
                )
                for scenario_id in scenarios for gamma in SP.GAMMAS
            ]
            if len(replicas) != 56:
                raise RuntimeError("offline macro-round requires 56 episodes")
            policy.eval()
            phi_policy = copy.deepcopy(policy).eval()
            for parameter in phi_policy.parameters():
                parameter.requires_grad_(False)
            gp, gp_ids, gp_selection = OE.gp_from_previous(
                phi_policy, previous_shard, round_i=round_i, ell=ell,
                cap=OE.CAP, lam=cfg.gp_lam, phi_s=cfg.phi_s, device=device,
                seed=cfg.seed + round_i * 101,
            )
            beta, calibrated_ess = OE._calibrate_beta(
                phi_policy, gp, replicas, cfg, device, round_i=round_i,
            )
            shard = OS.ExecutedRoundShard(round_i)
            gather = OE.gather_offline_round(
                policy, phi_policy, gp, beta, replicas, cfg, shard, device,
                executor, round_i=round_i,
            )
            shard_path = os.path.join(
                outdir, "round_shards", f"round_{round_i:02d}.pt",
            )
            shard_manifest = shard.save(shard_path)
            replay_start = time.perf_counter()
            replay = AUG.replay_with_mode(
                policy, optimizer, shard,
                mode=cfg.replay_mode, alpha=cfg.alpha,
                exposure_epochs=cfg.exposure_epochs, batch=cfg.batch,
                device=device, seed=cfg.seed + round_i * 1_000_003,
                executor=executor,
            )
            gather["timers"]["replay"] = time.perf_counter() - replay_start
            if BS.module_sha256(policy.enc_grid) != visual_encoder_sha:
                raise RuntimeError("visual encoder SHA changed")
            checkpoint_path = os.path.join(outdir, f"round_{round_i:02d}.pt")
            BX._save_checkpoint(policy, checkpoint_path, dict(
                round=round_i, experiment=cfg.arm_name,
                source_checkpoint=checkpoint, source_sha256=checkpoint_sha,
                encoder_sha256=visual_encoder_sha, recipe=asdict(cfg),
                ell=ell, ell0=ell0, cap=OE.CAP, beta=float(beta),
            ))
            record = dict(
                round=round_i, experiment=cfg.arm_name,
                scenarios=list(map(int, scenarios)),
                environment=environment, beta=float(beta),
                calibrated_normalized_ess_over_remaining=float(calibrated_ess),
                verifier=SM.verifier_manifest(),
                gp_buffer_ids=gp_ids, gp_selection=gp_selection,
                gp=gp.diagnostics(), gather=gather, replay=replay,
                shard=shard_manifest,
                checkpoint=os.path.abspath(checkpoint_path),
                checkpoint_sha256=OS.sha256_file(checkpoint_path),
                wall_seconds=time.perf_counter() - round_start,
            )
            history.append(record)
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(dict(
                round=round_i, experiment=cfg.arm_name,
                D=shard_manifest["D"], Dplus=shard_manifest["Dplus"],
                Dminus=shard_manifest["Dminus"], beta=float(beta),
                replay_mode=cfg.replay_mode,
                Adam_steps=int(replay["optimizer_steps"]),
                wall_seconds=record["wall_seconds"],
            )), flush=True)
            previous_shard = shard

    manifest = dict(
        status="CLAUDE_SFM_B1_OFFLINE_EXT_COMPLETE",
        experiment=cfg.arm_name,
        scientific_role="offline_expansion_data_collector_not_safe_controller",
        recipe=asdict(cfg),
        replay_rules=AUG.declared_rules(),
        constants=dict(
            ell=ell, ell0=ell0, ell_preflight=ell_preflight,
            gp_buffer_cap=OE.CAP, gp_lambda=OE.GP_LAMBDA,
            expected_checkpoint_sha256=OE.EXPECTED_CHECKPOINT_SHA256,
            replay_window_rounds=1,
        ),
        source=OE._source(),
        source_checkpoint=checkpoint,
        source_checkpoint_sha256=checkpoint_sha,
        environment=environment,
        frozen_parameters=frozen_parameters,
        visual_encoder_sha=visual_encoder_sha,
        history=history,
    )
    OE._write_json(os.path.join(outdir, "COMPLETE.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--exposure-epochs", type=int, required=True)
    parser.add_argument(
        "--selector", choices=OE.EXECUTION_SELECTORS, default="margin",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument(
        "--replay-mode", choices=AUG.REPLAY_MODES, default="original",
    )
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="ext")
    args = parser.parse_args(argv)
    cfg = ExtConfig(
        alpha=args.alpha, exposure_epochs=args.exposure_epochs,
        selector=args.selector, rounds=args.rounds, lr=args.lr,
        ess_target=args.ess_target, replay_mode=args.replay_mode,
        verifier_workers=args.verifier_workers, seed=args.seed,
        smoke=args.smoke, tag=args.tag,
    )
    run(args.checkpoint, args.outdir, cfg, device=args.device)


if __name__ == "__main__":
    main()
