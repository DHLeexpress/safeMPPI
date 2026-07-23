"""Isolated two-round max-margin B1 alpha/replay-epoch experiment.

This module intentionally does not change the authenticated Arm-A runner.  It
keeps the B1 gather path fixed and varies only

    alpha in {0, 0.01, 0.1}
    complete W=2 replay epochs in {1, 10, 100}.

One replay epoch visits every eligible record exactly once, accumulates the
hierarchically weighted objective over minibatches, and then takes one Adam
step.  Consequently ``replay_epochs`` is also the number of optimizer steps
per macro-round whenever positive support is non-empty.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_store as BS
import sfm_protocol as SP
import sfm_scene as SS
import sfm_metrics2 as SM


EXPECTED_CHECKPOINT_SHA256 = "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
ELL = 0.24210826720721101
CAP = 256
GP_LAMBDA = 1.0e-2
LEARNING_RATE = 1.0e-4
ROUNDS = 2
ALPHAS = (0.0, 0.01, 0.1)
REPLAY_EPOCHS = (1, 10, 100)


@dataclass(frozen=True)
class ExperimentConfig:
    alpha: float
    replay_epochs: int
    rounds: int = ROUNDS
    K: int = 16
    B: int = 4
    T: int = 180
    H: int = 10
    W: int = 2
    batch: int = 128
    lr: float = LEARNING_RATE
    ess_target: float = 0.5
    nfe: int = 8
    temp: float = 1.0
    phi_s: float = 0.9
    gp_lam: float = GP_LAMBDA
    selector: str = "margin"
    verifier_workers: int = 32
    seed: int = 20260723
    scene_profile: str = "double_density_velocity_ood"
    smoke: bool = False

    def validate(self):
        if float(self.alpha) not in ALPHAS:
            raise ValueError(f"alpha must be one of {ALPHAS}")
        if int(self.replay_epochs) not in REPLAY_EPOCHS:
            raise ValueError(f"replay_epochs must be one of {REPLAY_EPOCHS}")
        expected = (
            ROUNDS, 16, 4, 180, 10, 2, 128, LEARNING_RATE, 0.5, 8, 1.0,
            0.9, GP_LAMBDA, "margin", "double_density_velocity_ood", False,
        )
        actual = (
            self.rounds, self.K, self.B, self.T, self.H, self.W, self.batch,
            self.lr, self.ess_target, self.nfe, self.temp, self.phi_s,
            self.gp_lam, self.selector, self.scene_profile, self.smoke,
        )
        if actual != expected:
            raise ValueError("fixed two-round alpha/replay experiment contract changed")
        if int(self.verifier_workers) < 1:
            raise ValueError("verifier_workers must be positive")
        return self

    @property
    def arm_name(self):
        alpha = str(float(self.alpha)).replace(".", "p")
        return f"margin_alpha{alpha}_epochs{int(self.replay_epochs):03d}"


def _set_update_seed(seed):
    """Seed every RNG used by CFM noise, dropout, and replay ordering."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _records(recent, alpha):
    positives = [
        (shard, query)
        for shard, query in recent.positive_records()
        if query["train_eligible"]
    ]
    # Preserve the established alpha=0 contract exactly: D- is not read,
    # including for diagnostics. Its negative fixed-probe loss is therefore
    # explicitly reported as null in the alpha=0 arm.
    negatives = [] if float(alpha) == 0.0 else list(recent.negative_records())
    return positives, negatives


def _coverage(visited, eligible):
    identities = list(visited)
    return dict(
        eligible=int(eligible),
        visited=len(identities),
        unique_visited=len(set(identities)),
        exact_once=bool(
            len(identities) == int(eligible)
            and len(set(identities)) == int(eligible)
        ),
    )


def _compact_mass(accounting):
    """Keep exact mass checks without serializing every cell/context key."""
    gamma = dict(accounting.get("gamma", {}))
    gamma_values = list(map(float, gamma.values()))
    return dict(
        total=float(accounting.get("total", 0.0)),
        gamma=gamma,
        gamma_spread=(
            max(gamma_values) - min(gamma_values) if gamma_values else 0.0
        ),
        cells=len(accounting.get("cells", {})),
        contexts=len(accounting.get("contexts", {})),
    )


def _gradient_cosine(left, right):
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for name in set(left) | set(right):
        lvalue = left.get(name)
        rvalue = right.get(name)
        if lvalue is not None:
            left_sq += lvalue.to(torch.float64).square().sum().cpu()
        if rvalue is not None:
            right_sq += rvalue.to(torch.float64).square().sum().cpu()
        if lvalue is not None and rvalue is not None:
            dot += (lvalue.to(torch.float64) * rvalue.to(torch.float64)).sum().cpu()
    denominator = float((left_sq * right_sq).sqrt())
    return None if denominator <= 0.0 else float(dot) / denominator


def _group_gradient_norms(policy, snapshot):
    parameter_names = {id(parameter): name for name, parameter in policy.named_parameters()}
    result = {}
    for group_name, module in policy.module_groups().items():
        squared = torch.zeros((), dtype=torch.float64)
        for parameter in module.parameters():
            value = snapshot.get(parameter_names[id(parameter)])
            if value is not None:
                squared += value.to(torch.float64).square().sum().cpu()
        result[group_name] = float(squared.sqrt())
    return result


def _module_snapshot(policy):
    return {
        group: {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        }
        for group, module in policy.module_groups().items()
    }


def _module_relative_drift(before, after, eps=1.0e-12):
    result = {}
    for group in before:
        delta_sq = torch.zeros((), dtype=torch.float64)
        base_sq = torch.zeros((), dtype=torch.float64)
        for name, initial in before[group].items():
            final = after[group][name]
            delta_sq += (final.to(torch.float64) - initial.to(torch.float64)).square().sum()
            base_sq += initial.to(torch.float64).square().sum()
        result[group] = float(delta_sq.sqrt() / base_sq.sqrt().clamp_min(float(eps)))
    return result


@torch.no_grad()
def _fixed_probe_loss(policy, records, *, batch, device, seed):
    """Evaluate deterministic, dropout-disabled CFM loss on complete support."""
    if not records:
        return None
    mass, _ = BS.hierarchy_mass(records)
    was_training = policy.training
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    total = 0.0
    for values in BS._batches(records, int(batch)):
        grid, low, hist, controls = BS._tensor_batch(values, device)
        context = policy.ctx_from(grid, low, hist)
        count = len(values)
        x1 = (controls / policy.u_max).reshape(count, policy.d)
        x0 = torch.randn(
            x1.shape, dtype=x1.dtype, device=x1.device, generator=generator,
        )
        tau = torch.rand(
            count, dtype=x1.dtype, device=x1.device, generator=generator,
        ).clamp(1.0e-4, 1.0)
        x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        target = x1 - x0
        prediction = policy.forward(x_tau, tau, context)
        per = (prediction - target).square().mean(dim=1)
        weights = torch.as_tensor(
            [mass[(id(shard), int(query["query_id"]))] for shard, query in values],
            dtype=per.dtype, device=per.device,
        )
        total += float((per * weights).sum())
    policy.train(was_training)
    return float(total)


def _positive_epoch(policy, optimizer, positives, *, batch, device, seed, path):
    ordered = BS.hierarchical_order(positives, seed)
    mass, accounting = BS.hierarchy_mass(ordered)
    optimizer.zero_grad(set_to_none=True)
    if not ordered:
        return dict(
            path=path, optimizer_steps=0, positive_loss=0.0,
            positive_norm=0.0, positive_group_norms={},
            positive_coverage=_coverage([], 0),
            positive_mass=_compact_mass(accounting),
            negative_coverage=_coverage([], 0),
        )
    loss, visited = BS._accumulate_objective(policy, ordered, mass, batch, device)
    gradient = BS._gradient_snapshot(policy)
    norm = BS._gradient_norm(gradient)
    group_norms = _group_gradient_norms(policy, gradient)
    coverage = _coverage(visited, len(ordered))
    if not coverage["exact_once"]:
        raise RuntimeError("positive replay did not cover every eligible record exactly once")
    optimizer.step()
    return dict(
        path=path, optimizer_steps=1, positive_loss=float(loss),
        positive_norm=float(norm), positive_group_norms=group_norms,
        positive_coverage=coverage, positive_mass=_compact_mass(accounting),
        negative_coverage=_coverage([], 0),
    )


def _signed_epoch(policy, optimizer, positives, negatives, *, alpha, batch, device, seed):
    if float(alpha) == 0.0:
        return _positive_epoch(
            policy, optimizer, positives, batch=batch, device=device,
            seed=seed, path="positive_only",
        )
    if not negatives:
        return _positive_epoch(
            policy, optimizer, positives, batch=batch, device=device,
            seed=seed, path="positive_fallback_no_negative",
        )

    positive_order = BS.hierarchical_order(positives, seed)
    negative_order = BS.hierarchical_order(negatives, seed + 1)
    positive_mass, positive_accounting = BS.hierarchy_mass(positive_order)
    negative_mass, negative_accounting = BS.hierarchy_mass(negative_order)
    if not positive_order:
        optimizer.zero_grad(set_to_none=True)
        negative_loss, negative_visited = BS._accumulate_objective(
            policy, negative_order, negative_mass, batch, device,
        )
        negative_gradient = BS._gradient_snapshot(policy)
        optimizer.zero_grad(set_to_none=True)
        negative_coverage = _coverage(negative_visited, len(negative_order))
        if not negative_coverage["exact_once"]:
            raise RuntimeError("negative-only replay coverage failure")
        return dict(
            path="signed_no_positive", optimizer_steps=0, alpha=float(alpha),
            rho=0.0, gradient_cosine=None, positive_loss=0.0,
            negative_loss=float(negative_loss), positive_norm=0.0,
            negative_norm=BS._gradient_norm(negative_gradient),
            positive_group_norms={},
            negative_group_norms=_group_gradient_norms(policy, negative_gradient),
            positive_coverage=_coverage([], 0),
            negative_coverage=negative_coverage,
            positive_mass=_compact_mass(positive_accounting),
            negative_mass=_compact_mass(negative_accounting),
        )

    optimizer.zero_grad(set_to_none=True)
    positive_loss, positive_visited = BS._accumulate_objective(
        policy, positive_order, positive_mass, batch, device,
    )
    positive_gradient = BS._gradient_snapshot(policy)
    positive_norm = BS._gradient_norm(positive_gradient)
    optimizer.zero_grad(set_to_none=True)
    negative_loss, negative_visited = BS._accumulate_objective(
        policy, negative_order, negative_mass, batch, device,
    )
    negative_gradient = BS._gradient_snapshot(policy)
    negative_norm = BS._gradient_norm(negative_gradient)
    rho = float(alpha) * positive_norm / (negative_norm + 1.0e-12)
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        positive = positive_gradient.get(name)
        negative = negative_gradient.get(name)
        if positive is None and negative is None:
            parameter.grad = None
        elif positive is None:
            parameter.grad = -rho * negative
        elif negative is None:
            parameter.grad = positive
        else:
            parameter.grad = positive - rho * negative

    positive_coverage = _coverage(positive_visited, len(positive_order))
    negative_coverage = _coverage(negative_visited, len(negative_order))
    if not positive_coverage["exact_once"] or not negative_coverage["exact_once"]:
        raise RuntimeError("signed replay did not cover complete support exactly once")
    optimizer.step()
    return dict(
        path="signed", optimizer_steps=1, alpha=float(alpha), rho=float(rho),
        gradient_cosine=_gradient_cosine(positive_gradient, negative_gradient),
        positive_loss=float(positive_loss), negative_loss=float(negative_loss),
        positive_norm=float(positive_norm), negative_norm=float(negative_norm),
        positive_group_norms=_group_gradient_norms(policy, positive_gradient),
        negative_group_norms=_group_gradient_norms(policy, negative_gradient),
        positive_coverage=positive_coverage, negative_coverage=negative_coverage,
        positive_mass=_compact_mass(positive_accounting),
        negative_mass=_compact_mass(negative_accounting),
    )


def _numeric_summary(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return dict(
        first=values[0], last=values[-1], mean=float(np.mean(values)),
        minimum=min(values), maximum=max(values),
    )


def repeat_complete_replay(policy, optimizer, recent, cfg, *, device, round_i):
    positives, negatives = _records(recent, cfg.alpha)
    probe_seed = cfg.seed + int(round_i) * 1_000_003
    fixed_probe_before = dict(
        positive=_fixed_probe_loss(
            policy, positives, batch=cfg.batch, device=device, seed=probe_seed,
        ),
        negative=_fixed_probe_loss(
            policy, negatives, batch=cfg.batch, device=device, seed=probe_seed + 1,
        ),
    )
    modules_before = _module_snapshot(policy)
    encoder_before = BS.module_sha256(policy.enc_grid)
    epoch_rows = []
    for epoch_i in range(int(cfg.replay_epochs)):
        epoch_seed = cfg.seed + int(round_i) * 100_000 + epoch_i
        _set_update_seed(epoch_seed)
        row = _signed_epoch(
            policy, optimizer, positives, negatives, alpha=cfg.alpha,
            batch=cfg.batch, device=device, seed=epoch_seed,
        )
        epoch_rows.append(dict(epoch=epoch_i + 1, seed=epoch_seed, **row))
    encoder_after = BS.module_sha256(policy.enc_grid)
    if encoder_after != encoder_before:
        raise RuntimeError("visual encoder changed during isolated replay")
    fixed_probe_after = dict(
        positive=_fixed_probe_loss(
            policy, positives, batch=cfg.batch, device=device, seed=probe_seed,
        ),
        negative=_fixed_probe_loss(
            policy, negatives, batch=cfg.batch, device=device, seed=probe_seed + 1,
        ),
    )
    modules_after = _module_snapshot(policy)
    optimizer_steps = sum(int(row["optimizer_steps"]) for row in epoch_rows)
    expected_steps = int(cfg.replay_epochs) if positives else 0
    if optimizer_steps != expected_steps:
        raise RuntimeError(
            f"expected {expected_steps} optimizer steps, observed {optimizer_steps}"
        )
    if any(not row["positive_coverage"]["exact_once"] for row in epoch_rows if positives):
        raise RuntimeError("an epoch omitted or duplicated positive support")
    if (
        float(cfg.alpha) > 0.0
        and negatives
        and any(not row["negative_coverage"]["exact_once"] for row in epoch_rows)
    ):
        raise RuntimeError("an epoch omitted or duplicated negative support")
    summary_keys = (
        "positive_loss", "negative_loss", "positive_norm", "negative_norm",
        "rho", "gradient_cosine",
    )
    return dict(
        alpha=float(cfg.alpha), replay_epochs=int(cfg.replay_epochs),
        optimizer_steps=optimizer_steps,
        positive_eligible=len(positives), negative_eligible=len(negatives),
        positive_total_visits=sum(
            row["positive_coverage"]["visited"] for row in epoch_rows
        ),
        negative_total_visits=sum(
            row["negative_coverage"]["visited"] for row in epoch_rows
        ),
        negative_used_for_training=bool(float(cfg.alpha) > 0.0 and negatives),
        exact_complete_replay=True,
        fixed_probe=dict(before=fixed_probe_before, after=fixed_probe_after),
        module_relative_parameter_drift=_module_relative_drift(
            modules_before, modules_after,
        ),
        visual_encoder_sha_before=encoder_before,
        visual_encoder_sha_after=encoder_after,
        summaries={key: _numeric_summary(epoch_rows, key) for key in summary_keys},
        epochs=epoch_rows,
    )


def run(checkpoint, outdir, cfg, *, device):
    cfg.validate()
    checkpoint = os.path.abspath(checkpoint)
    outdir = os.path.abspath(outdir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = BS.sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse output directory: {outdir}")
    os.makedirs(outdir)
    environment = SS.scene_profile(cfg.scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    frozen_parameters = BS.configure_expansion_trainability(policy)
    visual_encoder_sha = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=cfg.lr,
    )
    recent = BS.RecentRounds(os.path.join(outdir, "round_shards"), cfg.W)
    proposal_generator = torch.Generator(device=device).manual_seed(cfg.seed)
    BX._save_checkpoint(policy, os.path.join(outdir, "round_00.pt"), dict(
        round=0, experiment=cfg.arm_name, source_checkpoint=checkpoint,
        source_sha256=checkpoint_sha, encoder_sha256=visual_encoder_sha,
        recipe=asdict(cfg),
    ))
    history = []
    with ProcessPoolExecutor(max_workers=cfg.verifier_workers) as executor:
        for round_i in range(1, cfg.rounds + 1):
            round_start = time.perf_counter()
            scenarios = SP.expansion_scenarios(round_i, smoke=False)
            replicas = [
                BX.Replica(
                    scenario_id, gamma, n_ped=environment["n_ped"],
                    ped_speed_range=tuple(environment["ped_speed_range"]),
                )
                for scenario_id in scenarios for gamma in SP.GAMMAS
            ]
            if len(replicas) != 56:
                raise RuntimeError("isolated experiment requires 56 macro-round replicas")
            policy.eval()
            phi_policy = copy.deepcopy(policy).eval()
            for parameter in phi_policy.parameters():
                parameter.requires_grad_(False)
            gp, gp_ids = BX.gp_from_recent(
                phi_policy, recent, ell=ELL, cap=CAP, lam=cfg.gp_lam,
                phi_s=cfg.phi_s, device=device, seed=cfg.seed + round_i * 101,
            )
            beta, calibrated_ess = BX._initial_beta(
                phi_policy, gp, replicas, cfg, device, cfg.seed + round_i * 1009,
            )
            shard = BS.RoundShard(round_i)
            gather = BX.gather_macro_round(
                policy, phi_policy, gp, beta, replicas, cfg, shard, device,
                executor, proposal_generator,
            )
            gather.pop("traces", None)
            shard_manifest = recent.append_and_save(shard)
            replay_start = time.perf_counter()
            replay = repeat_complete_replay(
                policy, optimizer, recent, cfg, device=device, round_i=round_i,
            )
            gather["timers"]["replay"] = time.perf_counter() - replay_start
            if BS.module_sha256(policy.enc_grid) != visual_encoder_sha:
                raise RuntimeError("visual encoder SHA changed")
            checkpoint_path = os.path.join(outdir, f"round_{round_i:02d}.pt")
            BX._save_checkpoint(policy, checkpoint_path, dict(
                round=round_i, experiment=cfg.arm_name,
                source_checkpoint=checkpoint, source_sha256=checkpoint_sha,
                encoder_sha256=visual_encoder_sha, recipe=asdict(cfg),
                ell=ELL, cap=CAP, beta=float(beta),
            ))
            record = dict(
                round=round_i, experiment=cfg.arm_name,
                scenarios=list(scenarios), environment=environment,
                beta=float(beta), calibrated_ess_over_K=float(calibrated_ess),
                verifier=SM.verifier_manifest(), gp_buffer_ids=gp_ids,
                gp=gp.diagnostics(), gather=gather, replay=replay,
                shard=shard_manifest, checkpoint=os.path.abspath(checkpoint_path),
                checkpoint_sha256=BS.sha256_file(checkpoint_path),
                wall_seconds=time.perf_counter() - round_start,
            )
            history.append(record)
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({
                "round": round_i, "experiment": cfg.arm_name,
                "beta": float(beta), "wall_seconds": record["wall_seconds"],
            }), flush=True)
    manifest = dict(
        status="R2_ALPHA_REPLAY_COMPLETE", experiment=cfg.arm_name,
        recipe=asdict(cfg), constants=dict(
            ell=ELL, cap=CAP, gp_lambda=GP_LAMBDA,
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        ),
        source_checkpoint=checkpoint, source_checkpoint_sha256=checkpoint_sha,
        environment=environment, frozen_parameters=frozen_parameters,
        visual_encoder_sha=visual_encoder_sha, history=history,
    )
    complete_path = os.path.join(outdir, "COMPLETE.json")
    temporary_complete = complete_path + ".tmp"
    with open(temporary_complete, "w") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(temporary_complete, complete_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--alpha", type=float, choices=ALPHAS, required=True)
    parser.add_argument(
        "--replay-epochs", type=int, choices=REPLAY_EPOCHS, required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--verifier-workers", type=int, default=32)
    args = parser.parse_args()
    cfg = ExperimentConfig(
        alpha=args.alpha, replay_epochs=args.replay_epochs,
        seed=args.seed, verifier_workers=args.verifier_workers,
    )
    run(args.checkpoint, args.outdir, cfg, device=args.device)


if __name__ == "__main__":
    main()
