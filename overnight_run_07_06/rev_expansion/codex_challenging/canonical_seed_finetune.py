#!/usr/bin/env python3
"""One-time certified canonical seed diagnostic for the Stage 5 cold start.

This is deliberately not the final recipe and is never written over the active
pretrained checkpoint.  It asks a narrow question: can a tiny set of exact-valid
SafeMPPI trajectories put the reverse-direction task inside the self-expansion
support?  The seed is used for a fixed number of CFM steps and then discarded;
there is no persistent demo fraction or LwF anchor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import gen_sg_data as SG
import grid_hp_expt as HP
import grid_scene as GS
from viz_style import GAMMAS


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
OVERNIGHT = HERE.parent / "codex_overnight"
sys.path.insert(0, str(OVERNIGHT))
import grid_metrics2 as GM2  # noqa: E402


def generate_seed(path: Path, trajectories_per_gamma: int, seed: int) -> dict:
    env = SG.SEEDS.make_walled_env(8)
    cfg = GS.mode1_config()
    start = np.array([0.05, 0.05], dtype=np.float32)
    goal = np.array([5.0, 5.0], dtype=np.float32)
    pieces = {key: [] for key in ("grid", "low5", "hist", "U")}
    gamma_ids: list[torch.Tensor] = []
    paths: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    attempts_by_gamma = {}
    for gamma_id, gamma in enumerate(GAMMAS):
        accepted = 0
        attempts = 0
        while accepted < trajectories_per_gamma and attempts < 30 * trajectories_per_gamma:
            attempt_seed = seed + gamma_id * 1000 + attempts
            states, actions, status, _, _ = SG.rollout_pair(
                env, cfg, start, goal, float(gamma), attempt_seed,
                device=torch.device("cpu"), reach=0.15, seed_base=seed,
                max_retries=0,
            )
            attempts += 1
            exact = bool(status["success"] and GM2.traj_valid2(states[:, :2], env, float(gamma)))
            if not exact:
                continue
            G, L, H, U = SG.goal_windows(states, actions, env, goal, float(gamma))
            count = len(U)
            for key, values in zip(("grid", "low5", "hist", "U"), (G, L, H, U)):
                pieces[key].append(torch.as_tensor(np.asarray(values), dtype=torch.float32))
            gamma_ids.append(torch.full((count,), gamma_id, dtype=torch.long))
            paths.append(states[:, :2].astype(np.float32))
            controls.append(actions.astype(np.float32))
            accepted += 1
        attempts_by_gamma[str(gamma)] = attempts
        if accepted != trajectories_per_gamma:
            raise RuntimeError(
                f"gamma={gamma}: only {accepted}/{trajectories_per_gamma} exact-valid experts "
                f"after {attempts} attempts"
            )
        print(f"[seed] gamma={gamma:g} exact={accepted}/{attempts}", flush=True)
    payload = {key: torch.cat(chunks) for key, chunks in pieces.items()}
    payload["gamma_id"] = torch.cat(gamma_ids)
    payload.update(
        paths=paths,
        controls=controls,
        gammas=list(GAMMAS),
        trajectories_per_gamma=trajectories_per_gamma,
        attempts_by_gamma=attempts_by_gamma,
        start=start,
        goal=goal,
        reach=0.15,
        wall_plugs=8,
        exact_valid2=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[seed] saved {len(payload['U']):,} windows -> {path}", flush=True)
    return payload


def grad_rms(parameters) -> float:
    values = [float(p.grad.detach().square().mean()) for p in parameters if p.grad is not None]
    return float(np.sqrt(np.mean(values))) if values else 0.0


@torch.no_grad()
def fixed_field(policy, batch, device):
    n = min(128, len(batch["U"]))
    G = batch["grid"][:n].to(device); L = batch["low5"][:n].to(device)
    H = batch["hist"][:n].to(device); U = batch["U"][:n].to(device)
    x = 0.5 * (U / policy.u_max).reshape(n, policy.d)
    tau = torch.full((n,), 0.5, device=device)
    return policy.forward(x, tau, policy._expand_ctx(policy.ctx_from(G, L, H), n)).detach()


def train(args, seed_data: dict) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    policy, source = HP.load_hp(args.checkpoint, device=device)
    policy.train()
    field = list(policy.trunk.parameters()) + list(policy.head.parameters())
    encoder = policy.encoder_modules()
    if args.freeze_encoder:
        for parameter in encoder:
            parameter.requires_grad_(False)
        groups = [{"params": field, "lr": args.lr}]
    else:
        for parameter in encoder:
            parameter.requires_grad_(True)
        groups = [
            {"params": field, "lr": args.lr},
            {"params": encoder, "lr": args.lr * args.enc_lr_mult},
        ]
    optimizer = torch.optim.Adam(groups)
    tensors = {key: seed_data[key] for key in ("grid", "low5", "hist", "U", "gamma_id")}
    reference = fixed_field(policy, tensors, device)
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        # Equal gamma quota first, then shuffle: the tiny seed cannot be dominated by
        # gamma=0.1's longer trajectories.
        picks = []
        per_gamma = max(1, args.batch // len(GAMMAS))
        for gamma_id in range(len(GAMMAS)):
            pool = torch.where(tensors["gamma_id"] == gamma_id)[0]
            choice = pool[torch.randint(len(pool), (per_gamma,), generator=generator)]
            picks.append(choice)
        index = torch.cat(picks)
        index = index[torch.randperm(len(index), generator=generator)[: args.batch]]
        G = tensors["grid"][index].to(device); L = tensors["low5"][index].to(device)
        H = tensors["hist"][index].to(device); U = tensors["U"][index].to(device)
        loss = policy.cfm_loss(U, policy.ctx_from(G, L, H))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        field_grad = grad_rms(field); encoder_grad = grad_rms(encoder)
        torch.nn.utils.clip_grad_norm_(field, 1.0)
        if not args.freeze_encoder:
            torch.nn.utils.clip_grad_norm_(encoder, 5.0)
        optimizer.step()
        current = fixed_field(policy, tensors, device)
        drift = float((current - reference).norm(dim=1).mean() /
                      reference.norm(dim=1).mean().clamp_min(1e-9))
        row = dict(step=step, loss=float(loss.detach()), field_grad_rms=field_grad,
                   encoder_grad_rms=encoder_grad, canonical_field_drift=drift)
        history.append(row)
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"[step {step:03d}] loss={row['loss']:.4f} field={field_grad:.3e} "
                  f"enc={encoder_grad:.3e} drift={drift:.4f}", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.outdir / "final.pt"
    HP.save_hp(
        policy.eval(), checkpoint_path,
        extra={
            "canonical_seed_diagnostic": True,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "seed_dataset": str(args.seed_data.resolve()),
            "steps": args.steps,
            "freeze_encoder": args.freeze_encoder,
            "enc_lr_mult": args.enc_lr_mult,
            "lr": args.lr,
            "history_tail": history[-1],
        },
    )
    summary = {
        "status": "PASS",
        "diagnostic_only": True,
        "persistent_anchor": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_schema": source["config"]["schema_version"],
        "windows": len(seed_data["U"]),
        "trajectories_per_gamma": seed_data["trajectories_per_gamma"],
        "steps": args.steps,
        "freeze_encoder": args.freeze_encoder,
        "enc_lr_mult": args.enc_lr_mult,
        "lr": args.lr,
        "wall_seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "pretrained_sg_walls8.pt")
    parser.add_argument("--seed-data", type=Path,
                        default=HERE / "stage_results/05_sanity/data/canonical_seed_windows.pt")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--trajectories-per-gamma", type=int, default=1)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch", type=int, default=63)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--enc-lr-mult", type=float, default=0.3)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.regenerate or not args.seed_data.exists():
        seed_data = generate_seed(args.seed_data, args.trajectories_per_gamma, args.seed)
    else:
        seed_data = torch.load(args.seed_data, map_location="cpu", weights_only=False)
        if not seed_data.get("exact_valid2", False):
            raise ValueError("seed dataset is not marked exact-valid2")
    train(args, seed_data)


if __name__ == "__main__":
    main()
