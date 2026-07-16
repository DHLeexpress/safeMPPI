#!/usr/bin/env python3
"""Balanced ID/OOD finite-sampler distillation before window expansion.

This stage preserves the original low5 + E(H_P) architecture and the requested
batch cap of 16.  Half of every batch is drawn from the geometrically balanced
ID corpus and half from the real successful SafeMPPI OOD demonstrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from giant_obstacle_ood import stage3_pretrain_balanced as S3  # noqa: E402
from giant_obstacle_ood.stage4_frozen_ood import CHECKPOINT as ID_CHECKPOINT  # noqa: E402
from giant_obstacle_ood.stage5_evaluate import evaluate_checkpoint  # noqa: E402
from giant_obstacle_ood import stage5_window_expand as S5  # noqa: E402
from viz_style import GAMMAS  # noqa: E402


W = S5.W
TARGET_GAMMAS = (0.1, 0.5, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ood_demo() -> dict:
    if not S5.DEMO_PATH.exists():
        S5.prepare_demo()
    cfg = W.CurConfig()
    cfg.demo_override_path = str(S5.DEMO_PATH.resolve())
    cfg.demo_stage_balanced = True
    cfg.giant_center = (2.5, 2.5)
    cfg.giant_radius = float(S5.RADIUS)
    demo = W._load_demo(cfg)
    if demo["U"].shape[0] != 2688:
        raise RuntimeError(f"expected 2,688 balanced OOD windows, got {len(demo['U'])}")
    return demo


def draw_id_rows(pool: dict[str, torch.Tensor], count: int) -> torch.Tensor:
    """Uniform gamma/route draw; replacement keeps every small batch balanced in expectation."""
    gamma_id = pool["gamma_id"].numpy()
    mode = pool["mode_label"].numpy()
    selected: list[int] = []
    strata = [(gid, sign) for gid in range(len(GAMMAS)) for sign in (-1.0, 1.0)]
    offset = int(np.random.randint(len(strata)))
    for row in range(count):
        gid, sign = strata[(offset + row) % len(strata)]
        choices = np.where((gamma_id == gid) & np.isclose(mode, sign))[0]
        selected.append(int(np.random.choice(choices)))
    np.random.shuffle(selected)
    return torch.as_tensor(selected, dtype=torch.long)


def draw_ood_rows(demo: dict, count: int, target_gammas: tuple[float, ...]) -> torch.Tensor:
    if not target_gammas:
        return W._draw_demo_rows(demo, count, (0.25, 0.25, 0.25, 0.25))
    gamma = demo["gamma"].numpy()
    stage = np.asarray(demo["sample_stage"], dtype=object)
    mode = np.asarray(demo["sample_mode"], dtype=object)
    stages = ("initial", "approach", "boundary", "post")
    modes = ("lower-right", "upper-left")
    # Interleave gamma inside each stage/route block.  With eight OOD rows in
    # a batch, the previous gamma-major ordering could draw an entire update
    # from only one gamma.  This ordering gives every small update all target
    # gammas while cycling geometric stage and route without changing their
    # long-run mass.
    strata = [(value, name, route) for name in stages for route in modes for value in target_gammas]
    offset = int(np.random.randint(len(strata)))
    selected: list[int] = []
    for row in range(count):
        value, name, route = strata[(offset + row) % len(strata)]
        choices = np.where(np.isclose(gamma, value) & (stage == name) & (mode == route))[0]
        if not len(choices):
            raise RuntimeError(f"empty OOD distillation stratum gamma={value} stage={name} mode={route}")
        selected.append(int(np.random.choice(choices)))
    np.random.shuffle(selected)
    return torch.as_tensor(selected, dtype=torch.long)


def save_checkpoint(policy, output: Path, *, step: int, recipe: dict) -> Path:
    checkpoint = output / f"ckpt_{step:05d}.pt"
    W.HP.save_hp(policy.cpu(), checkpoint, extra={
        "iter": int(step),
        "stage4b_ood_distill": recipe,
    })
    policy.to(recipe["device"])
    return checkpoint


def target_counts(summary: dict) -> dict[str, int]:
    return {
        str(gamma): int(summary["per_gamma"][str(gamma)]["successes"])
        for gamma in TARGET_GAMMAS
    }


def deployed_action_loss(policy, grid, low5, hist, controls, *, x0_scale: float,
                         nfe: int, mode_labels: torch.Tensor) -> torch.Tensor:
    """Exact finite-NFE loss on the only action executed by deployment.

    The ordinary endpoint objective treats all ten proposal actions equally,
    although receding-horizon rollout applies only ``proposal[0]``.  This term
    closes that objective mismatch without changing the sampler or adding a
    safety filter.  Every row retains the exact x/y-reflected partner and the
    same route-half-space Gaussian coupling used by evaluation.
    """
    rgrid, rlow, rhist, rcontrols = W._reflect_xy_batch(
        grid, low5, hist, controls,
    )
    batch = len(controls)
    sampled = float(x0_scale) * torch.randn(
        batch, policy.d, device=controls.device, dtype=controls.dtype,
    )
    sampled = W._couple_x0_to_control_mode(sampled, controls, mode_labels)
    reflected = W._reflect_flat_controls(sampled)
    context = policy.ctx_from(grid, low5, hist)
    reflected_context = policy.ctx_from(rgrid, rlow, rhist)
    step = 1.0 / int(nfe)
    for index in range(int(nfe)):
        tau = torch.full(
            (batch,), index / int(nfe), device=sampled.device,
            dtype=sampled.dtype,
        )
        sampled = sampled + step * policy.forward(
            sampled, tau, policy._expand_ctx(context, batch),
        )
        reflected = reflected + step * policy.forward(
            reflected, tau, policy._expand_ctx(reflected_context, batch),
        )
    target = (controls / policy.u_max).reshape(batch, policy.d)
    reflected_target = (rcontrols / policy.u_max).reshape(batch, policy.d)
    return 0.5 * (
        ((sampled[:, :2] - target[:, :2]) ** 2).mean()
        + ((reflected[:, :2] - reflected_target[:, :2]) ** 2).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ID_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--gate-every", type=int, default=200)
    parser.add_argument("--gate-m", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16, choices=(16,))
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-cfm-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--deployed-action-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-noise-repeats", type=int, default=1)
    parser.add_argument("--equivariance-weight", type=float, default=1.0)
    parser.add_argument("--ood-target-gammas", type=float, nargs="*", default=())
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.steps <= 0 or args.gate_every <= 0 or args.gate_m <= 0:
        parser.error("step and gate bounds must be positive")
    if args.endpoint_noise_repeats <= 0:
        parser.error("--endpoint-noise-repeats must be positive")
    if args.lr <= 0 or min(
        args.eval_cfm_weight, args.endpoint_weight, args.deployed_action_weight,
        args.equivariance_weight,
    ) < 0:
        parser.error("learning rate must be positive and loss weights nonnegative")
    if any(float(gamma) not in GAMMAS for gamma in args.ood_target_gammas):
        parser.error("--ood-target-gammas must be drawn from the configured gamma sweep")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy, source_payload = W.HP.load_hp(args.checkpoint.resolve(), device=device)
    if policy.config().get("raw_start_goal", False) or policy.ctx_dim != 37:
        raise RuntimeError("distillation must retain the original 37-D endpoint-free architecture")
    teacher = W.copy.deepcopy(policy).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    id_pool, id_sources = S3.load_pool(S3.DEFAULT_DATA)
    mode_lookup = torch.tensor(
        [1.0 if word.startswith("R") else -1.0 for word in S3.ALL_SIGNATURES],
        dtype=torch.float32,
    )
    id_pool["mode_label"] = mode_lookup[id_pool["window_signature_ids"].long()]
    ood = load_ood_demo()
    ood_mode = torch.as_tensor(
        [W._stable_route_label(mode) for mode in ood["sample_mode"]], dtype=torch.float32,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    recipe = {
        "status": "RUNNING",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": sha256(args.checkpoint.resolve()),
        "source_raw_start_goal": bool(source_payload["config"].get("raw_start_goal", False)),
        "architecture": "original low5 + E(H_P), ctx_dim=37",
        "batch": 16,
        "id_rows_per_batch": 8,
        "ood_rows_per_batch": 8,
        "ood_target_gammas": [float(gamma) for gamma in args.ood_target_gammas],
        "ood_batch_strata_order": "stage_then_route_then_gamma_interleaved",
        "gather_temperature_for_next_stage": 1.0,
        "training_noise_scales": [1.0, 0.5],
        "deployed_action_objective": {
            "weight": float(args.deployed_action_weight),
            "action_index": 0,
            "finite_nfe": 8,
            "temperature": 0.5,
            "reflection_paired": True,
            "noise_repeats_per_context": int(args.endpoint_noise_repeats),
        },
        "rollout_gate_temperature": 0.5,
        "nfe": 8,
        "mode_noise_coupling": True,
        "reflection_pairing": True,
        "id_sources": id_sources,
        "ood_source": str(S5.DEMO_PATH.resolve()),
        "ood_sha256": sha256(S5.DEMO_PATH),
        "args": vars(args) | {"output": str(output), "checkpoint": str(args.checkpoint)},
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2, default=str) + "\n")

    history: list[dict] = []
    best_score = (-1, -1, -1.0, -1.0)
    best_checkpoint: Path | None = None
    started = time.perf_counter()
    policy.train()
    for step in range(1, args.steps + 1):
        id_idx = draw_id_rows(id_pool, 8)
        ood_idx = draw_ood_rows(ood, 8, tuple(float(gamma) for gamma in args.ood_target_gammas))
        grid = torch.cat((id_pool["grid"][id_idx], ood["grid"][ood_idx])).to(device)
        low5 = torch.cat((id_pool["low5"][id_idx], ood["low5"][ood_idx])).to(device)
        hist = torch.cat((id_pool["hist"][id_idx], ood["hist"][ood_idx])).to(device)
        controls = torch.cat((id_pool["U"][id_idx], ood["U"][ood_idx])).to(device)
        labels = torch.cat((id_pool["mode_label"][id_idx], ood_mode[ood_idx])).to(device)

        standard, _, standard_eq = W._symmetric_cfm_loss_x0(
            policy, grid, low5, hist, controls, args.equivariance_weight,
            mode_noise_coupling=True, mode_labels=labels,
        )
        eval_cfm, _, eval_eq = W._symmetric_cfm_loss_x0(
            policy, grid, low5, hist, controls, args.equivariance_weight,
            x0_scale=0.5, mode_noise_coupling=True, mode_labels=labels,
        )
        repeats = int(args.endpoint_noise_repeats)
        repeat = lambda value: value.repeat_interleave(repeats, dim=0)
        endpoint_grid = repeat(grid[8:])
        endpoint_low5 = repeat(low5[8:])
        endpoint_hist = repeat(hist[8:])
        endpoint_controls = repeat(controls[8:])
        endpoint_labels = repeat(labels[8:])
        endpoint = W._symmetric_eval_endpoint_loss(
            policy, endpoint_grid, endpoint_low5, endpoint_hist, endpoint_controls,
            x0_scale=0.5, nfe=8, mode_noise_coupling=True,
            mode_labels=endpoint_labels,
        )
        deployed = deployed_action_loss(
            policy, endpoint_grid, endpoint_low5, endpoint_hist, endpoint_controls,
            x0_scale=0.5, nfe=8, mode_labels=endpoint_labels,
        ) if args.deployed_action_weight > 0 else endpoint.new_zeros(())
        loss = (
            standard + args.eval_cfm_weight * eval_cfm
            + args.endpoint_weight * endpoint
            + args.deployed_action_weight * deployed
        ) / (
            1.0 + args.eval_cfm_weight + args.endpoint_weight
            + args.deployed_action_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(
                f"DISTILL step={step}/{args.steps} loss={float(loss.detach()):.5f} "
                f"std={float(standard.detach()):.5f} eval={float(eval_cfm.detach()):.5f} "
                f"endpoint={float(endpoint.detach()):.5f} grad={gradient_norm:.4f} "
                f"deployed={float(deployed.detach()):.5f} "
                f"eq={float(0.5 * (standard_eq + eval_eq)):.5f}",
                flush=True,
            )

        if step % args.gate_every != 0 and step != args.steps:
            continue
        checkpoint = save_checkpoint(policy, output, step=step, recipe=recipe)
        policy = policy.to(device).train()
        records, summary = evaluate_checkpoint(
            checkpoint, temperature=0.5, repetitions=args.gate_m, device=device,
            method=f"OOD distill step {step}", seed0=92500,
            persistent_route_bit=True,
        )
        counts = target_counts(summary)
        score = (
            sum(value > 0 for value in counts.values()),
            sum(counts.values()),
            float(summary["overall"]["a_SR"]),
            -float(summary["overall"]["b_CR"]),
        )
        row = {
            "step": step,
            "checkpoint": str(checkpoint),
            "target_successes": counts,
            "target_gamma_nonzero": score[0],
            "overall_SR": summary["overall"]["a_SR"],
            "overall_CR": summary["overall"]["b_CR"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (output / "gate_history.json").write_text(json.dumps(history, indent=2) + "\n")
        (output / f"metrics_{step:05d}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"DISTILL_GATE step={step} g0.1={counts['0.1']} g0.5={counts['0.5']} "
            f"g1={counts['1.0']} overall_SR={summary['overall']['a_SR']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint
            W.HP.save_hp(policy.cpu(), output / "best.pt", extra={
                "iter": int(step), "stage4b_ood_distill": recipe,
                "gate_summary": summary,
            })
            policy = policy.to(device).train()
        if score[0] == len(TARGET_GAMMAS):
            recipe.update({
                "status": "PASS",
                "selected_checkpoint": str(checkpoint),
                "selected_step": step,
                "target_successes": counts,
                "elapsed_seconds": time.perf_counter() - started,
            })
            (output / "manifest.json").write_text(json.dumps(recipe, indent=2, default=str) + "\n")
            print(f"DISTILL_PASS checkpoint={checkpoint}", flush=True)
            return

    recipe.update({
        "status": "MAX_STEPS_WITHOUT_PASS",
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_score": list(best_score),
        "elapsed_seconds": time.perf_counter() - started,
    })
    (output / "manifest.json").write_text(json.dumps(recipe, indent=2, default=str) + "\n")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
