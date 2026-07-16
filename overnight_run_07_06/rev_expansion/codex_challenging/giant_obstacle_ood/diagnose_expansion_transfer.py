#!/usr/bin/env python3
"""Fixed-seed audit of OOD expert fitting and exact-start mode emergence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
REF = ROOT / "reference"
for path in (WORK, ROOT.parent, ROOT, REF):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import window_expand_hardtail as W  # noqa: E402
from giant_obstacle_ood.stage1b_smooth_expert import GOAL, START  # noqa: E402
from giant_obstacle_ood.stage5_window_expand import DEMO_PATH, make_config, make_env  # noqa: E402
from viz_style import GAMMAS  # noqa: E402


def open_loop_endpoint(controls: torch.Tensor, dt: float) -> torch.Tensor:
    position = torch.as_tensor(START, device=controls.device)[None].repeat(controls.shape[1], 1)
    velocity = torch.zeros_like(position)
    for action in controls:
        position = position + dt * velocity + 0.5 * dt * dt * action
        velocity = velocity + dt * action
    return position


def lateral_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_abs": float(np.abs(values).mean()),
        "upper_fraction_gt_0.02": float((values > 0.02).mean()),
        "lower_fraction_lt_minus_0.02": float((values < -0.02).mean()),
        "central_fraction": float((np.abs(values) <= 0.02).mean()),
    }


@torch.inference_mode()
def audit_checkpoint(path: Path, demo: dict, device: torch.device, samples: int) -> dict:
    policy, payload = W.HP.load_hp(path, device=device)
    policy.eval()
    env = make_env(300)
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    torch.manual_seed(77123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(77123)
    losses = []
    initial_losses = []
    for offset in range(0, len(demo["U"]), 256):
        sl = slice(offset, min(offset + 256, len(demo["U"])))
        G = demo["grid"][sl].to(device)
        L = demo["low5"][sl].to(device)
        H = demo["hist"][sl].to(device)
        U = demo["U"][sl].to(device)
        loss, _, _ = W._symmetric_cfm_loss_x0(policy, G, L, H, U, 1.0)
        losses.append((len(U), float(loss)))
    initial = torch.where(demo["window_step"] == 0)[0]
    for offset in range(0, len(initial), 64):
        idx = initial[offset:offset + 64]
        G = demo["grid"][idx].to(device)
        L = demo["low5"][idx].to(device)
        H = demo["hist"][idx].to(device)
        U = demo["U"][idx].to(device)
        loss, _, _ = W._symmetric_cfm_loss_x0(policy, G, L, H, U, 1.0)
        initial_losses.append((len(U), float(loss)))
    stage_losses = {}
    for stage in ("initial", "approach", "boundary", "post"):
        stage_index = np.where(np.asarray(demo["sample_stage"], dtype=object) == stage)[0]
        parts = []
        torch.manual_seed(77200)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(77200)
        for offset in range(0, len(stage_index), 256):
            idx = torch.as_tensor(stage_index[offset:offset + 256], dtype=torch.long)
            G = demo["grid"][idx].to(device)
            L = demo["low5"][idx].to(device)
            H = demo["hist"][idx].to(device)
            U = demo["U"][idx].to(device)
            loss, _, _ = W._symmetric_cfm_loss_x0(policy, G, L, H, U, 1.0)
            parts.append((len(U), float(loss)))
        stage_losses[stage] = {
            "rows": int(len(stage_index)),
            "symmetric_loss": float(sum(n * v for n, v in parts) / sum(n for n, _ in parts)),
        }
    result = {
        "checkpoint": str(path.resolve()),
        "iter": int(payload.get("iter", 0)),
        "expert_symmetric_loss": float(sum(n * v for n, v in losses) / sum(n for n, _ in losses)),
        "expert_initial_window_symmetric_loss": float(
            sum(n * v for n, v in initial_losses) / sum(n for n, _ in initial_losses)
        ),
        "expert_stage_symmetric_loss": stage_losses,
        "initial": {},
    }
    state = env.x0.detach().cpu().numpy()
    grid = torch.tensor(W.GF.axis_grid(state[:2], obstacles, rr), device=device)
    hist = torch.zeros(W.GF.K_HIST, 2, device=device)
    for gamma in GAMMAS:
        low = torch.tensor(W.GF.low5(state, GOAL, float(gamma)), device=device)
        torch.manual_seed(88000 + int(round(100 * gamma)))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(88000 + int(round(100 * gamma)))
        controls = policy.sample_window(grid, low, hist, n=samples, temp=0.5, nfe=8)
        endpoint = open_loop_endpoint(controls.transpose(0, 1), float(env.dt))
        lateral = (endpoint[:, 1] - endpoint[:, 0]).float().cpu().numpy()
        result["initial"][str(float(gamma))] = {
            "endpoint_lateral_y_minus_x": lateral_summary(lateral),
            "endpoint_progress_mean": float(
                (np.linalg.norm(START - GOAL) -
                 np.linalg.norm(endpoint.float().cpu().numpy() - GOAL[None], axis=1)).mean()
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = make_config(
        gather_temperature=1.0, evaluation_temperature=0.5,
        iters=1, start_iter=0, probe_only=False,
    )
    # ``state_from_low5`` uses the scene module's active goal.  Initialize the
    # fixed (4.5, 4.5) benchmark before deriving demo stage labels; otherwise
    # the legacy (5, 5) default shifts every reconstructed position by 0.5 m.
    make_env(300)
    demo = W._load_demo(cfg)
    rows = [audit_checkpoint(path.resolve(), demo, device, args.samples) for path in args.checkpoint]
    output = {
        "status": "PASS",
        "temperature": 0.5,
        "nfe": 8,
        "samples_per_gamma": args.samples,
        "demo": str(DEMO_PATH.resolve()),
        "checkpoints": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
