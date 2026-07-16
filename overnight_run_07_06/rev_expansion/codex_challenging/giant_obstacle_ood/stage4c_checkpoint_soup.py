#!/usr/bin/env python3
"""Screen convex merges of complementary endpoint-free policy checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from giant_obstacle_ood import stage5_window_expand as S5  # noqa: E402
from giant_obstacle_ood.stage5_evaluate import evaluate_checkpoint  # noqa: E402


HP = S5.W.HP
TARGETS = (0.1, 0.5, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(summary: dict) -> dict[str, int]:
    return {
        str(gamma): int(summary["per_gamma"][str(gamma)]["successes"])
        for gamma in TARGETS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--gate-m", type=int, default=6)
    parser.add_argument("--seed", type=int, default=92500)
    parser.add_argument(
        "--persistent-route-bit", action=argparse.BooleanOptionalAction, default=True,
    )
    args = parser.parse_args()
    if args.gate_m <= 0 or any(not 0.0 <= value <= 1.0 for value in args.alphas):
        parser.error("gate M must be positive and alphas must lie in [0,1]")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base, base_payload = HP.load_hp(args.base.resolve(), device="cpu")
    donor, donor_payload = HP.load_hp(args.donor.resolve(), device="cpu")
    if base_payload["config"] != donor_payload["config"]:
        raise RuntimeError("checkpoint configs differ; refusing an invalid merge")
    if base_payload["config"].get("raw_start_goal", False):
        raise RuntimeError("checkpoint merge must retain the endpoint-free architecture")
    base_state = base.state_dict()
    donor_state = donor.state_dict()
    if base_state.keys() != donor_state.keys():
        raise RuntimeError("checkpoint state dictionaries differ")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    best = (-1, -1, -1.0, -1.0)
    for index, alpha in enumerate(args.alphas):
        alpha = float(alpha)
        merged = {}
        for key, left in base_state.items():
            right = donor_state[key]
            if left.shape != right.shape:
                raise RuntimeError(f"state shape differs at {key}")
            merged[key] = (
                (1.0 - alpha) * left + alpha * right
                if left.is_floating_point() else left.clone()
            )
        base.load_state_dict(merged)
        checkpoint = output / f"alpha_{alpha:.4f}.pt"
        HP.save_hp(base, checkpoint, extra={
            "iter": 0,
            "stage4c_checkpoint_soup": {
                "base": str(args.base.resolve()),
                "base_sha256": sha256(args.base.resolve()),
                "donor": str(args.donor.resolve()),
                "donor_sha256": sha256(args.donor.resolve()),
                "alpha_donor": alpha,
                "raw_start_goal": False,
            },
        })
        _, summary = evaluate_checkpoint(
            checkpoint, temperature=0.5, repetitions=args.gate_m, device=device,
            method=f"checkpoint soup alpha={alpha:.4f}", seed0=args.seed,
            persistent_route_bit=args.persistent_route_bit,
        )
        target = counts(summary)
        score = (
            sum(value > 0 for value in target.values()), sum(target.values()),
            float(summary["overall"]["a_SR"]), -float(summary["overall"]["b_CR"]),
        )
        row = {
            "alpha_donor": alpha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "target_successes": target,
            "overall_SR": float(summary["overall"]["a_SR"]),
            "overall_CR": float(summary["overall"]["b_CR"]),
            "score": list(score),
        }
        rows.append(row)
        (output / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            f"SOUP alpha={alpha:.4f} "
            + " ".join(f"g{gamma}={target[str(gamma)]}" for gamma in TARGETS)
            + f" SR={summary['overall']['a_SR']:.3f}",
            flush=True,
        )
        if score > best:
            best = score
            HP.save_hp(base, output / "best.pt", extra={
                "iter": 0, "stage4c_checkpoint_soup": row,
            })
        if score[0] == len(TARGETS):
            manifest = {
                "status": "PASS", "selected": row, "gate_m": args.gate_m,
                "temperature": 0.5, "seed": args.seed,
                "persistent_route_bit": bool(args.persistent_route_bit),
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            return

    (output / "manifest.json").write_text(json.dumps({
        "status": "NO_PASS", "best_score": list(best), "gate_m": args.gate_m,
        "temperature": 0.5, "seed": args.seed,
        "persistent_route_bit": bool(args.persistent_route_bit),
    }, indent=2) + "\n")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
