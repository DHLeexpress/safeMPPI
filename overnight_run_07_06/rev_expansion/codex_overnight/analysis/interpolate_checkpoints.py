"""Create model-only checkpoints on the affine segment between two policy checkpoints.

The outputs are deliberately non-resumable: optimizer/query/RNG state is not meaningful
after weight interpolation.  This is used only for a read-only deployment gate/line search.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cand = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if base["state_dict"].keys() != cand["state_dict"].keys():
        raise RuntimeError("state_dict keys differ")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha outside [0,1]: {alpha}")
        state = {}
        for key, b in base["state_dict"].items():
            c = cand["state_dict"][key]
            state[key] = b + (c - b) * alpha if torch.is_floating_point(b) else b.clone()
        out = {
            "state_dict": state,
            "config": base["config"],
            "iter": float(base.get("iter", 0)) + alpha * (
                float(cand.get("iter", 0)) - float(base.get("iter", 0))
            ),
            "srcr": {},
            "recipe": {
                "algorithm": "model_weight_line_search",
                "base": str(Path(args.base)),
                "candidate": str(Path(args.candidate)),
                "alpha": alpha,
                "deployment_only": True,
            },
            "train_state": None,
            "resumable": False,
        }
        name = f"alpha_{str(alpha).replace('.', 'p')}.pt"
        torch.save(out, outdir / name)
        print(outdir / name)


if __name__ == "__main__":
    main()
