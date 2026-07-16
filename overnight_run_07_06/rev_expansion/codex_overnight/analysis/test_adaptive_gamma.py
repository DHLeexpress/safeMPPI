#!/usr/bin/env python3
"""Semantic smoke gates for deployment-only adaptive gamma."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

import adaptive_gamma_eval as AGE
import grid_expand_hardtail as HT
import grid_scene as GS


def main():
    ckpt = ROOT.parent.parent / "results/hp_repr/pretrained_a32uni.pt"
    policy, _ = HT.HP.load_hp(ckpt, device="cpu")
    env = GS.make_grid(); HT._apply_wall_plugs(env, 4)
    st = env.x0.numpy(); obs = env.obstacles.numpy()
    grid = HT.GF.axis_grid(st[:2], obs, float(env.r_robot))
    low5 = HT.GF.low5(st, env.goal.numpy(), .5)
    hist = HT.GF.hist_pad(np.zeros((0, 2)))

    torch.manual_seed(123)
    direct = policy.sample_window(torch.tensor(grid), torch.tensor(low5), torch.tensor(hist),
                                  n=1, temp=1.0, nfe=8)
    torch.manual_seed(123)
    latent = torch.randn(policy.d)
    explicit = AGE.windows_from_same_latent(policy, grid, low5, hist, [.5], latent, nfe=8)
    max_diff = float((direct - explicit).abs().max())
    assert torch.equal(direct, explicit), max_diff

    candidates = AGE.windows_from_same_latent(policy, grid, low5, hist, AGE.GAMMAS, latent, nfe=8)
    scored = AGE.verifier_scores(st, candidates.numpy(), AGE.GAMMAS, env)
    assert len(scored) == len(AGE.GAMMAS)
    assert all({"certificate", "face_margin", "progress", "score"} <= set(r) for r in scored)

    before = {k: v.clone() for k, v in policy.state_dict().items()}
    AGE.deploy(policy, env, "heuristic", seed=0, T=2)
    AGE.deploy(policy, env, "verifier", seed=0, T=1)
    assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items()), "deployment changed weights"

    result = dict(pass_count=4, fail_count=0, same_latent_max_abs_diff=max_diff,
                  gamma_candidates=[float(x) for x in AGE.GAMMAS],
                  score_fields=["exact validity", "certificate", "face margin", "progress"],
                  wall_plugs=4, weights_unchanged=True)
    out = HERE / "test_adaptive_gamma.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
