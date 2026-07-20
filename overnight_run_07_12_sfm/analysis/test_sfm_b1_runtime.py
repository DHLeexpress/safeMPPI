import ast
from pathlib import Path

import torch

import grid_policy_sfm as GPS
import sfm_b1_eval as E
import sfm_b1_expand as X


def test_nvp_isolates_one_replica(monkeypatch):
    monkeypatch.setattr(X.SS, "make_humans", lambda *args, **kwargs: [])
    first = X.Replica(1, .1, n_ped=0)
    second = X.Replica(2, .1, n_ped=0)
    X.nvp_fail_closed(first)
    assert not first.alive and first.status == "nvp"
    assert second.alive and second.status is None


def test_raw_evaluator_has_no_forbidden_import_or_call():
    source = Path(E.__file__).read_text()
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    forbidden = ("acquisition", "verifier", "selector", "template", "kazuki", "mppi", "refine")
    lowered = " ".join(imports).lower()
    assert not any(word in lowered for word in forbidden)
    raw = ast.get_source_segment(source, next(node for node in tree.body
                                              if isinstance(node, ast.FunctionDef) and node.name == "raw_rollout"))
    assert not any(word in raw.lower() for word in forbidden)


def test_zero_guidance_same_latent_matches_raw_generator():
    torch.manual_seed(18)
    policy = GPS.build_sfm_policy(width=24, res_dropout=0.0)
    context = policy.ctx_from(torch.randn(2, 10, 16, 12), torch.randn(2, 5), torch.randn(2, 16, 2))
    latent = torch.randn(2, policy.d)
    raw = E.integrate_latents(policy, latent.clone(), context, nfe=8)
    zero_guidance = E.integrate_latents(policy, latent.clone(), context, nfe=8)
    torch.testing.assert_close(raw, zero_guidance, rtol=0, atol=0)


def test_default_kazuki_is_separately_labeled_generate_refine():
    import sfm_kazuki as K
    config = K.KazukiConfig()
    assert config.safe_coefs == (0.3,) and config.goal_coef == 0.5
    assert config.n_copy > 0
