"""Round-local acquisition-temperature calibration for AFE uncertainty models."""
from __future__ import annotations

import time

import numpy as np
import torch

import afe2_calibration as BC
import afe_core as AC
import afe_rbf_core as RC
import grid_expand_afe2 as AFE2


def round_context_ids(store, round_i: int) -> list[int]:
    """Unique stored control contexts visited in one expansion round."""

    output = [
        context_id for context_id, meta in enumerate(store.ctx_meta)
        if int(meta[0]) == int(round_i)
    ]
    if not output:
        raise ValueError(f"round {round_i} contains no stored acquisition contexts")
    return output


@torch.no_grad()
def feature_pools(policy, store, cfg, device, round_i: int) -> tuple[list[torch.Tensor], dict]:
    """Generate beta-neutral K-pools at every context from one completed round."""

    context_ids = round_context_ids(store, round_i)
    pools: list[torch.Tensor] = []
    gamma_counts: dict[str, int] = {}
    chunk_size = 16
    for offset in range(0, len(context_ids), chunk_size):
        sids = context_ids[offset:offset + chunk_size]
        grid = store.grid3_of(sids).to(device)
        low = torch.stack([
            torch.from_numpy(store.ctx_low5[sid]) for sid in sids
        ]).to(device)
        hist = torch.stack([
            torch.from_numpy(store.ctx_hist[sid].astype(np.float32)) for sid in sids
        ]).to(device)
        context = policy.ctx_from(grid, low, hist)
        repeated = context.repeat_interleave(cfg.K, dim=0)
        with AC.isolated_random_state(AFE2.named_seed(
            cfg.seed, "adaptive_beta_candidates", round_i, offset
        )):
            controls = policy.sample(
                len(sids) * cfg.K,
                repeated,
                nfe=cfg.nfe,
                temp=cfg.temp,
            )
        features = RC.l2_normalize(
            policy.phi_s(controls, repeated, s=cfg.s)
        ).reshape(len(sids), cfg.K, -1)
        for local_index, sid in enumerate(sids):
            pools.append(features[local_index].detach())
            gamma = str(round(float(store.ctx_low5[sid][-1]), 2))
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + 1
    return pools, gamma_counts


@torch.no_grad()
def score_vectors(estimator, pools, cfg, round_i: int) -> list[np.ndarray]:
    """Sequential score vectors under beta-neutral random pending orders."""

    vectors: list[np.ndarray] = []
    for pool_index, features in enumerate(pools):
        rng = np.random.default_rng(AFE2.named_seed(
            cfg.seed, "adaptive_beta_order", round_i, pool_index
        ))
        order = torch.as_tensor(
            rng.permutation(cfg.K), device=features.device, dtype=torch.long
        )
        vectors.extend([
            values.detach().cpu().numpy()
            for values in estimator.sequential_score_vectors(
                features, order, min(cfg.B, cfg.K)
            )
        ])
    if not vectors:
        raise RuntimeError("round-local beta calibration produced no score vectors")
    return vectors


def calibrate_from_pools(estimator, pools, cfg, round_i: int, target: float) -> dict:
    """Solve beta for one estimator on already-generated current-policy pools."""

    started = time.perf_counter()
    vectors = score_vectors(estimator, pools, cfg, round_i)
    solution = BC.solve_beta_ragged(vectors, target=target)
    return {
        "status": "CALIBRATED_AFE_ROUND_LOCAL_ESS_V1",
        "round": int(round_i),
        "target": float(target),
        "beta": float(solution["beta"]),
        "solution": solution,
        "score_vector_sha256": BC.score_vectors_sha256(vectors),
        "score_vector_count": len(vectors),
        "context_count": len(pools),
        "verifier_queries": 0,
        "seconds": float(time.perf_counter() - started),
    }


def rbf_counterfactual_sweep(
    pools,
    buffer_features: torch.Tensor,
    cfg,
    round_i: int,
    target: float,
    *,
    lengthscale: float,
    multipliers=(0.5, 1.0, 2.0),
    caps=(128, 512),
) -> list[dict]:
    """Offline score-scale sweep; never selects or verifies an action."""

    rows = []
    full = RC.l2_normalize(buffer_features.detach())
    for cap in caps:
        count = min(int(cap), int(full.shape[0]))
        if count < 2:
            continue
        indices = torch.linspace(
            0, full.shape[0] - 1, steps=count, device=full.device
        ).round().to(torch.long)
        subset = full[indices]
        for multiplier in multipliers:
            gp = RC.RBFGPSigma(
                lengthscale=float(lengthscale) * float(multiplier),
                lam=cfg.gp_lam,
            )
            gp.set_buffer(subset)
            calibrated = calibrate_from_pools(
                gp, pools, cfg, round_i, target
            )
            rows.append({
                "cap": count,
                "lengthscale_multiplier": float(multiplier),
                "lengthscale": float(lengthscale) * float(multiplier),
                "beta": calibrated["beta"],
                "achieved": calibrated["solution"]["achieved"],
                "sigma_span_med": calibrated["solution"]["sigma_span_med"],
                "flat_pool_fraction": calibrated["solution"]["flat_pool_fraction"],
                "score_vector_sha256": calibrated["score_vector_sha256"],
            })
    return rows
