"""Small, explicit RBF-GP acquisition pieces for single-arm Safe Flow Expansion.

The acquisition buffer and the CFM replay store deliberately have different
semantics:

* ``D+`` remains cumulative and contains every full-window verifier positive.
* the exact GP contains at most ``cap`` positives from the immediately preceding
  round, selected without replacement and balanced across gamma values.

The GP is frozen while a round's closed-loop replicas are gathered.  This makes
parallel replicas order-independent and defines sigma as *novelty relative to the
previous round*, not as a calibrated probability of validity.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch


def l2_normalize(values: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def mean_pairwise_lengthscale(features: torch.Tensor) -> float:
    """Mean off-diagonal distance of the supplied normalized pretrained features."""

    values = l2_normalize(features.detach().to(torch.float64))
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("length-scale calibration requires at least two feature rows")
    distances = torch.pdist(values, p=2)
    ell = float(distances.mean())
    if not math.isfinite(ell) or ell <= 0.0:
        raise ValueError("pretrained feature distances do not define a positive RBF length scale")
    return ell


class RBFGPSigma:
    """Exact RBF-GP posterior standard deviation on an explicitly capped buffer."""

    def __init__(self, lengthscale: float, lam: float = 1.0e-2):
        if not math.isfinite(lengthscale) or lengthscale <= 0.0:
            raise ValueError("RBF length scale must be finite and positive")
        if not math.isfinite(lam) or lam <= 0.0:
            raise ValueError("GP noise must be finite and positive")
        self.ell = float(lengthscale)
        self.lam = float(lam)
        self.X: torch.Tensor | None = None
        self.L: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return 0 if self.X is None else int(self.X.shape[0])

    @staticmethod
    def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (
            (a * a).sum(dim=1, keepdim=True)
            + (b * b).sum(dim=1)[None]
            - 2.0 * a @ b.T
        ).clamp_min(0.0)

    def _kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sqdist(a, b) / (2.0 * self.ell * self.ell))

    @torch.no_grad()
    def set_buffer(self, features: torch.Tensor | None) -> None:
        if features is None or int(features.shape[0]) == 0:
            self.X = None
            self.L = None
            return
        self.X = l2_normalize(features.detach())
        kernel = self._kernel(self.X, self.X).to(torch.float64)
        eye = torch.eye(kernel.shape[0], dtype=torch.float64, device=kernel.device)
        jitter = self.lam
        last_error = None
        for _ in range(6):
            try:
                self.L = torch.linalg.cholesky(kernel + jitter * eye)
                return
            except RuntimeError as error:
                last_error = error
                jitter *= 10.0
        raise RuntimeError("RBF-GP Cholesky failed after jitter retries") from last_error

    @torch.no_grad()
    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        query = l2_normalize(features.detach())
        if self.X is None:
            return torch.ones(query.shape[0], dtype=query.dtype, device=query.device)
        cross = self._kernel(query, self.X)
        solved = torch.cholesky_solve(cross.T.to(torch.float64), self.L)
        reduction = (cross * solved.T.to(cross.dtype)).sum(dim=1)
        return (1.0 - reduction).clamp_min(0.0).sqrt()

    @torch.no_grad()
    def posterior_covariance(
        self,
        features: torch.Tensor,
        *,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        """Joint GP posterior covariance for one candidate batch."""

        query = l2_normalize(features.detach())
        covariance = self._kernel(query, query)
        if self.X is not None:
            cross = self._kernel(query, self.X)
            solved = torch.cholesky_solve(cross.T.to(torch.float64), self.L)
            covariance = covariance - cross @ solved.to(cross.dtype)
        covariance = 0.5 * (covariance + covariance.T)
        if include_observation_noise:
            covariance = covariance + self.lam * torch.eye(
                covariance.shape[0], dtype=covariance.dtype, device=covariance.device
            )
        return covariance

    @torch.no_grad()
    def conditional_variance(self, features: torch.Tensor, jitter: float = 1.0e-6) -> torch.Tensor:
        """Var(f_i | f_{-i}, GP buffer), matching the peptide implementation.

        For joint posterior covariance ``C``, the Schur-complement identity is
        ``Var(f_i | f_{-i}) = 1 / [C^{-1}]_ii``.  Conditioning on the rest of
        the K-pool makes near-duplicate candidates suppress one another even
        when their marginal variances are similar.
        """

        covariance = self.posterior_covariance(features)
        eye = torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        )
        covariance = covariance + float(jitter) * eye
        factor = torch.linalg.cholesky(covariance.to(torch.float64))
        inverse_factor = torch.linalg.solve_triangular(
            factor,
            eye.to(torch.float64),
            upper=False,
        )
        inverse_diagonal = inverse_factor.square().sum(dim=0).clamp_min(1.0e-12)
        conditional = (1.0 / inverse_diagonal).to(features.dtype)
        # Same prior-variance normalization used by the reference peptide code.
        return (conditional / (1.0 + self.lam)).clamp(0.0, 1.0)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, object]:
        if self.X is None:
            return {
                "n": 0,
                "kernel_effective_rank": 0.0,
                "kernel_eigenvalues": [],
                "buffer_sigma_mean": None,
            }
        kernel = self._kernel(self.X, self.X).to(torch.float64)
        eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0.0)
        effective_rank = float(
            eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1.0e-12)
        )
        return {
            "n": self.n,
            "kernel_effective_rank": effective_rank,
            "kernel_eigenvalues": [float(value) for value in eigenvalues.cpu()],
            "buffer_sigma_mean": float(self.sigma(self.X).mean()),
        }


def previous_round_positive_ids(store, round_i: int, cap: int, gammas, seed: int) -> list[int]:
    """Gamma-balanced, without-replacement compression of round ``round_i`` positives."""

    if cap <= 0:
        raise ValueError("GP buffer cap must be positive")
    groups: dict[float, list[int]] = defaultdict(list)
    for query_id in store.pos_ids:
        if int(store.q_round[query_id]) == int(round_i):
            groups[round(float(store.q_gamma[query_id]), 8)].append(int(query_id))
    all_ids = [query_id for values in groups.values() for query_id in values]
    if len(all_ids) <= cap:
        return sorted(all_ids)

    rng = np.random.default_rng(int(seed))
    gamma_keys = [round(float(gamma), 8) for gamma in gammas]
    quota, extra = divmod(cap, len(gamma_keys))
    selected: list[int] = []
    selected_set: set[int] = set()
    for index, gamma in enumerate(gamma_keys):
        candidates = np.asarray(groups.get(gamma, []), dtype=np.int64)
        take = min(len(candidates), quota + int(index < extra))
        if take:
            chosen = rng.choice(candidates, size=take, replace=False).tolist()
            selected.extend(int(value) for value in chosen)
            selected_set.update(int(value) for value in chosen)
    if len(selected) < cap:
        remaining = np.asarray(
            [query_id for query_id in all_ids if query_id not in selected_set],
            dtype=np.int64,
        )
        take = min(cap - len(selected), len(remaining))
        if take:
            selected.extend(int(value) for value in rng.choice(remaining, size=take, replace=False))
    if len(selected) != cap or len(set(selected)) != cap:
        raise RuntimeError("failed to construct the declared without-replacement GP buffer")
    return sorted(selected)


_VERIFY_ENV = None
_VERIFY_GOAL = None
_VERIFY_REACH = None
_VERIFY_N_THETA = None


def initialize_verifier_worker(scene_profile: str, reach: float, n_theta: int) -> None:
    """Process-pool initializer; each CPU worker builds one immutable scene."""

    global _VERIFY_ENV, _VERIFY_GOAL, _VERIFY_REACH, _VERIFY_N_THETA
    from afe2_scene_profiles import build_scene, get_scene_profile
    import grid_metrics2 as metrics2

    torch.set_num_threads(1)
    profile = get_scene_profile(scene_profile)
    _VERIFY_ENV = build_scene(profile)
    _VERIFY_GOAL = _VERIFY_ENV.goal.detach().cpu().numpy()
    _VERIFY_REACH = float(reach)
    _VERIFY_N_THETA = int(n_theta)
    metrics2.GOAL_XY = np.asarray(profile.goal, dtype=float)


def verify_in_worker(task):
    """Return the task identity plus one terminal-aware deterministic verifier result."""

    if _VERIFY_ENV is None:
        raise RuntimeError("verifier worker was not initialized")
    from afe_core import verify_plan_with_terminal

    episode_id, candidate_id, state, controls, gamma = task
    result = verify_plan_with_terminal(
        state,
        controls,
        _VERIFY_ENV,
        float(gamma),
        _VERIFY_GOAL,
        reach=_VERIFY_REACH,
        n_theta=_VERIFY_N_THETA,
    )
    return int(episode_id), int(candidate_id), result
