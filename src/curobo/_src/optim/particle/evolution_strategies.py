"""Portable Evolution Strategies compatible with the pinned cuRobo V2 API.

The CUDA implementation owns its distribution in ``ParticleOptCore``.  This
version keeps that state as ordinary, device-resident PyTorch tensors and uses
the same ES ingredients--z-score utilities and a natural-gradient mean
update--on CPU or MPS.  CUDA graph capture, Warp samplers, and packed CUDA
rollout buffers intentionally remain unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .mppi import MPPI, MPPICfg, jit_compute_total_cost
from ..components.gaussian_distribution import CovType
from ..components.particle_opt_core import SampleMode


@dataclass
class EvolutionStrategiesCfg(MPPICfg):
    """Configuration for portable natural-gradient evolution strategies."""

    solver_type: str = "es"
    solver_name: str = "es"
    learning_rate: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


class EvolutionStrategies(MPPI):
    """Deterministic ES with persistent Gaussian distribution state.

    The public lifecycle mirrors the V2 optimizer (warm starts, shifts,
    resizing, recorded rollouts, and sample modes) while avoiding raw CUDA
    graph and Warp dependencies.  Unlike the prior compatibility bridge this
    class performs a true ES distribution update instead of routing to CEM.
    """

    # ``MPPI.optimize`` deliberately routes non-MPPI strategies through a
    # generic particle helper.  ES supplies its own loop below.
    strategy = "es"

    def __init__(self, config: EvolutionStrategiesCfg, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        # V2 callers frequently inspect distribution buffers immediately after
        # construction.  Initialise the portable buffers from rollout metadata;
        # a concrete seed later re-shapes/re-dtypes them authoritatively.
        init = config.init_mean
        if init is None:
            init = torch.zeros(
                (config.num_problems, self.action_horizon, self.action_dim),
                device=config.device_cfg.device,
                dtype=config.device_cfg.dtype,
            )
        else:
            init = torch.as_tensor(init, device=config.device_cfg.device, dtype=config.device_cfg.dtype)
            if init.ndim == 2:
                init = init.unsqueeze(0)
        self._ensure_distribution(init, reset_mean=True)
        self._last_utilities: torch.Tensor | None = None

    # -- ES update -------------------------------------------------------------

    def _compute_total_cost(self, costs: torch.Tensor) -> torch.Tensor:
        """Discount a particle cost sequence, tolerating arbitrary horizons."""
        gamma = self.gamma_seq.to(device=costs.device, dtype=costs.dtype)
        if gamma.numel() != costs.shape[-1]:
            exponent = torch.arange(costs.shape[-1], device=costs.device, dtype=costs.dtype)
            gamma = torch.pow(torch.as_tensor(self.config.gamma, device=costs.device, dtype=costs.dtype), exponent)
        return jit_compute_total_cost(gamma, costs)

    def _exp_util(self, total_costs: torch.Tensor) -> torch.Tensor:
        return calc_exp(total_costs)

    def _exp_util_from_costs(self, costs: torch.Tensor) -> torch.Tensor:
        return self._exp_util(self._compute_total_cost(costs))

    def _compute_mean(self, utilities: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply the V2 ES natural-gradient mean step."""
        if self.config.cov_type not in (CovType.SIGMA_I, CovType.DIAG_A):
            raise NotImplementedError(
                f"portable ES supports SIGMA_I and DIAG_A covariance, not {self.config.cov_type!r}"
            )
        assert self._dist is not None
        return compute_es_mean(
            utilities,
            actions,
            self._dist.mean,
            self.full_inv_cov,
            self.config.num_particles,
            self.config.learning_rate,
        )

    def _compute_covariance(self, utilities: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute a finite diagonal covariance update from ES utilities.

        Signed z-score utilities are correct for a mean natural gradient but
        are not themselves a probability distribution.  Their magnitudes give
        a stable, deterministic importance weight for this portable covariance
        estimate, preventing a negative variance on both CPU and MPS.
        """
        assert self._dist is not None
        weights = utilities.abs()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(actions.dtype).eps)
        delta = actions - self._dist.mean.unsqueeze(1)
        covariance = (weights[..., None, None] * delta.square()).sum(dim=1).mean(dim=-2, keepdim=True)
        if self.config.cov_type == CovType.SIGMA_I:
            covariance = covariance.mean(dim=-1, keepdim=True).expand_as(self._dist.cov)
        current = self._dist.cov
        blended = (1.0 - float(self.config.step_size_cov)) * current + float(self.config.step_size_cov) * covariance
        return (blended + float(self.config.kappa)).clamp_min(torch.finfo(actions.dtype).eps)

    def _update_distribution(self, costs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Update mean and (optionally) covariance, returning ES utilities."""
        assert self._dist is not None
        utilities = self._exp_util_from_costs(costs)
        mean = self._compute_mean(utilities, actions)
        self._dist.mean = (
            (1.0 - float(self.config.step_size_mean)) * self._dist.mean
            + float(self.config.step_size_mean) * mean
        )
        if self.config.update_cov:
            self.cov_action = self._compute_covariance(utilities, actions)
        self._last_utilities = utilities.detach().clone()
        return utilities

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run ES on a batched action seed and return the selected action."""
        if not self.enabled:
            return seed_action
        seed = self._ensure_distribution(torch.as_tensor(seed_action), reset_mean=True)
        assert self._dist is not None
        problems, horizon, action_dim = self._dist.mean.shape
        particles = self.config.num_particles
        best_cost = torch.full((problems,), torch.inf, device=seed.device, dtype=seed.dtype)
        best = self._dist.mean.detach().clone()
        objective_history: list[torch.Tensor] = []
        mean_history: list[torch.Tensor] = []
        cov_history: list[torch.Tensor] = []

        for iteration in range(self.config.num_iters):
            noise = self._noise(
                problems, particles, horizon, action_dim,
                device=seed.device, dtype=seed.dtype, iteration=iteration,
            )
            population = self._project(self._dist.mean[:, None] + noise * self._dist.scale_tril[:, None])
            # Keep one exact mean trajectory, matching the useful V2 safety
            # invariant that ES can never sample *only* perturbations.
            population[:, 0] = self._dist.mean
            if self.null_per_problem:
                population[:, : self.null_per_problem] = 0.0
            costs = self._evaluate_costs(population)
            total = self._compute_total_cost(costs)
            finite_total = torch.where(torch.isfinite(total), total, torch.full_like(total, torch.inf))
            index = finite_total.argmin(dim=-1)
            candidate = population[torch.arange(problems, device=seed.device), index]
            candidate_cost = finite_total[torch.arange(problems, device=seed.device), index]
            improves = candidate_cost < best_cost
            best = torch.where(improves[:, None, None], candidate, best)
            best_cost = torch.where(improves, candidate_cost, best_cost)
            self._update_distribution(costs, population)

            if self.config.store_rollouts:
                count = min(20, particles)
                top_indices = finite_total.topk(count, largest=False).indices
                self._top_trajs = torch.gather(
                    population, 1, top_indices[:, :, None, None].expand(-1, -1, horizon, action_dim)
                ).detach().clone()
            self._last_rollouts = population.detach().clone()
            self._last_costs = costs.detach().clone()
            if self.config.store_debug:
                objective_history.append(best_cost.detach().clone())
                mean_history.append(self._dist.mean.detach().clone())
                cov_history.append(self._dist.cov.detach().clone())

        self._best_action = best.detach().clone()
        self._seed_action = seed.detach().clone()
        self.debug = None if not self.config.store_debug else {
            "objective": tuple(objective_history),
            "mean": tuple(mean_history),
            "covariance": tuple(cov_history),
            "utilities": None if self._last_utilities is None else self._last_utilities.detach().clone(),
        }
        if self.config.sample_mode == SampleMode.BEST:
            result = best
        elif self.config.sample_mode == SampleMode.SAMPLE:
            result = self._project(
                self._dist.mean
                + self._noise(problems, 1, horizon, action_dim, device=seed.device, dtype=seed.dtype,
                              iteration=self.config.num_iters)[:, 0] * self._dist.scale_tril
            )
        else:
            result = self._dist.mean
        return result.reshape(seed_action.shape)

    def _get_action_seq(self, mode: Any):
        """Return a distribution action using a V2 ``SampleMode``/string."""
        if self._dist is None:
            return None
        if isinstance(mode, str):
            mode = SampleMode[mode.upper()]
        if mode == SampleMode.BEST:
            return self.best_traj
        if mode == SampleMode.SAMPLE:
            return self.sample_actions(self._dist.mean)[: self.config.num_problems]
        return self.mean_action


def calc_exp(total_costs: torch.Tensor) -> torch.Tensor:
    """Return per-problem ES z-score utilities for lower-is-better costs.

    Non-finite populations and singleton populations deterministically produce
    zero utility rather than leaking NaNs into portable CPU/MPS optimizers.
    """
    if total_costs.ndim < 1:
        raise ValueError("total_costs must have a particle dimension")
    costs = torch.as_tensor(total_costs)
    finite = torch.where(torch.isfinite(costs), costs, torch.nan)
    valid = torch.isfinite(finite)
    count = valid.sum(dim=-1, keepdim=True)
    safe = torch.where(valid, -finite, torch.zeros_like(finite))
    mean = safe.sum(dim=-1, keepdim=True) / count.clamp_min(1)
    centred = torch.where(valid, safe - mean, torch.zeros_like(safe))
    # Upstream uses unbiased std; preserve it when there are at least two
    # finite particles while handling portable degenerate cases explicitly.
    denom = (count - 1).clamp_min(1)
    std = torch.sqrt(centred.square().sum(dim=-1, keepdim=True) / denom)
    valid_std = (count > 1) & (std > torch.finfo(costs.dtype).eps)
    return torch.where(valid_std, centred / std.clamp_min(torch.finfo(costs.dtype).eps), torch.zeros_like(costs))


def compute_es_mean(
    w: torch.Tensor,
    actions: torch.Tensor,
    mean_action: torch.Tensor,
    full_inv_cov: torch.Tensor | None,
    num_particles: int,
    learning_rate: float,
) -> torch.Tensor:
    """Compute the ES natural-gradient mean update on ``[P,N,H,D]`` actions."""
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if actions.ndim != 4 or mean_action.ndim != 3:
        raise ValueError("actions must be [problems, particles, horizon, action_dim] and mean_action [problems, horizon, action_dim]")
    if actions.shape[0] != mean_action.shape[0] or actions.shape[-2:] != mean_action.shape[-2:]:
        raise ValueError("actions and mean_action must share problem/horizon/action dimensions")
    utilities = torch.as_tensor(w, device=actions.device, dtype=actions.dtype)
    if utilities.shape == actions.shape[:2]:
        utilities = utilities[..., None, None]
    elif utilities.shape != (*actions.shape[:2], 1, 1):
        raise ValueError("w must be [problems, particles] or [problems, particles, 1, 1]")
    std_w = utilities.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(torch.finfo(actions.dtype).eps)
    centred_actions = (actions - mean_action.unsqueeze(1)) / std_w
    weighted = (utilities * centred_actions).sum(dim=1)
    if full_inv_cov is None:
        inv_diag = torch.ones_like(mean_action[:, :1, :])
    else:
        inv_cov = torch.as_tensor(full_inv_cov, device=actions.device, dtype=actions.dtype)
        if inv_cov.shape != (actions.shape[0], actions.shape[-1], actions.shape[-1]):
            raise ValueError("full_inv_cov must be [problems, action_dim, action_dim]")
        inv_diag = torch.diagonal(inv_cov, dim1=-2, dim2=-1).unsqueeze(-2)
    return mean_action + float(learning_rate) * weighted * inv_diag / float(num_particles)


__all__ = ["EvolutionStrategiesCfg", "EvolutionStrategies", "calc_exp", "compute_es_mean"]
