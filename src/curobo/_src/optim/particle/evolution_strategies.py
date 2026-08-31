"""Portable Evolution Strategies compatible with the pinned cuRobo V2 API.

The CUDA implementation owns its distribution in ``ParticleOptCore``.  This
version keeps that state as ordinary, device-resident PyTorch tensors and uses
the same ES ingredients--z-score utilities and a natural-gradient mean
update--on CPU or MPS.  CUDA graph capture, Warp samplers, and packed CUDA
rollout buffers intentionally remain unavailable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Any, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from .mppi import MPPI, MPPICfg, jit_compute_total_cost
from ..components.gaussian_distribution import CovType
from ..components.particle_opt_core import ParticleOptCore, SampleMode
from ..particle.mppi import jit_blend_cov, jit_blend_mean, jit_diag_a_cov_update
from ..particle.particle_opt_utils import SquashType, gaussian_entropy, scale_ctrl
from curobo._src.rollout.metrics import RolloutResult
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import stable_topk
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class _EvolutionStrategiesCfgPortable(MPPICfg):
    """Configuration for portable natural-gradient evolution strategies."""

    solver_type: str = "es"
    solver_name: str = "es"
    learning_rate: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a finite positive real number")


class _EvolutionStrategiesPortable(MPPI):
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

    def _population(
        self, problems: int, horizon: int, action_dim: int, *, device, dtype, iteration: int
    ) -> torch.Tensor:
        """Return an ES population in the shared V2 particle-buffer order.

        ES and MPPI use different distribution updates, not different particle
        layouts.  Routing through the common builder preserves the persistent
        ``fixed_samples`` cursor, one exact-mean candidate, and the configured
        negated/null candidates.  It also keeps batch sampling semantics
        identical for CPU and MPS rather than maintaining an ES-only RNG path.
        """
        return self._particle_population(
            problems, horizon, action_dim, device=device, dtype=dtype, iteration=iteration
        )

    def _sample_distribution_action(self, *, iteration: int) -> torch.Tensor:
        """Draw exactly one projected action per problem without mutating state."""
        assert self._dist is not None
        problems, horizon, action_dim = self._dist.mean.shape
        noise = self._noise(
            problems,
            1,
            horizon,
            action_dim,
            device=self._dist.mean.device,
            dtype=self._dist.mean.dtype,
            iteration=iteration,
        )[:, 0]
        return self._project(self._dist.mean + noise * self._dist.scale_tril)

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
            population = self._population(
                problems, horizon, action_dim, device=seed.device, dtype=seed.dtype, iteration=iteration
            )
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
            self._num_steps += 1

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
            result = self._sample_distribution_action(
                iteration=self.config.num_iters + 123 * self._num_steps
            )
        else:
            result = self._dist.mean
        return result.reshape(seed_action.shape)

    def _get_action_seq(self, mode: Any):
        """Return a distribution action using a V2 ``SampleMode``/string."""
        if self._dist is None:
            return None
        if isinstance(mode, str):
            try:
                mode = SampleMode[mode.upper()]
            except KeyError as exc:
                raise ValueError(f"unidentified ES sample mode: {mode!r}") from exc
        if mode == SampleMode.BEST:
            return self.best_traj
        if mode == SampleMode.SAMPLE:
            # ``sample_actions`` constructs a *full* particle population and,
            # when passed a seed, resets the warm-start distribution.  The V2
            # query contract asks for one action per problem and must be a
            # read-only operation.
            return self._sample_distribution_action(
                iteration=self.config.seed + 123 * self._num_steps
            )
        if mode == SampleMode.MEAN:
            return self.mean_action
        raise ValueError(f"unidentified ES sample mode: {mode!r}")


def calc_exp(total_costs):
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
    w,
    actions,
    mean_action,
    full_inv_cov,
    num_particles: int,
    learning_rate: float,
):
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


class EvolutionStrategiesCfg(_EvolutionStrategiesCfgPortable):
    """Pinned declaration façade for the portable ES configuration."""
    pass


class EvolutionStrategies(_EvolutionStrategiesPortable):
    """Pinned declaration façade for portable Evolution Strategies."""
    def __init__(self, config: EvolutionStrategiesCfg, rollout_list: List[Rollout], use_cuda_graph: bool=False): pass
    def action_bound_highs(self): pass
    def action_bound_lows(self): pass
    def action_dim(self): pass
    def action_horizon(self): pass
    def action_horizon_bounds_highs(self): pass
    def action_horizon_bounds_lows(self): pass
    def action_step_max(self): pass
    def best_traj(self): pass
    def best_traj(self, value): pass
    def compute_metrics(self, action): pass
    def config(self): pass
    def cov_action(self): pass
    def cov_action(self, value): pass
    def debug_dump(self, file_path=''): pass
    def device_cfg(self): pass
    def disable(self): pass
    def enable(self): pass
    def enabled(self): pass
    def full_inv_cov(self): pass
    def full_scale_tril(self): pass
    def gamma_seq(self): pass
    def generate_noise(self, shape, base_seed=None): pass
    def get_all_rollout_instances(self): pass
    def get_recorded_trace(self): pass
    def get_rollouts(self): pass
    def horizon(self): pass
    def initialize_samples(self): pass
    def inv_cov_action(self): pass
    def mean_action(self): pass
    def mean_action(self, value): pass
    def neg_per_problem(self): pass
    def null_act_seqs(self): pass
    def null_per_problem(self): pass
    def opt_dim(self): pass
    def opt_dt(self): pass
    def opt_dt(self, value): pass
    def optimize(self, seed_action): pass
    def outer_iters(self): pass
    def particles_per_problem(self): pass
    def problem_col(self): pass
    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False): pass
    def reset_covariance(self, reset_problem_ids=None): pass
    def reset_cuda_graph(self): pass
    def reset_distribution(self, reset_problem_ids=None): pass
    def reset_mean(self, reset_problem_ids=None): pass
    def reset_seed(self): pass
    def reset_shape(self): pass
    def rollout_fn(self): pass
    def sample_actions(self, init_act): pass
    def sample_lib(self): pass
    def sampled_particles_per_problem(self): pass
    def scale_tril(self): pass
    def scale_tril(self, value): pass
    def shift(self, shift_steps=0): pass
    def solve_time(self): pass
    def solver_names(self): pass
    def top_trajs(self): pass
    def total_num_particles(self): pass
    def update_goal_dt(self, goal): pass
    def update_init_mean(self, init_mean): pass
    def update_niters(self, niters): pass
    def update_num_problems(self, num_problems): pass
    def update_rollout_params(self, goal): pass
    def update_samples(self): pass
    def update_seed(self, init_act): pass
    def update_solver_params(self, solver_params): pass
    def use_cuda_graph(self): pass


def _install_portable_es_runtime():
    for public, portable in ((EvolutionStrategiesCfg, _EvolutionStrategiesCfgPortable), (EvolutionStrategies, _EvolutionStrategiesPortable)):
        for base in reversed(portable.__mro__):
            for name, value in base.__dict__.items():
                if not (name.startswith("__") and name != "__init__"):
                    setattr(public, name, value)


_install_portable_es_runtime()
__all__ = ["EvolutionStrategiesCfg", "EvolutionStrategies", "calc_exp", "compute_es_mean"]
