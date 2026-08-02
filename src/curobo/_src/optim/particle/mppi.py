"""Portable, deterministic MPPI compatible with the pinned cuRobo V2 surface.

The upstream implementation delegates sampling and distribution updates to CUDA
``ParticleOptCore`` kernels.  This module keeps the public lifecycle and the
MPPI update rule while running ordinary PyTorch tensors on CPU or MPS.  In
particular, the distribution is real persistent optimizer state; it is not a
one-shot call to a generic particle optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Optional

import torch

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer, _objective
from curobo._src.optim.components.gaussian_distribution import CovType, GaussianDistribution
from curobo._src.optim.components.particle_opt_core import SampleMode
from curobo._src.optim.particle.particle_opt_utils import SquashType, gaussian_entropy
from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg


class BaseActionType(Enum):
    """Value used to fill the tail exposed by :meth:`shift`."""

    REPEAT = "REPEAT"
    NULL = "NULL"
    RANDOM = "RANDOM"


@dataclass
class MPPICfg(PortableOptCfg):
    solver_type: str = "mppi"
    solver_name: str = "mppi"
    gamma: float = 1.0
    sample_mode: Any = SampleMode.MEAN
    seed: int = 0
    store_rollouts: bool = False
    null_act_frac: float = 0.0
    init_mean: Optional[torch.Tensor] = None
    init_cov: Any = 0.5
    base_action: BaseActionType = BaseActionType.REPEAT
    step_size_mean: float = 0.9
    step_size_cov: float = 0.1
    squash_fn: Any = SquashType.CLAMP
    cov_type: Any = CovType.DIAG_A
    sample_params: Any = None
    update_cov: bool = True
    random_mean: bool = False
    beta: float = 0.1
    alpha: float = 1.0
    kappa: float = 0.01
    sample_per_problem: bool = True

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if not 0.0 <= self.null_act_frac <= 1.0:
            raise ValueError("null_act_frac must lie in [0, 1]")
        if self.beta <= 0.0 or self.kappa < 0.0:
            raise ValueError("beta must be positive and kappa must be nonnegative")
        if isinstance(self.base_action, str):
            self.base_action = BaseActionType[self.base_action.upper()]
        if isinstance(self.cov_type, str):
            self.cov_type = CovType[self.cov_type.upper()]
        if isinstance(self.squash_fn, str):
            self.squash_fn = SquashType[self.squash_fn.upper()]
        if isinstance(self.sample_mode, str):
            self.sample_mode = SampleMode[self.sample_mode.upper()]
        if self.sample_params is None:
            self.sample_params = ParticleSamplerCfg(self.device_cfg, seed=self.seed)
        elif isinstance(self.sample_params, dict):
            self.sample_params = ParticleSamplerCfg(
                **self.sample_params, device_cfg=self.device_cfg
            )


class MPPI(PortableOptimizer):
    """Stateful Model Predictive Path Integral optimisation on CPU/MPS.

    Raw CUDA graph capture, Warp samplers, and the packed CUDA rollout ABI are
    deliberately not emulated.  The public distribution, sampled-rollout, and
    warm-start lifecycle instead run on device-resident PyTorch tensors.
    """

    strategy = "mppi"

    def __init__(self, config: MPPICfg, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._dist: GaussianDistribution | None = None
        self._distribution_shape: tuple[int, int] | None = None
        self._best_action: torch.Tensor | None = None
        self._top_trajs: torch.Tensor | None = None
        self._sample_set: torch.Tensor | None = None
        self._sample_iter: torch.Tensor | None = None
        self._seed_action: torch.Tensor | None = None
        self._last_rollouts: torch.Tensor | None = None
        self._last_costs: torch.Tensor | None = None

    # -- Persistent distribution -------------------------------------------------

    def _infer_layout(self, action: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        if action.ndim < 2:
            raise ValueError("MPPI seed_action must have trailing [horizon, action_dim] dimensions")
        horizon, action_dim = action.shape[-2:]
        problems = math.prod(action.shape[:-2]) if action.ndim > 2 else 1
        shaped = action.reshape(problems, horizon, action_dim)
        return shaped, problems, horizon, action_dim

    def _covariance(self, problems: int, action_dim: int, *, device, dtype) -> torch.Tensor:
        value = torch.as_tensor(self.config.init_cov, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(problems, 1, action_dim)
        elif value.ndim == 1:
            if value.numel() != action_dim:
                raise ValueError("init_cov vector must have action_dim elements")
            value = value.reshape(1, 1, action_dim).expand(problems, -1, -1)
        elif value.ndim == 2:
            if value.shape == (problems, action_dim):
                value = value.unsqueeze(-2)
            elif value.shape == (1, action_dim):
                value = value.unsqueeze(-2).expand(problems, -1, -1)
            else:
                raise ValueError("init_cov rank-2 tensor must be [problems, action_dim]")
        elif value.ndim == 3 and value.shape[-2:] == (1, action_dim):
            value = value.expand(problems, -1, -1)
        else:
            raise ValueError("init_cov must be scalar, [D], [P,D], or [P,1,D]")
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("init_cov must be finite and nonnegative")
        return value.clamp_min(torch.finfo(dtype).eps)

    def _ensure_distribution(self, action: torch.Tensor, *, reset_mean: bool = False) -> torch.Tensor:
        seed, problems, horizon, action_dim = self._infer_layout(action)
        layout = (horizon, action_dim)
        recreate = self._dist is None or self._distribution_shape != layout
        if recreate:
            # GaussianDistribution provides public sampler/covariance helpers.
            # We overwrite its buffers below so a concrete input tensor remains
            # authoritative for dtype/device, including MPS call sites.
            init_mean = torch.zeros((1, horizon, action_dim), device=seed.device, dtype=seed.dtype)
            self._dist = GaussianDistribution(
                self.config.device_cfg, horizon, action_dim, self.config.cov_type,
                init_mean, self.config.init_cov, self.config.sample_params,
                self.config.random_mean, self.config.seed,
            )
            self._distribution_shape = layout
        assert self._dist is not None
        current = self._dist.mean
        needs_reset = (
            recreate or reset_mean or current is None or current.shape != seed.shape
            or current.device != seed.device or current.dtype != seed.dtype
        )
        if needs_reset:
            configured = self.config.init_mean
            if configured is None:
                mean = seed.detach().clone()
            else:
                initial = torch.as_tensor(configured, device=seed.device, dtype=seed.dtype)
                if initial.ndim == 2:
                    initial = initial.unsqueeze(0)
                if initial.shape[-2:] != layout:
                    raise ValueError("init_mean must have [horizon, action_dim] trailing dimensions")
                mean = initial.expand(problems, -1, -1).clone()
            cov = self._covariance(problems, action_dim, device=seed.device, dtype=seed.dtype)
            self._dist.mean = mean
            self._dist.cov = cov
            self._dist.scale_tril = torch.sqrt(cov)
            self._dist.inv_cov = cov.reciprocal()
            self._best_action = mean.detach().clone()
        self.config.num_problems = problems
        return seed

    @property
    def mean_action(self):
        return None if self._dist is None else self._dist.mean

    @mean_action.setter
    def mean_action(self, value):
        seed = self._ensure_distribution(torch.as_tensor(value), reset_mean=False)
        assert self._dist is not None
        self._dist.mean = seed.detach().clone()

    @property
    def best_traj(self):
        return self._best_action

    @best_traj.setter
    def best_traj(self, value):
        self._best_action = None if value is None else torch.as_tensor(value).detach().clone()

    @property
    def cov_action(self):
        return None if self._dist is None else self._dist.cov

    @cov_action.setter
    def cov_action(self, value):
        if self._dist is None:
            raise RuntimeError("MPPI distribution is initialized by optimize or sample_actions")
        cov = torch.as_tensor(value, device=self._dist.mean.device, dtype=self._dist.mean.dtype)
        if cov.ndim == 2:
            cov = cov.unsqueeze(-2)
        if cov.shape != self._dist.cov.shape:
            raise ValueError("cov_action must match the current MPPI distribution")
        self._dist.cov = cov.clamp_min(torch.finfo(cov.dtype).eps)
        self._dist.scale_tril = self._dist.cov.sqrt()
        self._dist.inv_cov = self._dist.cov.reciprocal()

    @property
    def scale_tril(self):
        return None if self._dist is None else self._dist.scale_tril

    @scale_tril.setter
    def scale_tril(self, value):
        if self._dist is None:
            raise RuntimeError("MPPI distribution is initialized by optimize or sample_actions")
        scale = torch.as_tensor(value, device=self._dist.mean.device, dtype=self._dist.mean.dtype)
        if scale.ndim == 2:
            scale = scale.unsqueeze(-2)
        self._dist.scale_tril = scale
        self._dist.cov = scale.square().clamp_min(torch.finfo(scale.dtype).eps)
        self._dist.inv_cov = self._dist.cov.reciprocal()

    @property
    def inv_cov_action(self):
        return None if self._dist is None else self._dist.inv_cov

    @property
    def full_scale_tril(self):
        if self._dist is None:
            return None
        return torch.diag_embed(self._dist.scale_tril.squeeze(-2))

    @property
    def full_inv_cov(self):
        if self._dist is None:
            return None
        return torch.diag_embed(self._dist.inv_cov.squeeze(-2))

    @property
    def entropy(self):
        if self._dist is None:
            return None
        return gaussian_entropy(L=self.full_scale_tril)

    @property
    def squashed_mean(self):
        if self.mean_action is None:
            return None
        return self._project(self.mean_action)

    @property
    def particles_per_problem(self):
        return self.config.num_particles

    @property
    def sampled_particles_per_problem(self):
        return self.config.num_particles

    @property
    def null_per_problem(self):
        return int(round(self.config.num_particles * self.config.null_act_frac))

    @property
    def neg_per_problem(self):
        return 0

    @property
    def total_num_particles(self):
        return self.config.num_problems * self.config.num_particles

    @property
    def problem_col(self):
        return torch.arange(self.config.num_problems, device=self.device_cfg.device)

    @property
    def gamma_seq(self):
        if self._distribution_shape is None:
            horizon = self.action_horizon
        else:
            horizon = self._distribution_shape[0]
        exponent = torch.arange(horizon, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        return torch.pow(torch.as_tensor(self.config.gamma, device=exponent.device, dtype=exponent.dtype), exponent)

    @property
    def top_trajs(self):
        return self._top_trajs

    @property
    def sample_lib(self):
        return None if self._dist is None else self._dist.sample_lib

    @property
    def horizon(self):
        return self.action_horizon

    @property
    def solver_names(self):
        return [self.config.solver_name]

    # -- Sampling, evaluation, and update ---------------------------------------

    def _noise(self, problems: int, particles: int, horizon: int, action_dim: int, *, device, dtype, iteration: int):
        # CPU-seeded RNG gives bit-for-bit reproducible population generation
        # across CPU/MPS for a particular PyTorch release.  Moving the finished
        # noise tensor onto MPS avoids a CPU fallback operator.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed) + int(iteration))
        return torch.randn((problems, particles, horizon, action_dim), generator=generator, dtype=dtype, device="cpu").to(device)

    def _project(self, actions: torch.Tensor) -> torch.Tensor:
        low, high = self.action_bound_lows, self.action_bound_highs
        if low is None or high is None:
            return actions
        low = torch.as_tensor(low, device=actions.device, dtype=actions.dtype)
        high = torch.as_tensor(high, device=actions.device, dtype=actions.dtype)
        return torch.maximum(torch.minimum(actions, high), low)

    def _evaluate_costs(self, population: torch.Tensor) -> torch.Tensor:
        objective = _objective(self.rollout_fn)
        flat = population.flatten(0, 1)
        result = objective(flat)
        if hasattr(result, "costs_and_constraints"):
            result = result.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=False)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result, device=flat.device, dtype=flat.dtype)
        problems, particles = population.shape[:2]
        if result.numel() == problems * particles:
            return result.reshape(problems, particles, 1)
        if result.ndim >= 1 and result.shape[0] == problems * particles:
            return result.reshape(problems, particles, -1)
        if result.shape[:2] == (problems, particles):
            return result.reshape(problems, particles, -1)
        raise ValueError("MPPI rollout objective must return one scalar or cost sequence per action")

    def _distribution_update(self, costs: torch.Tensor, actions: torch.Tensor):
        assert self._dist is not None
        gamma = self.gamma_seq.to(device=costs.device, dtype=costs.dtype)
        if gamma.numel() != costs.shape[-1]:
            gamma = torch.pow(torch.as_tensor(self.config.gamma, device=costs.device, dtype=costs.dtype), torch.arange(costs.shape[-1], device=costs.device, dtype=costs.dtype))
        mean, covariance, scale = jit_mean_cov_diag_a(
            costs, actions, gamma, self._dist.mean, self._dist.cov,
            self.config.step_size_mean, self.config.step_size_cov,
            self.config.kappa, self.config.beta,
        )
        self._dist.mean = mean
        if self.config.update_cov:
            self._dist.cov, self._dist.scale_tril = covariance, scale
            self._dist.inv_cov = covariance.reciprocal()

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        # EvolutionStrategies intentionally retains its existing CEM bridge.
        if self.strategy != "mppi":
            return super().optimize(seed_action)
        if not self.enabled:
            return seed_action
        seed = self._ensure_distribution(seed_action, reset_mean=True)
        assert self._dist is not None
        problems, horizon, action_dim = self._dist.mean.shape
        particles = self.config.num_particles
        objective_history: list[torch.Tensor] = []
        mean_history: list[torch.Tensor] = []
        cov_history: list[torch.Tensor] = []
        best_cost = torch.full((problems,), torch.inf, device=seed.device, dtype=seed.dtype)
        best = self._dist.mean.detach().clone()
        for iteration in range(self.config.num_iters):
            noise = self._noise(problems, particles, horizon, action_dim, device=seed.device, dtype=seed.dtype, iteration=iteration)
            population = self._dist.mean[:, None] + noise * self._dist.scale_tril[:, None]
            population[:, 0] = self._dist.mean
            if self.null_per_problem:
                population[:, :self.null_per_problem] = 0.0
            population = self._project(population)
            costs = self._evaluate_costs(population)
            total = jit_compute_total_cost(self.gamma_seq.to(costs), costs)
            safe_total = torch.where(torch.isfinite(total), total, torch.full_like(total, torch.inf))
            index = safe_total.argmin(dim=-1)
            chosen = population[torch.arange(problems, device=seed.device), index]
            chosen_cost = safe_total[torch.arange(problems, device=seed.device), index]
            improve = chosen_cost < best_cost
            best = torch.where(improve[:, None, None], chosen, best)
            best_cost = torch.where(improve, chosen_cost, best_cost)
            self._distribution_update(costs, population)
            if self.config.store_rollouts:
                take = min(20, particles)
                ranking = safe_total.topk(take, largest=False).indices
                self._top_trajs = torch.gather(
                    population, 1, ranking[:, :, None, None].expand(-1, -1, horizon, action_dim)
                ).detach().clone()
            self._last_rollouts, self._last_costs = population.detach(), costs.detach()
            if self.config.store_debug:
                objective_history.append(best_cost.detach().clone())
                mean_history.append(self._dist.mean.detach().clone())
                cov_history.append(self._dist.cov.detach().clone())
        self._best_action = best.detach().clone()
        self._seed_action = seed.detach().clone()
        self.debug = None if not self.config.store_debug else {
            "objective": tuple(objective_history), "mean": tuple(mean_history), "covariance": tuple(cov_history),
        }
        if self.config.sample_mode == SampleMode.BEST:
            output = best
        elif self.config.sample_mode == SampleMode.SAMPLE:
            output = self._project(self._dist.mean + self._noise(problems, 1, horizon, action_dim, device=seed.device, dtype=seed.dtype, iteration=self.config.num_iters)[:, 0] * self._dist.scale_tril)
        else:
            output = self._dist.mean
        return output.reshape(seed_action.shape)

    def sample_actions(self, init_act):
        seed = self._ensure_distribution(torch.as_tensor(init_act), reset_mean=True)
        assert self._dist is not None
        problems, horizon, action_dim = self._dist.mean.shape
        noise = self._noise(problems, self.config.num_particles, horizon, action_dim, device=seed.device, dtype=seed.dtype, iteration=0)
        actions = self._project(self._dist.mean[:, None] + noise * self._dist.scale_tril[:, None])
        if self.null_per_problem:
            actions[:, :self.null_per_problem] = 0.0
        self._sample_set = actions.reshape(-1, horizon, action_dim)
        return self._sample_set

    def generate_noise(self, shape, base_seed=None):
        if self._distribution_shape is None:
            raise RuntimeError("MPPI distribution is initialized by optimize or sample_actions")
        count = int(shape[0])
        horizon, action_dim = self._distribution_shape
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed if base_seed is None else int(base_seed))
        device, dtype = self.mean_action.device, self.mean_action.dtype
        return torch.randn((count, horizon, action_dim), generator=generator, dtype=dtype, device="cpu").to(device)

    # -- Optimizer lifecycle ------------------------------------------------------

    def update_num_problems(self, num_problems: int):
        super().update_num_problems(num_problems)
        if self._dist is not None:
            self._dist = None

    def update_init_mean(self, init_mean):
        self.config.init_mean = torch.as_tensor(init_mean).detach().clone()
        if self._dist is not None:
            self._ensure_distribution(self._dist.mean, reset_mean=True)

    def update_seed(self, init_act):
        self._ensure_distribution(torch.as_tensor(init_act), reset_mean=True)
        return self.mean_action

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        del reset_num_iters
        seed = self._ensure_distribution(torch.as_tensor(action), reset_mean=False)
        assert self._dist is not None
        if mask is None:
            self._dist.mean.copy_(seed)
        else:
            selected = torch.as_tensor(mask, device=seed.device, dtype=torch.bool).reshape(-1)
            self._dist.mean[selected] = seed[selected]
        if clear_optimizer_state:
            self._cache.reset()
            self._best_action = self._dist.mean.detach().clone()
        return self.mean_action

    def reset_mean(self, reset_problem_ids=None):
        if self._dist is None:
            return None
        initial = self._seed_action if self._seed_action is not None else torch.zeros_like(self._dist.mean)
        if reset_problem_ids is None:
            self._dist.mean.copy_(initial)
        else:
            ids = torch.as_tensor(reset_problem_ids, device=initial.device, dtype=torch.long)
            self._dist.mean[ids] = initial[ids]
        return self._dist.mean

    def reset_covariance(self, reset_problem_ids=None):
        if self._dist is None:
            return None
        cov = self._covariance(*self._dist.mean.shape[:1], self._dist.mean.shape[-1], device=self._dist.mean.device, dtype=self._dist.mean.dtype)
        if reset_problem_ids is None:
            self._dist.cov.copy_(cov)
        else:
            ids = torch.as_tensor(reset_problem_ids, device=cov.device, dtype=torch.long)
            self._dist.cov[ids] = cov[ids]
        self._dist.scale_tril = self._dist.cov.sqrt()
        self._dist.inv_cov = self._dist.cov.reciprocal()
        return self._dist.cov

    def reset_distribution(self, reset_problem_ids=None):
        self.reset_mean(reset_problem_ids)
        return self.reset_covariance(reset_problem_ids)

    def reset_seed(self):
        if self._dist is not None:
            self._dist.reset_seed()

    def initialize_samples(self):
        if self.mean_action is None:
            return None
        return self.sample_actions(self.mean_action)

    update_samples = initialize_samples

    def shift(self, shift_steps: int = 0):
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        if self._dist is None or shift_steps == 0:
            return True
        horizon = self._dist.mean.shape[-2]
        if shift_steps >= horizon:
            shift_steps = horizon
        old = self._dist.mean.clone()
        self._dist.mean = torch.roll(old, shifts=-shift_steps, dims=-2)
        if self.config.base_action == BaseActionType.REPEAT and shift_steps:
            self._dist.mean[:, -shift_steps:] = old[:, -1:]
        elif self.config.base_action == BaseActionType.NULL and shift_steps:
            self._dist.mean[:, -shift_steps:] = 0.0
        elif self.config.base_action == BaseActionType.RANDOM and shift_steps:
            noise = self._noise(self._dist.mean.shape[0], 1, shift_steps, self._dist.mean.shape[-1], device=old.device, dtype=old.dtype, iteration=0)[:, 0]
            self._dist.mean[:, -shift_steps:] = noise * self._dist.scale_tril
        self._best_action = self._dist.mean.detach().clone()
        return True

    _shift = shift

    def get_all_rollout_instances(self):
        return self._rollout_list

    def get_rollouts(self):
        return self.get_all_rollout_instances()

    def compute_metrics(self, action):
        return _objective(self.rollout_fn)(action)

    def reset_shape(self):
        self._dist = None
        self._distribution_shape = None

    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA Graph capture is unavailable on CPU/MPS")

    def get_recorded_trace(self):
        return None

    def update_solver_params(self, solver_params):
        for key, value in dict(solver_params).items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def update_niters(self, niters):
        self.config.update_niters(niters)

    def debug_dump(self, file_path=""):
        if file_path:
            torch.save(self.debug, file_path)
        return self.debug


# -- Pinned tensor helpers -------------------------------------------------------

def jit_blend_cov(cov_action, cov_update, step_size_cov, kappa):
    """Exponential covariance blend with the V2 additive covariance floor."""
    return (1.0 - step_size_cov) * cov_action + step_size_cov * cov_update + kappa


def jit_blend_mean(mean_action, new_mean, step_size_mean):
    return (1.0 - step_size_mean) * mean_action + step_size_mean * new_mean


def jit_calculate_exp_util(beta, total_costs):
    beta_tensor = torch.as_tensor(beta, device=total_costs.device, dtype=total_costs.dtype)
    if bool((beta_tensor <= 0).any()):
        raise ValueError("beta must be positive")
    finite = torch.where(torch.isfinite(total_costs), total_costs, torch.full_like(total_costs, torch.inf))
    shifted = finite - finite.amin(dim=-1, keepdim=True)
    weights = torch.softmax(-shifted / beta_tensor, dim=-1)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)


def jit_compute_total_cost(gamma_seq, costs):
    gamma = torch.as_tensor(gamma_seq, device=costs.device, dtype=costs.dtype)
    while gamma.ndim < costs.ndim:
        gamma = gamma.unsqueeze(0)
    result = (costs * gamma).sum(dim=-1)
    # V2 normalises the discounted sequence by gamma[0].  The default value is
    # one; preserve it for non-default compatible gamma sequences too.
    first = gamma.reshape(-1, gamma.shape[-1])[0, 0].clamp_min(torch.finfo(costs.dtype).eps)
    return result / first


def jit_calculate_exp_util_from_costs(costs, gamma_seq, beta):
    return jit_calculate_exp_util(beta, jit_compute_total_cost(gamma_seq, costs))


def jit_diag_a_cov_update(w, actions, mean_action):
    """Per-action-dimension covariance, averaged over the action horizon."""
    while w.ndim < actions.ndim:
        w = w.unsqueeze(-1)
    delta = actions - mean_action.unsqueeze(-3)
    return (w * delta.square()).sum(dim=-3).mean(dim=-2, keepdim=True)


def jit_mean_cov_diag_a(
    costs, actions, gamma_seq, mean_action, cov_action, step_size_mean, step_size_cov, kappa, beta
):
    weights = jit_calculate_exp_util_from_costs(costs, gamma_seq, beta)
    expanded = weights
    while expanded.ndim < actions.ndim:
        expanded = expanded.unsqueeze(-1)
    weighted_mean = (expanded * actions).sum(dim=-3)
    new_mean = jit_blend_mean(mean_action, weighted_mean, step_size_mean)
    covariance = jit_diag_a_cov_update(expanded, actions, mean_action)
    new_cov = jit_blend_cov(cov_action, covariance, step_size_cov, kappa)
    return new_mean, new_cov, torch.sqrt(new_cov)


__all__ = [
    "BaseActionType", "MPPICfg", "MPPI", "jit_blend_cov", "jit_blend_mean",
    "jit_calculate_exp_util", "jit_calculate_exp_util_from_costs", "jit_compute_total_cost",
    "jit_diag_a_cov_update", "jit_mean_cov_diag_a",
]
