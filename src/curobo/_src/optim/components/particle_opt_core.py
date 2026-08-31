"""Portable shared lifecycle for V2 particle optimizers.

The upstream class is the stateful substrate used by MPPI and evolution
strategies.  Its CUDA graph executor and packed rollout ABI cannot be carried
to Metal, but its *observable* distribution, population, warm-start, batch,
and debugging lifecycle can.  This module deliberately keeps that behaviour
on ordinary device-resident PyTorch tensors and is useful to custom particle
optimizers as well as the public facades.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim._portable import PortableOptimizer, _objective
from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.components.gaussian_distribution import CovType, GaussianDistribution
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.optim.particle.particle_opt_utils import SquashType, gaussian_entropy, scale_ctrl
from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg
from curobo._src.rollout.metrics import RolloutResult
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class SampleMode(Enum):
    """How a particle optimizer extracts an action after an iteration."""

    MEAN = "MEAN"
    BEST = "BEST"
    SAMPLE = "SAMPLE"


@dataclass
class _PortableCosts:
    """Small ``RolloutResult``-like cost surface for callable rollouts.

    Existing production rollouts may return their own object and are passed
    through unchanged.  The wrapper makes custom callable rollout functions
    usable by the same update callbacks without pretending to implement the
    CUDA rollout ABI.
    """

    values: torch.Tensor

    def get_sum_cost_and_constraint(self, sum_horizon: bool = False) -> torch.Tensor:
        return self.values.sum(dim=-1) if sum_horizon else self.values


@dataclass
class _PortableParticleRollout:
    actions: torch.Tensor
    costs: torch.Tensor

    @property
    def costs_and_constraints(self) -> _PortableCosts:
        return _PortableCosts(self.costs)


class _ParticleOptCorePortable(PortableOptimizer):
    """Stateful, batched particle optimizer infrastructure for CPU/MPS.

    ``update_distribution_fn`` receives a rollout result after every sampled
    population.  A callback can update ``core._dist.mean`` / ``cov`` directly,
    or return a mapping/tuple containing ``mean`` and optional ``cov``.  This
    mirrors V2's owner-supplied distribution-update policy while retaining a
    portable, testable contract.

    Raw CUDA graph capture, Warp sample kernels, and packed CUDA rollout
    result buffers intentionally remain unavailable; requesting graph capture
    raises the standard portable optimizer boundary error.
    """

    _graphable_methods: set[str] = {"_opt_iters"}

    def __init__(
        self,
        config: Any,
        rollout_list: List[Any],
        update_distribution_fn: Callable[[Any], Any],
        use_cuda_graph: bool = False,
    ):
        if not callable(update_distribution_fn):
            raise TypeError("update_distribution_fn must be callable")
        expected = getattr(config, "num_rollout_instances", 1)
        if len(rollout_list) != expected:
            raise ValueError(
                f"num_rollout_instances {expected} != len(rollout_list) {len(rollout_list)}"
            )
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._update_distribution_fn = update_distribution_fn
        self._og_num_iters = int(config.num_iters)
        self._iteration_state: Optional[OptimizationIterationState] = None
        self._debug = DebugRecorder() if bool(getattr(config, "store_debug", False)) else None
        self._sample_iter_n = 0
        self.num_steps = 0
        self.problem_col: Optional[torch.Tensor] = None
        self.top_values: Optional[torch.Tensor] = None
        self.top_idx: Optional[torch.Tensor] = None
        self.top_trajs: Optional[torch.Tensor] = None
        self.visual_traj: Optional[torch.Tensor] = None
        self._last_population: Optional[torch.Tensor] = None
        self._last_costs: Optional[torch.Tensor] = None
        self._last_rollout: Optional[Any] = None
        self._bounds: Optional[ActionBounds] = None

        particles = int(getattr(config, "num_particles", 0) or 0)
        if particles <= 0:
            raise ValueError("particle optimizers require config.num_particles > 0")
        self._init_particle_counts(particles)
        self._dist = self._new_distribution()
        self.gamma_seq = self._make_gamma()
        self.update_num_problems(int(getattr(config, "num_problems", 1)))

    # -- Shape, distribution, and properties ---------------------------------

    @property
    def action_horizon(self) -> int:
        return int(getattr(self.rollout_fn, "action_horizon", 1))

    @property
    def action_dim(self) -> int:
        return int(getattr(self.rollout_fn, "action_dim", 1))

    @property
    def horizon(self) -> int:
        return int(getattr(self.rollout_fn, "horizon", self.action_horizon))

    @property
    def opt_dim(self) -> int:
        return self.action_horizon * self.action_dim

    @property
    def solver_names(self) -> list[str]:
        return [str(getattr(self.config, "solver_name", "particle"))]

    @property
    def total_num_particles(self) -> int:
        return int(self.config.num_problems) * self.particles_per_problem

    @property
    def action_bound_lows(self):
        return getattr(self.rollout_fn, "action_bound_lows", None)

    @property
    def action_bound_highs(self):
        return getattr(self.rollout_fn, "action_bound_highs", None)

    def _refresh_bounds(self) -> None:
        lows, highs = self.action_bound_lows, self.action_bound_highs
        if lows is None or highs is None:
            self._bounds = None
            return
        device, dtype = self.device_cfg.device, self.device_cfg.dtype
        low = torch.as_tensor(lows, device=device, dtype=dtype)
        high = torch.as_tensor(highs, device=device, dtype=dtype)
        self._bounds = ActionBounds(low, high, self.action_horizon, float(getattr(self.config, "step_scale", 1.0)))

    @property
    def action_step_max(self):
        self._refresh_bounds()
        return None if self._bounds is None else self._bounds.action_step_max

    @property
    def action_horizon_bounds_lows(self):
        self._refresh_bounds()
        return None if self._bounds is None else self._bounds.action_horizon_bounds_lows

    @property
    def action_horizon_bounds_highs(self):
        self._refresh_bounds()
        return None if self._bounds is None else self._bounds.action_horizon_bounds_highs

    def _initial_mean(self) -> torch.Tensor:
        value = getattr(self.config, "init_mean", None)
        if value is None:
            getter = getattr(self.rollout_fn, "get_initial_action", None)
            value = getter() if callable(getter) else None
        if value is None:
            value = torch.zeros((1, self.action_horizon, self.action_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        value = torch.as_tensor(value, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or tuple(value.shape[-2:]) != (self.action_horizon, self.action_dim):
            raise ValueError("init_mean must have shape [problems, action_horizon, action_dim]")
        return value

    def _initial_cov(self) -> torch.Tensor:
        value = torch.as_tensor(getattr(self.config, "init_cov", 1.0), device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        if value.ndim == 0:
            value = value.expand(self.action_dim)
        if value.ndim == 1:
            if value.numel() != self.action_dim:
                raise ValueError("init_cov must be scalar or contain action_dim elements")
            value = value.reshape(1, 1, self.action_dim)
        elif value.ndim == 2:
            if value.shape[-1] != self.action_dim:
                raise ValueError("init_cov must have action_dim trailing elements")
            value = value.unsqueeze(-2)
        elif value.ndim != 3 or value.shape[-2:] != (1, self.action_dim):
            raise ValueError("init_cov must be scalar, [D], [P,D], or [P,1,D]")
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("init_cov must be finite and nonnegative")
        return value.clamp_min(torch.finfo(value.dtype).eps)

    def _new_distribution(self) -> GaussianDistribution:
        sample_params = getattr(self.config, "sample_params", None)
        if sample_params is None:
            sample_params = ParticleSamplerCfg(self.device_cfg, seed=int(getattr(self.config, "seed", 0)))
            self.config.sample_params = sample_params
        cov_type = getattr(self.config, "cov_type", CovType.DIAG_A)
        if isinstance(cov_type, str):
            cov_type = CovType[cov_type.upper()]
        return GaussianDistribution(
            self.device_cfg, self.action_horizon, self.action_dim, cov_type,
            self._initial_mean(), self._initial_cov(), sample_params,
            bool(getattr(self.config, "random_mean", False)), int(getattr(self.config, "seed", 0)),
        )

    def _make_gamma(self) -> torch.Tensor:
        gamma = float(getattr(self.config, "gamma", 1.0))
        powers = torch.arange(self.horizon, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        return torch.pow(torch.as_tensor(gamma, device=powers.device, dtype=powers.dtype), powers).reshape(1, -1)

    def _init_particle_counts(self, particles: int) -> None:
        fraction = float(getattr(self.config, "null_act_frac", 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("null_act_frac must lie in [0, 1]")
        total_null = round(int(fraction * particles))
        self.null_per_problem = round(total_null * 0.5)
        self.neg_per_problem = total_null - self.null_per_problem
        self.sampled_particles_per_problem = particles - total_null
        self.particles_per_problem = particles

    def finish_init(self):
        """Compatibility no-op: persistent buffers replace CUDA graphs."""
        return self

    # -- Sampling and rollout evaluation --------------------------------------

    def _noise(self, shape: tuple[int, ...], *, iteration: int) -> torch.Tensor:
        # Generate deterministically on CPU, then move the fully materialized
        # sample tensor.  That avoids unsupported generator usage on MPS while
        # keeping the tensor work itself entirely device resident.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(self.config, "seed", 0)) + int(iteration))
        return torch.randn(shape, generator=generator, dtype=self.device_cfg.dtype, device="cpu").to(self.device_cfg.device)

    def _project(self, actions: torch.Tensor) -> torch.Tensor:
        low, high = self.action_horizon_bounds_lows, self.action_horizon_bounds_highs
        if low is None or high is None:
            return actions
        mode = getattr(self.config, "squash_fn", SquashType.CLAMP)
        return scale_ctrl(actions, low, high, squash_fn=mode)

    def initialize_samples(self):
        self._sample_iter_n = 0
        # Keep the visible Gaussian buffers in the conventional V2 shape.
        self._dist._sample_set = self._noise(
            (max(1, int(getattr(self.config, "num_iters", 1))), int(self.config.num_problems), self.sampled_particles_per_problem, self.action_horizon, self.action_dim),
            iteration=0,
        )
        if self.sampled_particles_per_problem:
            self._dist._sample_set[:, :, -1] = 0.0
        self._dist._sample_iter = torch.zeros(1, dtype=torch.long, device=self.device_cfg.device)
        return self._dist._sample_set

    def update_samples(self):
        return self.initialize_samples()

    def sample_actions(self, init_act=None) -> torch.Tensor:
        """Draw one device-resident population in V2 particle order.

        The sampled portion is followed by negative-mean and null sequences.
        The final ordinary sampled sequence is exactly the mean, preserving the
        useful safety invariant that every population contains its seed.
        """
        if init_act is not None:
            self.update_seed(init_act)
        if self._dist.mean is None:
            self.reset_distribution()
        assert self._dist.mean is not None and self._dist.scale_tril is not None
        problems = int(self.config.num_problems)
        count = self.sampled_particles_per_problem
        pieces: list[torch.Tensor] = []
        if count:
            sample_set = self._dist._sample_set
            if sample_set is None or sample_set.shape[1:3] != (problems, count):
                self.initialize_samples()
                sample_set = self._dist._sample_set
            assert sample_set is not None
            index = self._sample_iter_n % sample_set.shape[0]
            delta = sample_set[index]
            self._sample_iter_n += 1
            sampled = self._dist.mean[:, None] + delta * self._dist.scale_tril[:, None]
            sampled[:, -1] = self._dist.mean
            pieces.append(sampled)
        if self.neg_per_problem:
            pieces.append((-self._dist.mean).unsqueeze(1).expand(-1, self.neg_per_problem, -1, -1))
        if self.null_per_problem:
            pieces.append(torch.zeros((problems, self.null_per_problem, self.action_horizon, self.action_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype))
        population = torch.cat(pieces, dim=1)
        return self._project(population.reshape(self.total_num_particles, self.action_horizon, self.action_dim))

    def _cost_tensor(self, result: Any, actions: torch.Tensor) -> torch.Tensor:
        source = result
        if hasattr(source, "costs_and_constraints"):
            source = source.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=False)
        elif hasattr(source, "costs"):
            source = source.costs
        if not isinstance(source, torch.Tensor):
            source = torch.as_tensor(source, device=actions.device, dtype=actions.dtype)
        source = source.to(device=actions.device, dtype=actions.dtype)
        if source.numel() == actions.shape[0]:
            return source.reshape(actions.shape[0], 1)
        if source.ndim >= 1 and source.shape[0] == actions.shape[0]:
            return source.reshape(actions.shape[0], -1)
        raise ValueError("particle rollout must produce one scalar or cost sequence per sampled action")

    def _generate_rollouts(self):
        actions = self.sample_actions()
        evaluate = getattr(self.rollout_fn, "evaluate_action", None)
        raw = evaluate(actions) if callable(evaluate) else _objective(self.rollout_fn)(actions)
        costs = self._cost_tensor(raw, actions)
        self._last_population = actions.reshape(int(self.config.num_problems), self.particles_per_problem, self.action_horizon, self.action_dim)
        self._last_costs = costs.reshape(int(self.config.num_problems), self.particles_per_problem, -1)
        if hasattr(raw, "costs_and_constraints"):
            self._last_rollout = raw
        else:
            self._last_rollout = _PortableParticleRollout(actions, costs)
        return self._last_rollout

    def _apply_update_result(self, result: Any) -> None:
        if result is None:
            return
        mean = cov = None
        if isinstance(result, dict):
            mean, cov = result.get("mean"), result.get("cov", result.get("covariance"))
        elif isinstance(result, tuple):
            mean, cov = result[0], result[1] if len(result) > 1 else None
        elif isinstance(result, torch.Tensor):
            mean = result
        if mean is not None:
            value = torch.as_tensor(mean, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            if value.shape != self._dist.mean.shape:
                raise ValueError("distribution callback mean must match [problems, action_horizon, action_dim]")
            self._dist.mean = value
        if cov is not None:
            value = torch.as_tensor(cov, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            if value.ndim == 2:
                value = value.unsqueeze(-2)
            if value.shape != self._dist.cov.shape:
                raise ValueError("distribution callback cov must match [problems, 1, action_dim]")
            self._dist.cov = value.clamp_min(torch.finfo(value.dtype).eps)
            self._dist.scale_tril = self._dist.cov.sqrt()
            self._dist.inv_cov = self._dist.cov.reciprocal()

    def _total_costs(self) -> torch.Tensor:
        assert self._last_costs is not None
        costs = self._last_costs
        gamma = self.gamma_seq.to(device=costs.device, dtype=costs.dtype)
        if gamma.shape[-1] != costs.shape[-1]:
            gamma = torch.pow(torch.as_tensor(float(getattr(self.config, "gamma", 1.0)), device=costs.device, dtype=costs.dtype), torch.arange(costs.shape[-1], device=costs.device, dtype=costs.dtype)).reshape(1, 1, -1)
        return (costs * gamma).sum(dim=-1)

    def _get_action_seq(self, mode: Any):
        if isinstance(mode, str):
            mode = SampleMode[mode.upper()]
        if mode == SampleMode.MEAN:
            return self._dist.mean
        if mode == SampleMode.BEST:
            return self._dist.best_traj if self._dist.best_traj is not None else self._dist.mean
        if mode == SampleMode.SAMPLE:
            noise = self._noise((int(self.config.num_problems), 1, self.action_horizon, self.action_dim), iteration=123 + self.num_steps)[:, 0]
            return self._project(self._dist.mean + noise * self._dist.scale_tril)
        raise ValueError(f"unidentified sampling mode: {mode!r}")

    # -- Optimization and lifecycle -------------------------------------------

    def _opt_iters(self, iteration_state: OptimizationIterationState) -> OptimizationIterationState:
        self.update_seed(iteration_state.action)
        for _ in range(int(getattr(self.config, "inner_iters", 1))):
            trajectory = self._generate_rollouts()
            update = self._update_distribution_fn(trajectory)
            self._apply_update_result(update)
            totals = self._total_costs()
            safe = torch.where(torch.isfinite(totals), totals, torch.full_like(totals, torch.inf))
            index = safe.argmin(dim=-1)
            assert self._last_population is not None
            candidate = self._last_population[torch.arange(safe.shape[0], device=safe.device), index]
            candidate_cost = safe[torch.arange(safe.shape[0], device=safe.device), index]
            if self._dist.best_traj is None or iteration_state.best_cost is None:
                best, best_cost = candidate, candidate_cost
            else:
                prior = iteration_state.best_cost.reshape(-1)
                improved = candidate_cost < prior
                best = torch.where(improved[:, None, None], candidate, iteration_state.best_action)
                best_cost = torch.where(improved, candidate_cost, prior)
            self._dist.best_traj = best.detach().clone()
            ranking = safe.topk(min(20, self.particles_per_problem), largest=False).indices
            self.top_trajs = torch.gather(self._last_population, 1, ranking[:, :, None, None].expand(-1, -1, self.action_horizon, self.action_dim)).detach().clone()
            self.top_values, self.top_idx = safe.detach().clone(), ranking.detach().clone()
            action = self._get_action_seq(getattr(self.config, "sample_mode", SampleMode.MEAN))
            iteration_state = OptimizationIterationState(
                action=action.reshape(safe.shape[0], -1), cost=candidate_cost.unsqueeze(-1),
                exploration_action=action.reshape(safe.shape[0], -1), exploration_cost=candidate_cost.unsqueeze(-1),
                best_action=best, best_cost=best_cost.unsqueeze(-1),
            )
            if self._debug is not None:
                self._debug.record(iteration_state, self.action_horizon, self.action_dim)
        return iteration_state

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        seed = torch.as_tensor(seed_action, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        if seed.numel() != int(self.config.num_problems) * self.opt_dim:
            inferred = seed.reshape(-1, self.action_horizon, self.action_dim).shape[0]
            self.update_num_problems(inferred)
        seed = seed.reshape(int(self.config.num_problems), self.action_horizon, self.action_dim)
        started = time.perf_counter()
        state = OptimizationIterationState(action=seed.reshape(seed.shape[0], -1), best_action=seed, best_cost=None)
        for _ in range(self.outer_iters):
            state = self._opt_iters(state)
        if seed.device.type == "mps":
            torch.mps.synchronize()
        self.opt_dt = time.perf_counter() - started
        self._iteration_state = state
        self.num_steps += 1
        return state.best_action.reshape_as(seed)

    def update_seed(self, init_act):
        value = torch.as_tensor(init_act, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        if value.numel() != int(self.config.num_problems) * self.opt_dim:
            raise ValueError("seed action must contain [num_problems, action_horizon, action_dim]")
        self._dist.mean = value.reshape(int(self.config.num_problems), self.action_horizon, self.action_dim).detach().clone()
        self._dist.best_traj = self._dist.mean.detach().clone()
        return self._dist.mean

    def reset_distribution(self, reset_problem_ids=None):
        initial = self._initial_mean().expand(int(self.config.num_problems), -1, -1)
        covariance = self._initial_cov().expand(int(self.config.num_problems), -1, -1)
        if reset_problem_ids is None or self._dist.mean is None:
            self._dist.mean, self._dist.best_traj = initial.clone(), initial.clone()
            self._dist.cov = covariance.clone()
        else:
            ids = torch.as_tensor(reset_problem_ids, device=self.device_cfg.device, dtype=torch.long).reshape(-1)
            self._dist.mean[ids] = initial[ids]
            self._dist.best_traj[ids] = initial[ids]
            self._dist.cov[ids] = covariance[ids]
        self._dist.scale_tril = self._dist.cov.sqrt()
        self._dist.inv_cov = self._dist.cov.reciprocal()
        return self._dist.mean

    def reinitialize(self, action: torch.Tensor, mask: Optional[torch.Tensor] = None, clear_optimizer_state: bool = True, reset_num_iters: bool = False) -> None:
        if reset_num_iters:
            self.config.num_iters = self._og_num_iters
        seed = torch.as_tensor(action, device=self.device_cfg.device, dtype=self.device_cfg.dtype).reshape(int(self.config.num_problems), self.action_horizon, self.action_dim)
        if mask is None:
            self._dist.mean = seed.clone()
            self._dist.best_traj = seed.clone()
        else:
            selected = torch.as_tensor(mask, device=seed.device, dtype=torch.bool).reshape(-1)
            if selected.numel() != seed.shape[0]:
                raise ValueError("mask must have one value per problem")
            self._dist.mean[selected] = seed[selected]
            self._dist.best_traj[selected] = seed[selected]
        if clear_optimizer_state:
            self._sample_iter_n = 0
            self.initialize_samples()
            if self._debug is not None and mask is None:
                self._debug.clear()

    def shift(self, shift_steps: int = 0) -> bool:
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        if shift_steps == 0:
            return True
        steps = min(shift_steps, self.action_horizon)
        repeat = str(getattr(getattr(self.config, "base_action", ""), "name", getattr(self.config, "base_action", ""))).upper() == "REPEAT"
        for name in ("mean", "best_traj"):
            value = getattr(self._dist, name)
            if value is None:
                continue
            shifted = torch.roll(value, -steps, dims=-2)
            shifted[:, -steps:] = value[:, -1:] if repeat else 0.0
            setattr(self._dist, name, shifted)
        return True

    _shift = shift

    def update_num_problems(self, num_problems: int):
        if num_problems <= 0:
            raise ValueError("num_problems must be positive")
        self.config.num_problems = int(num_problems)
        self.problem_col = torch.arange(num_problems, device=self.device_cfg.device, dtype=torch.long)
        self.reset_distribution()
        self.initialize_samples()
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_batch_size", None)
            if callable(callback):
                try:
                    callback(batch_size=self.total_num_particles)
                except TypeError:
                    callback(self.total_num_particles)

    def update_rollout_params(self, goal):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_params", None)
            if callable(callback):
                try:
                    callback(goal, num_particles=self.particles_per_problem)
                except TypeError:
                    callback(goal)

    def update_goal_dt(self, goal):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_goal_dt", None)
            if callable(callback):
                callback(goal)

    def get_rollouts(self):
        return self.top_trajs

    def get_all_rollout_instances(self):
        return self._rollout_list

    def compute_metrics(self, action):
        callback = getattr(self.rollout_fn, "compute_metrics_from_action", None)
        if callable(callback):
            return callback(action)
        return _objective(self.rollout_fn)(action)

    def reset_shape(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_shape", None)
            if callable(callback):
                callback()

    def reset_seed(self) -> bool:
        self._sample_iter_n = 0
        self._dist.reset_seed()
        self.update_samples()
        return True

    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA Graph capture is unavailable on CPU/MPS")

    def get_recorded_trace(self) -> Dict[str, Any]:
        if self._debug is None:
            return {"debug": [], "debug_cost": []}
        return {"debug": self._debug.get_trace(), "debug_cost": []}

    def update_solver_params(self, solver_params: Dict[str, Dict[str, Any]]) -> bool:
        values = solver_params.get(getattr(self.config, "solver_name", "particle"))
        if values is None:
            raise ValueError(f"optimizer {getattr(self.config, 'solver_name', 'particle')} not found in solver_params")
        for name, value in values.items():
            setattr(self.config, name, value)
        return True

    def update_niters(self, niters: int):
        callback = getattr(self.config, "update_niters", None)
        if callable(callback):
            callback(niters)
        elif niters > 0:
            self.config.num_iters = niters
        else:
            raise ValueError("niters must be positive")

    def update_init_mean(self, init_mean):
        self.config.init_mean = torch.as_tensor(init_mean, device=self.device_cfg.device, dtype=self.device_cfg.dtype).detach().clone()
        return self.reset_distribution()

    def debug_dump(self, file_path: str = ""):
        trace = self.get_recorded_trace()
        if file_path:
            torch.save(trace, file_path)
        return trace


__all__ = [
    "ActionBounds", "CovType", "DebugRecorder", "GaussianDistribution", "OptimizationIterationState",
    "ParticleOptCore", "ParticleSamplerCfg", "SampleMode", "SquashType", "gaussian_entropy", "scale_ctrl",
]


class ParticleOptCore(_ParticleOptCorePortable):
    """Pinned declaration façade rebound to the portable particle lifecycle."""

    def __init__(self, config, rollout_list: List[Rollout], update_distribution_fn: Callable, use_cuda_graph: bool=False): pass
    def action_bound_highs(self): pass
    def action_bound_lows(self): pass
    def action_dim(self) -> int: pass
    def action_horizon(self) -> int: pass
    def action_horizon_bounds_highs(self): pass
    def action_horizon_bounds_lows(self): pass
    def action_step_max(self): pass
    def compute_metrics(self, action: torch.Tensor): pass
    def debug_dump(self, file_path: str=''): pass
    def disable(self): pass
    def enable(self): pass
    def enabled(self) -> bool: pass
    def finish_init(self): pass
    def get_all_rollout_instances(self) -> List[Rollout]: pass
    def get_recorded_trace(self) -> Dict[str, Any]: pass
    def get_rollouts(self): pass
    def horizon(self): pass
    def initialize_samples(self): pass
    def opt_dim(self) -> int: pass
    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor: pass
    def outer_iters(self) -> int: pass
    def reinitialize(self, action: torch.Tensor, mask: Optional[torch.Tensor]=None, clear_optimizer_state: bool=True, reset_num_iters: bool=False) -> None: pass
    def reset_cuda_graph(self): pass
    def reset_distribution(self, reset_problem_ids=None): pass
    def reset_seed(self) -> bool: pass
    def reset_shape(self): pass
    def sample_actions(self, init_act): pass
    def shift(self, shift_steps: int=0) -> bool: pass
    def solve_time(self) -> float: pass
    def solver_names(self): pass
    def update_goal_dt(self, goal): pass
    def update_niters(self, niters: int): pass
    def update_num_problems(self, num_problems: int): pass
    def update_rollout_params(self, goal): pass
    def update_samples(self): pass
    def update_seed(self, init_act): pass
    def update_solver_params(self, solver_params: Dict[str, Dict[str, Any]]) -> bool: pass


def _install_portable_particle_core_runtime():
    for base in reversed(_ParticleOptCorePortable.__mro__):
        for name, value in base.__dict__.items():
            if not (name.startswith("__") and name != "__init__"):
                setattr(ParticleOptCore, name, value)


_install_portable_particle_core_runtime()
