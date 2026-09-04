"""Portable scaled gradient descent compatible with cuRobo V2's public API.

The original implementation has CUDA graph and rollout-buffer special cases.
This version deliberately keeps the ordinary tensor/autograd behaviour on CPU
and MPS and rejects graph capture through :class:`PortableOptimizer`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

import curobo._src.runtime as curobo_runtime
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer, _objective

# These CUDA-only helpers are public upstream reexports.  Importing their
# modules initializes CUDA-oriented package state and creates a local circular
# import, so retain declaration aliases rather than claiming those backends.
ActionBounds = BestTracker = DebugRecorder = OptimizationIterationState = None
CudaEventTimer = GraphExecutor = None
Rollout = None
check_nan_last_dimension = create_graph_executor = log_and_raise = update_best_solution = None


@dataclass
class GradientDescentOptCfg(PortableOptCfg):
    solver_type: str = "gradient_descent"
    solver_name: str = "gradient_descent"
    inner_iters: int = 25
    step_scale: float = 1.0
    fixed_iters: bool = True
    cost_convergence: float = 1.0e-11
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    convergence_iteration: int = 0
    minimum_iters: int | None = None
    return_best_action: bool = True
    gradient_descent_step_scale: float = 0.001

    def __post_init__(self) -> None:
        for name in ("num_iters", "inner_iters", "num_problems"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.convergence_iteration, bool)
            or not isinstance(self.convergence_iteration, int)
            or self.convergence_iteration < 0
        ):
            raise ValueError("convergence_iteration must be a nonnegative integer")
        if self.num_particles is None:
            self.num_particles = 1
        if isinstance(self.num_particles, bool) or not isinstance(self.num_particles, int) or self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        for name in (
            "step_scale", "gradient_descent_step_scale", "cost_convergence",
            "cost_delta_threshold", "cost_relative_threshold", "converged_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.step_scale < 0 or self.gradient_descent_step_scale < 0:
            raise ValueError("gradient descent step scales must be nonnegative")
        if self.cost_convergence < 0 or self.cost_delta_threshold < 0 or self.cost_relative_threshold < 0:
            raise ValueError("cost convergence thresholds must be nonnegative")
        if not 0.0 <= self.converged_ratio <= 1.0:
            raise ValueError("converged_ratio must be in [0, 1]")
        if self.minimum_iters is not None and (
            isinstance(self.minimum_iters, bool)
            or not isinstance(self.minimum_iters, int)
            or self.minimum_iters < 0
        ):
            raise ValueError("minimum_iters must be nonnegative")
        if self.cost_relative_threshold >= 1.0:
            raise ValueError("cost_relative_threshold must be less than 1.0")
        for name in ("fixed_iters", "return_best_action"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0

    @property
    def outer_iters(self):
        return math.ceil(self.num_iters / self.inner_iters)

    @property
    def num_rollout_instances(self):
        return self._num_rollout_instances

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        return super().create_data_dict(data_dict, device_cfg, child_dict)

    def update_niters(self, niters: int):
        self.num_iters = niters


class _GradientDescentOptPortable(PortableOptimizer):
    """Autograd gradient descent with V2's per-problem lifecycle.

    The portable implementation intentionally has no CUDA graph or packed
    rollout-buffer path.  It does preserve the useful public behaviour: a
    fixed ``-gradient_descent_step_scale * gradient`` update, hard action
    bounds when a rollout publishes them, a finite best-action fallback, and
    per-problem convergence tracking for ordinary PyTorch rollouts.
    """

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        if len(rollout_list) != config.num_rollout_instances:
            raise ValueError(
                f"num_rollout_instances {config.num_rollout_instances} "
                f"!= len(rollout_list) {len(rollout_list)}"
            )
        if config.num_particles != 1:
            raise ValueError("GradientDescentOpt requires num_particles=1")
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._best_action: torch.Tensor | None = None
        self._best_cost: torch.Tensor | None = None
        self._converged: torch.Tensor | None = None
        self._convergence_count: torch.Tensor | None = None
        self._reinitialized_action: torch.Tensor | None = None
        self._iteration = 0
        self._original_num_iters = config.num_iters

    @property
    def horizon(self) -> int:
        return int(getattr(self.rollout_fn, "horizon", self.action_horizon))

    @property
    def solver_names(self) -> list[str]:
        return [self.config.solver_name]

    def _objective_value(self, action: torch.Tensor) -> torch.Tensor:
        value = _objective(self.rollout_fn)(action)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=action.device, dtype=action.dtype)
        batch = action.shape[0]
        # A rollout commonly retains horizon costs.  Every leading item is an
        # independent problem; only trailing cost dimensions are reduced.
        if value.ndim == 0:
            return value.expand(batch)
        if value.shape[0] != batch:
            raise ValueError(
                "rollout objective must return one value per problem or a scalar; "
                f"got {tuple(value.shape)} for {batch} problems"
            )
        return value.reshape(batch, -1).sum(dim=-1)

    def _canonical_action(self, action: torch.Tensor) -> torch.Tensor:
        """Validate V2's flattened or ``[B, H, D]`` seed convention."""

        if not isinstance(action, torch.Tensor):
            raise TypeError("seed_action must be a torch.Tensor")
        batch = int(self.config.num_problems)
        horizon, dim = self.action_horizon, self.action_dim
        # Lightweight callable rollouts do not publish a control-space shape.
        # In that case retain the historical optimizer convenience: infer the
        # complete seed layout instead of rejecting it because the portable
        # defaults are ``1 x 1 x 1``.  Rollouts that do publish a shape still
        # use the strict V2 validation below.
        if batch == 1 and horizon == 1 and dim == 1:
            if action.ndim == 3:
                return action
            if action.ndim == 2:
                return action.unsqueeze(1)
        if action.ndim == 2 and tuple(action.shape) == (batch, horizon * dim):
            return action.reshape(batch, horizon, dim)
        if action.ndim == 3 and tuple(action.shape) == (batch, horizon, dim):
            return action
        if batch == 1 and action.ndim == 2 and tuple(action.shape) == (horizon, dim):
            return action.unsqueeze(0)
        raise ValueError(
            "seed_action must have shape "
            f"[{batch}, {horizon}, {dim}] or [{batch}, {horizon * dim}], got {tuple(action.shape)}"
        )

    def _apply_action_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Project an action into rollout-published box bounds, if any."""

        low, high = self.action_bound_lows, self.action_bound_highs
        if low is None or high is None:
            return action
        low = torch.as_tensor(low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(high, device=action.device, dtype=action.dtype)
        if bool((low > high).any().item()):
            raise ValueError("action_bound_lows must not exceed action_bound_highs")
        try:
            torch.broadcast_shapes(action.shape, low.shape, high.shape)
        except RuntimeError as error:
            raise ValueError(
                "action bounds must broadcast to [num_problems, action_horizon, action_dim]"
            ) from error
        return torch.minimum(torch.maximum(action, low), high)

    def _should_stop(self, previous: torch.Tensor, current: torch.Tensor, iteration: int) -> bool:
        """Update V2-style per-problem convergence and decide batch exit."""

        if self.config.fixed_iters:
            return False
        finite = torch.isfinite(previous) & torch.isfinite(current)
        delta = (previous - current).abs()
        absolute = delta <= max(float(self.config.cost_convergence), float(self.config.cost_delta_threshold))
        relative_threshold = float(self.config.cost_relative_threshold)
        relative = torch.zeros_like(absolute)
        if relative_threshold > 0.0:
            relative = delta <= relative_threshold * previous.abs().clamp_min(torch.finfo(current.dtype).eps)
        stable = finite & (absolute | relative)
        if self._convergence_count is None or self._convergence_count.shape != stable.shape:
            self._convergence_count = torch.zeros_like(current, dtype=torch.int64)
        self._convergence_count = torch.where(
            stable, self._convergence_count + 1, torch.zeros_like(self._convergence_count)
        )
        required = max(1, int(self.config.convergence_iteration) + 1)
        self._converged = self._convergence_count >= required
        minimum = self.config.minimum_iters
        if minimum is not None and iteration < int(minimum):
            return False
        return bool(self._converged.to(torch.float32).mean().item() >= float(self.config.converged_ratio))

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        objective = _objective(self.rollout_fn)
        supplied = self._apply_action_bounds(self._canonical_action(seed_action)).detach()
        if self._reinitialized_action is not None:
            if self._reinitialized_action.shape != supplied.shape:
                raise ValueError("reinitialized action shape no longer matches optimizer seed shape")
            if self._reinitialized_action.device != supplied.device or self._reinitialized_action.dtype != supplied.dtype:
                raise ValueError("reinitialized action must share seed device and dtype")
            x = self._reinitialized_action
            self._reinitialized_action = None
        else:
            x = supplied
        best = x.detach().clone()
        best_cost = self._objective_value(best)
        previous_cost = best_cost
        self._converged = None
        self._convergence_count = None
        self._iteration = 0
        # V2 exposes a dedicated scale.  Older callers supplied ``step_scale``
        # before that field existed, so honor a non-default legacy value when
        # the dedicated field has not been changed from its default.
        step = float(self.config.gradient_descent_step_scale)
        if step == 0.001 and float(self.config.step_scale) != 1.0:
            step = float(self.config.step_scale)
        iterations: list[torch.Tensor] = []
        start = time.perf_counter()
        # Optimizers are expected to work inside a caller's ``no_grad``
        # inference scope: the update is still based on a local leaf graph.
        with torch.enable_grad():
            for iteration in range(self.config.num_iters):
                leaf = x.detach().requires_grad_(True)
                value = objective(leaf)
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value, device=leaf.device, dtype=leaf.dtype)
                scalar = torch.where(torch.isfinite(value), value, torch.zeros_like(value)).sum()
                gradient = (
                    torch.autograd.grad(scalar, leaf, create_graph=False, allow_unused=True)[0]
                    if scalar.requires_grad else None
                )
                if gradient is None:
                    gradient = torch.zeros_like(leaf)
                candidate = self._apply_action_bounds((leaf - step * gradient).detach())
                candidate_cost = self._objective_value(candidate)
                finite = torch.isfinite(candidate_cost)
                candidate_mask = finite.reshape((-1,) + (1,) * (candidate.ndim - 1))
                x = torch.where(candidate_mask, candidate, x.detach())
                reduced = self._objective_value(x)
                improved = torch.isfinite(reduced) & (reduced < best_cost)
                item_mask = improved.reshape((-1,) + (1,) * (best.ndim - 1))
                best = torch.where(item_mask, x, best)
                best_cost = torch.where(improved, reduced, best_cost)
                if self.config.store_debug:
                    iterations.append(best_cost.detach().clone())
                self._iteration += 1
                if self._should_stop(previous_cost, best_cost, iteration + 1):
                    break
                previous_cost = best_cost
        if x.device.type == "mps":
            torch.mps.synchronize()
        self.opt_dt = time.perf_counter() - start
        self._best_action = best.detach().clone()
        self._best_cost = best_cost.detach().clone()
        self.debug = (
            {"objective": tuple(iterations), "converged": self._converged.detach().clone() if self._converged is not None else None}
            if self.config.store_debug
            else None
        )
        current_finite = torch.isfinite(self._objective_value(x))
        current_mask = current_finite.reshape((-1,) + (1,) * (x.ndim - 1))
        return best if self.config.return_best_action else torch.where(current_mask, x, best)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        canonical = self._apply_action_bounds(self._canonical_action(action)).detach().clone()
        if mask is not None:
            mask = torch.as_tensor(mask, device=canonical.device, dtype=torch.bool)
            if mask.shape != (canonical.shape[0],):
                raise ValueError(f"mask must have shape [{canonical.shape[0]}]")
        super().reinitialize(canonical, mask, clear_optimizer_state, reset_num_iters)
        if reset_num_iters:
            self.config.update_niters(self._original_num_iters)
        if mask is None or self._reinitialized_action is None or self._reinitialized_action.shape != canonical.shape:
            self._reinitialized_action = canonical
        else:
            item_mask = mask.reshape((-1,) + (1,) * (canonical.ndim - 1))
            self._reinitialized_action = torch.where(item_mask, canonical, self._reinitialized_action)
        # Best/convergence records describe a completed solve, while the
        # pending action describes the next one.  Clear the former rather
        # than presenting a reinitialization seed as an already solved plan.
        self._best_action = None
        self._best_cost = None
        self._converged = None
        self._convergence_count = None
        self._iteration = 0

    def reset(self):
        super().reset()
        self._best_action = None
        self._best_cost = None
        self._converged = None
        self._convergence_count = None
        self._reinitialized_action = None
        self._iteration = 0
        self.debug = None

    def shift(self, shift_steps: int = 0):
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        self._best_action = None
        self._best_cost = None
        self._converged = None
        self._convergence_count = None
        self._reinitialized_action = None
        return True

    _shift = shift

    def get_recorded_trace(self):
        return self.debug if self.debug is not None else {"debug": [], "debug_cost": []}

    def update_solver_params(self, solver_params):
        if self.config.solver_name not in solver_params:
            raise ValueError(f"Optimizer {self.config.solver_name} not found in {solver_params}")
        values = solver_params[self.config.solver_name]
        if not isinstance(values, dict):
            raise TypeError("solver parameters must be a mapping")
        unknown = [name for name in values if not hasattr(self.config, name)]
        if unknown:
            raise ValueError("unknown optimizer parameter(s): " + ", ".join(sorted(unknown)))
        original = {name: getattr(self.config, name) for name in values}
        for name, value in solver_params[self.config.solver_name].items():
            setattr(self.config, name, value)
        try:
            self.config.__post_init__()
        except Exception:
            for name, value in original.items():
                setattr(self.config, name, value)
            self.config.__post_init__()
            raise
        return True

    def update_niters(self, niters: int):
        self.config.update_niters(niters)

    def update_num_problems(self, num_problems: int):
        if isinstance(num_problems, bool) or not isinstance(num_problems, int) or num_problems <= 0:
            raise ValueError("num_problems must be positive")
        super().update_num_problems(num_problems)
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_batch_size", None)
            if callable(callback):
                callback(batch_size=num_problems)
        self._best_action = self._best_cost = self._converged = self._convergence_count = None
        self._reinitialized_action = None
        self._iteration = 0

    def get_all_rollout_instances(self):
        return self._rollout_list

    def compute_metrics(self, action: torch.Tensor) -> Any:
        for name in ("compute_metrics_from_action", "compute_metrics"):
            callback = getattr(self.rollout_fn, name, None)
            if callable(callback):
                return callback(action)
        raise AttributeError("rollout does not provide compute_metrics_from_action or compute_metrics")

    def reset_seed(self):
        return True

    def reset_shape(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_shape", None)
            if callable(callback):
                callback()

    def reset_cuda_graph(self):
        # No graph is captured on portable devices.  Keep the upstream
        # lifecycle method so consumers can safely reset either backend.
        return None

    def debug_dump(self, file_path=""):
        del file_path
        return None


class GradientDescentOpt(_GradientDescentOptPortable):
    """Pinned declaration façade rebound to the portable tensor lifecycle."""

    def __init__(self, config: GradientDescentOptCfg, rollout_list: List[Rollout], use_cuda_graph: bool = False): pass
    def action_bound_highs(self): pass
    def action_bound_lows(self): pass
    def action_dim(self): pass
    def action_horizon(self): pass
    def action_step_max(self): pass
    def compute_metrics(self, action): pass
    def debug_dump(self, file_path=""): pass
    def disable(self): pass
    def enable(self): pass
    def enabled(self) -> bool: pass
    def get_all_rollout_instances(self): pass
    def get_recorded_trace(self): pass
    def horizon(self): pass
    def opt_dim(self): pass
    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor: pass
    def outer_iters(self): pass
    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False): pass
    def reset_cuda_graph(self): pass
    def reset_seed(self): pass
    def reset_shape(self): pass
    def shift(self, shift_steps=0): pass
    def solve_time(self): pass
    def solver_names(self): pass
    def update_goal_dt(self, goal): pass
    def update_niters(self, niters): pass
    def update_num_problems(self, num_problems): pass
    def update_rollout_params(self, goal): pass
    def update_solver_params(self, solver_params): pass


def _install_portable_gradient_descent_runtime():
    for base in reversed(_GradientDescentOptPortable.__mro__):
        for name, value in base.__dict__.items():
            if not (name.startswith("__") and name != "__init__"):
                setattr(GradientDescentOpt, name, value)


_install_portable_gradient_descent_runtime()


class LineSearchGradientDescentOpt(GradientDescentOpt):
    """Legacy V2 name; portable GD intentionally does not capture a graph."""
