"""Portable shared lifecycle for V2 gradient optimizers.

The upstream implementation couples this class to CUDA graph capture and a
packed rollout ABI.  Those are not portable to Metal, but the public pieces
that owners rely on are: device-resident cost/gradient evaluation, per-problem
best-action selection, callback-driven quasi-Newton state, warm starts, and
debug traces.  This implementation keeps those pieces ordinary PyTorch so the
same core can drive custom optimizers on CPU and MPS.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import torch

from curobo._src.optim._portable import PortableOptimizer, _objective
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState


class GradientOptCore(PortableOptimizer):
    """Stateful autograd/line-search substrate for portable gradient solvers.

    ``step_direction_fn`` receives an :class:`OptimizationIterationState` and
    returns a tensor with the action shape.  Returning a state object is also
    accepted for V2 owner callbacks; its ``step_direction`` is used when
    present, otherwise steepest descent is used.  CUDA graph execution and
    raw CUDA line-search kernels deliberately remain unsupported.
    """

    _graphable_methods = {"_opt_iters", "_prepare_initial_iteration_state"}

    def __init__(
        self,
        config: Any,
        rollout_list: list[Any],
        step_direction_fn: Callable[[OptimizationIterationState], Any],
        *,
        on_reinitialize: Optional[Callable[[Optional[torch.Tensor]], None]] = None,
        on_initial_state: Optional[Callable[[OptimizationIterationState, Optional[torch.Tensor]], None]] = None,
        on_resize: Optional[Callable[[int], None]] = None,
        on_shift: Optional[Callable[[int], None]] = None,
        use_cuda_graph: bool = False,
    ):
        expected = int(getattr(config, "num_rollout_instances", 1))
        if len(rollout_list) != expected:
            raise ValueError(f"num_rollout_instances {expected} != len(rollout_list) {len(rollout_list)}")
        if not callable(step_direction_fn):
            raise TypeError("step_direction_fn must be callable")
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._step_direction_fn = step_direction_fn
        self._on_reinitialize = on_reinitialize
        self._on_initial_state = on_initial_state
        self._on_resize = on_resize
        self._on_shift = on_shift
        self._executors: dict[str, Any] = {}
        self._iteration_state: OptimizationIterationState | None = None
        self._og_num_iters = int(config.num_iters)
        self._best_action: torch.Tensor | None = None
        self._best_cost: torch.Tensor | None = None
        self._best_iteration: torch.Tensor | None = None
        self._converged: torch.Tensor | None = None
        self._convergence_count: torch.Tensor | None = None
        self._debug_trace: list[OptimizationIterationState] = []
        self._iteration = 0

    # -- public shape and lifecycle surfaces ---------------------------------

    def finish_init(self):
        """Compatibility no-op; portable execution is eager and cache-backed."""

        return self

    @property
    def horizon(self):
        return int(getattr(self.rollout_fn, "horizon", self.action_horizon))

    @property
    def solver_names(self):
        return [str(getattr(self.config, "solver_name", "gradient"))]

    def get_all_rollout_instances(self):
        return self._rollout_list

    def _rollout_callback(self, name: str, *args, **kwargs):
        result = None
        for rollout in self._rollout_list:
            callback = getattr(rollout, name, None)
            if callable(callback):
                result = callback(*args, **kwargs)
        return result

    def compute_metrics(self, action):
        callback = getattr(self.rollout_fn, "compute_metrics_from_action", None)
        if not callable(callback):
            callback = getattr(self.rollout_fn, "compute_metrics", None)
        return callback(action) if callable(callback) else None

    def reset_shape(self):
        self._rollout_callback("reset_shape")
        self._iteration_state = None

    def reset_seed(self):
        return True

    def reset_cuda_graph(self):
        # The base constructor rejects actual graph capture.  Retain this
        # method because callers commonly reset both graph and eager solvers.
        self._executors.clear()
        self._rollout_callback("reset_cuda_graph")

    def get_recorded_trace(self):
        if not bool(getattr(self.config, "store_debug", False)):
            return {"debug": [], "debug_cost": []}
        return {
            "debug": tuple(self._debug_trace),
            "debug_cost": tuple(state.cost.detach().clone() for state in self._debug_trace),
        }

    def update_niters(self, niters):
        self.config.update_niters(int(niters))

    def update_solver_params(self, solver_params):
        name = str(getattr(self.config, "solver_name", "gradient"))
        if name not in solver_params:
            raise ValueError(f"Optimizer {name} not found in solver parameters")
        for key, value in solver_params[name].items():
            if not hasattr(self.config, key):
                raise ValueError(f"unknown optimizer parameter: {key}")
            setattr(self.config, key, value)
        inner = int(getattr(self.config, "inner_iters", 1))
        total = int(self.config.num_iters)
        if inner <= 0 or total <= 0 or inner > total or total % inner:
            raise ValueError("num_iters must be a positive multiple of inner_iters")
        return True

    def update_goal_dt(self, goal):
        self._goal_dt = goal
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_goal_dt", None)
            if not callable(callback):
                callback = getattr(rollout, "update_dt", None)
            if callable(callback):
                callback(goal)

    def update_rollout_params(self, goal):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_params", None)
            if callable(callback):
                try:
                    callback(goal, num_particles=getattr(self.config, "num_particles", 1))
                except TypeError:
                    callback(goal)

    def update_num_problems(self, num_problems):
        num_problems = int(num_problems)
        if num_problems <= 0:
            raise ValueError("num_problems must be positive")
        self.config.num_problems = num_problems
        self._iteration_state = None
        self._best_action = self._best_cost = self._best_iteration = None
        self._converged = self._convergence_count = None
        if self._on_resize:
            self._on_resize(num_problems)
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_batch_size", None)
            if callable(callback):
                callback(batch_size=num_problems * int(getattr(self.config, "num_particles", 1) or 1))

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        action = self._canonical_action(action)
        batch = action.shape[0]
        checked_mask = self._mask(mask, batch, action.device)
        if reset_num_iters:
            self.config.update_niters(self._og_num_iters)
        if clear_optimizer_state:
            self._cache.reset()
        if self._on_reinitialize:
            self._on_reinitialize(checked_mask)
        state = self._prepare_initial_iteration_state(action)
        if self._iteration_state is not None and checked_mask is not None:
            state = self._merge_state(self._iteration_state, state, checked_mask)
        self._iteration_state = state
        self._set_best_from_state(state)
        if checked_mask is None:
            self._debug_trace.clear()

    def reset(self):
        super().reset()
        self._iteration_state = None
        self._best_action = self._best_cost = self._best_iteration = None
        self._converged = self._convergence_count = None
        self._debug_trace.clear()
        self.debug = None
        self._iteration = 0

    def shift(self, shift_steps=0):
        shift_steps = int(shift_steps)
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        if shift_steps and self._on_shift:
            self._on_shift(shift_steps)
        # A shifted action is a new optimization problem: retain no stale
        # best cost, while owners can retain quasi-Newton buffers via hook.
        self._iteration_state = None
        self._best_action = self._best_cost = self._best_iteration = None
        self._converged = self._convergence_count = None
        return True

    _shift = shift

    def debug_dump(self, file_path=""):
        del file_path
        # CUDA graph DOT dumps have no eager PyTorch equivalent.
        return None

    # -- tensor helpers -------------------------------------------------------

    def _canonical_action(self, action: torch.Tensor) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            raise TypeError("seed_action must be a torch.Tensor")
        batch = int(getattr(self.config, "num_problems", action.shape[0] if action.ndim else 1))
        horizon, dimension = self.action_horizon, self.action_dim
        if action.ndim == 3 and tuple(action.shape[1:]) == (horizon, dimension):
            if action.shape[0] != batch:
                raise ValueError(f"seed batch {action.shape[0]} != num_problems {batch}")
            return action
        if action.ndim == 2 and tuple(action.shape) == (batch, horizon * dimension):
            return action.reshape(batch, horizon, dimension)
        if batch == 1 and action.ndim == 2 and tuple(action.shape) == (horizon, dimension):
            return action.unsqueeze(0)
        raise ValueError(
            "seed_action must have shape "
            f"[{batch}, {horizon}, {dimension}] or [{batch}, {horizon * dimension}], got {tuple(action.shape)}"
        )

    @staticmethod
    def _mask(mask, batch: int, device: torch.device) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
        if tuple(mask.shape) != (batch,):
            raise ValueError(f"mask must have shape [{batch}]")
        return mask

    def _apply_action_bounds(self, action: torch.Tensor) -> torch.Tensor:
        low, high = self.action_bound_lows, self.action_bound_highs
        if low is None or high is None:
            return action
        low = torch.as_tensor(low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(high, device=action.device, dtype=action.dtype)
        if bool((low > high).any().item()):
            raise ValueError("action_bound_lows must not exceed action_bound_highs")
        try:
            return torch.maximum(torch.minimum(action, high), low)
        except RuntimeError as error:
            raise ValueError("action bounds must broadcast to [B, H, D]") from error

    def _objective_value(self, action: torch.Tensor, rollout=None) -> torch.Tensor:
        value = _objective(self.rollout_fn if rollout is None else rollout)(action)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=action.device, dtype=action.dtype)
        if value.ndim == 0:
            return value.expand(action.shape[0])
        if value.shape[0] != action.shape[0]:
            raise ValueError(
                "rollout objective must return one value per problem or a scalar; "
                f"got {tuple(value.shape)} for {action.shape[0]} problems"
            )
        return value.reshape(action.shape[0], -1).sum(-1)

    def _cost_and_gradient(self, action: torch.Tensor, rollout=None):
        with torch.enable_grad():
            leaf = action.detach().requires_grad_(True)
            cost = self._objective_value(leaf, rollout)
            finite_cost = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))
            if finite_cost.requires_grad:
                gradient = torch.autograd.grad(finite_cost.sum(), leaf, allow_unused=True)[0]
            else:
                gradient = None
        return cost.detach(), (torch.zeros_like(action) if gradient is None else gradient.detach())

    def _make_state(self, action: torch.Tensor, rollout=None) -> OptimizationIterationState:
        cost, gradient = self._cost_and_gradient(action, rollout)
        batch = action.shape[0]
        current = torch.full((batch,), self._iteration, device=action.device, dtype=torch.long)
        best_action = action.detach().clone()
        best_cost = cost.detach().clone()
        finite = torch.isfinite(best_cost)
        best_cost = torch.where(finite, best_cost, torch.full_like(best_cost, torch.inf))
        return OptimizationIterationState(
            action=action.detach().clone(), cost=cost, gradient=gradient,
            exploration_action=action.detach().clone(), exploration_gradient=gradient.detach().clone(),
            exploration_cost=cost.detach().clone(), step_direction=-gradient.detach().clone(),
            best_action=best_action, best_cost=best_cost,
            best_iteration=current.clone(), current_iteration=current,
            converged=torch.zeros(batch, device=action.device, dtype=torch.bool),
        )

    def _set_best_from_state(self, state: OptimizationIterationState) -> None:
        self._best_action = state.best_action.detach().clone()
        self._best_cost = state.best_cost.detach().clone()
        self._best_iteration = state.best_iteration.detach().clone()
        self._converged = state.converged.detach().clone() if state.converged is not None else None

    @staticmethod
    def _merge_state(old: OptimizationIterationState, new: OptimizationIterationState, mask: torch.Tensor) -> OptimizationIterationState:
        output = old.clone()
        for name, new_value in vars(new).items():
            old_value = getattr(output, name)
            if isinstance(old_value, torch.Tensor) and isinstance(new_value, torch.Tensor) and old_value.shape == new_value.shape and old_value.ndim:
                shape = (mask.shape[0],) + (1,) * (old_value.ndim - 1)
                setattr(output, name, torch.where(mask.reshape(shape), new_value, old_value))
            elif old_value is None:
                setattr(output, name, new_value)
        return output

    # -- optimization ---------------------------------------------------------

    def _prepare_initial_iteration_state(self, action: torch.Tensor) -> OptimizationIterationState:
        rollout = self._rollout_list[1] if len(self._rollout_list) > 1 else self.rollout_fn
        state = self._make_state(self._apply_action_bounds(action), rollout)
        if self._on_initial_state:
            self._on_initial_state(state, None)
        return state

    def _direction(self, state: OptimizationIterationState) -> torch.Tensor:
        value = self._step_direction_fn(state)
        if isinstance(value, OptimizationIterationState):
            value = value.step_direction
        if value is None:
            value = -state.gradient
        if not isinstance(value, torch.Tensor) or value.shape != state.action.shape:
            raise ValueError("step_direction_fn must return an action-shaped tensor or state")
        # Non-finite directions cannot produce a trustworthy candidate.  Keep
        # their corresponding coordinates stationary, independently per item.
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value)).detach()

    def _line_search_scales(self, action: torch.Tensor) -> torch.Tensor:
        scales = getattr(self.config, "line_search_scale", None)
        if scales is None:
            scales = [float(getattr(self.config, "step_scale", 1.0))]
        scales = torch.as_tensor(scales, device=action.device, dtype=action.dtype).reshape(-1)
        if scales.numel() == 0 or bool((~torch.isfinite(scales)).any()) or bool((scales < 0).any()):
            raise ValueError("line_search_scale must contain finite nonnegative values")
        return scales

    def _opt_step(self, state: OptimizationIterationState) -> OptimizationIterationState:
        direction = self._direction(state)
        scales = self._line_search_scales(state.action)
        candidates = self._apply_action_bounds(
            state.action[:, None] + scales.reshape((1, -1, 1, 1)) * direction[:, None]
        )
        batch, count = candidates.shape[:2]
        flat = candidates.reshape(batch * count, *candidates.shape[2:])
        costs, gradients = self._cost_and_gradient(flat)
        costs = costs.reshape(batch, count)
        gradients = gradients.reshape_as(candidates)
        selected = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, torch.inf)).argmin(dim=1)
        gather = selected.reshape(batch, 1, 1, 1).expand(batch, 1, *state.action.shape[1:])
        action = candidates.gather(1, gather).squeeze(1).detach()
        gradient = gradients.gather(1, gather).squeeze(1).detach()
        cost = costs.gather(1, selected[:, None]).squeeze(1).detach()
        prior_best = state.best_cost
        improved = torch.isfinite(cost) & (cost < prior_best)
        event_mask = improved.reshape((batch,) + (1,) * (action.ndim - 1))
        best_action = torch.where(event_mask, action, state.best_action)
        best_cost = torch.where(improved, cost, prior_best)
        current = state.current_iteration + 1
        best_iteration = torch.where(improved, current, state.best_iteration)
        delta = (cost - state.cost).abs()
        absolute = delta <= max(float(getattr(self.config, "cost_convergence", 0.0)), float(getattr(self.config, "cost_delta_threshold", 0.0)))
        relative_threshold = float(getattr(self.config, "cost_relative_threshold", 0.0))
        relative = delta <= relative_threshold * state.cost.abs().clamp_min(torch.finfo(cost.dtype).eps) if relative_threshold else torch.zeros_like(absolute)
        stable = torch.isfinite(cost) & torch.isfinite(state.cost) & (absolute | relative)
        if self._convergence_count is None or self._convergence_count.shape != stable.shape:
            self._convergence_count = torch.zeros_like(current)
        self._convergence_count = torch.where(stable, self._convergence_count + 1, torch.zeros_like(self._convergence_count))
        required = max(1, int(getattr(self.config, "convergence_iteration", 0)) + 1)
        converged = self._convergence_count >= required
        return OptimizationIterationState(
            action=action, cost=cost, gradient=gradient,
            exploration_action=action.clone(), exploration_gradient=gradient.clone(), exploration_cost=cost.clone(),
            step_direction=-gradient.clone(), best_action=best_action, best_cost=best_cost,
            best_iteration=best_iteration, current_iteration=current, converged=converged,
        )

    def _opt_iters(self, state: OptimizationIterationState) -> OptimizationIterationState:
        return self._opt_step(state)

    def _record(self, state: OptimizationIterationState) -> None:
        if bool(getattr(self.config, "store_debug", False)):
            self._debug_trace.append(state.clone())

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        seed = self._canonical_action(seed_action)
        if self._iteration_state is None:
            state = self._prepare_initial_iteration_state(seed)
        else:
            state = self._iteration_state
            self._iteration_state = None
        self._set_best_from_state(state)
        self._record(state)
        start = time.perf_counter()
        iterations = int(self.config.num_iters)
        fixed = bool(getattr(self.config, "fixed_iters", True))
        for _ in range(iterations):
            state = self._opt_step(state)
            self._iteration += 1
            self._record(state)
            if not fixed:
                ratio = float(getattr(self.config, "converged_ratio", 1.0))
                if bool(state.converged.to(torch.float32).mean() >= ratio):
                    break
        if seed.device.type == "mps":
            torch.mps.synchronize()
        self.opt_dt = time.perf_counter() - start
        self._set_best_from_state(state)
        self.debug = self.get_recorded_trace() if bool(getattr(self.config, "store_debug", False)) else None
        if bool(getattr(self.config, "return_best_action", True)):
            return state.best_action.detach().clone()
        finite_action = torch.isfinite(state.action).reshape(state.action.shape[0], -1).all(-1)
        return torch.where(finite_action.reshape((-1, 1, 1)), state.action, state.best_action).detach().clone()
