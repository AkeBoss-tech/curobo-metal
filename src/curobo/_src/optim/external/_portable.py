"""Shared portable machinery for external cuRobo optimizer adapters.

The upstream wrappers use CUDA graphs around rollout evaluation.  This module
keeps the useful part of that API--a rollout-owning optimization lifecycle--on
ordinary PyTorch tensors, while deliberately not presenting an execution cache
as a CUDA graph.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.util.cuda_event_timer import CudaEventTimer


class UnsupportedExternalOptimizerFeature(NotImplementedError):
    """A CUDA-only external-optimizer option was requested on CPU/MPS."""


def create_data_dict(cls, data_dict: dict[str, Any], device_cfg, child_dict=None) -> dict[str, Any]:
    values = dict(data_dict if child_dict is None else child_dict)
    values["device_cfg"] = device_cfg
    values.setdefault("num_particles", None)
    allowed = {entry.name for entry in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


class ExternalOptimizerBase:
    """Rollout lifecycle common to the portable Torch and SciPy adapters."""

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        if use_cuda_graph:
            raise UnsupportedExternalOptimizerFeature(
                "CUDA Graph capture is unavailable on CPU/MPS; use eager portable optimization"
            )
        if len(rollout_list) != config.num_rollout_instances:
            raise ValueError(
                f"num_rollout_instances {config.num_rollout_instances} != len(rollout_list) {len(rollout_list)}"
            )
        if not rollout_list:
            raise ValueError("rollout_list must contain at least one rollout")
        self.config = config
        self.device_cfg = config.device_cfg
        self.rollout_fn = rollout_list[0]
        self._rollout_list = list(rollout_list)
        self._enabled = True
        self.use_cuda_graph = False
        self.opt_dt = 0.0
        self._og_num_iters = config.num_iters
        self._debug = DebugRecorder() if config.store_debug else None
        self._iteration_state = None
        self._goal_dt = None
        self.best_cost = None
        self.best_q = None
        self._optimization_variable = None
        self.l_vec = None

        low, high = self.action_bound_lows, self.action_bound_highs
        self._bounds = (
            ActionBounds(low, high, self.action_horizon, config.step_scale)
            if isinstance(low, torch.Tensor) and isinstance(high, torch.Tensor)
            else None
        )
        self.update_num_problems(config.num_problems)

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def action_horizon(self):
        return int(getattr(self.rollout_fn, "action_horizon", 1))

    @property
    def action_dim(self):
        return int(getattr(self.rollout_fn, "action_dim", 1))

    @property
    def opt_dim(self):
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self):
        return self.config.outer_iters

    @property
    def horizon(self):
        return int(getattr(self.rollout_fn, "horizon", self.action_horizon))

    @property
    def action_bound_lows(self):
        return getattr(self.rollout_fn, "action_bound_lows", None)

    @property
    def action_bound_highs(self):
        return getattr(self.rollout_fn, "action_bound_highs", None)

    @property
    def action_horizon_bounds_lows(self):
        return None if self._bounds is None else self._bounds.action_horizon_bounds_lows

    @property
    def action_horizon_bounds_highs(self):
        return None if self._bounds is None else self._bounds.action_horizon_bounds_highs

    @property
    def solve_time(self):
        return self.opt_dt

    @property
    def solver_names(self):
        return [self.config.solver_name]

    def _action_view(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize only known rollout actions, preserving generic callables."""
        if hasattr(self.rollout_fn, "action_horizon") and hasattr(self.rollout_fn, "action_dim"):
            expected = self.config.num_problems * self.action_horizon * self.action_dim
            if value.numel() == expected:
                return value.reshape(self.config.num_problems, self.action_horizon, self.action_dim)
        return value

    @staticmethod
    def _reduce_cost(value: torch.Tensor, batch: int) -> torch.Tensor:
        if value.ndim == 0:
            return value.expand(batch)
        if value.shape[0] != batch:
            if value.numel() == batch:
                return value.reshape(batch)
            return value.sum().expand(batch)
        return value.reshape(batch, -1).sum(dim=-1)

    def _rollout_values(self, action: torch.Tensor, *, constraints: bool = False):
        """Return one differentiable scalar cost (and optional constraint) per batch.

        It accepts both small callable test objectives and the real cuRobo
        ``evaluate_action``/``CostsAndConstraints`` protocol.
        """
        batch = int(action.shape[0]) if action.ndim else 1
        if callable(self.rollout_fn):
            cost = self._reduce_cost(self.rollout_fn(action), batch)
            return cost, None
        for name in ("objective", "cost_fn"):
            callback = getattr(self.rollout_fn, name, None)
            if callable(callback):
                return self._reduce_cost(callback(action), batch), None
        evaluate = getattr(self.rollout_fn, "evaluate_action", None)
        if not callable(evaluate):
            raise TypeError("rollout must be callable or expose objective/cost_fn/evaluate_action")
        result = evaluate(action)
        collection = result.costs_and_constraints
        cost = collection.get_sum_cost(sum_horizon=True, include_all_hybrid=False)
        cost = self._reduce_cost(cost, batch)
        constraint = None
        if constraints and getattr(collection, "constraints", None) is not None:
            raw = collection.get_sum_constraint(sum_horizon=True, include_all_hybrid=False)
            if raw is not None:
                # SciPy defines feasible inequalities as g(x) >= 0; cuRobo
                # stores positive violation costs, hence the sign inversion.
                constraint = -self._reduce_cost(raw, batch)
        return cost, constraint

    def _record(self, action: torch.Tensor, cost: torch.Tensor, best_action=None, best_cost=None):
        state = OptimizationIterationState(
            action=action.detach().clone(),
            cost=cost.detach().clone(),
            best_action=None if best_action is None else best_action.detach().clone(),
            best_cost=None if best_cost is None else best_cost.detach().clone(),
        )
        self._iteration_state = state
        if self._debug is not None:
            self._debug.record(state, self.action_horizon, self.action_dim)
        return state

    def _timed(self, callback):
        timer = CudaEventTimer().start()
        result = callback()
        self.opt_dt = timer.stop()
        return result

    def _reset_storage_for_action(self, action: torch.Tensor):
        """Allocate visible best-state buffers for an actual caller seed.

        A generic callable has no action-horizon metadata, so its trailing
        dimensions must come from the seed rather than the rollout protocol.
        """
        if self._optimization_variable is not None and self._optimization_variable.shape == action.shape:
            return False
        self.config.num_problems = int(action.shape[0]) if action.ndim else 1
        kwargs = {"device": self.device_cfg.device, "dtype": self.device_cfg.dtype}
        self._optimization_variable = torch.zeros_like(action, **kwargs, requires_grad=True)
        self.best_q = torch.zeros_like(action, **kwargs)
        self.best_cost = torch.full((self.config.num_problems,), 5.0e6, **kwargs)
        self.l_vec = torch.ones((self.config.num_problems,), **kwargs)
        return True

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        del action, clear_optimizer_state
        if mask is None and self._debug is not None:
            self._debug.clear()
        if reset_num_iters:
            self.config.num_iters = self._og_num_iters
        if self.best_cost is not None:
            self.best_cost.fill_(5.0e10)
        return True

    def shift(self, shift_steps=0):
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        return True

    _shift = shift

    def update_num_problems(self, num_problems):
        if num_problems <= 0:
            raise ValueError("num_problems must be positive")
        self.config.num_problems = int(num_problems)
        shape = (num_problems, self.action_horizon, self.action_dim)
        kwargs = {"device": self.device_cfg.device, "dtype": self.device_cfg.dtype}
        self._optimization_variable = torch.zeros(shape, **kwargs, requires_grad=True)
        self.best_cost = torch.full((num_problems,), 5.0e6, **kwargs)
        self.best_q = torch.zeros(shape, **kwargs)
        self.l_vec = torch.ones((num_problems,), **kwargs)
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_batch_size", None)
            if callable(callback):
                callback(batch_size=num_problems * int(self.config.num_particles or 1))

    def update_rollout_params(self, goal):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_params", None)
            if callable(callback):
                try:
                    callback(goal, num_particles=self.config.num_particles)
                except TypeError:
                    callback(goal)

    def update_goal_dt(self, goal):
        self._goal_dt = goal
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_goal_dt", None) or getattr(rollout, "update_dt", None)
            if callable(callback):
                callback(goal)

    def get_all_rollout_instances(self):
        return self._rollout_list

    def compute_metrics(self, action):
        callback = getattr(self.rollout_fn, "compute_metrics_from_action", None)
        return callback(action) if callable(callback) else None

    def reset_shape(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_shape", None)
            if callable(callback):
                callback()

    def reset_seed(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_seed", None)
            if callable(callback):
                callback()
        return True

    def reset_cuda_graph(self):
        # This is a lifecycle no-op, intentionally not an emulated CUDA graph.
        return None

    def get_recorded_trace(self):
        trace = [] if self._debug is None else self._debug.get_trace()
        return {"debug": trace, "debug_cost": [state.cost for state in trace]}

    def update_solver_params(self, solver_params):
        if self.config.solver_name not in solver_params:
            raise ValueError(f"Optimizer {self.config.solver_name} not found in {solver_params}")
        for key, value in solver_params[self.config.solver_name].items():
            if not hasattr(self.config, key):
                raise ValueError(f"unknown optimizer parameter: {key}")
            setattr(self.config, key, value)
        return True

    def update_niters(self, niters):
        self.config.update_niters(niters)

    def debug_dump(self, file_path=""):
        del file_path
        return self.get_recorded_trace()


__all__ = ["ExternalOptimizerBase", "UnsupportedExternalOptimizerFeature", "create_data_dict"]
