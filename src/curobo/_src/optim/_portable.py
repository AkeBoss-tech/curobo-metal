"""Portable optimizer compatibility core for the pinned cuRoboV2 API."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Callable, Optional

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo_metal.optim import (
    ExecutionCache,
    LBFGSConfig,
    ParticleConfig,
    lbfgs_optimize,
    particle_optimize,
)


class UnsupportedOptimizerFeature(NotImplementedError):
    """Raised when a CUDA-specific optimizer feature has no portable meaning."""


@dataclass
class PortableOptCfg:
    num_iters: int = 100
    solver_type: str = "portable"
    solver_name: str = "portable"
    device_cfg: DeviceCfg = DeviceCfg()
    store_debug: bool = False
    debug_info: Any = None
    num_problems: int = 1
    num_particles: Optional[int] = None
    sync_cuda_time: bool = True
    use_coo_sparse: bool = True
    step_scale: float = 1.0
    inner_iters: int = 1
    _num_rollout_instances: int = 1

    @property
    def num_rollout_instances(self) -> int:
        return self._num_rollout_instances

    @property
    def outer_iters(self) -> int:
        return math.ceil(self.num_iters / self.inner_iters)

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        values = dict(data_dict if child_dict is None else child_dict)
        values["device_cfg"] = device_cfg
        valid = {item.name for item in fields(cls)}
        return {key: value for key, value in values.items() if key in valid}

    def update_niters(self, niters: int):
        if niters <= 0:
            raise ValueError("niters must be positive")
        self.num_iters = niters


def _objective(rollout: object) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(rollout):
        return rollout
    for name in ("objective", "cost_fn"):
        value = getattr(rollout, name, None)
        if callable(value):
            return value
    evaluate_action = getattr(rollout, "evaluate_action", None)
    if callable(evaluate_action):
        def evaluate_objective(action: torch.Tensor) -> torch.Tensor:
            trajectories = evaluate_action(action)
            costs_and_constraints = getattr(trajectories, "costs_and_constraints", None)
            if costs_and_constraints is None:
                raise TypeError("rollout evaluate_action result has no costs_and_constraints")
            if getattr(costs_and_constraints, "constraints", None) is None:
                value = costs_and_constraints.get_sum_cost(sum_horizon=True)
            else:
                value = costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
            if value is None:
                raise ValueError("rollout produced no costs or constraints")
            return value

        return evaluate_objective
    raise TypeError("portable optimizer rollout must be callable or expose objective/cost_fn")


class PortableOptimizer:
    """Shared cuRobo lifecycle over device-resident portable optimizer functions."""

    strategy = "lbfgs"

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        if use_cuda_graph:
            raise UnsupportedOptimizerFeature(
                "CUDA Graph capture is unavailable on CPU/MPS; persistent shape caches are used"
            )
        if not rollout_list:
            raise ValueError("rollout_list must contain at least one rollout")
        self.config = config
        self.device_cfg = config.device_cfg
        self._rollout_list = list(rollout_list)
        self.rollout_fn = self._rollout_list[0]
        self._enabled = True
        # Keep the upstream capability declaration available for callers that
        # inspect optimizer structure.  MPS executes these methods eagerly;
        # the declaration does not claim that a CUDA graph was captured.
        self._graphable_methods = {"_opt_iters"}
        # Keep this state visible because callers use it when deciding whether
        # an optimizer may be re-used.  It deliberately remains false on the
        # portable backend: an ExecutionCache is not a CUDA graph.
        self.use_cuda_graph = False
        self._cache = ExecutionCache()
        self.opt_dt = 0.0
        self.debug = None
        self._goal_dt = None

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def action_horizon(self):
        return getattr(self.rollout_fn, "action_horizon", 1)

    @property
    def action_dim(self):
        return getattr(self.rollout_fn, "action_dim", 1)

    @property
    def opt_dim(self):
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self):
        return self.config.outer_iters

    @property
    def solve_time(self):
        return self.opt_dt

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        objective = _objective(self.rollout_fn)
        if self.strategy == "lbfgs":
            result = lbfgs_optimize(
                objective,
                seed_action,
                config=LBFGSConfig(
                    iterations=self.config.num_iters,
                    history_size=self.config.history,
                    learning_rate=self.config.step_scale,
                    tolerance_change=max(self.config.cost_convergence, 0.0),
                    line_search=self.config.portable_line_search,
                    record_debug=self.config.store_debug,
                ),
                event_ndim=2 if seed_action.ndim >= 3 else 1,
                cache=self._cache,
                warm_start=True,
            )
        else:
            result = particle_optimize(
                objective,
                seed_action,
                config=ParticleConfig(
                    iterations=self.config.num_iters,
                    particles=self.config.num_particles,
                    elite_count=max(1, min(self.config.num_particles, self.config.num_particles // 4)),
                    initial_std=float(self.config.init_cov),
                    seed=self.config.seed,
                    strategy=self.strategy,
                    temperature=self.config.beta,
                    record_debug=self.config.store_debug,
                ),
                event_ndim=2 if seed_action.ndim >= 3 else 1,
                cache=self._cache,
            )
        self.debug = result.debug
        return result.solution

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        del action, mask, reset_num_iters
        if clear_optimizer_state:
            self._cache.reset()

    def reset(self):
        self._cache.reset()

    def update_num_problems(self, num_problems: int):
        if num_problems <= 0:
            raise ValueError("num_problems must be positive")
        self.config.num_problems = num_problems

    def shift(self, shift_steps: int = 0):
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        return True

    _shift = shift

    def update_rollout_params(self, goal):
        callback = getattr(self.rollout_fn, "update_params", None)
        if callback is not None:
            callback(goal)

    def update_goal_dt(self, goal_dt):
        """Propagate a changed rollout timestep without rebuilding state."""
        self._goal_dt = goal_dt
        callback = getattr(self.rollout_fn, "update_dt", None)
        if callable(callback):
            callback(goal_dt)

    # These are populated by cuRobo rollouts.  Keeping the attributes stable
    # makes optimizer setup code portable even for lightweight callable
    # rollouts that do not impose action bounds.
    @property
    def action_bound_lows(self):
        return getattr(self.rollout_fn, "action_bound_lows", None)

    @property
    def action_bound_highs(self):
        return getattr(self.rollout_fn, "action_bound_highs", None)

    @property
    def action_step_max(self):
        return getattr(self.rollout_fn, "action_step_max", None)

    @property
    def action_horizon_bounds_lows(self):
        return getattr(self.rollout_fn, "action_horizon_bounds_lows", None)

    @property
    def action_horizon_bounds_highs(self):
        return getattr(self.rollout_fn, "action_horizon_bounds_highs", None)

    @property
    def action_horizon_step_max(self):
        return getattr(self.rollout_fn, "action_horizon_step_max", None)

    def get_debug(self):
        return self.debug
