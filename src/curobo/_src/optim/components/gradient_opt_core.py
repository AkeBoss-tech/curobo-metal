"""Portable common lifecycle for V2 gradient optimizer facades."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from curobo._src.optim._portable import PortableOptimizer, _objective


class GradientOptCore(PortableOptimizer):
    """Tensor/autograd substitute for the V2 line-search core.

    It preserves callbacks and state/lifecycle operations used by the public
    L-BFGS/LSR1/CG facades.  CUDA graph executors and raw CUDA line-search
    kernels remain intentionally outside the portable contract.
    """

    _graphable_methods = {"_opt_iters", "_prepare_initial_iteration_state"}

    def __init__(
        self, config, rollout_list, step_direction_fn: Callable,
        *, on_reinitialize: Optional[Callable] = None,
        on_initial_state: Optional[Callable] = None,
        on_resize: Optional[Callable] = None,
        on_shift: Optional[Callable] = None,
        use_cuda_graph: bool = False,
    ):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._step_direction_fn = step_direction_fn
        self._on_reinitialize = on_reinitialize
        self._on_initial_state = on_initial_state
        self._on_resize = on_resize
        self._on_shift = on_shift
        self._executors = {}
        self._iteration_state = None
        self._og_num_iters = config.num_iters

    def finish_init(self):
        return self
    @property
    def horizon(self): return self.action_horizon
    @property
    def solver_names(self): return [self.config.solver_name]
    def get_all_rollout_instances(self): return self._rollout_list
    def compute_metrics(self, action):
        fn = getattr(self.rollout_fn, "compute_metrics_from_action", None)
        if not callable(fn):
            fn = getattr(self.rollout_fn, "compute_metrics", None)
        return fn(action) if callable(fn) else None
    def reset_shape(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_shape", None)
            if callable(callback):
                callback()
    def reset_seed(self): return True
    def reset_cuda_graph(self): return None
    def get_recorded_trace(self): return self.debug if self.debug is not None else {"debug": [], "debug_cost": []}
    def update_niters(self, niters): self.config.update_niters(niters)
    def update_solver_params(self,solver_params):
        for k,v in solver_params.get(self.config.solver_name,{}).items(): setattr(self.config,k,v)
    def update_goal_dt(self, goal):
        return super().update_goal_dt(goal)
    def update_num_problems(self, num_problems):
        super().update_num_problems(num_problems)
        if self._on_resize:
            self._on_resize(num_problems)
    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        super().reinitialize(action, mask, clear_optimizer_state, reset_num_iters)
        if self._on_reinitialize:
            self._on_reinitialize(mask)
        if self._on_initial_state:
            self._on_initial_state(action, mask)
    def shift(self, shift_steps=0):
        result = super().shift(shift_steps)
        if shift_steps and self._on_shift:
            self._on_shift(shift_steps)
        return result
    _shift = shift
    def debug_dump(self, file_path=""):
        del file_path
        return None
