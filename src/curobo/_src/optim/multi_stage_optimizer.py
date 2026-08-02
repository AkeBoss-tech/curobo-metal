"""Sequential optimizer composition compatible with cuRoboV2."""

from typing import List, Optional


class MultiStageOptimizer:
    def __init__(self, optimizers: List, rollout_list: Optional[List] = None):
        if not optimizers:
            raise ValueError("optimizers must not be empty")
        self.optimizers = list(optimizers)
        self._rollout_list = rollout_list or self.optimizers[-1]._rollout_list
        self.rollout_fn = self._rollout_list[0]
        self.config = self.optimizers[-1].config
        self.device_cfg = self.config.device_cfg
        self._enabled = True
        self.opt_dt = 0.0

    @property
    def enabled(self): return self._enabled
    def enable(self): self._enabled = True
    def disable(self): self._enabled = False
    @property
    def action_horizon(self): return self.optimizers[-1].action_horizon
    @property
    def action_dim(self): return self.optimizers[-1].action_dim
    @property
    def opt_dim(self): return self.action_horizon * self.action_dim
    @property
    def outer_iters(self): return 1
    @property
    def solver_names(self): return [x.config.solver_name for x in self.optimizers]
    @property
    def solve_time(self): return self.opt_dt

    def optimize(self, seed_action):
        value = seed_action
        if self.enabled:
            for optimizer in self.optimizers:
                value = optimizer.optimize(value)
        return value

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        for optimizer in self.optimizers:
            optimizer.reinitialize(action, mask, clear_optimizer_state, reset_num_iters)

    def shift(self, shift_steps=0):
        return all(optimizer.shift(shift_steps) for optimizer in self.optimizers)
    _shift = shift

    def update_num_problems(self, num_problems):
        for optimizer in self.optimizers:
            optimizer.update_num_problems(num_problems)

    def update_rollout_params(self, goal):
        for optimizer in self.optimizers:
            optimizer.update_rollout_params(goal)

    def reset(self):
        for optimizer in self.optimizers:
            optimizer.reset()

    def get_all_rollout_instances(self):
        return self._rollout_list

    def compute_metrics(self, action):
        callback = getattr(self.rollout_fn, "compute_metrics", None)
        return callback(action) if callable(callback) else None

    def reset_shape(self):
        for optimizer in self.optimizers:
            callback = getattr(optimizer, "reset_shape", optimizer.reset)
            callback()

    def reset_seed(self):
        for optimizer in self.optimizers:
            callback = getattr(optimizer, "reset_seed", optimizer.reset)
            callback()

    def reset_cuda_graph(self):
        # No graph capture exists on CPU/MPS; this resets the portable cache.
        self.reset()

    def get_recorded_trace(self):
        return [getattr(optimizer, "get_recorded_trace", lambda: None)() for optimizer in self.optimizers]

    def update_niters(self, niters):
        for optimizer in self.optimizers:
            callback = getattr(optimizer, "update_niters", None)
            if callable(callback):
                callback(niters)

    def update_solver_params(self, solver_params):
        for optimizer in self.optimizers:
            callback = getattr(optimizer, "update_solver_params", None)
            if callable(callback):
                callback(solver_params)

    def update_goal_dt(self, goal_dt):
        for optimizer in self.optimizers:
            callback = getattr(optimizer, "update_goal_dt", None)
            if callable(callback):
                callback(goal_dt)

    def debug_dump(self, file_path=""):
        raise NotImplementedError("portable optimizer debug serialization is unavailable")


__all__ = ["MultiStageOptimizer"]
