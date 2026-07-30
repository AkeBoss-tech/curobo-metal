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


__all__ = ["MultiStageOptimizer"]
