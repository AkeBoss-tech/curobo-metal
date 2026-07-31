"""Action bound expansion and validation."""
import torch

class ActionBounds:
    def __init__(self, action_bound_lows, action_bound_highs, action_horizon, step_scale=1.0):
        self._compute(action_bound_lows, action_bound_highs, action_horizon, step_scale)

    def _compute(self, lows: torch.Tensor, highs: torch.Tensor, action_horizon: int, step_scale: float):
        if lows.shape != highs.shape or bool((lows > highs).any()):
            raise ValueError("action bounds must have equal shapes and lows <= highs")
        self.action_bound_lows, self.action_bound_highs = lows, highs
        self.action_horizon_bounds_lows = lows.expand(action_horizon, -1).clone()
        self.action_horizon_bounds_highs = highs.expand(action_horizon, -1).clone()
        self.action_step_max = (highs - lows) * step_scale
        self.action_horizon_step_max = self.action_step_max.expand(action_horizon, -1).clone()
        return self

    def refresh(self, action_bound_lows, action_bound_highs, action_horizon):
        return self._compute(action_bound_lows, action_bound_highs, action_horizon, 1.0)
