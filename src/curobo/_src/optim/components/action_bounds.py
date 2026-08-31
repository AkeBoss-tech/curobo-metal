"""Action-bound expansion used by portable optimizers.

The CUDA implementation stores flattened, horizon-tiled buffers because its
kernels consume packed actions.  Keeping those names here makes callers that
use the public component directly work on CPU and MPS too; matrix-shaped
aliases are retained for the portable optimizers in this package.
"""

from __future__ import annotations

from typing import Optional

import torch


class ActionBounds:
    """Cache validated per-action and horizon-expanded bounds.

    ``lows`` and ``highs`` must be one-dimensional, finite tensors on the
    same device.  A zero ``step_scale`` is useful for callers that deliberately
    disable line-search movement, so it is accepted.  Negative values are not
    meaningful and are rejected rather than silently reversing a limit.
    """

    def __init__(
        self,
        action_bound_lows: torch.Tensor,
        action_bound_highs: torch.Tensor,
        action_horizon: int,
        step_scale: float,
    ):
        self._action_horizon = 0
        self._step_scale = self._validate_scale(step_scale)
        self.lows: torch.Tensor
        self.highs: torch.Tensor
        self._compute(action_bound_lows, action_bound_highs, action_horizon, self._step_scale)

    @staticmethod
    def _validate_scale(value: float) -> float:
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value < 0.0:
            raise ValueError("step_scale must be finite and nonnegative")
        return value

    @staticmethod
    def _validate(lows: torch.Tensor, highs: torch.Tensor, action_horizon: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not isinstance(lows, torch.Tensor) or not isinstance(highs, torch.Tensor):
            raise TypeError("action bounds must be torch.Tensor values")
        if lows.ndim != 1 or highs.ndim != 1 or lows.shape != highs.shape:
            raise ValueError("action bounds must have matching shape [action_dim]")
        if lows.numel() == 0:
            raise ValueError("action bounds must contain at least one action dimension")
        if lows.device != highs.device:
            raise ValueError("action bound lows and highs must be on the same device")
        if lows.dtype != highs.dtype:
            raise TypeError("action bound lows and highs must have the same dtype")
        if not lows.is_floating_point():
            raise TypeError("action bounds must use a floating-point dtype")
        if not bool(torch.isfinite(lows).all()) or not bool(torch.isfinite(highs).all()):
            raise ValueError("action bounds must be finite")
        if bool((lows > highs).any()):
            raise ValueError("action bounds must satisfy lows <= highs")
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        return lows, highs, action_horizon

    def _compute(
        self,
        lows: torch.Tensor,
        highs: torch.Tensor,
        action_horizon: int,
        step_scale: float,
    ) -> "ActionBounds":
        lows, highs, action_horizon = self._validate(lows, highs, action_horizon)
        self._action_horizon = action_horizon
        self._step_scale = self._validate_scale(step_scale)
        self.lows, self.highs = lows, highs

        # CUDA-facing V2 names are flattened.  ``expand`` followed by
        # ``reshape`` materializes a contiguous tensor on CPU and MPS.
        tiled_lows = lows.unsqueeze(0).expand(action_horizon, -1).clone()
        tiled_highs = highs.unsqueeze(0).expand(action_horizon, -1).clone()
        self.horizon_lows = tiled_lows.reshape(-1)
        self.horizon_highs = tiled_highs.reshape(-1)
        self.step_max = self._step_scale * (highs - lows).abs()
        self.horizon_step_max = self._step_scale * (self.horizon_highs - self.horizon_lows).abs()

        # Earlier portable code exposed rank-two variants.  Keep those
        # aliases without making users care which execution backend is used.
        self.action_bound_lows = lows
        self.action_bound_highs = highs
        self.action_horizon_bounds_lows = tiled_lows
        self.action_horizon_bounds_highs = tiled_highs
        self.action_step_max = self.step_max
        self.action_horizon_step_max = self.horizon_step_max.reshape(action_horizon, -1)
        return self

    def refresh(
        self,
        action_bound_lows: torch.Tensor,
        action_bound_highs: torch.Tensor,
        action_horizon: int,
    ):
        """Refresh cached values after a rollout changes its bounds or horizon.

        Upstream refreshes only on a horizon change.  Refreshing changed bound
        values as well avoids stale limits for mutable portable rollouts while
        preserving the same result for immutable inputs.
        """
        self._compute(
            action_bound_lows,
            action_bound_highs,
            action_horizon,
            self._step_scale,
        )


__all__ = ["ActionBounds"]
