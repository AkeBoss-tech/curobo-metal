"""Deterministic trajectory seed generation."""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


def interpolate_kernel(h, int_steps, device_cfg: DeviceCfg):
    if h < 2 or int_steps < 2:
        raise ValueError("h and int_steps must be >= 2")
    coordinate = torch.linspace(0, h - 1, (h - 1) * int_steps, **device_cfg.as_torch_dict())
    low = coordinate.floor().to(torch.int64).clamp_max(h - 2)
    fraction = coordinate - low
    matrix = torch.zeros((coordinate.numel(), h), **device_cfg.as_torch_dict())
    rows = torch.arange(coordinate.numel(), device=device_cfg.device)
    matrix[rows, low] = 1 - fraction
    matrix[rows, low + 1] = fraction
    return matrix


class TrajectorySeedGenerator:
    def __init__(self, action_horizon: int, action_dim: int, device_cfg: DeviceCfg):
        if action_horizon < 2 or action_dim < 1:
            raise ValueError("action_horizon must be >= 2 and action_dim must be positive")
        self.device_cfg = device_cfg
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def generate_constant_seeds(self, constant_position: torch.Tensor, num_seeds: int):
        self._validate_constant_inputs(constant_position, num_seeds)
        return constant_position[:, None, None].expand(
            -1, num_seeds, self.action_horizon, -1
        ).clone()

    def generate_interpolated_seeds(
        self, start_position: torch.Tensor, goal_position: torch.Tensor, num_seeds: int
    ):
        self._validate_interpolation_inputs(start_position, goal_position, num_seeds)
        phase = torch.linspace(
            0, 1, self.action_horizon, **self.device_cfg.as_torch_dict()
        ).view(1, 1, -1, 1)
        return start_position[:, None, None] * (1 - phase) + goal_position[:, :, None] * phase

    def generate_deceleration_seeds(
        self,
        current_state: JointState,
        num_seeds: int,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
    ) -> torch.Tensor:
        self._validate_deceleration_inputs(current_state, num_seeds)
        if deceleration_profile not in ("linear", "exponential", "smooth"):
            raise ValueError("deceleration_profile must be linear, exponential, or smooth")
        dt = float(current_state.dt.reshape(-1)[0]) if current_state.dt is not None else 0.02
        total = deceleration_time or dt * (self.action_horizon - 1)
        t = torch.linspace(0, total, self.action_horizon, **self.device_cfg.as_torch_dict())
        phase = t / max(total, torch.finfo(t.dtype).eps)
        if deceleration_profile == "linear":
            velocity_scale = 1 - phase
        elif deceleration_profile == "smooth":
            velocity_scale = 0.5 * (1 + torch.cos(torch.pi * phase))
        else:
            velocity_scale = (torch.exp(-3 * phase) - torch.exp(t.new_tensor(-3.0))) / (
                1 - torch.exp(t.new_tensor(-3.0))
            )
        velocity = current_state.velocity
        if velocity is None:
            velocity = torch.zeros_like(current_state.position)
        increments = velocity[:, None] * velocity_scale[None, :, None] * dt
        position = current_state.position[:, None] + torch.cumsum(increments, dim=1)
        position[:, 0] = current_state.position
        return position[:, None].expand(-1, num_seeds, -1, -1).clone()

    def _validate_interpolation_inputs(self, start_position, goal_position, num_seeds):
        if start_position.ndim != 2 or goal_position.shape != (
            start_position.shape[0], num_seeds, self.action_dim
        ):
            raise ValueError("expected start [B,J] and goal [B,N,J]")
        if start_position.shape[-1] != self.action_dim or num_seeds < 1:
            raise ValueError("seed dimensions are invalid")

    def _validate_constant_inputs(self, constant_position, num_seeds):
        if constant_position.ndim != 2 or constant_position.shape[-1] != self.action_dim:
            raise ValueError("constant_position must have shape [B,J]")
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")

    def _validate_deceleration_inputs(self, current_state, num_seeds):
        if current_state.position.ndim != 2 or current_state.position.shape[-1] != self.action_dim:
            raise ValueError("current_state.position must have shape [B,J]")
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")


__all__ = ["TrajectorySeedGenerator", "interpolate_kernel"]
