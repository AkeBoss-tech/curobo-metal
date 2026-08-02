"""Deterministic, differentiable seed generation for portable TrajOpt."""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def interpolate_kernel(h, int_steps, device_cfg: DeviceCfg):
    """Create the pinned piecewise-linear interpolation stencil on ``device_cfg``."""
    if h < 2 or int_steps < 2:
        raise ValueError("h and int_steps must be >= 2")
    delta = torch.linspace(0, 1, int_steps, **device_cfg.as_torch_dict())
    matrix = torch.zeros(((h - 1) * int_steps, h), **device_cfg.as_torch_dict())
    for index in range(h - 1):
        segment = slice(index * int_steps, (index + 1) * int_steps)
        matrix[segment, index] = 1 - delta
        matrix[segment, index + 1] = delta
    return matrix


class TrajectorySeedGenerator:
    def __init__(self, action_horizon: int, action_dim: int, device_cfg: DeviceCfg):
        if action_horizon < 2 or action_dim < 1:
            raise ValueError("action_horizon must be >= 2 and action_dim must be positive")
        self.device_cfg = device_cfg
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self._interpolation_weights = interpolate_kernel(2, action_horizon, device_cfg).reshape(
            1, action_horizon, 2, 1
        )

    def generate_constant_seeds(self, constant_position: torch.Tensor, num_seeds: int):
        self._validate_constant_inputs(constant_position, num_seeds)
        return constant_position[:, None, None].expand(
            -1, num_seeds, self.action_horizon, -1
        ).clone()

    def generate_interpolated_seeds(
        self, start_position: torch.Tensor, goal_position: torch.Tensor, num_seeds: int
    ):
        self._validate_interpolation_inputs(start_position, goal_position, num_seeds)
        starts = start_position[:, None].expand(-1, num_seeds, -1)
        return self._interpolate_trajectory(starts, goal_position)

    def _interpolate_trajectory(
        self, start_position_seeds: torch.Tensor, goal_position_seeds: torch.Tensor
    ) -> torch.Tensor:
        if start_position_seeds.shape != goal_position_seeds.shape or start_position_seeds.ndim != 3:
            raise ValueError("start_position_seeds and goal_position_seeds must both be [B,N,J]")
        weights = self._interpolation_weights.to(
            device=start_position_seeds.device, dtype=start_position_seeds.dtype
        )
        return (
            weights[:, :, 0] * start_position_seeds.reshape(-1, 1, self.action_dim)
            + weights[:, :, 1] * goal_position_seeds.reshape(-1, 1, self.action_dim)
        ).reshape(
            start_position_seeds.shape[0], start_position_seeds.shape[1],
            self.action_horizon, self.action_dim,
        )

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
        position = current_state.position
        velocity = current_state.velocity if current_state.velocity is not None else torch.zeros_like(position)
        acceleration = (
            current_state.acceleration
            if current_state.acceleration is not None
            else torch.zeros_like(position)
        )
        dt = self._state_dt(current_state, position)
        acceleration_profile = self._generate_deceleration_acceleration_profile(
            velocity, acceleration, deceleration_profile, dt, deceleration_time
        )
        trajectory = self._integrate_acceleration_to_trajectory(
            position, velocity, acceleration_profile, dt
        )
        return trajectory[:, None].expand(-1, num_seeds, -1, -1).clone()

    def _state_dt(self, current_state: JointState, position: torch.Tensor) -> torch.Tensor:
        value = current_state.dt
        if value is None:
            return torch.full((position.shape[0],), 0.02, device=position.device, dtype=position.dtype)
        value = torch.as_tensor(value, device=position.device, dtype=position.dtype)
        if value.ndim == 0:
            value = value.expand(position.shape[0])
        if value.ndim != 1 or value.numel() not in (1, position.shape[0]):
            raise ValueError("current_state.dt must be scalar or [batch]")
        if value.numel() == 1:
            value = value.expand(position.shape[0])
        if bool((value <= 0).any().item()):
            raise ValueError("current_state.dt must be positive")
        return value

    def _generate_linear_deceleration_profile(self, decel_steps: int) -> torch.Tensor:
        return self._profile_steps(decel_steps, "linear")

    def _generate_exponential_deceleration_profile(self, decel_steps: int) -> torch.Tensor:
        return self._profile_steps(decel_steps, "exponential")

    def _generate_smooth_deceleration_profile(self, decel_steps: int) -> torch.Tensor:
        return self._profile_steps(decel_steps, "smooth")

    def _profile_steps(self, decel_steps: int, profile: str) -> torch.Tensor:
        if decel_steps <= 0:
            return torch.empty(0, **self.device_cfg.as_torch_dict())
        phase = torch.linspace(0, 1, decel_steps, **self.device_cfg.as_torch_dict())
        if profile == "linear":
            return 1 - phase
        if profile == "smooth":
            return 0.5 * (1 + torch.cos(torch.pi * phase))
        return torch.exp(-3 * phase)

    def _generate_deceleration_acceleration_profile(
        self,
        current_vel: torch.Tensor,
        current_acc: torch.Tensor,
        deceleration_profile: str,
        dt: torch.Tensor,
        deceleration_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Produce bounded acceleration that brings every moving joint to rest.

        A normalized profile distributes the required velocity reduction over
        the requested horizon.  Direction clipping in the integrator prevents
        an overshoot from reversing a joint.
        """
        profile = self._profile_steps(self.action_horizon - 1, deceleration_profile).to(
            current_vel
        )
        if deceleration_time is not None:
            if deceleration_time <= 0:
                raise ValueError("deceleration_time must be positive")
            duration = torch.full_like(dt, float(deceleration_time))
        else:
            duration = dt * (self.action_horizon - 1)
        # Per-batch normalized weights integrate to exactly -current_vel.
        weighted_dt = (profile[None, :] * dt[:, None]).sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(current_vel.dtype).eps
        )
        desired = -current_vel[:, None, :] * profile[None, :, None] / weighted_dt[:, :, None]
        # Respect a nonzero initial acceleration without introducing a jump;
        # the correction is then redistributed so its integral stays exact.
        blend = torch.linspace(0, 1, profile.numel(), device=current_vel.device, dtype=current_vel.dtype)
        candidate = current_acc[:, None, :] * (1 - blend)[None, :, None] + desired * blend[None, :, None]
        correction = (candidate * dt[:, None, None]).sum(dim=1, keepdim=True) + current_vel[:, None, :]
        candidate = candidate - correction / duration[:, None, None].clamp_min(
            torch.finfo(current_vel.dtype).eps
        )
        return torch.cat((candidate[:, :1], candidate), dim=1)

    def _integrate_acceleration_to_trajectory(
        self,
        current_pos: torch.Tensor,
        current_vel: torch.Tensor,
        acceleration_profile: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        positions = [current_pos]
        velocity = current_vel
        initial_sign = torch.sign(current_vel)
        for step in range(1, self.action_horizon):
            positions.append(positions[-1] + velocity * dt[:, None])
            next_velocity = velocity + acceleration_profile[:, step - 1] * dt[:, None]
            reversed_direction = (initial_sign != 0) & (torch.sign(next_velocity) != initial_sign)
            velocity = torch.where(reversed_direction, torch.zeros_like(next_velocity), next_velocity)
        return torch.stack(positions, dim=1)

    def _validate_interpolation_inputs(self, start_position, goal_position, num_seeds):
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")
        if start_position.ndim != 2 or goal_position.ndim != 3:
            raise ValueError("expected start [B,J] and goal [B,N,J]")
        if start_position.shape[0] != goal_position.shape[0] or start_position.shape[-1] != self.action_dim:
            raise ValueError("start and goal batch/dof dimensions are invalid")
        if goal_position.shape[1:] != (num_seeds, self.action_dim):
            raise ValueError("goal_position must have shape [B,num_seeds,action_dim]")

    def _validate_constant_inputs(self, constant_position, num_seeds):
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")
        if constant_position.ndim != 2 or constant_position.shape[-1] != self.action_dim:
            raise ValueError("constant_position must have shape [B,J]")

    def _validate_deceleration_inputs(self, current_state, num_seeds):
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")
        if current_state.position.ndim != 2 or current_state.position.shape[-1] != self.action_dim:
            raise ValueError("current_state.position must have shape [B,J]")


__all__ = ["TrajectorySeedGenerator", "interpolate_kernel"]
