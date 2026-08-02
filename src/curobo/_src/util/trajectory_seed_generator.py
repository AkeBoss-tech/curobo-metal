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
        # V2 documents a singleton start state as valid for a batch of goal
        # sets. Expanding it preserves autograd and MPS residency.
        if start_position.shape[0] == 1 and goal_position.shape[0] != 1:
            start_position = start_position.expand(goal_position.shape[0], -1)
        starts = start_position[:, None].expand(-1, num_seeds, -1)
        return self._interpolate_trajectory(starts, goal_position)

    def _interpolate_trajectory(
        self, start_position_seeds: torch.Tensor, goal_position_seeds: torch.Tensor
    ) -> torch.Tensor:
        if start_position_seeds.shape != goal_position_seeds.shape or start_position_seeds.ndim != 3:
            raise ValueError("start_position_seeds and goal_position_seeds must both be [B,N,J]")
        if start_position_seeds.shape[-1] != self.action_dim:
            raise ValueError("start_position_seeds and goal_position_seeds must end in action_dim")
        if start_position_seeds.device != goal_position_seeds.device:
            raise ValueError("start_position_seeds and goal_position_seeds must share a device")
        if start_position_seeds.dtype != goal_position_seeds.dtype:
            raise ValueError("start_position_seeds and goal_position_seeds must share a dtype")
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
        elif value.ndim == 2 and value.shape == (position.shape[0], 1):
            # State/trajectory code commonly represents a per-problem scalar
            # dt as [B, 1]. It is unambiguous for seed integration.
            value = value[:, 0]
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
        """Return a discrete, velocity-integrating deceleration schedule.

        The first ``action_horizon - 1`` values are interval accelerations;
        the final zero preserves V2's historical ``[B,H,J]`` layout. Every
        moving joint integrates to rest at a knot, including with per-problem
        ``dt``. CUDA's graph-captured rollout buffer is not emulated here.
        """
        if current_vel.ndim != 2 or current_acc.shape != current_vel.shape:
            raise ValueError("current_vel and current_acc must both be [B,J]")
        if dt.ndim != 1 or dt.numel() != current_vel.shape[0]:
            raise ValueError("dt must be [B]")
        if deceleration_profile not in ("linear", "exponential", "smooth"):
            raise ValueError("deceleration_profile must be linear, exponential, or smooth")

        intervals = self.action_horizon - 1
        profile = self._profile_steps(intervals, deceleration_profile).to(current_vel)
        batch_size = current_vel.shape[0]
        full_duration = dt * intervals
        if deceleration_time is None:
            duration = full_duration
        else:
            requested = torch.as_tensor(
                deceleration_time, device=current_vel.device, dtype=current_vel.dtype
            )
            if requested.ndim == 0:
                requested = requested.expand(batch_size)
            if requested.ndim != 1 or requested.numel() not in (1, batch_size):
                raise ValueError("deceleration_time must be a positive scalar or [batch]")
            if requested.numel() == 1:
                requested = requested.expand(batch_size)
            if bool((requested <= 0).any().item()):
                raise ValueError("deceleration_time must be positive")
            duration = torch.minimum(requested, full_duration)

        # Fractional final intervals support a requested duration that is not
        # a multiple of dt. The correction keeps the discrete integral exact.
        elapsed = torch.arange(intervals, device=current_vel.device, dtype=current_vel.dtype)
        elapsed = elapsed[None, :] * dt[:, None]
        active_fraction = ((duration[:, None] - elapsed) / dt[:, None]).clamp(0, 1)
        weights = profile[None, :] * active_fraction
        weighted_dt = (weights * dt[:, None]).sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(current_vel.dtype).eps
        )
        desired = -current_vel[:, None, :] * weights[:, :, None] / weighted_dt[:, :, None]

        # Fade an observed acceleration into the target profile, then project
        # it so the exact discrete integral cancels the incoming velocity.
        blend = torch.linspace(0, 1, intervals, device=current_vel.device, dtype=current_vel.dtype)
        candidate = desired * blend[None, :, None] + current_acc[:, None, :] * (
            1 - blend
        )[None, :, None]
        candidate = candidate * active_fraction[:, :, None]
        correction = (candidate * dt[:, None, None]).sum(dim=1) + current_vel
        candidate = candidate - correction[:, None, :] * weights[:, :, None] / weighted_dt[:, :, None]
        moving = current_vel.abs() > torch.finfo(current_vel.dtype).eps
        candidate = torch.where(moving[:, None, :], candidate, torch.zeros_like(candidate))
        return torch.cat((candidate, torch.zeros_like(candidate[:, :1])), dim=1)

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
        if start_position.shape[0] not in (1, goal_position.shape[0]) or start_position.shape[-1] != self.action_dim:
            raise ValueError("start and goal batch/dof dimensions are invalid")
        if goal_position.shape[1:] != (num_seeds, self.action_dim):
            raise ValueError("goal_position must have shape [B,num_seeds,action_dim]")
        if start_position.device != goal_position.device or start_position.dtype != goal_position.dtype:
            raise ValueError("start_position and goal_position must share device and dtype")

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
        for name in ("velocity", "acceleration"):
            value = getattr(current_state, name, None)
            if value is not None and (
                value.shape != current_state.position.shape
                or value.device != current_state.position.device
                or value.dtype != current_state.position.dtype
            ):
                raise ValueError(f"current_state.{name} must match position shape, device, and dtype")


__all__ = ["TrajectorySeedGenerator", "interpolate_kernel"]
