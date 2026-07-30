"""Stateful trajectory command buffer compatible with cuRoboV2."""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.state.state_joint import JointState


class TrajectoryExecutionManager:
    def __init__(
        self, interpolation_steps: int, command_start_idx: int = 0,
        command_end_idx: Optional[int] = None,
    ):
        if interpolation_steps < 1:
            raise ValueError("interpolation_steps must be positive")
        self.interpolation_steps = interpolation_steps
        self.command_start_idx = command_start_idx
        self.command_end_idx = command_end_idx or interpolation_steps
        self._index = command_start_idx
        self._state = self._actions = self._metrics = self._robot_state = None

    def get_current_metrics(self):
        return self._metrics

    def update_robot_state_trajectory(self, robot_state_trajectory):
        self._robot_state = robot_state_trajectory
        self._index = self.command_start_idx

    def update_state_action_buffers(self, state_trajectory, action_trajectory):
        self._state, self._actions = state_trajectory, action_trajectory
        self._index = self.command_start_idx

    def update_state_action_metrics_buffers(
        self, state_trajectory, action_trajectory, metrics,
    ):
        self.update_state_action_buffers(state_trajectory, action_trajectory)
        self._metrics = metrics

    def get_next_command(self) -> JointState:
        if not self.has_valid_next_command():
            raise IndexError("trajectory command buffer is exhausted")
        command = self._state[..., self._index, :]
        self._index += 1
        return command

    def get_command_sequence(self) -> torch.Tensor:
        if self._state is None:
            raise ValueError("state trajectory is not initialized")
        return self._state.position[..., self.command_start_idx:self.command_end_idx, :]

    def has_valid_next_command(self) -> bool:
        return self._state is not None and self._index < min(
            self.command_end_idx, self._state.position.shape[-2]
        )

    def has_valiaction_dim_buffer(self) -> bool:
        return self._actions is not None

    def get_action_buffer(self) -> torch.Tensor:
        if self._actions is None:
            raise ValueError("action buffer is not initialized")
        return self._actions

    def get_robot_state_sequence(self):
        if self._robot_state is None:
            raise ValueError("robot state trajectory is not initialized")
        return self._robot_state

    def has_valid_robot_state_trajectory(self) -> bool:
        return self._robot_state is not None

    def get_shifteaction_dim_buffer(self) -> torch.Tensor:
        action = self.get_action_buffer()
        return torch.cat((action[..., 1:, :], action[..., -1:, :]), dim=-2)

    def update_action_buffer(self, action_buffer: torch.Tensor):
        self._actions = action_buffer


__all__ = ["TrajectoryExecutionManager"]
