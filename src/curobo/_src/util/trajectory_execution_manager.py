"""Stateful trajectory command buffer compatible with cuRoboV2."""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import (
    get_joint_state_at_horizon_index,
    trim_joint_state_trajectory,
)
from curobo._src.state.state_robot import RobotState
from curobo._src.util.logging import log_and_raise


class TrajectoryExecutionManager:
    def __init__(
        self, interpolation_steps: int, command_start_idx: int = 0,
        command_end_idx: Optional[int] = None,
    ):
        if interpolation_steps < 1:
            raise ValueError("interpolation_steps must be positive")
        self.interpolation_steps = interpolation_steps
        self.command_start_idx = command_start_idx
        self.command_end_idx = command_end_idx if command_end_idx is not None else interpolation_steps * 2
        self._index = 0
        self._state = self._actions = self._metrics = self._robot_state = None

    def get_current_metrics(self):
        return self._metrics

    def update_robot_state_trajectory(self, robot_state_trajectory: RobotState):
        self._robot_state = robot_state_trajectory

    def update_state_action_buffers(self, state_trajectory: JointState, action_trajectory: torch.Tensor):
        self._state, self._actions = state_trajectory, action_trajectory
        self._index = 0
        self._metrics = None

    def update_state_action_metrics_buffers(
        self, state_trajectory: JointState, action_trajectory: torch.Tensor, metrics: RolloutMetrics,
    ):
        self.update_state_action_buffers(state_trajectory, action_trajectory)
        self._metrics = metrics

    def get_next_command(self) -> JointState:
        if not self.has_valid_next_command():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        command = get_joint_state_at_horizon_index(
            self._state, self.command_start_idx + self._index
        )
        self._index += 1
        return command

    def get_command_sequence(self) -> torch.Tensor:
        if self._state is None:
            raise ValueError("state trajectory is not initialized")
        return trim_joint_state_trajectory(self._state, self.command_start_idx, self.command_end_idx)

    def has_valid_next_command(self) -> bool:
        return self._actions is not None and self._index < self.interpolation_steps

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
        self._index = 0
        self._state = None


__all__ = ["TrajectoryExecutionManager"]
