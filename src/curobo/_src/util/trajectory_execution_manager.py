"""Stateful, portable trajectory-command execution.

The pinned API was originally implemented around CUDA rollout buffers.  This
version deliberately keeps the same tensor lifecycle on CPU/MPS and does not
pretend that a CUDA graph is involved.
"""

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
    """Expose an interpolated trajectory one command at a time.

    ``interpolation_steps`` is the command window after every optimization
    update.  Action buffers are shifted by one action when callers prepare a
    warm-start; command progress only determines whether enough actions remain
    to make that operation valid, matching the pinned public lifecycle.
    """

    def __init__(
        self,
        interpolation_steps: int,
        command_start_idx: int = 0,
        command_end_idx: Optional[int] = None,
    ):
        if interpolation_steps < 1:
            raise ValueError("interpolation_steps must be positive")
        if command_start_idx < 0:
            raise ValueError("command_start_idx must be non-negative")
        if command_end_idx is not None and command_end_idx < command_start_idx:
            raise ValueError("command_end_idx must not precede command_start_idx")
        self._current_robot_state_trajectory: Optional[RobotState] = None
        self._current_joint_state_trajectory: Optional[JointState] = None
        self._current_action_trajectory: Optional[torch.Tensor] = None
        self._current_metrics: Optional[RolloutMetrics] = None
        self._current_command_idx = 0
        self.interpolation_steps = interpolation_steps
        self.command_start_idx = command_start_idx
        self.command_end_idx = (
            command_end_idx if command_end_idx is not None else interpolation_steps * 2
        )

    # Backward-compatible private aliases for the first portable release.
    @property
    def _state(self):
        return self._current_joint_state_trajectory

    @property
    def _actions(self):
        return self._current_action_trajectory

    @property
    def _metrics(self):
        return self._current_metrics

    @property
    def _robot_state(self):
        return self._current_robot_state_trajectory

    @property
    def _index(self):
        return self._current_command_idx

    def get_current_metrics(self) -> Optional[RolloutMetrics]:
        return self._current_metrics

    def update_robot_state_trajectory(self, robot_state_trajectory: RobotState):
        self._current_robot_state_trajectory = robot_state_trajectory

    def update_state_action_buffers(
        self, state_trajectory: JointState, action_trajectory: torch.Tensor
    ):
        if state_trajectory.position.shape[:-2] != action_trajectory.shape[:-2]:
            raise ValueError("state and action trajectories must have matching leading dimensions")
        self._current_joint_state_trajectory = state_trajectory
        self._current_action_trajectory = action_trajectory
        self._current_command_idx = 0
        self._current_metrics = None

    def update_state_action_metrics_buffers(
        self, state_trajectory: JointState, action_trajectory: torch.Tensor, metrics: RolloutMetrics
    ):
        self.update_state_action_buffers(state_trajectory, action_trajectory)
        self._current_metrics = metrics

    def get_next_command(self) -> JointState:
        if not self.has_valid_next_command():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        assert self._current_joint_state_trajectory is not None
        horizon_index = self.command_start_idx + self._current_command_idx
        command = get_joint_state_at_horizon_index(self._current_joint_state_trajectory, horizon_index)
        self._current_command_idx += 1
        return command

    def get_command_sequence(self) -> JointState:
        if not self.has_valiaction_dim_buffer() or self._current_joint_state_trajectory is None:
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        return trim_joint_state_trajectory(
            self._current_joint_state_trajectory,
            self.command_start_idx,
            self.command_end_idx,
        )

    def has_valid_next_command(self) -> bool:
        if not self.has_valiaction_dim_buffer() or self._current_joint_state_trajectory is None:
            return False
        horizon_index = self.command_start_idx + self._current_command_idx
        return (
            self._current_command_idx < self.interpolation_steps
            and horizon_index < self._current_joint_state_trajectory.position.shape[-2]
        )

    def has_valiaction_dim_buffer(self) -> bool:
        return self._current_action_trajectory is not None

    def get_action_buffer(self) -> torch.Tensor:
        if not self.has_valiaction_dim_buffer():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        assert self._current_action_trajectory is not None
        return self._current_action_trajectory

    def get_robot_state_sequence(self) -> RobotState:
        if not self.has_valid_robot_state_trajectory():
            log_and_raise("No valid robot state trajectory, call update_robot_state_trajectory first")
        assert self._current_robot_state_trajectory is not None
        return self._current_robot_state_trajectory

    def has_valid_robot_state_trajectory(self) -> bool:
        return self._current_robot_state_trajectory is not None

    def get_shifteaction_dim_buffer(self) -> torch.Tensor:
        """Return a one-action warm start, repeating the terminal command.

        The misspelled method name is part of the upstream public surface.
        """
        action = self.get_action_buffer()
        action_index = self._current_command_idx // self.interpolation_steps
        if action_index >= action.shape[-2]:
            log_and_raise("Action index out of bounds, call update_action_trajectory first")
        shifted = action.clone()
        if shifted.shape[-2] > 1:
            shifted = shifted.roll(-1, dims=-2)
            shifted[..., -1:, :] = shifted[..., -2:-1, :]
        return shifted

    def update_action_buffer(self, action_buffer: torch.Tensor):
        if action_buffer.ndim < 2:
            raise ValueError("action_buffer must have an action and dof dimension")
        self._current_action_trajectory = action_buffer
        self._current_command_idx = 0
        self._current_joint_state_trajectory = None


__all__ = ["TrajectoryExecutionManager"]
