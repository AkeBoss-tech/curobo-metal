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

    @staticmethod
    def _validate_state_action_buffers(
        state_trajectory: JointState, action_trajectory: torch.Tensor
    ) -> None:
        """Fail before changing queue state when an update is not executable.

        V2's CUDA rollout normally constructs these buffers together, so shape
        mistakes otherwise surface later as an opaque device indexing error.
        The portable implementation accepts both ``[horizon, dof]`` and
        ``[..., horizon, dof]`` layouts, while retaining that common leading
        dimensions, device, and dtype are all required.  State and action
        widths deliberately need not match: rollouts can expose position state
        while optimizing a lower-dimensional control/action space.
        """
        if not isinstance(state_trajectory, JointState):
            raise TypeError("state_trajectory must be a JointState")
        if not isinstance(action_trajectory, torch.Tensor):
            raise TypeError("action_trajectory must be a torch.Tensor")
        state = state_trajectory.position
        if state.ndim < 2:
            raise ValueError("state_trajectory must have horizon and dof dimensions")
        if action_trajectory.ndim < 2:
            raise ValueError("action_trajectory must have action and dof dimensions")
        if state.shape[:-2] != action_trajectory.shape[:-2]:
            raise ValueError("state and action trajectories must have matching leading dimensions")
        if state.device != action_trajectory.device:
            raise ValueError("state and action trajectories must be on the same device")
        if state.dtype != action_trajectory.dtype:
            raise ValueError("state and action trajectories must have matching dtype")
        if state.shape[-2] < 1 or action_trajectory.shape[-2] < 1:
            raise ValueError("state and action trajectories must be non-empty")

    @property
    def command_index(self) -> int:
        """Number of commands consumed since the last state/action update."""
        return self._current_command_idx

    @property
    def remaining_commands(self) -> int:
        """Commands still executable from the currently exposed state window."""
        if not self.has_valiaction_dim_buffer() or self._current_joint_state_trajectory is None:
            return 0
        usable_horizon = max(
            0,
            self._current_joint_state_trajectory.position.shape[-2] - self.command_start_idx,
        )
        return max(0, min(self.interpolation_steps, usable_horizon) - self._current_command_idx)

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
        self._validate_state_action_buffers(state_trajectory, action_trajectory)
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
        command = self.peek_next_command()
        self._current_command_idx += 1
        return command

    def peek_next_command(self) -> JointState:
        """Return the next stream command without consuming it.

        This is useful for device-side robot control loops that must validate
        a command before committing it to an external actuator.  It has no
        CUDA-stream implication; returned tensors stay on their source CPU or
        MPS device and retain normal PyTorch autograd connectivity.
        """
        if not self.has_valid_next_command():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        assert self._current_joint_state_trajectory is not None
        horizon_index = self.command_start_idx + self._current_command_idx
        return get_joint_state_at_horizon_index(self._current_joint_state_trajectory, horizon_index)

    def get_command_sequence(self) -> JointState:
        if not self.has_valiaction_dim_buffer() or self._current_joint_state_trajectory is None:
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        return trim_joint_state_trajectory(
            self._current_joint_state_trajectory,
            self.command_start_idx,
            self.command_end_idx,
        )

    def has_valid_next_command(self) -> bool:
        return self.remaining_commands > 0

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
        if action_buffer.shape[-2] < 1:
            raise ValueError("action_buffer must be non-empty")
        self._current_action_trajectory = action_buffer
        self._current_command_idx = 0
        self._current_joint_state_trajectory = None
        self._current_metrics = None

    def reset_command_index(self) -> None:
        """Rewind an installed trajectory without replacing its tensor buffers."""
        self._current_command_idx = 0

    def clear_buffers(self, *, clear_robot_state: bool = False) -> None:
        """Atomically invalidate the command queue and its rollout metrics.

        ``RobotState`` is intentionally retained by default: callers often
        continue to use the most recently evaluated kinematics while a new
        optimizer result is pending.  Set ``clear_robot_state`` for a complete
        lifecycle reset.  External robot transports/CUDA streams are outside
        this manager's portable contract.
        """
        self._current_joint_state_trajectory = None
        self._current_action_trajectory = None
        self._current_metrics = None
        self._current_command_idx = 0
        if clear_robot_state:
            self._current_robot_state_trajectory = None


__all__ = ["TrajectoryExecutionManager"]
