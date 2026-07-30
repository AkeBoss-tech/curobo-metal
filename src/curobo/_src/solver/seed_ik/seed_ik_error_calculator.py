"""Portable seeded-IK error/Jacobian calculator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


@dataclass
class ErrorJacobianResult:
    position_errors: torch.Tensor
    orientation_errors: torch.Tensor
    jTerror: torch.Tensor
    jacobian: torch.Tensor
    error_norm: torch.Tensor
    joint_position: torch.Tensor


class SeedIKErrorCalculator:
    def __init__(self, robot_model, config, action_min, action_max, device_cfg):
        self.robot_model, self.config = robot_model, config
        self.action_min, self.action_max = action_min, action_max
        self.device_cfg = device_cfg
        self.batch_size, self.num_seeds = 1, 1
        self._criteria: Dict[str, object] = {}

    def setup_batch_tensors(self, batch_size: int, num_seeds: int = 1):
        self.batch_size, self.num_seeds = batch_size, num_seeds

    def compute_error_and_jacobian(
        self, joint_position: torch.Tensor, goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor, current_position: Optional[torch.Tensor] = None,
        current_velocity: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ):
        del current_position, current_velocity, dt, velocity_clamping_active
        state = self.robot_model.compute_kinematics(
            JointState.from_position(joint_position)
        )
        position = state.tool_poses.position[:, 0, -1]
        quaternion = state.tool_poses.quaternion[:, 0, -1]
        target_p = goal_poses.position[:, 0, -1, 0][idxs_goal]
        target_q = goal_poses.quaternion[:, 0, -1, 0][idxs_goal]
        position_vector = position - target_p
        position_error = torch.linalg.vector_norm(position_vector, dim=-1)
        orientation_error = 2 * torch.acos(
            (quaternion * target_q).sum(-1).abs().clamp(max=1)
        )
        jacobian = state.tool_jacobians[:, 0, -1]
        residual = torch.cat(
            (position_vector, torch.zeros_like(position_vector)), dim=-1
        )
        jterror = torch.einsum("bij,bi->bj", jacobian, residual)
        error_norm = position_error.square() + orientation_error.square()
        return ErrorJacobianResult(
            position_error, orientation_error, jterror, jacobian,
            error_norm, joint_position,
        )

    def update_tool_pose_criteria(self, tool_pose_criteria):
        self._criteria = dict(tool_pose_criteria)

    def stream_context(self, stream_name: str):
        if stream_name not in ("default", "kinematics", "cost"):
            raise ValueError(f"unknown portable stream: {stream_name}")
        from contextlib import nullcontext
        return nullcontext()


__all__ = ["ErrorJacobianResult", "SeedIKErrorCalculator"]
