"""Robot state combining joints, torques, and optional kinematics output."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Union
import torch
from .state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import ToolPose


@dataclass
class RobotState:
    joint_state: JointState
    joint_torque: Optional[torch.Tensor] = None
    cuda_robot_model_state: Optional[Any] = None

    def data_ptr(self): return self.joint_state.data_ptr()
    def __len__(self): return len(self.joint_state)
    def __getitem__(self, idx: Union[int, torch.Tensor]):
        def index(value): return None if value is None else value[idx]
        return type(self)(index(self.joint_state), index(self.joint_torque), index(self.cuda_robot_model_state))
    def detach(self):
        def detach(value): return None if value is None else value.detach()
        return type(self)(detach(self.joint_state), detach(self.joint_torque), detach(self.cuda_robot_model_state))
    robot_spheres = property(lambda self: None if self.cuda_robot_model_state is None else self.cuda_robot_model_state.robot_spheres)
    link_poses = property(lambda self: None if self.cuda_robot_model_state is None else self.cuda_robot_model_state.tool_poses)
    tool_poses = property(lambda self: None if self.cuda_robot_model_state is None else self.cuda_robot_model_state.tool_poses)
    tool_frames = property(lambda self: [] if self.tool_poses is None else self.tool_poses.tool_frames)
    def get_link_pose(self, link_name: str) -> Pose:
        if self.tool_poses is None: raise ValueError("Link poses are not set")
        return self.tool_poses.get_link_pose(link_name)
    def clone(self):
        def clone(value): return None if value is None else value.clone()
        return type(self)(clone(self.joint_state), clone(self.joint_torque), clone(self.cuda_robot_model_state))
    def copy_(self, other: "RobotState"):
        self.joint_state.copy_(other.joint_state)
        if self.joint_torque is not None: self.joint_torque.copy_(other.joint_torque)
        if self.cuda_robot_model_state is not None: self.cuda_robot_model_state.copy_(other.cuda_robot_model_state)
        return self

    def copy_only_index(self, other: "RobotState", index: Union[int, torch.Tensor]):
        self.joint_state[index] = other.joint_state[index]
        if self.robot_spheres is not None: self.robot_spheres[index] = other.robot_spheres[index]
        if self.tool_poses is not None:
            self.tool_poses.position[index] = other.tool_poses.position[index]
            self.tool_poses.quaternion[index] = other.tool_poses.quaternion[index]
        if self.joint_torque is not None: self.joint_torque[index] = other.joint_torque[index]
        return self

    def copy_at_batch_seed_indices(self, other: "RobotState", batch_idx: torch.Tensor, seed_idx: torch.Tensor):
        self.joint_state.position[batch_idx, seed_idx] = other.joint_state.position[batch_idx, seed_idx]
        for field in ("velocity", "acceleration", "jerk"):
            target, source = getattr(self.joint_state, field), getattr(other.joint_state, field)
            if target is not None: target[batch_idx, seed_idx] = source[batch_idx, seed_idx]
        if self.joint_torque is not None: self.joint_torque[batch_idx, seed_idx] = other.joint_torque[batch_idx, seed_idx]
        return self


__all__ = ["RobotState"]
