"""Kinematics result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from curobo._src.types.pose import Pose
from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry


class ToolPose(Pose):
    def __init__(self, tool_frames: List[str], position: torch.Tensor, quaternion: torch.Tensor):
        super().__init__(position, quaternion)
        self.tool_frames = list(tool_frames)

    def get_link_pose(self, link_name: str) -> Pose:
        index = self.tool_frames.index(link_name)
        return Pose(self.position[..., index, :], self.quaternion[..., index, :])

    def clone(self) -> "ToolPose":
        return ToolPose(self.tool_frames, self.position.clone(), self.quaternion.clone())

    def detach(self) -> "ToolPose":
        return ToolPose(self.tool_frames, self.position.detach(), self.quaternion.detach())

    def copy_(self, other: "ToolPose") -> "ToolPose":
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)
        return self

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> "ToolPose":
        return ToolPose(self.tool_frames, self.position[idx], self.quaternion[idx])


@dataclass
class KinematicsState:
    tool_poses: Optional[ToolPose] = None
    tool_jacobians: Optional[torch.Tensor] = None
    robot_spheres: Optional[torch.Tensor] = None
    robot_com: Optional[torch.Tensor] = None
    robot_collision_geometry: Optional[object] = None

    @property
    def tool_frames(self) -> List[str]:
        return [] if self.tool_poses is None else self.tool_poses.tool_frames

    def get_link_spheres(self) -> torch.Tensor:
        return self.robot_spheres

    def clone(self) -> "KinematicsState":
        return KinematicsState(
            None if self.tool_poses is None else self.tool_poses.clone(),
            None if self.tool_jacobians is None else self.tool_jacobians.clone(),
            None if self.robot_spheres is None else self.robot_spheres.clone(),
            None if self.robot_com is None else self.robot_com.clone(),
            self.robot_collision_geometry,
        )

    def detach(self) -> "KinematicsState":
        return KinematicsState(
            None if self.tool_poses is None else self.tool_poses.detach(),
            None if self.tool_jacobians is None else self.tool_jacobians.detach(),
            None if self.robot_spheres is None else self.robot_spheres.detach(),
            None if self.robot_com is None else self.robot_com.detach(),
            self.robot_collision_geometry,
        )

    def copy_(self, other: "KinematicsState") -> "KinematicsState":
        for name in ("tool_poses", "tool_jacobians", "robot_spheres", "robot_com"):
            target, source = getattr(self, name), getattr(other, name)
            if target is not None and source is not None:
                target.copy_(source)
        return self

    def __len__(self) -> int:
        for value in (self.robot_spheres, self.tool_poses, self.tool_jacobians):
            if value is not None:
                return value.position.shape[0] if isinstance(value, ToolPose) else value.shape[0]
        return 0

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> "KinematicsState":
        return KinematicsState(
            None if self.tool_poses is None else self.tool_poses[idx],
            None if self.tool_jacobians is None else self.tool_jacobians[idx],
            None if self.robot_spheres is None else self.robot_spheres[idx],
            None if self.robot_com is None else self.robot_com[idx],
            self.robot_collision_geometry,
        )
