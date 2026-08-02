"""Portable SDF detector facade over deterministic mesh registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose

from .detection_result import DetectionResult
from .mesh_robot import RobotMesh
from .pose_detector import PoseDetector
from .sdf_pose_detector_cfg import SDFDetectorCfg


@dataclass
class SDFRefinementState:
    position: torch.Tensor
    quaternion: torch.Tensor
    loss: torch.Tensor
    iterations: int = 0

    def clone(self) -> "SDFRefinementState":
        return type(self)(self.position.clone(), self.quaternion.clone(), self.loss.clone(), self.iterations)

    def copy_(self, other: "SDFRefinementState") -> "SDFRefinementState":
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)
        self.loss.copy_(other.loss)
        self.iterations = other.iterations
        return self


class SDFPoseDetector(PoseDetector):
    def __init__(self, robot_mesh: RobotMesh, config: Optional[SDFDetectorCfg] = None) -> None:
        super().__init__(robot_mesh, config or SDFDetectorCfg())

    def detect(self, camera_obs: CameraObservation, config: Optional[torch.Tensor] = None,
               initial_pose: Optional[Pose] = None) -> DetectionResult:
        from .util import extract_observed_points
        return self.detect_from_points(extract_observed_points(camera_obs), config, initial_pose)

    def detect_from_points(self, observed_points: torch.Tensor, config: Optional[torch.Tensor] = None,
                           initial_pose: Optional[Pose] = None) -> DetectionResult:
        return super().detect_from_points(observed_points, config, initial_pose)


__all__ = ["SDFPoseDetector", "SDFRefinementState"]
