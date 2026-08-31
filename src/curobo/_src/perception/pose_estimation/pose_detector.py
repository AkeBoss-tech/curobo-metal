"""Deterministic CPU/MPS point-cloud pose detector."""

from __future__ import annotations

import time
from typing import Optional, Union

import torch

from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose

from .detection_result import DetectionResult
from .geometry import ArticulatedRobotGeometry, RigidObjectGeometry
from .mesh_robot import RobotMesh
from .pose_detector_cfg import DetectorCfg
from .util import (
    compute_pose_point_to_plane_cholesky,
    compute_pose_point_to_plane_svd,
    extract_observed_points,
    find_nearest_neighbors,
    resample_points,
)


class PoseDetector:
    """Portable centroid-registration detector for bounded scene geometry.

    This intentionally does not emulate CUDA/Warp global rotation sampling;
    callers that supply an ``initial_pose`` get deterministic local centroid
    registration on ordinary PyTorch tensors.
    """

    def __init__(
        self,
        geometry: Union[RigidObjectGeometry, ArticulatedRobotGeometry, RobotMesh],
        config: DetectorCfg,
    ):
        self.geometry = geometry
        self.config = config

    def _model_points(self, n_points: int) -> torch.Tensor:
        if isinstance(self.geometry, RobotMesh):
            return self.geometry.sample_surface_points(n_points)[0]
        return self.geometry.sample_surface_points(n_points)[0]

    def detect_from_points(self, observed_points: torch.Tensor, config: torch.Tensor,
                           initial_pose: Optional[Pose] = None) -> DetectionResult:
        started = time.perf_counter()
        points = torch.as_tensor(observed_points, device=self.config.device_cfg.device,
                                 dtype=self.config.device_cfg.dtype).reshape(-1, 3)
        if len(points) == 0:
            raise ValueError("observed_points must contain at least one point")
        model = self._model_points(min(self.config.n_mesh_points_coarse, max(1, len(points)))).to(points)
        if len(model) == 0:
            raise ValueError("detector geometry contains no sampleable points")
        initial_translation = points.new_zeros(3) if initial_pose is None else initial_pose.position.reshape(-1, 3)[0].to(points)
        translation = points.mean(0) - model.mean(0) + initial_translation
        quaternion = points.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        pose = Pose(translation[None], quaternion)
        shifted = model + translation
        error = torch.cdist(points, shifted).min(-1).values.mean()
        confidence = float(torch.exp(-error).detach().cpu())
        return DetectionResult(pose, config, confidence, float(error.detach().cpu()), 1, time.perf_counter() - started)

    def detect(self, camera_obs: CameraObservation, config: torch.Tensor) -> DetectionResult:
        return self.detect_from_points(extract_observed_points(camera_obs), config)


__all__ = ["PoseDetector"]
