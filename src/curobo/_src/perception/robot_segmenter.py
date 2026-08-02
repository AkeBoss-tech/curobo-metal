"""CPU/MPS robot-depth segmentation via configured collision spheres."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg


class RobotSegmenter:
    def __init__(self, kinematics, distance_threshold: float = 0.05,
                 use_cuda_graph: bool = True, ops_dtype: torch.dtype = torch.bfloat16) -> None:
        del use_cuda_graph, ops_dtype
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be nonnegative")
        self._kinematics = kinematics
        self.distance_threshold = float(distance_threshold)

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def base_link(self) -> str:
        return str(getattr(getattr(self._kinematics, "config", None), "base_link", "base_link"))

    @classmethod
    def from_robot_file(cls, robot_file: Union[str, Dict], collision_sphere_buffer: Optional[float] = None,
                        distance_threshold: float = 0.05, use_cuda_graph: bool = True,
                        device_cfg: DeviceCfg = DeviceCfg()) -> "RobotSegmenter":
        del collision_sphere_buffer
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        cfg = KinematicsCfg.from_robot_yaml_file(robot_file, device_cfg=device_cfg)
        return cls(Kinematics(cfg, compute_spheres=True), distance_threshold, use_cuda_graph)

    def get_pointcloud_from_depth(self, camera_obs: CameraObservation) -> torch.Tensor:
        return camera_obs.get_pointcloud(project_to_pose=True)

    def update_camera_projection(self, camera_obs: CameraObservation) -> None:
        camera_obs.update_projection_rays()

    def get_robot_mask_from_active_js(self, camera_obs: CameraObservation,
                                      active_joint_state: JointState) -> Tuple[torch.Tensor, torch.Tensor]:
        points = self.get_pointcloud_from_depth(camera_obs)
        state = self._kinematics.compute_kinematics(active_joint_state)
        spheres = state.robot_spheres
        if spheres is None:
            raise ValueError("kinematics was created without collision spheres")
        # Kinematics stores [B,H,S,4]. Camera observations use [B,H,W,3].
        if points.ndim == 3:
            points = points.unsqueeze(0)
        if spheres.ndim == 4:
            spheres = spheres[:, 0]
        if points.shape[0] not in (1, spheres.shape[0]):
            raise ValueError("camera and joint-state batch sizes are incompatible")
        if points.shape[0] == 1 and spheres.shape[0] > 1:
            points = points.expand(spheres.shape[0], -1, -1, -1)
        flat = points.reshape(points.shape[0], -1, 3)
        centers, radii = spheres[..., :3], spheres[..., 3]
        distance = torch.cdist(flat, centers) - radii[:, None]
        mask = (distance.min(-1).values < self.distance_threshold).reshape(points.shape[:-1])
        depth = camera_obs.depth_image
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        filtered = torch.where(mask.to(depth.device), torch.zeros_like(depth), depth)
        return mask, filtered[0] if camera_obs.depth_image.ndim == 2 else filtered

    def get_robot_mask(self, camera_obs: CameraObservation, joint_state: JointState) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_robot_mask_from_active_js(camera_obs, joint_state)


__all__ = ["RobotSegmenter"]
