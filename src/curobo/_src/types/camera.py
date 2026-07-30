"""Portable RGB-D camera observation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from .pose import Pose


@dataclass
class CameraObservation:
    name: str = "camera_image"
    rgb_image: Optional[torch.Tensor] = None
    depth_image: Optional[torch.Tensor] = None
    image_segmentation: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    projection_rays: Optional[torch.Tensor] = None
    resolution: Optional[List[int]] = None
    pose: Optional[Pose] = None
    intrinsics: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None
    depth_to_meter: float = 0.001
    feature_grid: Optional[torch.Tensor] = None

    def filter_depth(self, distance: float = 0.01):
        if self.depth_image is None: raise ValueError("depth_image is None, cannot filter depth")
        self.depth_image = torch.where(self.depth_image < distance, 0, self.depth_image)

    @property
    def shape(self):
        if self.rgb_image is None: raise ValueError("rgb_image is None, cannot get shape")
        return self.rgb_image.shape

    def copy_(self, new_data: "CameraObservation"):
        for field in ("rgb_image", "depth_image", "image_segmentation", "projection_matrix",
                      "projection_rays", "timestamp", "feature_grid"):
            target, source = getattr(self, field), getattr(new_data, field)
            if target is not None and source is not None: target.copy_(source)
        if self.pose is not None: self.pose.copy_(new_data.pose)
        self.depth_to_meter, self.resolution = new_data.depth_to_meter, new_data.resolution

    def clone(self):
        def clone(value): return None if value is None else value.clone()
        return type(self)(self.name, clone(self.rgb_image), clone(self.depth_image),
                          clone(self.image_segmentation), clone(self.projection_matrix),
                          clone(self.projection_rays), self.resolution, clone(self.pose),
                          clone(self.intrinsics), clone(self.timestamp), self.depth_to_meter,
                          clone(self.feature_grid))

    def to(self, device: torch.device):
        for field in ("rgb_image", "depth_image", "image_segmentation", "projection_matrix",
                      "projection_rays", "intrinsics", "timestamp", "feature_grid"):
            value = getattr(self, field)
            if value is not None: setattr(self, field, value.to(device=device))
        if self.pose is not None: self.pose.to(device=device)
        return self

    def update_projection_rays(self):
        if self.depth_image is None: raise ValueError("depth_image is None, cannot update projection rays")
        if self.intrinsics is None: raise ValueError("intrinsics is None, cannot update projection rays")
        intrinsics = self.intrinsics.unsqueeze(0) if self.intrinsics.ndim == 2 else self.intrinsics
        height, width = self.depth_image.shape[-2:]
        y, x = torch.meshgrid(torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype),
                              torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype),
                              indexing="ij")
        x = (x[None] - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None]
        y = (y[None] - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None]
        rays = torch.stack((x.expand_as(y), y, torch.ones_like(y)), -1) * self.depth_to_meter
        if self.projection_rays is None: self.projection_rays = rays
        else: self.projection_rays.copy_(rays)

    def get_pointcloud(self, project_to_pose: bool = False):
        if self.depth_image is None: raise ValueError("depth_image is None, cannot generate pointcloud")
        if self.projection_rays is None: self.update_projection_rays()
        depth = self.depth_image.unsqueeze(0) if self.depth_image.ndim == 2 else self.depth_image
        cloud = depth[..., None] * self.projection_rays
        if project_to_pose and self.pose is not None: cloud = self.pose.batch_transform_points(cloud)
        return cloud

    def extract_depth_from_structured_pointcloud(self, pointcloud, output_image=None):
        if pointcloud.ndim == 3: pointcloud = pointcloud.unsqueeze(0)
        depth = pointcloud[..., 2]
        if output_image is None: return depth.clone()
        target = output_image.unsqueeze(0) if output_image.ndim == 2 else output_image
        target.copy_(depth); return target

    def stack(self, new_observation: "CameraObservation", dim: int = 0):
        def stack(field):
            left, right = getattr(self, field), getattr(new_observation, field)
            return torch.stack((left, right), dim=dim) if left is not None else None
        return type(self)(self.name, stack("rgb_image"), stack("depth_image"),
                          stack("image_segmentation"), stack("projection_matrix"),
                          stack("projection_rays"), self.resolution,
                          self.pose.stack(new_observation.pose) if self.pose is not None else None,
                          stack("intrinsics"), stack("timestamp"))

    def save_to_file(self, file_path: str):
        torch.save({"rgb_image": self.rgb_image, "depth_image": self.depth_image,
                    "intrinsics": self.intrinsics, "pose": self.pose.tolist(),
                    "timestamp": self.timestamp, "depth_to_meter": self.depth_to_meter}, file_path)


__all__ = ["CameraObservation"]
