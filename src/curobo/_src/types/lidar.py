"""Portable structured LiDAR observation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from .pose import Pose


@dataclass
class LidarObservation:
    name: str = "lidar_range_image"
    range_image: Optional[torch.Tensor] = None
    rgb_image: Optional[torch.Tensor] = None
    feature_grid: Optional[torch.Tensor] = None
    pose: Optional[Pose] = None
    valid_range_m: Optional[torch.Tensor] = None
    elevation_range_rad: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None

    @property
    def shape(self):
        if self.range_image is None:
            raise ValueError("range_image is None, cannot get shape")
        return self.range_image.shape

    def copy_(self, new_data: "LidarObservation"):
        for field in ("range_image", "rgb_image", "feature_grid", "valid_range_m",
                      "elevation_range_rad", "timestamp"):
            target, source = getattr(self, field), getattr(new_data, field)
            if target is not None and source is not None:
                target.copy_(source)
        if self.pose is not None:
            self.pose.copy_(new_data.pose)
        return self

    def clone(self):
        def clone(value):
            return None if value is None else value.clone()
        return type(self)(self.name, clone(self.range_image), clone(self.rgb_image),
                          clone(self.feature_grid), clone(self.pose), clone(self.valid_range_m),
                          clone(self.elevation_range_rad), clone(self.timestamp))

    def to(self, device: torch.device):
        for field in ("range_image", "rgb_image", "feature_grid", "valid_range_m",
                      "elevation_range_rad", "timestamp"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.to(device=device))
        if self.pose is not None:
            self.pose.to(device=device)
        return self


__all__ = ["LidarObservation"]
