"""Portable structured LiDAR observation value type.

This is deliberately a tensor data model, not a LiDAR driver or a Warp range
integration kernel. It can validate, move, serialize, convert, and feed range
images into the portable dense or block-sparse mapper on CPU/MPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch.profiler import record_function

from curobo._src.util.logging import log_and_raise
from .camera import _pose_from_payload, _pose_to_payload
from .pose import Pose


_TENSOR_FIELDS = (
    "range_image", "rgb_image", "feature_grid", "valid_range_m",
    "elevation_range_rad", "timestamp",
)


class _LidarObservationPortableMixin:
    @property
    def device(self) -> torch.device:
        return self._device_portable

    def validate(self, **requirements) -> "LidarObservation":
        return self._validate_portable(**requirements)

    def valid_mask(self) -> torch.Tensor:
        return self._valid_mask_portable()

    def to_pointcloud(self, *, project_to_pose: bool = False) -> torch.Tensor:
        return self._to_pointcloud_portable(project_to_pose=project_to_pose)

    def detach(self) -> "LidarObservation":
        return self._detach_portable()

    def requires_grad_(self, requires_grad: bool = True) -> "LidarObservation":
        return self._requires_grad_portable(requires_grad)

    def as_dict(self) -> dict[str, Any]:
        return self._as_dict_portable()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LidarObservation":
        return cls._from_dict_portable(value)

    def save_to_file(self, file_path: str | Path) -> None:
        self._save_to_file_portable(file_path)

    @classmethod
    def load_from_file(cls, file_path: str | Path, *, map_location=None) -> "LidarObservation":
        return cls._load_from_file_portable(file_path, map_location=map_location)


@dataclass
class LidarObservation(_LidarObservationPortableMixin):
    """Structured range-image observation with CPU/MPS-safe lifecycle helpers.

    Range values are Euclidean metres from the sensor origin.  ``to_pointcloud``
    assumes equally spaced azimuth samples over ``[-pi, pi)``; drivers with an
    arbitrary per-pixel calibration must supply their own ray table.
    """

    name: str = "lidar_range_image"
    range_image: Optional[torch.Tensor] = None
    rgb_image: Optional[torch.Tensor] = None
    feature_grid: Optional[torch.Tensor] = None
    pose: Optional[Pose] = None
    valid_range_m: Optional[torch.Tensor] = None
    elevation_range_rad: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"{field} must be a torch.Tensor")
        if self.pose is not None and not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose or None")

    @property
    def shape(self):
        if self.range_image is None:
            raise ValueError("range_image is None, cannot get shape")
        return self.range_image.shape

    @property
    def _device_portable(self) -> torch.device:
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None:
                return value.device
        if self.pose is not None:
            return self.pose.device
        raise ValueError("empty LidarObservation has no device")

    def _validate_portable(
        self, *, require_range: bool = False, require_rgb: bool = False,
        require_pose: bool = False, require_calibration: bool = False,
    ) -> "LidarObservation":
        """Validate portable range-image layout without touching CUDA/Warp."""
        if require_range and self.range_image is None:
            raise ValueError("range_image is required")
        if require_rgb and self.rgb_image is None:
            raise ValueError("rgb_image is required")
        if require_pose and self.pose is None:
            raise ValueError("pose is required")
        if require_calibration and (self.valid_range_m is None or self.elevation_range_rad is None):
            raise ValueError("valid_range_m and elevation_range_rad are required")
        ranges = self.range_image
        if ranges is not None:
            if ranges.ndim != 3:
                raise ValueError("range_image must have shape [num_lidars,H,W]")
            if not ranges.is_floating_point():
                raise TypeError("range_image must have a floating-point dtype")
            if ranges.shape[0] == 0 or ranges.shape[1] == 0 or ranges.shape[2] == 0:
                raise ValueError("range_image dimensions must be positive")
        if self.rgb_image is not None:
            if self.rgb_image.ndim != 4 or self.rgb_image.shape[-1] != 3:
                raise ValueError("rgb_image must have shape [num_lidars,H,W,3]")
            if self.rgb_image.dtype != torch.uint8:
                raise TypeError("rgb_image must have dtype torch.uint8")
        if self.feature_grid is not None and self.feature_grid.ndim != 4:
            raise ValueError("feature_grid must have shape [num_lidars,H,W,C]")
        for field in ("valid_range_m", "elevation_range_rad"):
            value = getattr(self, field)
            if value is not None:
                if value.ndim != 2 or value.shape[-1] != 2:
                    raise ValueError(f"{field} must have shape [num_lidars,2]")
                if not value.is_floating_point():
                    raise TypeError(f"{field} must have a floating-point dtype")
        if ranges is not None:
            number, height, width = ranges.shape
            if self.rgb_image is not None and tuple(self.rgb_image.shape) != (number, height, width, 3):
                raise ValueError("rgb_image shape must match range_image")
            if self.feature_grid is not None and tuple(self.feature_grid.shape[:3]) != (number, height, width):
                raise ValueError("feature_grid leading shape must match range_image")
            for field in ("valid_range_m", "elevation_range_rad"):
                value = getattr(self, field)
                if value is not None and value.shape[0] != number:
                    raise ValueError(f"{field} batch dimension must match range_image")
            for field in _TENSOR_FIELDS:
                value = getattr(self, field)
                if value is not None and value.device != ranges.device:
                    raise ValueError(f"{field} must share the range_image device")
            if self.pose is not None:
                if self.pose.device != ranges.device:
                    raise ValueError("pose must share the range_image device")
                if self.pose.position is None or self.pose.position.reshape(-1, 3).shape[0] != number:
                    raise ValueError("pose batch dimension must match range_image")
        if self.elevation_range_rad is not None and ranges is not None and ranges.shape[1] == 1:
            # Mathematical comparison is intentionally here rather than in
            # construction: this is operation-boundary validation.
            if not bool(torch.allclose(self.elevation_range_rad[:, 0], self.elevation_range_rad[:, 1])):
                raise ValueError("planar LiDAR requires equal elevation_range_rad bounds")
        return self

    def _valid_mask_portable(self) -> torch.Tensor:
        """Return finite in-range pixels, preserving batch/rank/device."""
        self.validate(require_range=True)
        assert self.range_image is not None
        mask = torch.isfinite(self.range_image) & (self.range_image > 0)
        if self.valid_range_m is not None:
            lower = self.valid_range_m[:, 0, None, None]
            upper = self.valid_range_m[:, 1, None, None]
            mask = mask & (self.range_image >= lower) & (self.range_image <= upper)
        return mask

    def _to_pointcloud_portable(self, *, project_to_pose: bool = False) -> torch.Tensor:
        """Convert calibrated-elevation range pixels to differentiable XYZ points.

        The returned ``[N,H,W,3]`` points are in LiDAR frame unless
        ``project_to_pose=True``.  This portable helper does not claim the
        source's raw CUDA range-projection kernel ABI.
        """
        self.validate(require_range=True, require_calibration=True)
        assert self.range_image is not None and self.elevation_range_rad is not None
        number, height, width = self.range_image.shape
        azimuth = torch.arange(width, device=self.range_image.device, dtype=self.range_image.dtype)
        azimuth = azimuth * (2.0 * torch.pi / width) - torch.pi
        if height == 1:
            elevation = self.elevation_range_rad[:, :1]
        else:
            interpolation = torch.linspace(0.0, 1.0, height, device=self.range_image.device,
                                           dtype=self.range_image.dtype)[None]
            elevation = self.elevation_range_rad[:, 1:] - interpolation * (
                self.elevation_range_rad[:, 1:] - self.elevation_range_rad[:, :1]
            )
        cos_elevation = torch.cos(elevation)[:, :, None]
        sin_elevation = torch.sin(elevation)[:, :, None]
        cos_azimuth, sin_azimuth = torch.cos(azimuth)[None, None], torch.sin(azimuth)[None, None]
        ranges = self.range_image
        points = torch.stack((ranges * cos_elevation * cos_azimuth,
                              ranges * cos_elevation * sin_azimuth,
                              ranges * sin_elevation), dim=-1)
        points = torch.where(self.valid_mask()[..., None], points, torch.zeros_like(points))
        if project_to_pose:
            if self.pose is None:
                raise ValueError("pose is required when project_to_pose=True")
            points = self.pose.batch_transform_points(points)
        return points

    def copy_(self, new_data: LidarObservation):
        if not isinstance(new_data, LidarObservation):
            raise TypeError("new_data must be a LidarObservation")
        for field in _TENSOR_FIELDS:
            source, target = getattr(new_data, field), getattr(self, field)
            if source is None:
                setattr(self, field, None)
            elif target is None or target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
                setattr(self, field, source.clone())
            else:
                target.copy_(source)
        if new_data.pose is None:
            self.pose = None
        elif self.pose is None:
            self.pose = new_data.pose.clone()
        else:
            self.pose.copy_(new_data.pose)
        return self

    def clone(self):
        return type(self)(name=self.name).copy_(self)

    def _detach_portable(self) -> "LidarObservation":
        value = self.clone()
        for field in _TENSOR_FIELDS:
            tensor = getattr(value, field)
            if tensor is not None:
                setattr(value, field, tensor.detach())
        if value.pose is not None:
            value.pose = value.pose.detach()
        return value

    def _requires_grad_portable(self, requires_grad: bool = True) -> "LidarObservation":
        for field in _TENSOR_FIELDS:
            tensor = getattr(self, field)
            if tensor is not None and (tensor.is_floating_point() or tensor.is_complex()):
                tensor.requires_grad_(requires_grad)
        if self.pose is not None:
            self.pose.requires_grad_(requires_grad)
        return self

    def to(self, device: torch.device):
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.to(device=device))
        if self.pose is not None:
            self.pose.to(device=device)
        return self

    def _as_dict_portable(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in _TENSOR_FIELDS}
        result.update({"name": self.name, "pose": _pose_to_payload(self.pose)})
        return result

    @classmethod
    def _from_dict_portable(cls, value: Mapping[str, Any]) -> "LidarObservation":
        if not isinstance(value, Mapping):
            raise TypeError("LiDAR observation payload must be a mapping")
        fields = {field: value.get(field) for field in _TENSOR_FIELDS}
        return cls(name=value.get("name", "lidar_range_image"),
                   pose=_pose_from_payload(value.get("pose")), **fields)

    def _save_to_file_portable(self, file_path: str | Path) -> None:
        torch.save(self.as_dict(), file_path)

    @classmethod
    def _load_from_file_portable(
        cls, file_path: str | Path, *, map_location=None
    ) -> "LidarObservation":
        try:
            payload = torch.load(file_path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(file_path, map_location=map_location)
        return cls.from_dict(payload)


__all__ = ["LidarObservation"]
