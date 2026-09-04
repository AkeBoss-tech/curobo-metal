"""Portable RGB-D camera observation value type.

This module intentionally contains only ordinary PyTorch tensor operations.
It is therefore usable on CPU and MPS without loading Warp, CUDA, a camera
driver, or a mapper.  The hardware-specific integration kernels remain an
explicit boundary in :mod:`curobo._src.perception.mapper`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

import torch
from torch.profiler import record_function

from .pose import Pose
from curobo._src.util.logging import log_and_raise


_TENSOR_FIELDS = (
    "rgb_image", "depth_image", "image_segmentation", "projection_matrix",
    "projection_rays", "intrinsics", "timestamp", "feature_grid",
)


def _require_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _get_projection_rays_portable(
    height: int, width: int, intrinsics: torch.Tensor, depth_to_meter: float = 0.001
) -> torch.Tensor:
    """Build differentiable pinhole rays in the camera frame.

    ``intrinsics`` has shape ``[3, 3]`` or ``[B, 3, 3]`` and the result has
    shape ``[B, H, W, 3]``.  Multiplying a raw depth image by these rays gives
    meter-valued camera-frame points when ``depth_to_meter`` is the raw depth
    unit conversion.  The function deliberately allows a one-camera ray grid
    to broadcast over a batched depth image.
    """
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    intrinsics = _require_tensor(intrinsics, "intrinsics")
    if intrinsics.ndim not in (2, 3) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [3,3] or [B,3,3]")
    if not intrinsics.is_floating_point():
        raise TypeError("intrinsics must have a floating-point dtype")
    if not isinstance(depth_to_meter, (float, int)) or not float(depth_to_meter) > 0.0:
        raise ValueError("depth_to_meter must be positive")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    y, x = torch.meshgrid(
        torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype),
        torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype),
        indexing="ij",
    )
    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    if bool(torch.any(fx == 0).item()) or bool(torch.any(fy == 0).item()):
        raise ValueError("intrinsics focal lengths fx and fy must be non-zero")
    px = (x[None] - intrinsics[:, 0, 2, None, None]) / fx
    py = (y[None] - intrinsics[:, 1, 2, None, None]) / fy
    return torch.stack((px.expand_as(py), py, torch.ones_like(py)), -1) * float(depth_to_meter)


def _project_depth_using_rays_portable(
    depth_image: torch.Tensor, projection_rays: torch.Tensor
) -> torch.Tensor:
    """Project a structured raw-depth image through pinhole rays.

    This preserves autograd with respect to both depth and intrinsics-derived
    rays.  Depth is ``[H,W]`` or ``[B,H,W]``; rays must be ``[1|B,H,W,3]``.
    """
    depth_image = _require_tensor(depth_image, "depth_image")
    projection_rays = _require_tensor(projection_rays, "projection_rays")
    if depth_image.ndim not in (2, 3):
        raise ValueError("depth_image must have shape [H,W] or [B,H,W]")
    if projection_rays.ndim not in (3, 4) or projection_rays.shape[-1] != 3:
        raise ValueError("projection_rays must have shape [B,H*W,3] or [B,H,W,3]")
    depth = depth_image.unsqueeze(0) if depth_image.ndim == 2 else depth_image
    if projection_rays.ndim == 3:
        height, width = depth.shape[-2:]
        if projection_rays.shape[-2] != height * width:
            raise ValueError("depth_image and projection_rays spatial shapes must match")
        projection_rays = projection_rays.reshape(projection_rays.shape[0], height, width, 3)
    if tuple(depth.shape[-2:]) != tuple(projection_rays.shape[-3:-1]):
        raise ValueError("depth_image and projection_rays spatial shapes must match")
    if projection_rays.shape[0] not in (1, depth.shape[0]):
        raise ValueError("projection_rays batch dimension must be one or match depth_image")
    if depth.device != projection_rays.device:
        raise ValueError("depth_image and projection_rays must share a device")
    if depth.dtype != projection_rays.dtype:
        raise ValueError("depth_image and projection_rays must share a dtype")
    return depth[..., None] * projection_rays


def _extract_depth_from_structured_pointcloud_portable(
    pointcloud: torch.Tensor, output_image: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Extract camera-frame Z depth from ``[H,W,3]`` or ``[B,H,W,3]`` points."""
    pointcloud = _require_tensor(pointcloud, "pointcloud")
    if pointcloud.ndim not in (3, 4) or pointcloud.shape[-1] != 3:
        raise ValueError("pointcloud must have shape [H,W,3] or [B,H,W,3]")
    points = pointcloud.unsqueeze(0) if pointcloud.ndim == 3 else pointcloud
    depth = points[..., 2]
    if output_image is None:
        return depth.clone()
    output = _require_tensor(output_image, "output_image")
    output = output.unsqueeze(0) if output.ndim == 2 else output
    if output.shape != depth.shape:
        raise ValueError(f"output_image must have shape {tuple(depth.shape)}")
    if output.device != depth.device or output.dtype != depth.dtype:
        raise ValueError("output_image must share pointcloud device and dtype")
    output.copy_(depth)
    return output


# V2 exposes these as geometry-module imports.  Keep the portable tensor
# implementations private and re-export aliases with the same declaration
# category, so callers retain the symbols without treating this module as the
# owner of a competing callable surface.
get_projection_rays = _get_projection_rays_portable
project_depth_using_rays = _project_depth_using_rays_portable
extract_depth_from_structured_pointcloud = _extract_depth_from_structured_pointcloud_portable


def _pose_to_payload(pose: Optional[Pose]) -> Optional[dict[str, Any]]:
    if pose is None:
        return None
    return {
        "position": pose.position,
        "quaternion": pose.quaternion,
        "rotation": pose.rotation,
        "name": pose.name,
    }


def _pose_from_payload(value: Optional[Pose | Mapping[str, Any]]) -> Optional[Pose]:
    if value is None or isinstance(value, Pose):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("pose payload must be a Pose, mapping, or None")
    return Pose(
        value.get("position"), value.get("quaternion"), value.get("rotation"),
        name=value.get("name", "pose"), normalize_rotation=False,
    )


@dataclass
class CameraObservation:
    """RGB-D observation with an explicit tensor-only CPU/MPS lifecycle.

    Construction stays permissive like cuRobo V2.  Call :meth:`validate` at
    a component boundary when all fields required by a particular operation
    must be present.  This avoids rejecting RGB-only texture observations.
    """

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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.depth_to_meter, (float, int)) or not float(self.depth_to_meter) > 0.0:
            raise ValueError("depth_to_meter must be positive")
        self.depth_to_meter = float(self.depth_to_meter)
        if self.resolution is not None:
            if len(self.resolution) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.resolution):
                raise ValueError("resolution must be [height, width] with positive integers")
            self.resolution = list(self.resolution)
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None:
                _require_tensor(value, field)
        if self.pose is not None and not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose or None")

    @property
    def shape(self):
        if self.rgb_image is None:
            raise ValueError("rgb_image is None, cannot get shape")
        return self.rgb_image.shape

    @property
    def _device_portable(self) -> torch.device:
        """Device of the first tensor field, or the pose for pose-only frames."""
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None:
                return value.device
        if self.pose is not None:
            return self.pose.device
        raise ValueError("empty CameraObservation has no device")

    def _validate_portable(
        self, *, require_depth: bool = False, require_intrinsics: bool = False,
        require_pose: bool = False, require_rgb: bool = False,
    ) -> "CameraObservation":
        """Validate tensor ranks, batch compatibility, and device consistency.

        The check intentionally does not read tensor values (other than shape),
        so it does not require a CPU fallback or a device-to-host copy on MPS.
        """
        if require_depth and self.depth_image is None:
            raise ValueError("depth_image is None")
        if require_intrinsics and self.intrinsics is None:
            raise ValueError("intrinsics is None")
        if require_pose and self.pose is None:
            raise ValueError("pose is required")
        if require_rgb and self.rgb_image is None:
            raise ValueError("rgb_image is required")
        depth = self.depth_image
        if depth is not None:
            if depth.ndim not in (2, 3):
                raise ValueError("depth_image must have shape [H,W] or [B,H,W]")
            if not depth.is_floating_point():
                raise TypeError("depth_image must have a floating-point dtype")
        if self.intrinsics is not None:
            if self.intrinsics.ndim not in (2, 3) or self.intrinsics.shape[-2:] != (3, 3):
                raise ValueError("intrinsics must have shape [3,3] or [B,3,3]")
            if not self.intrinsics.is_floating_point():
                raise TypeError("intrinsics must have a floating-point dtype")
        if self.projection_rays is not None:
            if self.projection_rays.ndim not in (3, 4) or self.projection_rays.shape[-1] != 3:
                raise ValueError("projection_rays must have shape [B,H*W,3] or [B,H,W,3]")
        if self.projection_matrix is not None and self.projection_matrix.shape[-2:] != (4, 4):
            raise ValueError("projection_matrix must end in shape [4,4]")
        if self.rgb_image is not None and self.rgb_image.ndim not in (3, 4):
            raise ValueError("rgb_image must have shape [H,W,C] or [B,H,W,C]")
        if self.image_segmentation is not None and self.image_segmentation.ndim not in (2, 3):
            raise ValueError("image_segmentation must have shape [H,W] or [B,H,W]")
        if self.feature_grid is not None and self.feature_grid.ndim != 4:
            raise ValueError("feature_grid must have shape [B,H,W,C]")

        primary = depth if depth is not None else self.rgb_image
        if primary is not None:
            device = primary.device
            for field in _TENSOR_FIELDS:
                value = getattr(self, field)
                if value is not None and value.device != device:
                    raise ValueError(f"{field} must share the observation device")
            if self.pose is not None and self.pose.device != device:
                raise ValueError("pose must share the observation device")
        if depth is not None and self.intrinsics is not None:
            depth_batch = 1 if depth.ndim == 2 else depth.shape[0]
            intrinsics_batch = 1 if self.intrinsics.ndim == 2 else self.intrinsics.shape[0]
            if intrinsics_batch not in (1, depth_batch):
                raise ValueError("intrinsics batch dimension must be one or match depth_image")
        if depth is not None and self.projection_rays is not None:
            expected = (depth.shape[-2], depth.shape[-1])
            rays_match = (
                tuple(self.projection_rays.shape[-3:-1]) == expected
                if self.projection_rays.ndim == 4
                else self.projection_rays.shape[-2] == expected[0] * expected[1]
            )
            if not rays_match:
                raise ValueError("projection_rays spatial dimensions must match depth_image")
        if self.resolution is not None and depth is not None and tuple(self.resolution) != tuple(depth.shape[-2:]):
            raise ValueError("resolution must match depth_image [height, width]")
        return self

    def filter_depth(self, distance: float = 0.01):
        if self.depth_image is None:
            raise ValueError("depth_image is None, cannot filter depth")
        if not isinstance(distance, (float, int)) or float(distance) < 0.0:
            raise ValueError("distance must be non-negative")
        self.depth_image = torch.where(self.depth_image < float(distance), 0, self.depth_image)

    @record_function("camera/copy_")
    def copy_(self, new_data: CameraObservation):
        """Copy source data into fields already allocated by this observation.

        This intentionally follows the pinned cuRobo behavior: ``copy_`` is
        an in-place operation over destination-owned buffers.  In particular,
        a destination field that is ``None`` is not implicitly allocated from
        the source, and camera intrinsics are metadata owned by the existing
        destination rather than copied here.
        """
        if not isinstance(new_data, CameraObservation):
            raise TypeError("new_data must be a CameraObservation")
        for field in (
            "rgb_image", "depth_image", "image_segmentation",
            "projection_matrix", "projection_rays", "timestamp",
        ):
            target = getattr(self, field)
            if target is not None:
                source = getattr(new_data, field)
                if source is None:
                    target.copy_(source)
                elif target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
                    raise ValueError(f"cannot copy {field} into a destination with different shape, dtype, or device")
                else:
                    target.copy_(source)
        if self.pose is not None:
            self.pose.copy_(new_data.pose)
        if self.feature_grid is not None and new_data.feature_grid is not None:
            self.feature_grid.copy_(new_data.feature_grid)
        self.depth_to_meter = new_data.depth_to_meter
        self.resolution = new_data.resolution

    @record_function("camera/clone")
    def clone(self):
        return type(self)(
            name=self.name,
            resolution=None if self.resolution is None else list(self.resolution),
            pose=None if self.pose is None else self.pose.clone(),
            depth_to_meter=self.depth_to_meter,
            **{
                field: None if getattr(self, field) is None else getattr(self, field).clone()
                for field in _TENSOR_FIELDS
            },
        )

    def _detach_portable(self) -> "CameraObservation":
        value = self.clone()
        for field in _TENSOR_FIELDS:
            tensor = getattr(value, field)
            if tensor is not None:
                setattr(value, field, tensor.detach())
        if value.pose is not None:
            value.pose = value.pose.detach()
        return value

    def _requires_grad_portable(self, requires_grad: bool = True) -> "CameraObservation":
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

    def update_projection_rays(self):
        self.validate(require_depth=True, require_intrinsics=True)
        assert self.depth_image is not None and self.intrinsics is not None
        intrinsics = self.intrinsics.unsqueeze(0) if self.intrinsics.ndim == 2 else self.intrinsics
        height, width = self.depth_image.shape[-2:]
        rays = get_projection_rays(height, width, intrinsics, self.depth_to_meter).reshape(
            intrinsics.shape[0], height * width, 3
        )
        if self.projection_rays is None or self.projection_rays.shape != rays.shape or \
                self.projection_rays.device != rays.device or self.projection_rays.dtype != rays.dtype:
            self.projection_rays = rays
        else:
            self.projection_rays.copy_(rays)

    def get_pointcloud(self, project_to_pose: bool = False):
        self.validate(require_depth=True)
        if self.projection_rays is None:
            self.update_projection_rays()
        assert self.depth_image is not None and self.projection_rays is not None
        cloud = project_depth_using_rays(self.depth_image, self.projection_rays)
        if project_to_pose:
            if self.pose is None:
                raise ValueError("pose is required when project_to_pose=True")
            cloud = self.pose.batch_transform_points(cloud)
        return cloud

    def extract_depth_from_structured_pointcloud(
        self, pointcloud, output_image: Optional[torch.Tensor] = None
    ):
        return extract_depth_from_structured_pointcloud(pointcloud, output_image)

    def stack(self, new_observation: CameraObservation, dim: int = 0):
        if not isinstance(new_observation, CameraObservation):
            raise TypeError("new_observation must be a CameraObservation")

        def stack_field(field: str) -> Optional[torch.Tensor]:
            left, right = getattr(self, field), getattr(new_observation, field)
            if left is None and right is None:
                return None
            if left is None or right is None:
                raise ValueError(f"cannot stack observations with different {field} availability")
            if left.device != right.device or left.dtype != right.dtype:
                raise ValueError(f"cannot stack {field} with different device or dtype")
            return torch.stack((left, right), dim=dim)

        pose = None
        if self.pose is not None or new_observation.pose is not None:
            if self.pose is None or new_observation.pose is None:
                raise ValueError("cannot stack observations with different pose availability")
            pose = self.pose.stack(new_observation.pose)
        return type(self)(
            name=self.name, rgb_image=stack_field("rgb_image"), depth_image=stack_field("depth_image"),
            image_segmentation=stack_field("image_segmentation"), projection_matrix=stack_field("projection_matrix"),
            projection_rays=stack_field("projection_rays"),
            resolution=None if self.resolution is None else list(self.resolution), pose=pose,
            intrinsics=stack_field("intrinsics"), timestamp=stack_field("timestamp"),
            depth_to_meter=self.depth_to_meter, feature_grid=stack_field("feature_grid"),
        )

    def _as_dict_portable(self) -> dict[str, Any]:
        """Return a tensor-preserving, ``torch.save``-compatible snapshot."""
        result = {field: getattr(self, field) for field in _TENSOR_FIELDS}
        result.update({
            "name": self.name, "resolution": None if self.resolution is None else list(self.resolution),
            "pose": _pose_to_payload(self.pose), "depth_to_meter": self.depth_to_meter,
        })
        return result

    @classmethod
    def _from_dict_portable(cls, value: Mapping[str, Any]) -> "CameraObservation":
        if not isinstance(value, Mapping):
            raise TypeError("camera observation payload must be a mapping")
        fields = {field: value.get(field) for field in _TENSOR_FIELDS}
        return cls(
            name=value.get("name", "camera_image"), resolution=value.get("resolution"),
            pose=_pose_from_payload(value.get("pose")), depth_to_meter=value.get("depth_to_meter", 0.001),
            **fields,
        )

    def save_to_file(self, file_path: str):
        """Persist the portable tensor payload without sensor-driver dependencies."""
        torch.save(self._as_dict_portable(), file_path)

    @classmethod
    def _load_from_file_portable(cls, file_path: str | Path, *, map_location=None) -> "CameraObservation":
        try:
            payload = torch.load(file_path, map_location=map_location, weights_only=False)
        except TypeError:  # torch versions before weights_only support
            payload = torch.load(file_path, map_location=map_location)
        return cls._from_dict_portable(payload)


# Extra tensor-lifecycle affordances are portable extensions rather than part
# of the pinned V2 declaration surface.  Install them from private helpers
# after the class is created so static callers see the V2-owned methods above,
# while CPU/MPS users retain the richer runtime convenience API.
CameraObservation.device = CameraObservation.__dict__["_device_portable"]
CameraObservation.validate = CameraObservation._validate_portable
CameraObservation.detach = CameraObservation._detach_portable
CameraObservation.requires_grad_ = CameraObservation._requires_grad_portable
CameraObservation.as_dict = CameraObservation._as_dict_portable
CameraObservation.from_dict = CameraObservation.__dict__["_from_dict_portable"]
CameraObservation.load_from_file = CameraObservation.__dict__["_load_from_file_portable"]


__all__ = [
    "CameraObservation", "extract_depth_from_structured_pointcloud",
    "get_projection_rays", "project_depth_using_rays",
]
