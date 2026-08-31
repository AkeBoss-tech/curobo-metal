"""Tensor records returned by portable robot forward kinematics.

The state deliberately keeps the FK ``[batch, horizon, ...]`` axes.  It is a
small value object rather than a CUDA output buffer: ordinary PyTorch views,
autograd, and device transfers therefore work the same way on CPU and MPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import ToolPose


def _tensor_like_values(state: "KinematicsState"):
    """Yield materialized state tensors in a stable public-field order."""
    if state.tool_poses is not None:
        yield "tool_poses.position", state.tool_poses.position
        yield "tool_poses.quaternion", state.tool_poses.quaternion
    for name in ("tool_jacobians", "robot_spheres", "robot_com"):
        value = getattr(state, name)
        if value is not None:
            yield name, value


class _KinematicsStatePortableMixin:
    @property
    def batch_size(self) -> int:
        return self._batch_size_value

    @property
    def horizon(self) -> int:
        return self._horizon_value

    @property
    def device(self) -> torch.device:
        return self._device_value

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype_value

    def contiguous(self):
        return self._contiguous()

    def to(self, *args, **kwargs):
        return self._to(*args, **kwargs)

    def requires_grad_(self, *args, **kwargs):
        return self._requires_grad(*args, **kwargs)


@dataclass
class KinematicsState(_KinematicsStatePortableMixin):
    """Kinematic state of a robot.

    The primary FK tensors retain ``[B, H, ...]`` layout: tool poses are
    ``[B,H,L,3/4]``, Jacobians ``[B,H,L,6,dof]``, spheres ``[B,H,S,4]``, and
    centre of mass ``[B,H,4]``.  Indexing follows pinned cuRobo behavior: an
    integer selects a batch entry (with :class:`ToolPose` preserving its
    singleton batch axis), while a tensor index returns the corresponding
    zero-copy PyTorch tensor view/index result.
    """

    tool_poses: Optional[ToolPose] = None
    tool_jacobians: Optional[torch.Tensor] = None
    robot_spheres: Optional[torch.Tensor] = None
    robot_com: Optional[torch.Tensor] = None
    robot_collision_geometry: Optional[RobotCollisionGeometry] = None

    def __post_init__(self) -> None:
        if self.tool_poses is not None and not isinstance(self.tool_poses, ToolPose):
            raise TypeError("tool_poses must be a ToolPose")
        for name, value in _tensor_like_values(self):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        # The pinned ``__getitem__(int)`` convention deliberately keeps the
        # ToolPose batch axis but lets raw tensor fields follow normal tensor
        # indexing.  Do not impose a global rank/layout validation here: it
        # would make a valid indexed view impossible to represent.

    @property
    def tool_frames(self) -> List[str]:
        return [] if self.tool_poses is None else self.tool_poses.tool_frames

    @property
    def _batch_size_value(self) -> int:
        """Number of FK batch entries, or zero for an empty state."""
        return len(self)

    @property
    def _horizon_value(self) -> int:
        """FK horizon length, or zero for an empty state."""
        for _, value in _tensor_like_values(self):
            return int(value.shape[1])
        return 0

    @property
    def _device_value(self) -> torch.device:
        for _, value in _tensor_like_values(self):
            return value.device
        raise ValueError("empty KinematicsState has no device")

    @property
    def _dtype_value(self) -> torch.dtype:
        for _, value in _tensor_like_values(self):
            return value.dtype
        raise ValueError("empty KinematicsState has no dtype")

    def get_link_spheres(self) -> torch.Tensor:
        """Return collision spheres in ``[B,H,S,4]`` layout without copying."""
        return self.robot_spheres

    def clone(self):
        """Deep-copy all materialized tensor and geometry records."""
        return type(self)(
            tool_poses=None if self.tool_poses is None else self.tool_poses.clone(),
            tool_jacobians=None if self.tool_jacobians is None else self.tool_jacobians.clone(),
            robot_spheres=None if self.robot_spheres is None else self.robot_spheres.clone(),
            robot_com=None if self.robot_com is None else self.robot_com.clone(),
            robot_collision_geometry=(
                None
                if self.robot_collision_geometry is None
                else self.robot_collision_geometry.clone()
            ),
        )

    def detach(self):
        """Detach tensor and collision-geometry payloads from autograd."""
        return type(self)(
            tool_poses=None if self.tool_poses is None else self.tool_poses.detach(),
            tool_jacobians=(
                None if self.tool_jacobians is None else self.tool_jacobians.detach()
            ),
            robot_spheres=None if self.robot_spheres is None else self.robot_spheres.detach(),
            robot_com=None if self.robot_com is None else self.robot_com.detach(),
            robot_collision_geometry=(
                None
                if self.robot_collision_geometry is None
                else self.robot_collision_geometry.detach()
            ),
        )

    def _contiguous(self) -> "KinematicsState":
        """Materialize contiguous tensor storage while retaining state metadata."""
        poses = None
        if self.tool_poses is not None:
            poses = self.tool_poses.contiguous()
        return type(self)(
            tool_poses=poses,
            tool_jacobians=(
                None if self.tool_jacobians is None else self.tool_jacobians.contiguous()
            ),
            robot_spheres=(
                None if self.robot_spheres is None else self.robot_spheres.contiguous()
            ),
            robot_com=None if self.robot_com is None else self.robot_com.contiguous(),
            robot_collision_geometry=self.robot_collision_geometry,
        )

    def _to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KinematicsState":
        """Return a state on ``device``/``dtype`` using ordinary PyTorch copies.

        ``DeviceCfg`` is accepted for cuRobo call-site compatibility.  Geometry
        metadata is moved alongside the FK tensors, rather than being silently
        left on CPU after a state has moved to MPS.
        """
        if device_cfg is not None and (device is not None or dtype is not None):
            raise ValueError("pass either device_cfg or device/dtype to KinematicsState.to()")
        if device_cfg is not None:
            options = device_cfg.as_torch_dict()
        else:
            if device is None and dtype is None:
                return self
            options = {}
            if device is not None:
                options["device"] = device
            if dtype is not None:
                options["dtype"] = dtype

        def move(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.to(**options)

        poses = None
        if self.tool_poses is not None:
            poses = ToolPose(
                self.tool_poses.tool_frames.copy(),
                self.tool_poses.position.to(**options),
                self.tool_poses.quaternion.to(**options),
            )
        geometry = self.robot_collision_geometry
        if geometry is not None:
            geometry = RobotCollisionGeometry(
                geometry.link_sphere_idx_map.to(
                    device=options.get("device", geometry.link_sphere_idx_map.device)
                ),
                geometry.num_links,
            )
        return type(self)(poses, move(self.tool_jacobians), move(self.robot_spheres), move(self.robot_com), geometry)

    def _requires_grad(self, requires_grad: bool = True) -> "KinematicsState":
        """Set gradients for every floating-point FK tensor in place."""
        if self.tool_poses is not None:
            self.tool_poses.requires_grad_(requires_grad)
        for name in ("tool_jacobians", "robot_spheres", "robot_com"):
            value = getattr(self, name)
            if value is not None and (value.is_floating_point() or value.is_complex()):
                value.requires_grad_(requires_grad)
        return self

    def copy_(self, other: KinematicsState):
        """Copy materialized fields into a preallocated state buffer.

        A target field that exists must have a corresponding source field. This
        turns a formerly silent partial copy into a precise buffer-layout error.
        """
        if not isinstance(other, KinematicsState):
            raise TypeError("other must be a KinematicsState")
        for name in ("tool_poses", "tool_jacobians", "robot_spheres", "robot_com"):
            target, source = getattr(self, name), getattr(other, name)
            if target is None:
                continue
            if source is None:
                raise ValueError(f"cannot copy missing source field: {name}")
            target.copy_(source)
        if self.robot_collision_geometry is not None and other.robot_collision_geometry is not None:
            self.robot_collision_geometry.copy_(other.robot_collision_geometry)
        return self

    def __len__(self) -> int:
        for _, value in _tensor_like_values(self):
            return int(value.shape[0])
        return 0

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> "KinematicsState":
        """Return the pinned batch-indexed FK state view/index result."""
        return type(self)(
            tool_poses=None if self.tool_poses is None else self.tool_poses[idx],
            tool_jacobians=None if self.tool_jacobians is None else self.tool_jacobians[idx],
            robot_spheres=None if self.robot_spheres is None else self.robot_spheres[idx],
            robot_com=None if self.robot_com is None else self.robot_com[idx],
            robot_collision_geometry=self.robot_collision_geometry,
        )


__all__ = ["KinematicsState", "ToolPose"]
