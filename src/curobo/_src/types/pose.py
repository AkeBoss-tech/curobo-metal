"""Portable pose implementation for the pinned internal import path."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from curobo_metal.types.math import Pose as _MetalPose

from .device_cfg import DeviceCfg


class Pose(_MetalPose):
    """Pose using xyz positions and ``wxyz`` quaternions."""

    @property
    def shape(self) -> torch.Size:
        if self.position is None:
            raise ValueError("empty Pose has no shape")
        return self.position.shape

    @classmethod
    def from_numpy(
        cls,
        position: np.ndarray,
        quaternion: np.ndarray,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "Pose":
        return cls(device_cfg.to_device(position), device_cfg.to_device(quaternion))

    @classmethod
    def from_batch_list(
        cls,
        pose: list[list[float]],
        device_cfg: DeviceCfg = DeviceCfg(),
        q_xyzw: bool = False,
    ) -> "Pose":
        value = np.asarray(pose)
        if value.ndim != 2 or value.shape[-1] != 7:
            raise ValueError("pose batch must have shape [batch, 7]")
        quaternion = value[..., 3:]
        if q_xyzw:
            quaternion = quaternion[..., [3, 0, 1, 2]]
        return cls(device_cfg.to_device(value[..., :3]), device_cfg.to_device(quaternion))

    @classmethod
    def from_euler_xyz(
        cls, euler_xyz: torch.Tensor, position: Optional[torch.Tensor] = None
    ) -> "Pose":
        rx, ry, rz = (euler_xyz[..., index : index + 1] * 0.5 for index in range(3))
        cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
        sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)
        quaternion = torch.cat(
            (
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
            ),
            dim=-1,
        )
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        if position is None:
            position = torch.zeros(
                (*quaternion.shape[:-1], 3), device=euler_xyz.device, dtype=euler_xyz.dtype
            )
        elif position.ndim == 1:
            position = position.unsqueeze(0)
        return cls(position, quaternion)

    def to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device] = None,
    ) -> "Pose":
        if device_cfg is None and device is None:
            raise ValueError("Pose.to() requires device_cfg or device")
        options = device_cfg.as_torch_dict() if device_cfg is not None else {"device": device}
        for field in ("position", "quaternion", "rotation"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.to(**options))
        return self

    def detach(self) -> "Pose":
        return type(self)(
            None if self.position is None else self.position.detach(),
            None if self.quaternion is None else self.quaternion.detach(),
            None if self.rotation is None else self.rotation.detach(),
        )

    def get_rotation_matrix(self) -> torch.Tensor | None:
        return self.get_rotation()

    def stack(self, other_pose: "Pose") -> "Pose":
        return type(self)(
            torch.vstack((self.position, other_pose.position)),
            torch.vstack((self.quaternion, other_pose.quaternion)),
        )

    def repeat(self, n: int) -> "Pose":
        if n <= 1:
            return self
        return type(self)(self.position.repeat(n, 1), self.quaternion.repeat(n, 1))

    def repeat_seeds(self, num_seeds: int) -> "Pose":
        if self.position is None or self.quaternion is None or num_seeds <= 1:
            return type(self)(self.position, self.quaternion)
        return type(self)(
            self.position.view(self.batch_size, 1, 3)
            .repeat(1, num_seeds, 1)
            .reshape(self.batch_size * num_seeds, 3),
            self.quaternion.view(self.batch_size, 1, 4)
            .repeat(1, num_seeds, 1)
            .reshape(self.batch_size * num_seeds, 4),
        )


__all__ = ["Pose"]
