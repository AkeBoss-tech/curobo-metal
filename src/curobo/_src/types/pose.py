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

    @staticmethod
    def _euler_xyz_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
        return Pose.from_euler_xyz(euler_xyz).quaternion.squeeze(0) if euler_xyz.ndim == 1 else Pose.from_euler_xyz(euler_xyz).quaternion

    @staticmethod
    def _euler_xyz_intrinsic_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
        rx, ry, rz = (euler_xyz[..., index : index + 1] * 0.5 for index in range(3))
        cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
        sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)
        return torch.cat((cx*cy*cz-sx*sy*sz, sx*cy*cz+cx*sy*sz,
                          cx*sy*cz-sx*cy*sz, cx*cy*sz+sx*sy*cz), dim=-1)

    @classmethod
    def from_euler_xyz_intrinsic(
        cls, euler_xyz: torch.Tensor, position: Optional[torch.Tensor] = None
    ) -> "Pose":
        quaternion = cls._euler_xyz_intrinsic_to_quaternion(euler_xyz)
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        if position is None:
            position = torch.zeros((*quaternion.shape[:-1], 3), device=euler_xyz.device,
                                   dtype=euler_xyz.dtype)
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

    def requires_grad_(self, requires_grad: bool):
        if self.position is not None: self.position.requires_grad_(requires_grad)
        if self.quaternion is not None: self.quaternion.requires_grad_(requires_grad)

    def stack(self, other_pose: "Pose") -> "Pose":
        return type(self)(
            torch.vstack((self.position, other_pose.position)),
            torch.vstack((self.quaternion, other_pose.quaternion)),
        )

    def unsqueeze(self, dim: int = -1) -> "Pose":
        for field in ("position", "quaternion", "rotation"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.unsqueeze(dim))
        self._update_shape_params()
        return self

    def squeeze(self, dim: int = -1) -> "Pose":
        for field in ("position", "quaternion", "rotation"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.squeeze(dim))
        return self

    def _update_shape_params(self) -> None:
        if self.position is not None and self.position.ndim > 1:
            self.batch_size = self.position.shape[0]

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

    def get_index(self, b: int, n: Optional[int] = None) -> "Pose":
        index = (b, slice(None)) if n is None else (b, n, slice(None))
        return type(self)(self.position[index], self.quaternion[index])

    def __setitem__(self, idx: int | torch.Tensor, value: "Pose"):
        self.position[idx] = value.position
        self.quaternion[idx] = value.quaternion

    def apply_kernel(self, kernel_mat: torch.Tensor) -> "Pose":
        if self.position is None:
            return self
        return type(self)(kernel_mat @ self.position, kernel_mat @ self.quaternion)

    def get_affine_matrix(self, out_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        matrix = self.get_matrix()
        affine = matrix[..., :3, :]
        if out_matrix is not None:
            out_matrix.copy_(affine)
            return out_matrix
        return affine

    def get_matrix(self, out_matrix: Optional[torch.Tensor] = None):
        matrix = super().get_matrix()
        if out_matrix is not None:
            out_matrix.copy_(matrix)
            return out_matrix
        return matrix

    def get_numpy_affine_matrix(self):
        return self.get_affine_matrix().cpu().numpy()

    def get_numpy_matrix(self):
        return self.get_matrix().cpu().numpy()

    def copy_(self, pose: "Pose"):
        if pose.position is None and pose.quaternion is None:
            raise ValueError("Pose.copy_(): pose.position and pose.quaternion are None")
        if self.position.shape != pose.position.shape or self.quaternion.shape != pose.quaternion.shape:
            raise ValueError(f"Copy not possible due to shape mismatch: {pose.position.shape} != {self.position.shape}")
        self.position.copy_(pose.position); self.quaternion.copy_(pose.quaternion)

    @staticmethod
    def cat(pose_list: list["Pose"]) -> "Pose":
        return Pose(torch.cat([x.position for x in pose_list]),
                    torch.cat([x.quaternion for x in pose_list]))

    def linear_distance(self, other_pose: "Pose"):
        return torch.linalg.norm(self.position - other_pose.position, dim=-1)

    def angular_distance(self, other_pose: "Pose", use_phi3: bool = False):
        left = self.quaternion / torch.linalg.vector_norm(self.quaternion, dim=-1, keepdim=True)
        right = other_pose.quaternion / torch.linalg.vector_norm(other_pose.quaternion, dim=-1, keepdim=True)
        dot = torch.abs(torch.sum(left * right, dim=-1)).clamp(max=1)
        return 1 - dot if use_phi3 else 2 * torch.acos(dot)

    def distance(self, other_pose: "Pose", use_phi3: bool = False):
        return self.linear_distance(other_pose), self.angular_distance(other_pose, use_phi3)

    def multiply(self, other_pose: "Pose", out_position: Optional[torch.Tensor] = None,
                 out_quaternion: Optional[torch.Tensor] = None):
        result = super().multiply(other_pose)
        if out_position is not None:
            out_position.copy_(result.position); result.position = out_position
        if out_quaternion is not None:
            out_quaternion.copy_(result.quaternion); result.quaternion = out_quaternion
        return result

    def transform_point(self, points: torch.Tensor, out_buffer: Optional[torch.Tensor] = None,
                        gp_out: Optional[torch.Tensor] = None,
                        gq_out: Optional[torch.Tensor] = None,
                        gpt_out: Optional[torch.Tensor] = None):
        return self.transform_points(points, out_buffer, gp_out, gq_out, gpt_out)

    def transform_points(self, points: torch.Tensor, out_buffer: Optional[torch.Tensor] = None,
                         gp_out: Optional[torch.Tensor] = None,
                         gq_out: Optional[torch.Tensor] = None,
                         gpt_out: Optional[torch.Tensor] = None):
        if points.ndim > 2:
            points = points.view(-1, 3)
        output = super().transform_points(points)
        if out_buffer is not None:
            out_buffer.copy_(output); return out_buffer
        return output

    def batch_transform_points(self, points: torch.Tensor, out_buffer: Optional[torch.Tensor] = None,
                               gp_out=None, gq_out=None, gpt_out=None):
        if points.ndim <= 2:
            raise ValueError("batch_transform requires points to be b,n,3 shape")
        rotation = self.get_rotation().view(-1, 3, 3)
        output = torch.einsum("bij,b...j->b...i", rotation, points) + self.position.view(-1, 1, 3)
        if out_buffer is not None:
            out_buffer.copy_(output); return out_buffer
        return output

    def batch_transform_points_inverse(self, points: torch.Tensor,
                                       out_buffer: Optional[torch.Tensor] = None,
                                       gp_out=None, gq_out=None, gpt_out=None):
        rotation = self.get_rotation().view(-1, 3, 3).transpose(-1, -2)
        output = torch.einsum("bij,b...j->b...i", rotation,
                              points - self.position.view(-1, 1, 3))
        if out_buffer is not None:
            out_buffer.copy_(output); return out_buffer
        return output

    def compute_offset_pose(self, offset: "Pose") -> "Pose":
        return self.multiply(offset)

    def compute_local_pose(self, world_pose: "Pose") -> "Pose":
        return self.inverse().multiply(world_pose)

    def contiguous(self) -> "Pose":
        return type(self)(self.position.contiguous(), self.quaternion.contiguous())


__all__ = ["Pose"]
