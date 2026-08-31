"""Pose value type using cuRobo's wxyz quaternion convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from .device import DeviceCfg


@dataclass
class Pose(Sequence["Pose"]):
    position: torch.Tensor | None = None
    quaternion: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    batch_size: int = 1
    name: str = "ee_link"
    normalize_rotation: bool = False

    def __post_init__(self) -> None:
        if self.position is None:
            return
        if self.position.shape[-1] != 3:
            raise ValueError("position must end in dimension 3")
        if self.position.ndim == 1:
            self.position = self.position.unsqueeze(0)
        if self.quaternion is None:
            self.quaternion = self.position.new_zeros((*self.position.shape[:-1], 4))
            self.quaternion[..., 0] = 1
        elif self.quaternion.ndim == 1:
            self.quaternion = self.quaternion.unsqueeze(0)
        if self.quaternion.shape[:-1] != self.position.shape[:-1] or self.quaternion.shape[-1] != 4:
            raise ValueError("quaternion must match position batch dimensions and end in 4")
        if self.quaternion.device != self.position.device or self.quaternion.dtype != self.position.dtype:
            raise ValueError("position and quaternion must share device and dtype")
        if self.normalize_rotation:
            norm = torch.linalg.vector_norm(self.quaternion, dim=-1, keepdim=True)
            # Match pinned cuRobo: a zero quaternion is retained instead of
            # raising, while nonzero values are normalized and canonicalized
            # to a non-negative scalar component.
            norm = torch.where(norm > 1e-7, norm, torch.ones_like(norm))
            self.quaternion = self.quaternion / norm
            sign = torch.sign(self.quaternion[..., :1])
            self.quaternion = self.quaternion * torch.where(
                sign == 0, torch.ones_like(sign), sign
            )
        self.batch_size = self.position.shape[0]

    @classmethod
    def from_list(
        cls, value: Sequence[float], device_cfg: DeviceCfg = DeviceCfg(), q_xyzw: bool = False
    ) -> "Pose":
        if len(value) != 7:
            raise ValueError("pose list must contain xyz and four quaternion values")
        quaternion = value[3:]
        if q_xyzw:
            quaternion = [quaternion[3], *quaternion[:3]]
        return cls(device_cfg.to_device(value[:3]), device_cfg.to_device(quaternion))

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "Pose":
        if matrix.shape[-2:] != (4, 4):
            raise ValueError("matrix must end in shape [4,4]")
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        rotation = matrix[..., :3, :3]
        quaternion = _matrix_to_quaternion(rotation)
        return cls(matrix[..., :3, 3], quaternion, rotation)

    @property
    def device(self) -> torch.device:
        if self.position is None:
            raise ValueError("empty Pose has no device")
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        if self.position is None:
            raise ValueError("empty Pose has no dtype")
        return self.position.dtype

    @property
    def ndim(self) -> int:
        return 0 if self.position is None else self.position.ndim

    def clone(self) -> "Pose":
        return type(self)(
            None if self.position is None else self.position.clone(),
            None if self.quaternion is None else self.quaternion.clone(),
            None if self.rotation is None else self.rotation.clone(),
            name=self.name,
        )

    def to(
        self,
        device_cfg: DeviceCfg | torch.device | str | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Pose":
        if isinstance(device_cfg, DeviceCfg):
            options = device_cfg.as_torch_dict()
        else:
            target = device if device is not None else device_cfg
            options = {"device": self.device if target is None else target, "dtype": dtype or self.dtype}
        return type(self)(
            None if self.position is None else self.position.to(**options),
            None if self.quaternion is None else self.quaternion.to(**options),
            None if self.rotation is None else self.rotation.to(**options),
            name=self.name,
        )

    def get_rotation(self) -> torch.Tensor | None:
        if self.rotation is not None:
            return self.rotation
        return None if self.quaternion is None else _quaternion_to_matrix(self.quaternion)

    def get_matrix(self) -> torch.Tensor:
        if self.position is None:
            raise ValueError("empty Pose has no matrix")
        rotation = self.get_rotation()
        assert rotation is not None
        output = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            *self.position.shape[:-1], 4, 4
        ).clone()
        output[..., :3, :3] = rotation
        output[..., :3, 3] = self.position
        return output

    def get_pose_vector(self) -> torch.Tensor:
        if self.position is None or self.quaternion is None:
            raise ValueError("empty Pose has no vector")
        return torch.cat((self.position, self.quaternion), dim=-1)

    def inverse(self) -> "Pose":
        """Return the differentiable inverse rigid transform."""
        if self.position is None or self.quaternion is None:
            raise ValueError("empty Pose has no inverse")
        q = self.quaternion / torch.linalg.vector_norm(
            self.quaternion, dim=-1, keepdim=True
        )
        inverse_q = torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
        inverse_rotation = _quaternion_to_matrix(inverse_q)
        inverse_position = -(inverse_rotation @ self.position[..., None]).squeeze(-1)
        return type(self)(inverse_position, inverse_q, inverse_rotation, name=self.name)

    def multiply(self, other: "Pose") -> "Pose":
        """Compose ``self`` with ``other`` using broadcastable batch dimensions."""
        if not isinstance(other, Pose):
            raise TypeError("other must be a Pose")
        if (
            self.position is None or self.quaternion is None
            or other.position is None or other.quaternion is None
        ):
            raise ValueError("cannot multiply an empty Pose")
        if self.device != other.device or self.dtype != other.dtype:
            raise ValueError("poses must share device and dtype")
        left_q, right_q = torch.broadcast_tensors(self.quaternion, other.quaternion)
        left_p, right_p = torch.broadcast_tensors(self.position, other.position)
        position = left_p + (
            _quaternion_to_matrix(left_q) @ right_p[..., None]
        ).squeeze(-1)
        quaternion = _quaternion_multiply(left_q, right_q)
        return type(self)(position, quaternion, name=other.name)

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the pose to points with standard torch broadcasting."""
        if self.position is None:
            raise ValueError("empty Pose cannot transform points")
        if not isinstance(points, torch.Tensor) or points.shape[-1] != 3:
            raise ValueError("points must be a tensor ending in dimension 3")
        if points.device != self.device or points.dtype != self.dtype:
            raise ValueError("points and pose must share device and dtype")
        rotation = self.get_rotation()
        assert rotation is not None
        return (rotation @ points[..., None]).squeeze(-1) + self.position

    def __mul__(self, other: "Pose") -> "Pose":
        return self.multiply(other)

    def tolist(self, q_xyzw: bool = False) -> list[float]:
        vector = self.get_pose_vector().squeeze().detach().cpu().tolist()
        if q_xyzw:
            return vector[:3] + vector[4:] + [vector[3]]
        return vector

    to_list = tolist

    def __getitem__(self, index: Any) -> "Pose":
        if self.position is None or self.quaternion is None:
            raise IndexError("empty Pose")
        return type(self)(self.position[index], self.quaternion[index], name=self.name)

    def __len__(self) -> int:
        return self.batch_size

    def __iter__(self) -> Iterator["Pose"]:
        return (self[index] for index in range(len(self)))


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    matrix = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)
    # Warp's pinned zero-quaternion edge case yields -I.  Preserve that
    # observable contract without sacrificing the numerically stabler
    # normalized-quaternion formula for valid rotations.
    zero = (q * q).sum(dim=-1, keepdim=True) == 0
    negative_identity = -torch.eye(3, dtype=q.dtype, device=q.device).expand_as(matrix)
    return torch.where(zero.unsqueeze(-1), negative_identity, matrix)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # Stable branch-free conversion; copysign fixes the vector signs.
    q = torch.stack((
        torch.sqrt(torch.clamp(1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2], min=0)) / 2,
        torch.sqrt(torch.clamp(1 + matrix[..., 0, 0] - matrix[..., 1, 1] - matrix[..., 2, 2], min=0)) / 2,
        torch.sqrt(torch.clamp(1 - matrix[..., 0, 0] + matrix[..., 1, 1] - matrix[..., 2, 2], min=0)) / 2,
        torch.sqrt(torch.clamp(1 - matrix[..., 0, 0] - matrix[..., 1, 1] + matrix[..., 2, 2], min=0)) / 2,
    ), dim=-1)
    q[..., 1] = torch.copysign(q[..., 1], matrix[..., 2, 1] - matrix[..., 1, 2])
    q[..., 2] = torch.copysign(q[..., 2], matrix[..., 0, 2] - matrix[..., 2, 0])
    q[..., 3] = torch.copysign(q[..., 3], matrix[..., 1, 0] - matrix[..., 0, 1])
    return q
