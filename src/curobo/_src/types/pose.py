"""Portable pose implementation for the pinned internal import path."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from curobo_metal.types.math import Pose as _MetalPose
from curobo_metal.types.math import _quaternion_to_matrix

from .device_cfg import DeviceCfg


class Pose(_MetalPose):
    """Pose using xyz positions and ``wxyz`` quaternions."""

    def __post_init__(self) -> None:
        """Validate all pose representations before the base tensor wrapper.

        The upstream constructor accepts a rotation matrix in lieu of a
        quaternion.  The portable value type deliberately keeps the supplied
        matrix too: callers that use it as an output buffer should not lose
        that representation simply by constructing a :class:`Pose`.

        This is ordinary PyTorch tensor work, so conversion and the resulting
        rotation/quaternion graph remain available on both CPU and MPS.
        """
        if self.rotation is not None:
            if not isinstance(self.rotation, torch.Tensor) or self.rotation.shape[-2:] != (3, 3):
                raise ValueError("rotation must be a tensor ending in shape [3,3]")
            if self.quaternion is None:
                self.quaternion = matrix_to_quaternion(self.rotation)

        if self.position is not None and self.rotation is not None:
            if self.rotation.device != self.position.device or self.rotation.dtype != self.position.dtype:
                raise ValueError("position and rotation must share device and dtype")
            if self.rotation.shape[:-2] != self.position.shape[:-1]:
                raise ValueError("rotation batch dimensions must match position")
            # _MetalPose normalizes one-dimensional position/quaternion inputs
            # to a batch of one.  Keep a user-provided rotation aligned with
            # that convention as well.
            if self.position.ndim == 1:
                self.rotation = self.rotation.unsqueeze(0)

        super().__post_init__()

        if self.rotation is not None and self.position is not None:
            if self.rotation.shape[:-2] != self.position.shape[:-1]:
                raise ValueError("rotation batch dimensions must match position")

    def __eq__(self, other: object) -> bool:
        """Compare rigid transforms with cuRobo's 1e-6 pose tolerance.

        Dataclass-generated equality is not valid for batched tensors because
        ``Tensor.__bool__`` is ambiguous.  This explicit implementation
        matches the V2 public contract and has deterministic scalar behavior.
        """
        if not isinstance(other, Pose):
            return NotImplemented
        if self.position is None or self.quaternion is None:
            return self.position is other.position and self.quaternion is other.quaternion
        if other.position is None or other.quaternion is None:
            return False
        if self.position.shape != other.position.shape or self.quaternion.shape != other.quaternion.shape:
            return False
        if self.device != other.device or self.position.dtype != other.position.dtype:
            return False
        linear, angular = self.distance(other)
        return bool(torch.all(linear <= 1e-6).item() and torch.all(angular <= 1e-6).item())

    @property
    def shape(self) -> torch.Size:
        if self.position is None:
            raise ValueError("empty Pose has no shape")
        return self.position.shape

    @property
    def batch(self):
        return self.batch_size

    @property
    def device(self):
        if self.position is None:
            raise ValueError("empty Pose has no device")
        return self.position.device

    @property
    def ndim(self):
        return 0 if self.position is None else self.position.ndim

    @staticmethod
    def from_matrix(matrix: Union[np.ndarray, torch.Tensor]):
        if not isinstance(matrix, torch.Tensor):
            matrix = DeviceCfg().to_device(matrix)
        if matrix.shape[-2:] != (4, 4):
            raise ValueError("matrix must end in shape [4,4]")
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        rotation = matrix[..., :3, :3].contiguous()
        return Pose(
            position=matrix[..., :3, 3].contiguous(),
            quaternion=matrix_to_quaternion(rotation),
            rotation=rotation,
            normalize_rotation=True,
        )

    @classmethod
    def from_list(cls, pose: List[float], device_cfg: DeviceCfg = DeviceCfg(), q_xyzw=False):
        if len(pose) != 7:
            raise ValueError("pose list must contain xyz and four quaternion values")
        quaternion = pose[3:]
        if q_xyzw:
            quaternion = [quaternion[3], *quaternion[:3]]
        return cls(device_cfg.to_device(pose[:3]), device_cfg.to_device(quaternion))

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
        pose: List[List[float]],
        device_cfg: DeviceCfg = DeviceCfg(),
        q_xyzw=False,
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
            name=self.name,
            normalize_rotation=False,
        )

    def get_rotation_matrix(self) -> torch.Tensor | None:
        return self.get_rotation()

    def get_rotation(self):
        if self.rotation is not None:
            return self.rotation
        return None if self.quaternion is None else quaternion_to_matrix(self.quaternion)

    def get_pose_vector(self):
        if self.position is None or self.quaternion is None:
            raise ValueError("empty Pose has no vector")
        return torch.cat((self.position, self.quaternion), dim=-1)

    def clone(self):
        return type(self)(
            None if self.position is None else self.position.clone(),
            None if self.quaternion is None else self.quaternion.clone(),
            None if self.rotation is None else self.rotation.clone(),
            name=self.name,
            normalize_rotation=False,
        )

    def inverse(self):
        return super().inverse()

    def requires_grad_(self, requires_grad: bool):
        if self.position is not None: self.position.requires_grad_(requires_grad)
        if self.quaternion is not None: self.quaternion.requires_grad_(requires_grad)
        if self.rotation is not None: self.rotation.requires_grad_(requires_grad)
        return self

    def stack(self, other_pose: Pose):
        if not isinstance(other_pose, Pose):
            raise TypeError("other_pose must be a Pose")
        if self.position is None or self.quaternion is None:
            raise ValueError("cannot stack an empty Pose")
        if other_pose.position is None or other_pose.quaternion is None:
            raise ValueError("cannot stack an empty Pose")
        if self.device != other_pose.device or self.position.dtype != other_pose.position.dtype:
            raise ValueError("poses must share device and dtype")
        rotation = None
        if self.rotation is not None and other_pose.rotation is not None:
            rotation = torch.vstack((self.rotation, other_pose.rotation))
        return type(self)(
            torch.vstack((self.position, other_pose.position)),
            torch.vstack((self.quaternion, other_pose.quaternion)),
            rotation=rotation,
            name=self.name,
        )

    def unsqueeze(self, dim=-1):
        for field in ("position", "quaternion", "rotation"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.unsqueeze(dim))
        self._update_shape_params()
        return self

    def squeeze(self, dim=-1):
        for field in ("position", "quaternion", "rotation"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.squeeze(dim))
        return self

    def _update_shape_params(self) -> None:
        if self.position is not None and self.position.ndim > 1:
            self.batch_size = self.position.shape[0]

    def repeat(self, n):
        if n <= 1:
            return self
        rotation = None if self.rotation is None else self.rotation.repeat(n, 1, 1)
        return type(self)(
            self.position.repeat(n, 1), self.quaternion.repeat(n, 1), rotation=rotation, name=self.name
        )

    def repeat_seeds(self, num_seeds: int) -> "Pose":
        if self.position is None or self.quaternion is None or num_seeds <= 1:
            return type(self)(self.position, self.quaternion, self.rotation, name=self.name)
        position = (
            self.position.reshape(self.batch_size, 1, 3)
            .repeat(1, num_seeds, 1)
            .reshape(self.batch_size * num_seeds, 3)
        )
        quaternion = (
            self.quaternion.reshape(self.batch_size, 1, 4)
            .repeat(1, num_seeds, 1)
            .reshape(self.batch_size * num_seeds, 4)
        )
        rotation = None
        if self.rotation is not None:
            rotation = (
                self.rotation.reshape(self.batch_size, 1, 3, 3)
                .repeat(1, num_seeds, 1, 1)
                .reshape(self.batch_size * num_seeds, 3, 3)
            )
        return type(self)(position, quaternion, rotation=rotation, name=self.name)

    def __getitem__(self, index) -> "Pose":
        """Index every materialized representation while retaining batch form."""
        if self.position is None or self.quaternion is None:
            raise IndexError("empty Pose")
        rotation = None if self.rotation is None else self.rotation[index]
        return type(self)(self.position[index], self.quaternion[index], rotation, name=self.name)

    def get_index(self, b: int, n: Optional[int] = None) -> "Pose":
        index = (b, slice(None)) if n is None else (b, n, slice(None))
        rotation_index = (b, slice(None), slice(None)) if n is None else (b, n, slice(None), slice(None))
        rotation = None if self.rotation is None else self.rotation[rotation_index]
        return type(self)(self.position[index], self.quaternion[index], rotation, name=self.name)

    def __setitem__(self, idx: int | torch.Tensor, value: "Pose"):
        if not isinstance(value, Pose):
            raise TypeError("Pose assignment requires a Pose value")
        if self.position is None or self.quaternion is None:
            raise ValueError("cannot assign into an empty Pose")
        if value.position is None or value.quaternion is None:
            raise ValueError("cannot assign an empty Pose")
        self.position[idx] = value.position
        self.quaternion[idx] = value.quaternion
        # A matrix-backed pose is a mutable value buffer in cuRobo.  Do not
        # leave its cached representation stale when the source was built
        # from a quaternion rather than a matrix.
        if self.rotation is not None:
            source_rotation = value.get_rotation()
            assert source_rotation is not None
            self.rotation[idx] = source_rotation

    def apply_kernel(self, kernel_mat):
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

    def copy_(self, pose: Pose):
        if not isinstance(pose, Pose):
            raise TypeError("Pose.copy_() requires a Pose")
        if pose.position is None or pose.quaternion is None:
            raise ValueError("Pose.copy_(): pose.position and pose.quaternion are None")
        if self.position is None or self.quaternion is None:
            raise ValueError("Pose.copy_(): destination position and quaternion must be materialized")
        if self.position.shape != pose.position.shape or self.quaternion.shape != pose.quaternion.shape:
            raise ValueError(f"Copy not possible due to shape mismatch: {pose.position.shape} != {self.position.shape}")
        self.position.copy_(pose.position); self.quaternion.copy_(pose.quaternion)
        if self.rotation is not None:
            source_rotation = pose.get_rotation()
            assert source_rotation is not None
            if self.rotation.shape != source_rotation.shape:
                raise ValueError(
                    f"Copy not possible due to rotation shape mismatch: {source_rotation.shape} != {self.rotation.shape}"
                )
            self.rotation.copy_(source_rotation)

    @staticmethod
    def cat(pose_list: List[Pose]):
        if not pose_list:
            raise ValueError("pose_list cannot be empty")
        if any(x.position is None or x.quaternion is None for x in pose_list):
            raise ValueError("cannot concatenate an empty Pose")
        rotations = [x.rotation for x in pose_list]
        rotation = torch.cat(rotations) if all(x is not None for x in rotations) else None
        first = pose_list[0]
        return type(first)(
            torch.cat([x.position for x in pose_list]),
            torch.cat([x.quaternion for x in pose_list]),
            rotation=rotation,
            name=first.name,
        )

    def linear_distance(self, other_pose: Pose):
        return torch.linalg.norm(self.position - other_pose.position, dim=-1)

    def angular_distance(self, other_pose: Pose, use_phi3: bool = False):
        left = self.quaternion / torch.linalg.vector_norm(self.quaternion, dim=-1, keepdim=True)
        right = other_pose.quaternion / torch.linalg.vector_norm(other_pose.quaternion, dim=-1, keepdim=True)
        dot = torch.abs(torch.sum(left * right, dim=-1)).clamp(max=1)
        return 1 - dot if use_phi3 else 2 * torch.acos(dot)

    def distance(self, other_pose: Pose, use_phi3: bool = False):
        return self.linear_distance(other_pose), self.angular_distance(other_pose, use_phi3)

    def multiply(self, other_pose: Pose, out_position: Optional[torch.Tensor] = None,
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
        if self.position is None or self.quaternion is None:
            raise ValueError("empty Pose cannot transform points")
        if not isinstance(points, torch.Tensor) or points.shape[-1] != 3:
            raise ValueError("points must be a tensor ending in dimension 3")
        if points.device != self.device or points.dtype != self.position.dtype:
            raise ValueError("points and pose must share device and dtype")
        # V2's legacy point transform flattens point ranks above two.  The
        # CUDA op treats a matching flattened pose batch pairwise, rather
        # than letting torch broadcasting create an N-by-N cross-product.
        flattened = points.reshape(-1, 3) if points.ndim > 2 else points
        position = self.position.reshape(-1, 3)
        quaternion = self.quaternion.reshape(-1, 4)
        if position.shape[0] not in (1, flattened.shape[0]):
            raise ValueError(
                "transform_points requires one pose or one pose per flattened point; "
                f"got {position.shape[0]} poses and {flattened.shape[0]} points"
            )
        output = _pairwise_transform_points(position, quaternion, flattened)
        if out_buffer is not None:
            out_buffer.copy_(output); return out_buffer
        return output

    def batch_transform_points(self, points: torch.Tensor, out_buffer: Optional[torch.Tensor] = None,
                               gp_out: Optional[torch.Tensor] = None,
                               gq_out: Optional[torch.Tensor] = None,
                               gpt_out: Optional[torch.Tensor] = None):
        if self.position is None or self.quaternion is None:
            raise ValueError("empty Pose cannot transform points")
        if not isinstance(points, torch.Tensor) or points.ndim <= 2 or points.shape[-1] != 3:
            raise ValueError("batch_transform requires points to be b,n,3 shape")
        if points.device != self.device or points.dtype != self.position.dtype:
            raise ValueError("points and pose must share device and dtype")
        rotation = self.get_rotation().reshape(-1, 3, 3)
        if rotation.shape[0] != points.shape[0]:
            raise ValueError(
                "batch_transform requires the flattened pose batch to match points; "
                f"got {rotation.shape[0]} poses and {points.shape[0]} point batches"
            )
        output = torch.einsum("bij,b...j->b...i", rotation, points) + self.position.reshape(-1, 1, 3)
        if out_buffer is not None:
            out_buffer.copy_(output); return out_buffer
        return output

    def batch_transform_points_inverse(self, points: torch.Tensor,
                                       out_buffer: Optional[torch.Tensor] = None,
                                       gp_out: Optional[torch.Tensor] = None,
                                       gq_out: Optional[torch.Tensor] = None,
                                       gpt_out: Optional[torch.Tensor] = None):
        if self.position is None or self.quaternion is None:
            raise ValueError("empty Pose cannot transform points")
        if not isinstance(points, torch.Tensor) or points.ndim <= 2 or points.shape[-1] != 3:
            raise ValueError("batch_transform_inverse requires points to be b,n,3 shape")
        if points.device != self.device or points.dtype != self.position.dtype:
            raise ValueError("points and pose must share device and dtype")
        rotation = self.get_rotation().reshape(-1, 3, 3).transpose(-1, -2)
        if rotation.shape[0] != points.shape[0]:
            raise ValueError(
                "batch_transform_inverse requires the flattened pose batch to match points; "
                f"got {rotation.shape[0]} poses and {points.shape[0]} point batches"
            )
        output = torch.einsum("bij,b...j->b...i", rotation,
                              points - self.position.reshape(-1, 1, 3))
        if out_buffer is not None:
            out_buffer.copy_(output); return out_buffer
        return output

    def compute_offset_pose(self, offset: Pose):
        return self.multiply(offset)

    def compute_local_pose(self, world_pose: Pose):
        return self.inverse().multiply(world_pose)

    def contiguous(self) -> "Pose":
        return type(self)(
            self.position.contiguous(),
            self.quaternion.contiguous(),
            None if self.rotation is None else self.rotation.contiguous(),
            name=self.name,
        )

    def to_list(self, q_xyzw=False):
        return self.tolist(q_xyzw)

    def tolist(self, q_xyzw=False):
        vector = self.get_pose_vector().detach().cpu().squeeze().tolist()
        if q_xyzw:
            return vector[:3] + vector[4:] + [vector[3]]
        return vector


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize wxyz quaternions with a deterministic zero-norm error."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm == 0).any().item()):
        raise ValueError("quaternion must be nonzero")
    return quaternion / norm


def _pairwise_transform_points(
    position: torch.Tensor, quaternion: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Transform one point per pose, or broadcast a singleton pose.

    The compiled cuRobo transform path has pairwise [N, 3] semantics.  Plain
    ``torch.matmul`` instead interprets two [N, ...] operands as independent
    batch dimensions in this layout, producing an unintended [N, N, 3]
    result.  Keeping this helper in the value layer makes the portable CPU
    and MPS paths agree while retaining an ordinary differentiable graph.
    """
    rotation = quaternion_to_matrix(quaternion)
    if position.shape[0] == 1:
        rotation = rotation.expand(points.shape[0], -1, -1)
        position = position.expand(points.shape[0], -1)
    return torch.einsum("bij,bj->bi", rotation, points) + position


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    return _quaternion_to_matrix(normalize_quaternion(quaternion))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must end in shape [3,3]")
    # Keep every candidate construction out-of-place.  The older lightweight
    # value implementation assigned quaternion signs in-place, which breaks
    # a matrix-backed Pose's autograd graph after the values are saved.
    m = matrix
    # Clamp before ``sqrt`` (rather than after it) so unselected zero-valued
    # candidates cannot contribute an infinite derivative through the gather.
    q_abs = torch.sqrt(torch.clamp(torch.stack((
        1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
        1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
    ), dim=-1), min=torch.finfo(m.dtype).eps))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0].square(), m[..., 2, 1] - m[..., 1, 2],
                     m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), dim=-1),
        torch.stack((m[..., 2, 1] - m[..., 1, 2], q_abs[..., 1].square(),
                     m[..., 1, 0] + m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0]), dim=-1),
        torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1],
                     q_abs[..., 2].square(), m[..., 2, 1] + m[..., 1, 2]), dim=-1),
        torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 2, 0] + m[..., 0, 2],
                     m[..., 2, 1] + m[..., 1, 2], q_abs[..., 3].square()), dim=-1),
    ), dim=-2)
    index = q_abs.argmax(dim=-1)
    selected = candidates.gather(
        -2, index[..., None, None].expand(index.shape + (1, 4))
    ).squeeze(-2)
    denom = (2 * q_abs).clamp_min(torch.finfo(m.dtype).eps)
    quaternion = selected / denom.gather(-1, index[..., None])
    quaternion = normalize_quaternion(quaternion)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def pose_to_matrix(
    position: Pose | torch.Tensor,
    quaternion: Optional[torch.Tensor] = None,
    out_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a homogeneous matrix from either a Pose or raw ``(p, q)`` tensors.

    The raw form has the pinned cuRobo signature.  The Pose form is retained
    as a backwards-compatible convenience for the earlier portable facade.
    """
    if isinstance(position, Pose):
        if out_matrix is not None and quaternion is not None:
            raise TypeError("Pose form accepts one output buffer")
        return position.get_matrix(out_matrix if out_matrix is not None else quaternion)
    if quaternion is None:
        raise TypeError("raw pose_to_matrix requires position and quaternion")
    rotation = quaternion_to_matrix(quaternion)
    matrix = torch.zeros(
        rotation.shape[:-2] + (4, 4), dtype=rotation.dtype, device=rotation.device
    )
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = position
    matrix[..., 3, 3] = 1
    if out_matrix is not None:
        out_matrix.copy_(matrix)
        return out_matrix
    return matrix


def pose_to_affine_matrix(
    position: Pose | torch.Tensor,
    quaternion: Optional[torch.Tensor] = None,
    out_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(position, Pose):
        if out_matrix is not None and quaternion is not None:
            raise TypeError("Pose form accepts one output buffer")
        return position.get_affine_matrix(out_matrix if out_matrix is not None else quaternion)
    if quaternion is None:
        raise TypeError("raw pose_to_affine_matrix requires position and quaternion")
    matrix = pose_to_matrix(position, quaternion)[..., :3, :]
    if out_matrix is not None:
        out_matrix.copy_(matrix)
        return out_matrix
    return matrix


def pose_inverse(
    position: Pose | torch.Tensor,
    quaternion: Optional[torch.Tensor] = None,
    out_position: Optional[torch.Tensor] = None,
    out_quaternion: Optional[torch.Tensor] = None,
) -> Pose | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(position, Pose):
        if quaternion is not None or out_position is not None or out_quaternion is not None:
            raise TypeError("Pose form of pose_inverse does not accept tensor buffers")
        return position.inverse()
    if quaternion is None:
        raise TypeError("raw pose_inverse requires position and quaternion")
    normalized = normalize_quaternion(quaternion)
    inverse_quaternion = torch.cat((normalized[..., :1], -normalized[..., 1:]), dim=-1)
    inverse_position = -torch.matmul(
        position.unsqueeze(-2), quaternion_to_matrix(inverse_quaternion).transpose(-1, -2)
    ).squeeze(-2)
    if out_position is not None:
        out_position.copy_(inverse_position)
        inverse_position = out_position
    if out_quaternion is not None:
        out_quaternion.copy_(inverse_quaternion)
        inverse_quaternion = out_quaternion
    return inverse_position, inverse_quaternion


def pose_multiply(
    position: Pose | torch.Tensor,
    quaternion: Pose | torch.Tensor,
    position2: Optional[torch.Tensor] = None,
    quaternion2: Optional[torch.Tensor] = None,
    out_position: Optional[torch.Tensor] = None,
    out_quaternion: Optional[torch.Tensor] = None,
) -> Pose | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(position, Pose):
        if not isinstance(quaternion, Pose) or any(x is not None for x in (position2, quaternion2, out_position, out_quaternion)):
            raise TypeError("Pose form requires exactly two Pose arguments")
        return position.multiply(quaternion)
    if not isinstance(quaternion, torch.Tensor) or position2 is None or quaternion2 is None:
        raise TypeError("raw pose_multiply requires position, quaternion, position2, quaternion2")
    composed_position = transform_points(position, quaternion, position2.unsqueeze(-2)).squeeze(-2)
    lw, lx, ly, lz = quaternion.unbind(-1)
    rw, rx, ry, rz = quaternion2.unbind(-1)
    composed_quaternion = normalize_quaternion(torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1))
    if out_position is not None:
        out_position.copy_(composed_position)
        composed_position = out_position
    if out_quaternion is not None:
        out_quaternion.copy_(composed_quaternion)
        composed_quaternion = out_quaternion
    return composed_position, composed_quaternion


def transform_points(
    position: Pose | torch.Tensor,
    quaternion: torch.Tensor,
    points: Optional[torch.Tensor] = None,
    out_points: Optional[torch.Tensor] = None,
    adj_position: Optional[torch.Tensor] = None,
    adj_quaternion: Optional[torch.Tensor] = None,
    adj_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(position, Pose):
        if out_points is not None and points is not None:
            raise TypeError("Pose form accepts one output buffer")
        return position.transform_points(quaternion, out_points if out_points is not None else points)
    if points is None:
        raise TypeError("raw transform_points requires position, quaternion, points")
    del adj_position, adj_quaternion, adj_points
    if (
        not isinstance(position, torch.Tensor)
        or not isinstance(quaternion, torch.Tensor)
        or not isinstance(points, torch.Tensor)
        or position.shape[-1] != 3
        or quaternion.shape[-1] != 4
        or points.shape[-1] != 3
    ):
        raise ValueError("position, quaternion, and points must end in 3, 4, and 3 respectively")
    if position.device != quaternion.device or position.device != points.device:
        raise ValueError("position, quaternion, and points must share a device")
    if position.dtype != quaternion.dtype or position.dtype != points.dtype:
        raise ValueError("position, quaternion, and points must share a dtype")
    rotation = quaternion_to_matrix(quaternion)
    if points.ndim == position.ndim:
        # Pointwise transforms, including the common [N, 3] flattened
        # launch layout.  A singleton pose broadcasts over the point batch.
        if position.ndim == 2:
            if position.shape[0] not in (1, points.shape[0]):
                raise ValueError(
                    "pointwise transform requires one pose or one pose per point; "
                    f"got {position.shape[0]} poses and {points.shape[0]} points"
                )
            output = _pairwise_transform_points(position, quaternion, points)
        else:
            output = torch.einsum("...ij,...j->...i", rotation, points) + position
    elif points.ndim == position.ndim + 1:
        # [batch..., n, 3] point clouds: retain batch/horizon dimensions and
        # apply the corresponding transform to every point in the cloud.
        output = torch.einsum("...ij,...nj->...ni", rotation, points) + position.unsqueeze(-2)
    else:
        raise ValueError(
            "points must have either the pose rank for pointwise transforms or "
            "one additional point-set dimension"
        )
    if out_points is not None:
        out_points.copy_(output)
        return out_points
    return output


def batch_transform_points(
    position: Pose | torch.Tensor,
    quaternion: torch.Tensor,
    points: Optional[torch.Tensor] = None,
    out_points: Optional[torch.Tensor] = None,
    adj_position: Optional[torch.Tensor] = None,
    adj_quaternion: Optional[torch.Tensor] = None,
    adj_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(position, Pose):
        if out_points is not None and points is not None:
            raise TypeError("Pose form accepts one output buffer")
        return position.batch_transform_points(quaternion, out_points if out_points is not None else points)
    if points is None:
        raise TypeError("raw batch_transform_points requires position, quaternion, points")
    return transform_points(
        position, quaternion, points, out_points, adj_position, adj_quaternion, adj_points
    )


def batch_transform_points_inverse(
    position: Pose | torch.Tensor,
    quaternion: torch.Tensor,
    points: Optional[torch.Tensor] = None,
    out_points: Optional[torch.Tensor] = None,
    adj_position: Optional[torch.Tensor] = None,
    adj_quaternion: Optional[torch.Tensor] = None,
    adj_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(position, Pose):
        if out_points is not None and points is not None:
            raise TypeError("Pose form accepts one output buffer")
        return position.batch_transform_points_inverse(quaternion, out_points if out_points is not None else points)
    if points is None:
        raise TypeError("raw batch_transform_points_inverse requires position, quaternion, points")
    del adj_position, adj_quaternion, adj_points
    output = torch.matmul(
        points - position.unsqueeze(-2), quaternion_to_matrix(quaternion)
    )
    if out_points is not None:
        out_points.copy_(output)
        return out_points
    return output


def angular_distance_phi3(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, right = normalize_quaternion(left), normalize_quaternion(right)
    return 1 - torch.abs(torch.sum(left * right, dim=-1)).clamp(max=1)


def angular_distance_axis_angle(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, right = normalize_quaternion(left), normalize_quaternion(right)
    return 2 * torch.acos(torch.abs(torch.sum(left * right, dim=-1)).clamp(max=1))


__all__ = [
    "Pose", "angular_distance_axis_angle", "angular_distance_phi3", "batch_transform_points",
    "batch_transform_points_inverse", "matrix_to_quaternion", "normalize_quaternion", "pose_inverse",
    "pose_multiply", "pose_to_affine_matrix", "pose_to_matrix", "quaternion_to_matrix", "transform_points",
]
