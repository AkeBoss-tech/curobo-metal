"""Portable robot collision-geometry ownership metadata.

The pinned cuRobo V2 record is deliberately small: one integer link index per
robot sphere.  CUDA uses that tensor as an opaque launch buffer; the Metal
backend instead keeps it as ordinary PyTorch indexing metadata.  That makes
the object safe to clone, move to MPS, and reuse with a caller-owned buffer
without turning its link map into floating-point geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from curobo._src.types.device_cfg import DeviceCfg


_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


@dataclass
class RobotCollisionGeometry:
    """Map every collision sphere to the link that owns it.

    ``link_sphere_idx_map`` is a rank-one integer tensor of shape
    ``[num_spheres]``.  Its values are link indices in ``[0, num_links)``.
    The layout intentionally has no batch or horizon axis: a kinematics
    result may contain many batch/horizon samples, but the robot topology is
    shared by all of them.  This mirrors the pinned V2 data record while
    retaining regular PyTorch CPU/MPS semantics.
    """

    link_sphere_idx_map: torch.Tensor
    num_links: int

    def __post_init__(self) -> None:
        mapping = self.link_sphere_idx_map
        if not isinstance(mapping, torch.Tensor):
            raise TypeError("link_sphere_idx_map must be a torch.Tensor")
        if mapping.ndim != 1:
            raise ValueError("link_sphere_idx_map must have shape [num_spheres]")
        if mapping.dtype not in _INTEGER_DTYPES:
            raise TypeError("link_sphere_idx_map must use an integer dtype")
        if isinstance(self.num_links, bool) or not isinstance(self.num_links, int):
            raise TypeError("num_links must be an integer")
        if self.num_links < 0:
            raise ValueError("num_links must be non-negative")
        if mapping.numel() and self.num_links == 0:
            raise ValueError("num_links must be positive when collision spheres are present")
        if mapping.numel():
            # Convert only for the validation reduction.  The stored map
            # preserves the caller's index dtype (V2 commonly uses int32,
            # while PyTorch indexing sites commonly use int64).
            long_map = mapping.to(dtype=torch.int64)
            if bool(((long_map < 0) | (long_map >= self.num_links)).any().item()):
                raise ValueError("link_sphere_idx_map contains an out-of-range link index")

    @property
    def num_spheres(self) -> int:
        """Number of collision spheres represented by this topology."""
        return int(self.link_sphere_idx_map.numel())

    @property
    def device(self) -> torch.device:
        """Device on which the reusable topology index buffer resides."""
        return self.link_sphere_idx_map.device

    @property
    def dtype(self) -> torch.dtype:
        """Integer dtype of the link-index buffer."""
        return self.link_sphere_idx_map.dtype

    def link_mask(self, link_index: int) -> torch.Tensor:
        """Return a device-resident boolean mask for one owning link.

        A mask is useful for selecting the same sphere subset from any
        ``[B,H,S,...]`` collision tensor without manufacturing a per-batch
        ownership map.  Invalid link indices are rejected before launching an
        MPS comparison operation.
        """
        if isinstance(link_index, bool) or not isinstance(link_index, int):
            raise TypeError("link_index must be an integer")
        if link_index < 0 or link_index >= self.num_links:
            raise ValueError("link_index is out of range")
        return self.link_sphere_idx_map == link_index

    def clone(self) -> "RobotCollisionGeometry":
        """Deep-copy the link-index buffer while retaining link metadata."""
        return RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map.clone(),
            num_links=self.num_links,
        )

    def copy_(self, other: "RobotCollisionGeometry") -> None:
        """Copy into a compatible caller-owned topology buffer.

        Pinned V2 treats ``num_links`` as immutable metadata for a launch
        buffer.  Preserve that behavior rather than silently changing the
        target topology during a tensor-only copy.
        """
        if not isinstance(other, RobotCollisionGeometry):
            raise TypeError("other must be a RobotCollisionGeometry")
        if self.num_links != other.num_links:
            raise ValueError("copy_ requires matching num_links")
        if self.link_sphere_idx_map.shape != other.link_sphere_idx_map.shape:
            raise ValueError("copy_ requires matching link_sphere_idx_map shapes")
        if self.link_sphere_idx_map.device != other.link_sphere_idx_map.device:
            raise ValueError("copy_ requires link_sphere_idx_map tensors on the same device")
        self.link_sphere_idx_map.copy_(other.link_sphere_idx_map)

    def detach(self) -> "RobotCollisionGeometry":
        """Detach the index buffer while preserving storage/device metadata."""
        return RobotCollisionGeometry(
            link_sphere_idx_map=self.link_sphere_idx_map.detach(),
            num_links=self.num_links,
        )

    def contiguous(self) -> "RobotCollisionGeometry":
        """Materialize contiguous index storage when a sliced map is reused."""
        if self.link_sphere_idx_map.is_contiguous():
            return self
        return RobotCollisionGeometry(self.link_sphere_idx_map.contiguous(), self.num_links)

    def to(
        self,
        device: Union[DeviceCfg, torch.device, str, None] = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> "RobotCollisionGeometry":
        """Return topology metadata on a CPU/MPS device without dtype casting.

        ``DeviceCfg`` is accepted for normal cuRobo call sites.  A collision
        ownership map must stay integer-valued, so dtype arguments are
        deliberately absent rather than allowing an accidental float map.
        """
        if device is None and not copy:
            return self
        target = device.device if isinstance(device, DeviceCfg) else device
        moved = self.link_sphere_idx_map.to(
            device=target if target is not None else self.device,
            non_blocking=non_blocking,
            copy=copy,
        )
        return RobotCollisionGeometry(moved, self.num_links)


__all__ = ["RobotCollisionGeometry"]
