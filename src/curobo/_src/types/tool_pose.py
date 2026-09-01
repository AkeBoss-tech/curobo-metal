"""Portable value types for batched tool-frame poses.

The pinned V2 surface represents FK output as :class:`ToolPose` with a
``[batch, horizon, link, xyz|wxyz]`` layout, and target poses as
:class:`GoalToolPose` with an additional goalset dimension.  These types are
intentionally ordinary PyTorch tensor containers: they do not rely on CUDA
buffers, Warp structs, or a graph capture runtime, so the same value and
gradient semantics hold on CPU and MPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional, Sequence, Union

import torch

from .device_cfg import DeviceCfg
from .pose import Pose
from ..util.logging import log_and_raise  # pinned public re-export


_Index = Union[int, slice, List[int], torch.Tensor]


def _validate_frames(tool_frames: List[str]) -> None:
    if not isinstance(tool_frames, list) or not all(isinstance(name, str) and name for name in tool_frames):
        raise TypeError("tool_frames must be a list of non-empty strings")
    # A duplicate silently loses a frame in ``to_dict`` and makes named goal
    # tracking ambiguous.  Reject it at the value boundary rather than later
    # in a solver/cost update.
    if len(set(tool_frames)) != len(tool_frames):
        raise ValueError("tool_frames must not contain duplicate names")


def _validate_tensors(position: torch.Tensor, quaternion: torch.Tensor, rank: int, layout: str) -> None:
    if not isinstance(position, torch.Tensor) or not isinstance(quaternion, torch.Tensor):
        raise TypeError("position and quaternion must be torch.Tensor instances")
    if position.ndim != rank:
        raise ValueError(f"position must be {rank}D {layout}, got {tuple(position.shape)}")
    if quaternion.ndim != rank:
        raise ValueError(f"quaternion must be {rank}D {layout[:-2]}4], got {tuple(quaternion.shape)}")
    if position.shape[-1] != 3:
        raise ValueError(f"position must have shape {layout}, got {tuple(position.shape)}")
    if quaternion.shape[-1] != 4:
        raise ValueError(f"quaternion must have shape {layout[:-2]}4], got {tuple(quaternion.shape)}")
    if position.shape[:-1] != quaternion.shape[:-1]:
        raise ValueError(
            "position and quaternion must have identical leading dimensions; "
            f"got {tuple(position.shape)} and {tuple(quaternion.shape)}"
        )
    if position.device != quaternion.device:
        raise ValueError("position and quaternion must reside on the same device")
    if position.dtype != quaternion.dtype:
        raise ValueError("position and quaternion must use the same dtype")
    if not (position.is_floating_point() and quaternion.is_floating_point()):
        raise TypeError("position and quaternion must use floating dtypes")


def _to_options(
    device_cfg: Optional[DeviceCfg], device: Optional[torch.device | str], dtype: Optional[torch.dtype]
) -> Dict[str, object]:
    if device_cfg is not None:
        if device is not None or dtype is not None:
            raise ValueError("pass either device_cfg or device/dtype")
        return device_cfg.as_torch_dict()
    options: Dict[str, object] = {}
    if device is not None:
        options["device"] = device
    if dtype is not None:
        options["dtype"] = dtype
    return options


def _portable_requires_grad_default(method):
    """Keep the portable no-argument convenience without changing its declaration."""

    @wraps(method)
    def wrapped(self, requires_grad: bool = True):
        return method(self, requires_grad)

    return wrapped


class _ToolPosePortableMixin:
    """Portable conveniences kept outside the pinned direct class surface."""

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    def to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ToolPose":
        options = _to_options(device_cfg, device, dtype)
        return type(self)(self.tool_frames.copy(), self.position.to(**options), self.quaternion.to(**options))

    def cpu(self) -> "ToolPose":
        return self.to(device="cpu")


class _GoalToolPosePortableMixin:
    """Portable conveniences kept outside the pinned direct class surface."""

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    def contiguous(self) -> "GoalToolPose":
        return type(self)(self.tool_frames.copy(), self.position.contiguous(), self.quaternion.contiguous())

    def to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GoalToolPose":
        options = _to_options(device_cfg, device, dtype)
        return type(self)(self.tool_frames.copy(), self.position.to(**options), self.quaternion.to(**options))

    def cpu(self) -> "GoalToolPose":
        return self.to(device="cpu")

    def get_goalset(self, goalset_index: int) -> "ToolPose":
        """Return one target from every batch/horizon/link as a 4D ToolPose."""
        if not isinstance(goalset_index, int):
            raise TypeError("goalset_index must be an integer")
        if not -self.num_goalset <= goalset_index < self.num_goalset:
            raise IndexError(f"goalset_index {goalset_index} is outside [0, {self.num_goalset})")
        return ToolPose(
            self.tool_frames.copy(),
            self.position[:, :, :, goalset_index, :],
            self.quaternion[:, :, :, goalset_index, :],
        )


@dataclass
class ToolPose(_ToolPosePortableMixin, Sequence):
    """4D FK output with layout ``[B, H, L, 3/4]``."""

    tool_frames: List[str]
    position: torch.Tensor
    quaternion: torch.Tensor

    def __post_init__(self) -> None:
        _validate_frames(self.tool_frames)
        if isinstance(self.position, torch.Tensor) and self.position.ndim != 4:
            raise ValueError(f"ToolPose position must be 4D [B,H,L,3], got {self.position.shape}")
        if isinstance(self.quaternion, torch.Tensor) and self.quaternion.ndim != 4:
            raise ValueError(f"ToolPose quaternion must be 4D [B,H,L,4], got {self.quaternion.shape}")
        _validate_tensors(self.position, self.quaternion, 4, "[B,H,L,3]")
        if self.position.shape[2] != len(self.tool_frames):
            raise ValueError(
                f"num_links dim ({self.position.shape[2]}) != len(tool_frames) ({len(self.tool_frames)})"
            )

    @property
    def batch_size(self) -> int:
        return self.position.shape[0]

    @property
    def horizon(self) -> int:
        return self.position.shape[1]

    @property
    def num_links(self) -> int:
        return self.position.shape[2]

    @property
    def shape(self):
        return self.position.shape

    @property
    def ndim(self):
        return self.position.ndim

    @property
    def device(self):
        return self.position.device

    def get_link_pose(self, link_name: str, make_contiguous: bool = False) -> Pose:
        """Extract a named link as a flattened 2D ``Pose`` of shape ``[B*H, ...]``."""
        try:
            link_index = self.tool_frames.index(link_name)
        except ValueError as error:
            raise ValueError(f"Link {link_name} not found in {self.tool_frames}") from error
        position = self.position[:, :, link_index, :].reshape(-1, 3)
        quaternion = self.quaternion[:, :, link_index, :].reshape(-1, 4)
        if make_contiguous:
            position, quaternion = position.contiguous(), quaternion.contiguous()
        return Pose(position=position, quaternion=quaternion, name=link_name, normalize_rotation=False)

    def to_dict(self, make_contiguous: bool = True) -> Dict[str, Pose]:
        return {name: self.get_link_pose(name, make_contiguous) for name in self.tool_frames}

    def copy_(self, other: ToolPose):
        if not isinstance(other, ToolPose):
            raise TypeError("other must be a ToolPose")
        if self.position.shape != other.position.shape or self.quaternion.shape != other.quaternion.shape:
            raise ValueError("ToolPose.copy_ requires matching position and quaternion shapes")
        if self.device != other.device or self.dtype != other.dtype:
            raise ValueError("ToolPose.copy_ requires matching device and dtype")
        self.tool_frames = other.tool_frames.copy()
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)
        return self

    @_portable_requires_grad_default
    def requires_grad_(self, requires_grad: bool):
        self.position.requires_grad_(requires_grad)
        self.quaternion.requires_grad_(requires_grad)
        return self

    def clone(self) -> ToolPose:
        return type(self)(self.tool_frames.copy(), self.position.clone(), self.quaternion.clone())

    def detach(self) -> ToolPose:
        return type(self)(self.tool_frames.copy(), self.position.detach(), self.quaternion.detach())

    def contiguous(self) -> ToolPose:
        return type(self)(self.tool_frames.copy(), self.position.contiguous(), self.quaternion.contiguous())

    def __len__(self) -> int:
        return len(self.tool_frames)

    def __getitem__(self, idx: Union[str, _Index]) -> Union[Pose, "ToolPose"]:
        if isinstance(idx, str):
            return self.get_link_pose(idx)
        if isinstance(idx, tuple):
            raise IndexError("ToolPose indexing must select only the batch dimension")
        position, quaternion = self.position[idx], self.quaternion[idx]
        # Integer and scalar-tensor indexing remove B; portable value types
        # preserve the public 4D layout by restoring that singleton batch.
        if position.ndim == 3:
            position, quaternion = position.unsqueeze(0), quaternion.unsqueeze(0)
        if position.ndim != 4:
            raise IndexError("ToolPose indexing must select only the batch dimension")
        return type(self)(self.tool_frames.copy(), position, quaternion)

    def reorder_links(self, ordered_tool_frames: List[str]) -> ToolPose:
        _validate_frames(ordered_tool_frames)
        if not set(ordered_tool_frames).issubset(self.tool_frames):
            raise ValueError(f"Ordered link names {ordered_tool_frames} not a subset of {self.tool_frames}")
        if self.tool_frames == ordered_tool_frames:
            return self
        indices = torch.tensor([self.tool_frames.index(name) for name in ordered_tool_frames], device=self.device)
        return type(self)(
            ordered_tool_frames.copy(),
            self.position.index_select(2, indices).contiguous(),
            self.quaternion.index_select(2, indices).contiguous(),
        )

    def as_goal(self, ordered_tool_frames: Optional[List[str]] = None) -> GoalToolPose:
        value = self.reorder_links(ordered_tool_frames) if ordered_tool_frames is not None else self
        return GoalToolPose(value.tool_frames.copy(), value.position.unsqueeze(3), value.quaternion.unsqueeze(3))


@dataclass
class GoalToolPose(_GoalToolPosePortableMixin, Sequence):
    """5D target pose value with layout ``[B, H, L, G, 3/4]``."""

    tool_frames: List[str]
    position: torch.Tensor
    quaternion: torch.Tensor

    def __post_init__(self) -> None:
        _validate_frames(self.tool_frames)
        if isinstance(self.position, torch.Tensor) and self.position.ndim != 5:
            raise ValueError(f"GoalToolPose position must be 5D [B,H,L,G,3], got {self.position.shape}")
        if isinstance(self.quaternion, torch.Tensor) and self.quaternion.ndim != 5:
            raise ValueError(f"GoalToolPose quaternion must be 5D [B,H,L,G,4], got {self.quaternion.shape}")
        _validate_tensors(self.position, self.quaternion, 5, "[B,H,L,G,3]")
        if self.position.shape[2] != len(self.tool_frames):
            raise ValueError(
                f"num_links dim ({self.position.shape[2]}) != len(tool_frames) ({len(self.tool_frames)})"
            )

    @property
    def batch_size(self) -> int:
        return self.position.shape[0]

    @property
    def horizon(self) -> int:
        return self.position.shape[1]

    @property
    def num_links(self) -> int:
        return self.position.shape[2]

    @property
    def num_goalset(self) -> int:
        return self.position.shape[3]

    @property
    def shape(self):
        return self.position.shape

    @property
    def ndim(self):
        return self.position.ndim

    @property
    def device(self):
        return self.position.device

    @classmethod
    def from_poses(
        cls,
        pose_dict: Dict[str, Pose],
        ordered_tool_frames: Optional[List[str]] = None,
        num_goalset: int = 1,
    ) -> GoalToolPose:
        if not pose_dict:
            raise ValueError("pose_dict cannot be empty")
        if not isinstance(num_goalset, int) or num_goalset < 1:
            raise ValueError("num_goalset must be a positive integer")
        frames = list(ordered_tool_frames) if ordered_tool_frames is not None else list(pose_dict)
        _validate_frames(frames)
        missing = set(frames).difference(pose_dict)
        if missing:
            raise ValueError(f"Missing poses for links: {sorted(missing)}")
        first = pose_dict[frames[0]]
        if not isinstance(first, Pose) or first.position is None or first.quaternion is None:
            raise TypeError("pose_dict values must be materialized Pose instances")
        if first.position.ndim not in (2, 3) or first.position.shape[-1] != 3:
            raise ValueError("pose_dict values must have position shape [N,3] or [B,G,3]")
        if first.quaternion.ndim != first.position.ndim or first.quaternion.shape[:-1] != first.position.shape[:-1] or first.quaternion.shape[-1] != 4:
            raise ValueError("pose_dict values must have matching quaternion shape ending in 4")
        total_batch = first.position.numel() // 3
        if total_batch == 0 or total_batch % num_goalset:
            raise ValueError("Pose batch size must be non-zero and divisible by num_goalset")
        batch = total_batch // num_goalset
        positions, quaternions = [], []
        for frame in frames:
            pose = pose_dict[frame]
            if not isinstance(pose, Pose) or pose.position is None or pose.quaternion is None:
                raise TypeError(f"pose_dict[{frame!r}] must be a materialized Pose")
            if pose.position.shape[:-1] != first.position.shape[:-1] or pose.position.shape[-1] != 3 or pose.quaternion.shape != (*pose.position.shape[:-1], 4):
                raise ValueError(f"pose_dict[{frame!r}] must match the first pose's leading dimensions")
            if pose.position.device != first.position.device or pose.position.dtype != first.position.dtype:
                raise ValueError("all poses must share device and dtype")
            positions.append(pose.position.reshape(batch, num_goalset, 3))
            quaternions.append(pose.quaternion.reshape(batch, num_goalset, 4))
        return cls(frames, torch.stack(positions, 1).unsqueeze(1), torch.stack(quaternions, 1).unsqueeze(1))

    def get_link_pose(self, link_name: str, make_contiguous: bool = False) -> Pose:
        try:
            link_index = self.tool_frames.index(link_name)
        except ValueError as error:
            raise ValueError(f"Link {link_name} not found in {self.tool_frames}") from error
        position = self.position[:, :, link_index, :, :].reshape(-1, 3)
        quaternion = self.quaternion[:, :, link_index, :, :].reshape(-1, 4)
        if make_contiguous:
            position, quaternion = position.contiguous(), quaternion.contiguous()
        return Pose(position=position, quaternion=quaternion, name=link_name, normalize_rotation=False)

    def to_dict(self, make_contiguous: bool = True) -> Dict[str, Pose]:
        return {name: self.get_link_pose(name, make_contiguous) for name in self.tool_frames}

    def copy_(self, other: GoalToolPose):
        if not isinstance(other, GoalToolPose):
            raise TypeError("other must be a GoalToolPose")
        if self.position.shape != other.position.shape or self.quaternion.shape != other.quaternion.shape:
            raise ValueError("GoalToolPose.copy_ requires matching position and quaternion shapes")
        if self.device != other.device or self.dtype != other.dtype:
            raise ValueError("GoalToolPose.copy_ requires matching device and dtype")
        self.tool_frames = other.tool_frames.copy()
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)
        return self

    @_portable_requires_grad_default
    def requires_grad_(self, requires_grad: bool):
        self.position.requires_grad_(requires_grad)
        self.quaternion.requires_grad_(requires_grad)
        return self

    def clone(self) -> GoalToolPose:
        return type(self)(self.tool_frames.copy(), self.position.clone(), self.quaternion.clone())

    def detach(self) -> GoalToolPose:
        return type(self)(self.tool_frames.copy(), self.position.detach(), self.quaternion.detach())

    def __len__(self) -> int:
        return len(self.tool_frames)

    def __getitem__(self, idx: Union[str, _Index]) -> Union[Pose, "GoalToolPose"]:
        if isinstance(idx, str):
            return self.get_link_pose(idx)
        if isinstance(idx, tuple):
            raise IndexError("GoalToolPose indexing must select only the batch dimension")
        position, quaternion = self.position[idx], self.quaternion[idx]
        if position.ndim == 4:
            position, quaternion = position.unsqueeze(0), quaternion.unsqueeze(0)
        if position.ndim != 5:
            raise IndexError("GoalToolPose indexing must select only the batch dimension")
        return type(self)(self.tool_frames.copy(), position, quaternion)

    def reorder_links(self, ordered_tool_frames: List[str]) -> GoalToolPose:
        _validate_frames(ordered_tool_frames)
        if not set(ordered_tool_frames).issubset(self.tool_frames):
            raise ValueError(f"Ordered link names {ordered_tool_frames} not a subset of {self.tool_frames}")
        if self.tool_frames == ordered_tool_frames:
            return self
        indices = torch.tensor([self.tool_frames.index(name) for name in ordered_tool_frames], device=self.device)
        return type(self)(
            ordered_tool_frames.copy(),
            self.position.index_select(2, indices).contiguous(),
            self.quaternion.index_select(2, indices).contiguous(),
        )

__all__ = ["ToolPose", "GoalToolPose"]
