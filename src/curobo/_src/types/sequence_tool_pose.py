"""Time-first goal-tool-pose values used by portable retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from .device_cfg import DeviceCfg
from .tool_pose import GoalToolPose, _to_options, _validate_frames, _validate_tensors
from ..util.logging import log_and_raise  # pinned public re-export


@dataclass
class SequenceGoalToolPose:
    """Goal poses arranged as ``[T, B, L, G, 3/4]`` for contiguous frame views.

    The type is deliberately a tensor value container.  All movement,
    cloning, and first-order autograd paths use regular PyTorch operators and
    therefore remain available on MPS without CUDA or Warp fallback.
    """

    tool_frames: List[str]
    position: torch.Tensor
    quaternion: torch.Tensor

    def __post_init__(self) -> None:
        _validate_frames(self.tool_frames)
        if isinstance(self.position, torch.Tensor) and self.position.ndim != 5:
            raise ValueError(f"SequenceGoalToolPose position must be 5D [T,B,L,G,3], got {self.position.shape}")
        if isinstance(self.quaternion, torch.Tensor) and self.quaternion.ndim != 5:
            raise ValueError(f"SequenceGoalToolPose quaternion must be 5D [T,B,L,G,4], got {self.quaternion.shape}")
        _validate_tensors(self.position, self.quaternion, 5, "[T,B,L,G,3]")
        if self.position.shape[2] != len(self.tool_frames):
            raise ValueError(
                f"num_links dim ({self.position.shape[2]}) does not match len(tool_frames) ({len(self.tool_frames)})"
            )

    @property
    def num_frames(self) -> int:
        return self.position.shape[0]

    @property
    def num_envs(self) -> int:
        return self.position.shape[1]

    @property
    def num_links(self) -> int:
        return self.position.shape[2]

    @property
    def num_goalset(self) -> int:
        return self.position.shape[3]

    @property
    def shape(self) -> torch.Size:
        return self.position.shape

    @property
    def ndim(self) -> int:
        return self.position.ndim

    @property
    def device(self) -> torch.device:
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    def get_frame(self, t: int) -> GoalToolPose:
        """Return a view of a single frame as ``GoalToolPose[B, 1, L, G, ...]``."""
        if not isinstance(t, int):
            raise TypeError("frame index must be an integer")
        if not -self.num_frames <= t < self.num_frames:
            raise IndexError(f"frame index {t} is outside [0, {self.num_frames})")
        return GoalToolPose(self.tool_frames.copy(), self.position[t].unsqueeze(1), self.quaternion[t].unsqueeze(1))

    def clone(self) -> "SequenceGoalToolPose":
        return type(self)(self.tool_frames.copy(), self.position.clone(), self.quaternion.clone())

    def detach(self) -> "SequenceGoalToolPose":
        return type(self)(self.tool_frames.copy(), self.position.detach(), self.quaternion.detach())

    def contiguous(self) -> "SequenceGoalToolPose":
        return type(self)(self.tool_frames.copy(), self.position.contiguous(), self.quaternion.contiguous())

    def requires_grad_(self, requires_grad: bool = True) -> "SequenceGoalToolPose":
        self.position.requires_grad_(requires_grad)
        self.quaternion.requires_grad_(requires_grad)
        return self

    def to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SequenceGoalToolPose":
        options = _to_options(device_cfg, device, dtype)
        return type(self)(self.tool_frames.copy(), self.position.to(**options), self.quaternion.to(**options))

    def cpu(self) -> "SequenceGoalToolPose":
        return self.to(device="cpu")

    def copy_(self, other: "SequenceGoalToolPose") -> "SequenceGoalToolPose":
        if not isinstance(other, SequenceGoalToolPose):
            raise TypeError("other must be a SequenceGoalToolPose")
        if self.position.shape != other.position.shape or self.quaternion.shape != other.quaternion.shape:
            raise ValueError("SequenceGoalToolPose.copy_ requires matching position and quaternion shapes")
        if self.device != other.device or self.dtype != other.dtype:
            raise ValueError("SequenceGoalToolPose.copy_ requires matching device and dtype")
        self.tool_frames = other.tool_frames.copy()
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)
        return self

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: Union[int, slice, List[int], torch.Tensor]) -> Union[GoalToolPose, "SequenceGoalToolPose"]:
        if isinstance(index, int):
            return self.get_frame(index)
        if isinstance(index, tuple):
            raise IndexError("SequenceGoalToolPose indexing must select only the frame dimension")
        position, quaternion = self.position[index], self.quaternion[index]
        if position.ndim != 5:
            raise IndexError("SequenceGoalToolPose indexing must select only the frame dimension")
        return type(self)(self.tool_frames.copy(), position, quaternion)


__all__ = ["SequenceGoalToolPose"]
