"""Tool pose types for FK output and goal specification."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import torch
from .pose import Pose


@dataclass
class ToolPose(Sequence):
    tool_frames: List[str]
    position: torch.Tensor
    quaternion: torch.Tensor

    def __post_init__(self):
        if self.position.ndim != 4:
            raise ValueError(f"ToolPose position must be 4D [B,H,L,3], got {self.position.shape}")
        if self.quaternion.ndim != 4:
            raise ValueError(f"ToolPose quaternion must be 4D [B,H,L,4], got {self.quaternion.shape}")
        if self.position.shape[2] != len(self.tool_frames):
            raise ValueError(f"num_links dim ({self.position.shape[2]}) != len(tool_frames) ({len(self.tool_frames)})")

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
        if link_name not in self.tool_frames:
            raise ValueError(f"Link {link_name} not found in {self.tool_frames}")
        index = self.tool_frames.index(link_name)
        position = self.position[:, :, index, :].reshape(-1, 3)
        quaternion = self.quaternion[:, :, index, :].reshape(-1, 4)
        if make_contiguous:
            position, quaternion = position.contiguous(), quaternion.contiguous()
        return Pose(position=position, quaternion=quaternion, name=link_name, normalize_rotation=False)

    def to_dict(self, make_contiguous: bool = True) -> Dict[str, Pose]:
        return {name: self.get_link_pose(name, make_contiguous) for name in self.tool_frames}

    def copy_(self, other: ToolPose):
        self.tool_frames = other.tool_frames
        self.position.copy_(other.position); self.quaternion.copy_(other.quaternion)

    def requires_grad_(self, requires_grad: bool):
        self.position.requires_grad_(requires_grad); self.quaternion.requires_grad_(requires_grad)

    def clone(self) -> ToolPose:
        return type(self)(self.tool_frames.copy(), self.position.clone(), self.quaternion.clone())

    def detach(self) -> ToolPose:
        return type(self)(self.tool_frames.copy(), self.position.detach(), self.quaternion.detach())

    def contiguous(self) -> ToolPose:
        return type(self)(self.tool_frames, self.position.contiguous(), self.quaternion.contiguous())
    def __len__(self): return len(self.tool_frames)

    def __getitem__(self, idx: Union[int, str, torch.Tensor]):
        if isinstance(idx, str): return self.get_link_pose(idx)
        if isinstance(idx, int):
            return type(self)(self.tool_frames, self.position[idx].unsqueeze(0), self.quaternion[idx].unsqueeze(0))
        return type(self)(self.tool_frames, self.position[idx], self.quaternion[idx])

    def reorder_links(self, ordered_tool_frames: List[str]):
        if not set(ordered_tool_frames).issubset(self.tool_frames):
            raise ValueError(f"Ordered link names {ordered_tool_frames} not a subset of {self.tool_frames}")
        if self.tool_frames == ordered_tool_frames: return self
        indices = [self.tool_frames.index(name) for name in ordered_tool_frames]
        return type(self)(ordered_tool_frames, self.position[:, :, indices, :].contiguous(),
                          self.quaternion[:, :, indices, :].contiguous())

    def as_goal(self, ordered_tool_frames: Optional[List[str]] = None) -> "GoalToolPose":
        value = self.reorder_links(ordered_tool_frames) if ordered_tool_frames else self
        return GoalToolPose(value.tool_frames, value.position.unsqueeze(3), value.quaternion.unsqueeze(3))


@dataclass
class GoalToolPose(Sequence):
    tool_frames: List[str]
    position: torch.Tensor
    quaternion: torch.Tensor

    def __post_init__(self):
        if self.position.ndim != 5:
            raise ValueError(f"GoalToolPose position must be 5D [B,H,L,G,3], got {self.position.shape}")
        if self.quaternion.ndim != 5:
            raise ValueError(f"GoalToolPose quaternion must be 5D [B,H,L,G,4], got {self.quaternion.shape}")
        if self.position.shape[2] != len(self.tool_frames):
            raise ValueError(f"num_links dim ({self.position.shape[2]}) != len(tool_frames) ({len(self.tool_frames)})")

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
    def from_poses(cls, pose_dict: Dict[str, Pose],
                   ordered_tool_frames: Optional[List[str]] = None,
                   num_goalset: int = 1):
        if not pose_dict: raise ValueError("pose_dict cannot be empty")
        frames = list(ordered_tool_frames) if ordered_tool_frames else list(pose_dict)
        missing = set(frames) - set(pose_dict)
        if missing: raise ValueError(f"Missing poses for links: {missing}")
        total_batch = pose_dict[frames[0]].position.shape[0]
        batch = total_batch // num_goalset
        position = torch.stack([pose_dict[x].position.view(batch, num_goalset, 3) for x in frames], 1)
        quaternion = torch.stack([pose_dict[x].quaternion.view(batch, num_goalset, 4) for x in frames], 1)
        return cls(frames, position.unsqueeze(1), quaternion.unsqueeze(1))

    def get_link_pose(self, link_name: str, make_contiguous: bool = False):
        if link_name not in self.tool_frames: raise ValueError(f"Link {link_name} not found in {self.tool_frames}")
        index = self.tool_frames.index(link_name)
        position = self.position[:, :, index, :, :].reshape(-1, 3)
        quaternion = self.quaternion[:, :, index, :, :].reshape(-1, 4)
        if make_contiguous: position, quaternion = position.contiguous(), quaternion.contiguous()
        return Pose(position=position, quaternion=quaternion, name=link_name, normalize_rotation=False)

    def to_dict(self, make_contiguous: bool = True):
        return {name: self.get_link_pose(name, make_contiguous) for name in self.tool_frames}
    def copy_(self, other: GoalToolPose):
        self.tool_frames = other.tool_frames; self.position.copy_(other.position); self.quaternion.copy_(other.quaternion)
    def requires_grad_(self, requires_grad: bool):
        self.position.requires_grad_(requires_grad); self.quaternion.requires_grad_(requires_grad)
    def clone(self) -> GoalToolPose:
        return type(self)(self.tool_frames.copy(), self.position.clone(), self.quaternion.clone())

    def detach(self) -> GoalToolPose:
        return type(self)(self.tool_frames.copy(), self.position.detach(), self.quaternion.detach())
    def __len__(self): return len(self.tool_frames)
    def __getitem__(self, idx):
        if isinstance(idx, str): return self.get_link_pose(idx)
        if isinstance(idx, int): return type(self)(self.tool_frames, self.position[idx].unsqueeze(0), self.quaternion[idx].unsqueeze(0))
        return type(self)(self.tool_frames, self.position[idx], self.quaternion[idx])
    def reorder_links(self, ordered_tool_frames: List[str]) -> GoalToolPose:
        if not set(ordered_tool_frames).issubset(self.tool_frames):
            raise ValueError(f"Ordered link names {ordered_tool_frames} not a subset of {self.tool_frames}")
        if self.tool_frames == ordered_tool_frames: return self
        indices = [self.tool_frames.index(x) for x in ordered_tool_frames]
        return type(self)(ordered_tool_frames, self.position[:, :, indices, :, :].contiguous(),
                          self.quaternion[:, :, indices, :, :].contiguous())


__all__ = ["ToolPose", "GoalToolPose"]
