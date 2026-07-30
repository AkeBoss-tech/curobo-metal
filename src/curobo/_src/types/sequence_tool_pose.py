from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .tool_pose import GoalToolPose


@dataclass
class SequenceGoalToolPose:
    tool_frames: List[str]
    position: torch.Tensor
    quaternion: torch.Tensor

    def __post_init__(self):
        if self.position.ndim != 5 or self.position.shape[-1] != 3:
            raise ValueError("position must have shape [T,B,L,G,3]")
        if self.quaternion.ndim != 5 or self.quaternion.shape[-1] != 4:
            raise ValueError("quaternion must have shape [T,B,L,G,4]")
        if self.position.shape[2] != len(self.tool_frames):
            raise ValueError("tool_frames must match link dimension")

    num_frames = property(lambda self: self.position.shape[0])
    num_envs = property(lambda self: self.position.shape[1])
    num_links = property(lambda self: self.position.shape[2])
    num_goalset = property(lambda self: self.position.shape[3])
    device = property(lambda self: self.position.device)

    def get_frame(self, t: int):
        return GoalToolPose(
            self.tool_frames, self.position[t].unsqueeze(1),
            self.quaternion[t].unsqueeze(1),
        )

    def clone(self):
        return type(self)(
            self.tool_frames.copy(), self.position.clone(), self.quaternion.clone()
        )


__all__ = ["SequenceGoalToolPose"]
