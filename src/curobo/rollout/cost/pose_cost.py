"""Established pose-cost metric used by motion-planning applications."""

from dataclasses import dataclass
from typing import Optional

import torch

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg


@dataclass
class PoseCostMetric:
    hold_partial_pose: bool = False
    release_partial_pose: bool = False
    hold_vec_weight: Optional[torch.Tensor] = None
    reach_partial_pose: bool = False
    reach_full_pose: bool = False
    reach_vec_weight: Optional[torch.Tensor] = None
    offset_position: Optional[torch.Tensor] = None
    offset_rotation: Optional[torch.Tensor] = None
    offset_tstep_fraction: float = -1.0
    remove_offset_waypoint: bool = False
    include_link_pose: bool = False
    project_to_goal_frame: Optional[bool] = None

    def clone(self):
        values = {
            name: (value.clone() if isinstance(value, torch.Tensor) else value)
            for name, value in vars(self).items()
        }
        return type(self)(**values)

    @classmethod
    def reset_metric(cls):
        return cls(remove_offset_waypoint=True, reach_full_pose=True,
                   release_partial_pose=True)


PoseCost = ToolPoseCost
PoseCostConfig = ToolPoseCostCfg
__all__ = ["PoseCost", "PoseCostConfig", "PoseCostMetric"]
