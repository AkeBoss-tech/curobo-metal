"""Per-tool pose tracking criteria."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class ToolPoseCriteria:
    terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None
    non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None
    terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None
    non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None
    project_distance_to_goal: Union[torch.Tensor, bool] = False
    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self):
        specs = (("terminal_pose_axes_weight_factor", 6, [1.] * 6),
                 ("non_terminal_pose_axes_weight_factor", 6, [0.] * 6),
                 ("terminal_pose_convergence_tolerance", 2, [0.] * 2),
                 ("non_terminal_pose_convergence_tolerance", 2, [0.] * 2))
        for name, size, default in specs:
            value = getattr(self, name)
            if value is None: value = default
            elif len(value) != size: raise ValueError(f"{name} must be a list of {size} floats, got {value}")
            setattr(self, name, self.device_cfg.to_device(value))
        if isinstance(self.project_distance_to_goal, bool):
            self.project_distance_to_goal = torch.tensor([self.project_distance_to_goal],
                                                         device=self.device_cfg.device, dtype=torch.uint8)
        elif not isinstance(self.project_distance_to_goal, torch.Tensor):
            raise ValueError(f"project_distance_to_goal must be a bool or a torch.Tensor, got {self.project_distance_to_goal}")

    def clone(self):
        return type(self)(self.terminal_pose_axes_weight_factor.clone(),
                          self.non_terminal_pose_axes_weight_factor.clone(),
                          self.terminal_pose_convergence_tolerance.clone(),
                          self.non_terminal_pose_convergence_tolerance.clone(),
                          self.project_distance_to_goal.clone(), self.device_cfg)
    def copy_(self, other):
        if self.device_cfg != other.device_cfg: raise ValueError(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")
        for field in ("terminal_pose_axes_weight_factor", "non_terminal_pose_axes_weight_factor",
                      "terminal_pose_convergence_tolerance", "non_terminal_pose_convergence_tolerance",
                      "project_distance_to_goal"):
            getattr(self, field).copy_(getattr(other, field))
    @staticmethod
    def track_position(xyz: List[float] = [1., 1., 1.]):
        return ToolPoseCriteria([*xyz, 0., 0., 0.], [*xyz, 0., 0., 0.])
    @staticmethod
    def track_orientation(rpy: List[float] = [.001, .001, .001], non_terminal_scale: float = 1.):
        return ToolPoseCriteria([0., 0., 0., *rpy], [0., 0., 0., *(non_terminal_scale*x for x in rpy)])
    @staticmethod
    def track_position_and_orientation(xyz: List[float] = [1., 1., 1.],
                                       rpy: List[float] = [1., 1., 1.],
                                       non_terminal_scale: float = .1):
        terminal = [*xyz, *rpy]
        return ToolPoseCriteria(terminal, [non_terminal_scale*x for x in terminal])
    @staticmethod
    def linear_motion(axis: str = "z", non_terminal_scale: float = 1.,
                      project_distance_to_goal: bool = True):
        if axis not in "xyz" or len(axis) != 1: raise ValueError(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")
        weights = [non_terminal_scale] * 6; weights["xyz".index(axis)] = 0.
        return ToolPoseCriteria([1.] * 6, weights, project_distance_to_goal=project_distance_to_goal)
    @staticmethod
    def disabled(): return ToolPoseCriteria([0.] * 6, [0.] * 6)


__all__ = ["ToolPoseCriteria"]
