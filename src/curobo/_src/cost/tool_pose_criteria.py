"""Per-tool pose tracking criteria."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
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

@dataclass
class StackedToolPoseCriteria:
    tool_frames: List[str]
    terminal_pose_axes_weight_factor: torch.Tensor
    non_terminal_pose_axes_weight_factor: torch.Tensor
    terminal_pose_convergence_tolerance: torch.Tensor
    non_terminal_pose_convergence_tolerance: torch.Tensor
    project_distance_to_goal: torch.Tensor
    device_cfg: DeviceCfg = DeviceCfg()
    _tool_pose_criteria: Optional[Dict[str, ToolPoseCriteria]] = None

    @classmethod
    def from_tool_pose_criteria(cls, tool_pose_criteria):
        frames = list(tool_pose_criteria)
        values = [tool_pose_criteria[name] for name in frames]
        return cls(
            frames,
            torch.stack([x.terminal_pose_axes_weight_factor for x in values]),
            torch.stack([x.non_terminal_pose_axes_weight_factor for x in values]),
            torch.stack([x.terminal_pose_convergence_tolerance for x in values]),
            torch.stack([x.non_terminal_pose_convergence_tolerance for x in values]),
            torch.stack([x.project_distance_to_goal for x in values]),
            values[0].device_cfg if values else DeviceCfg(),
            dict(tool_pose_criteria),
        )

    def clone(self):
        return type(self)(
            self.tool_frames.copy(),
            self.terminal_pose_axes_weight_factor.clone(),
            self.non_terminal_pose_axes_weight_factor.clone(),
            self.terminal_pose_convergence_tolerance.clone(),
            self.non_terminal_pose_convergence_tolerance.clone(),
            self.project_distance_to_goal.clone(),
            self.device_cfg,
            None if self._tool_pose_criteria is None else
            {k: v.clone() for k, v in self._tool_pose_criteria.items()},
        )

    def update_tool_pose_criteria(self, tool_pose_criteria):
        if not isinstance(tool_pose_criteria, dict):
            raise TypeError("tool_pose_criteria must be a mapping")
        for name, criteria in tool_pose_criteria.items():
            if name not in self.tool_frames:
                raise ValueError(f"link_name {name!r} not found in tool_frames")
            if not isinstance(criteria, ToolPoseCriteria):
                raise TypeError(f"criterion for {name!r} must be a ToolPoseCriteria")
            index = self.tool_frames.index(name)
            # Keep the stacked tensors allocated by the original setup.  This
            # mirrors cuRobo's in-place criteria update and lets a long-lived
            # cost change one link without invalidating the others.
            self.terminal_pose_axes_weight_factor[index].copy_(criteria.terminal_pose_axes_weight_factor)
            self.non_terminal_pose_axes_weight_factor[index].copy_(criteria.non_terminal_pose_axes_weight_factor)
            self.terminal_pose_convergence_tolerance[index].copy_(criteria.terminal_pose_convergence_tolerance)
            self.non_terminal_pose_convergence_tolerance[index].copy_(criteria.non_terminal_pose_convergence_tolerance)
            self.project_distance_to_goal[index].copy_(criteria.project_distance_to_goal)
            if self._tool_pose_criteria is not None:
                self._tool_pose_criteria[name].copy_(criteria)


__all__ = ["ToolPoseCriteria", "StackedToolPoseCriteria"]
