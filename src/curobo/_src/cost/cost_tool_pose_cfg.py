"""Validated portable configuration and façade for multi-link tool-pose costs.

The CUDA/Warp implementation behind the pinned module is intentionally not
replicated.  The actual pose residual is the differentiable PyTorch cost in
``cost.portable``; this module makes its public configuration lifecycle
strict, stable, and safe for batched CPU/MPS callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch

from .portable import BaseCostCfg, ToolPoseCost as _PortableToolPoseCost
from .portable import ToolPoseCostCfg as _PortableToolPoseCostCfg
from .tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


class ToolPoseCost(_PortableToolPoseCost):
    """Portable pose cost with explicit public input/device validation."""

    def _apply_weight(self, value: torch.Tensor) -> torch.Tensor:
        # A configuration can be constructed before the caller chooses a
        # rollout device.  Keep that common CPU-config/MPS-execution pattern
        # device-local without mutating the reusable configuration tensor.
        weight = self._weight.to(device=value.device, dtype=value.dtype)
        weight = weight if weight.numel() == 1 else weight.mean()
        result = value * weight
        if self.config.convert_to_binary:
            result = (result > 0).to(result.dtype)
        return result if self.enabled else result * 0

    def forward(self, current_tool_poses: ToolPose, goal_tool_poses: GoalToolPose, idxs_goal=None, **kwargs):
        if not isinstance(current_tool_poses, ToolPose):
            raise TypeError("current_tool_poses must be a ToolPose")
        if not isinstance(goal_tool_poses, GoalToolPose):
            raise TypeError("goal_tool_poses must be a GoalToolPose")
        tensors = (
            current_tool_poses.position,
            current_tool_poses.quaternion,
            goal_tool_poses.position,
            goal_tool_poses.quaternion,
        )
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("tool pose positions and quaternions must be floating tensors")
        devices = {value.device for value in tensors}
        if len(devices) != 1:
            raise ValueError("current and goal tool poses must share one device")
        if idxs_goal is not None:
            if not isinstance(idxs_goal, torch.Tensor) or idxs_goal.device != current_tool_poses.position.device:
                raise ValueError("idxs_goal must be a tensor on the tool-pose device")
            if idxs_goal.ndim != 1 or idxs_goal.numel() != current_tool_poses.position.shape[0]:
                raise ValueError("idxs_goal must have shape [batch]")
            if idxs_goal.dtype not in (torch.int32, torch.int64):
                raise TypeError("idxs_goal must have an integer dtype")
            if idxs_goal.numel() and (bool((idxs_goal < 0).any()) or bool((idxs_goal >= goal_tool_poses.position.shape[0]).any())):
                raise ValueError("idxs_goal contains an out-of-range goal batch index")
        return super().forward(current_tool_poses, goal_tool_poses, idxs_goal, **kwargs)

    __call__ = forward


@dataclass
class ToolPoseCostCfg(_PortableToolPoseCostCfg):
    """Configuration with stable frame/criterion ownership and clone semantics."""

    class_type: Type[ToolPoseCost] = field(default_factory=lambda: ToolPoseCost)
    tool_frames: Optional[List[str]] = None
    tool_pose_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = None if self.tool_frames is None else list(self.tool_frames)
        supplied = dict(self.tool_pose_criteria)
        if frames is not None:
            if not all(isinstance(name, str) and name for name in frames):
                raise ValueError("tool_frames must contain non-empty strings")
            if len(set(frames)) != len(frames):
                raise ValueError("tool_frames must be unique")
        if not isinstance(self.use_lie_group, bool):
            raise TypeError("use_lie_group must be bool")
        if frames is None and supplied:
            raise ValueError("tool_pose_criteria requires tool_frames")
        unknown = set(supplied).difference(frames or [])
        if unknown:
            raise ValueError(f"tool_pose_criteria contains unknown frames: {sorted(unknown)}")
        for name, criteria in supplied.items():
            if not isinstance(criteria, ToolPoseCriteria):
                raise TypeError(f"criterion for {name!r} must be a ToolPoseCriteria")
            if criteria.device_cfg != self.device_cfg:
                raise ValueError(f"criterion for {name!r} has a different device_cfg")

        # Build a detached default template once and preserve caller-provided
        # per-link criteria rather than overwriting them in set_tool_frames.
        template = self._pose_criteria
        if template is None:
            template = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=self._terminal_pose_axes_weight_factor,
                non_terminal_pose_axes_weight_factor=self._non_terminal_pose_axes_weight_factor,
                terminal_pose_convergence_tolerance=self._terminal_pose_convergence_tolerance,
                non_terminal_pose_convergence_tolerance=self._non_terminal_pose_convergence_tolerance,
                project_distance_to_goal=self._project_distance_to_goal,
                device_cfg=self.device_cfg,
            )
        if not isinstance(template, ToolPoseCriteria):
            raise TypeError("_pose_criteria must be a ToolPoseCriteria or None")
        if template.device_cfg != self.device_cfg:
            raise ValueError("_pose_criteria has a different device_cfg")

        BaseCostCfg.__post_init__(self)
        self._pose_criteria = template.clone()
        self.tool_frames = frames
        self.tool_pose_criteria = {}
        if frames is not None:
            for name in frames:
                self.tool_pose_criteria[name] = supplied.get(name, template).clone()

    def set_tool_frames(self, tool_frames: List[str]):
        if not isinstance(tool_frames, (list, tuple)) or not all(isinstance(name, str) and name for name in tool_frames):
            raise ValueError("tool_frames must contain non-empty strings")
        frames = list(tool_frames)
        if len(set(frames)) != len(frames):
            raise ValueError("tool_frames must be unique")
        if self._pose_criteria is None:
            raise ValueError("pose criteria must be initialized before setting tool frames")
        existing = self.tool_pose_criteria
        self.tool_frames = frames
        self.tool_pose_criteria = {
            name: (existing[name].clone() if name in existing else self._pose_criteria.clone())
            for name in frames
        }

    def clone(self):
        return type(self)(
            weight=self.weight.clone(),
            class_type=self.class_type,
            device_cfg=self.device_cfg.clone(),
            convert_to_binary=self.convert_to_binary,
            use_grad_input=self.use_grad_input,
            tool_frames=None if self.tool_frames is None else self.tool_frames.copy(),
            tool_pose_criteria={name: criteria.clone() for name, criteria in self.tool_pose_criteria.items()},
            use_lie_group=self.use_lie_group,
            _terminal_pose_convergence_tolerance=None,
            _non_terminal_pose_convergence_tolerance=None,
            _terminal_pose_axes_weight_factor=None,
            _non_terminal_pose_axes_weight_factor=None,
            _project_distance_to_goal=False,
            _pose_criteria=None if self._pose_criteria is None else self._pose_criteria.clone(),
        )


__all__ = ["BaseCostCfg", "ToolPoseCost", "ToolPoseCostCfg", "ToolPoseCriteria"]
