"""Adapter-friendly planning result records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .state import JointState


class MotionGenStatus(str, Enum):
    SUCCESS = "success"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    IK_FAILED = "ik_failed"
    GRAPH_FAILED = "graph_failed"
    TRAJECTORY_FAILED = "trajectory_failed"
    UNSUPPORTED = "unsupported"


@dataclass
class PlanningResult:
    success: torch.Tensor | bool
    status: MotionGenStatus | str | tuple[MotionGenStatus | str, ...]
    solution: JointState | None = None
    solve_time: float = 0.0
    debug_info: object | None = None

    def to(self, *args: object, **kwargs: object) -> "PlanningResult":
        return type(self)(
            self.success.to(*args, **kwargs) if isinstance(self.success, torch.Tensor) else self.success,
            self.status,
            None if self.solution is None else self.solution.to(*args, **kwargs),
            self.solve_time,
            self.debug_info,
        )


@dataclass
class IKResult(PlanningResult):
    goal_index: torch.Tensor | None = None


@dataclass
class GraphResult(PlanningResult):
    path_length: torch.Tensor | None = None


@dataclass
class TrajectoryResult(PlanningResult):
    optimized_dt: torch.Tensor | float | None = None


@dataclass
class MotionGenMetrics:
    duration: float | torch.Tensor = 0.0
    path_length: torch.Tensor | None = None
    maximum_velocity: torch.Tensor | None = None
    maximum_acceleration: torch.Tensor | None = None
    maximum_jerk: torch.Tensor | None = None
    minimum_clearance: torch.Tensor | None = None
    maximum_limit_violation: torch.Tensor | None = None
    graph_used: bool | torch.Tensor = False


@dataclass
class MotionGenResult:
    success: torch.Tensor | bool
    status: MotionGenStatus | str | tuple[MotionGenStatus | str, ...]
    optimized_plan: JointState | None = None
    interpolated_plan: JointState | None = None
    interpolation_dt: float = 0.0
    metrics: MotionGenMetrics | None = None
    trajectory_result: object | None = None
    ik_result: object | None = None
    graph_result: object | None = None
    solve_time: float = 0.0

    def get_interpolated_plan(self) -> JointState | None:
        return self.interpolated_plan
