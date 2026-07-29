"""Stable result/status compatibility names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from curobo_metal.motion_gen import JointState


class MotionGenStatus(str, Enum):
    SUCCESS = "success"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    IK_FAIL = "ik_failed"
    GRAPH_FAIL = "graph_failed"
    TRAJOPT_FAIL = "trajectory_failed"
    TIMEOUT = "timeout"
    DT_EXCEPTION = "dt_exception"


@dataclass(frozen=True)
class MotionGenResult:
    success: torch.Tensor
    status: MotionGenStatus | tuple[MotionGenStatus, ...]
    optimized_plan: JointState | None = None
    interpolated_plan: JointState | None = None
    interpolation_dt: float | None = None
    solve_time: float = 0.0
    attempts: int = 1
    used_graph: bool = False
    debug_info: Any = None

    @property
    def motion_time(self) -> float:
        if self.interpolated_plan is None or self.interpolation_dt is None:
            return 0.0
        return max(0, self.interpolated_plan.position.shape[-2] - 1) * self.interpolation_dt

    @property
    def optimized_dt(self) -> float | None:
        return self.interpolation_dt

    def get_interpolated_plan(self) -> JointState | None:
        return self.interpolated_plan
