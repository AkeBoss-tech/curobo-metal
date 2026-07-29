"""Small, stable public value types for motion generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class MotionGenStatus(str, Enum):
    SUCCESS = "success"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    IK_FAILED = "ik_failed"
    GRAPH_FAILED = "graph_failed"
    TRAJECTORY_FAILED = "trajectory_failed"


class UnsupportedMotionGenFeature(NotImplementedError):
    """Requested cuRobo feature is intentionally not implemented."""


@dataclass(frozen=True)
class JointState:
    position: torch.Tensor
    joint_names: tuple[str, ...] | None = None
    velocity: torch.Tensor | None = None
    acceleration: torch.Tensor | None = None
    jerk: torch.Tensor | None = None

    @classmethod
    def from_position(
        cls, position: torch.Tensor, joint_names: tuple[str, ...] | None = None
    ) -> "JointState":
        return cls(position=position, joint_names=joint_names)


@dataclass(frozen=True)
class Pose:
    position: torch.Tensor
    quaternion: torch.Tensor


@dataclass(frozen=True)
class MotionGenMetrics:
    duration: float | torch.Tensor
    path_length: torch.Tensor
    maximum_velocity: torch.Tensor
    maximum_acceleration: torch.Tensor
    maximum_jerk: torch.Tensor
    minimum_clearance: torch.Tensor
    maximum_limit_violation: torch.Tensor
    graph_used: bool | torch.Tensor


@dataclass(frozen=True)
class MotionGenResult:
    success: torch.Tensor
    status: MotionGenStatus | tuple[MotionGenStatus, ...]
    optimized_plan: JointState | None
    interpolated_plan: JointState | None
    interpolation_dt: float
    metrics: MotionGenMetrics | None
    trajectory_result: object | None = None
    ik_result: object | None = None
    graph_result: object | None = None

    def get_interpolated_plan(self) -> JointState | None:
        return self.interpolated_plan
