"""End-to-end joint or pose goal motion generation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from curobo_metal.ops.ik import IKProblem, IKResult, solve_ik

from .core import TrajectoryProblem, TrajectoryResult, optimize_trajectory


@dataclass(frozen=True)
class MotionGenerationResult:
    trajectory: TrajectoryResult | None
    ik: IKResult | None
    success: bool
    status: str


def generate_motion(
    problem: TrajectoryProblem,
    *,
    ik_problem: IKProblem | None = None,
    differentiable: bool = False,
) -> MotionGenerationResult:
    """Generate start-to-goal motion, optionally deriving joint goals with IK."""
    ik_result = None
    trajectory_problem = problem
    if ik_problem is not None:
        ik_result = solve_ik(ik_problem, differentiable=differentiable)
        selected = ik_result.selected_seed
        if not isinstance(selected, int):
            return MotionGenerationResult(None, ik_result, False, "ik_failed")
        goal = ik_result.solutions[selected]
        trajectory_problem = replace(problem, goal=goal)
    result = optimize_trajectory(trajectory_problem, differentiable=differentiable)
    selected = result.selected_seed
    success = isinstance(selected, int)
    status = "success" if success else (result.status[0] if result.status else "no_seeds")
    return MotionGenerationResult(result, ik_result, success, status)
