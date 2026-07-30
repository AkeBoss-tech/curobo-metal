"""Batched facade over the portable high-level planner."""

from __future__ import annotations

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose

from .motion_planner import MotionPlanner


class BatchMotionPlanner(MotionPlanner):
    @property
    def batch_size(self):
        return self.config.ik_solver_config.max_batch_size

    def warmup(self, enable_graph: bool = True, num_warmup_iterations: int = 5):
        return super().warmup(
            enable_graph=enable_graph,
            num_warmup_iterations=num_warmup_iterations,
        )

    def plan_pose(
        self, goal_tool_poses: GoalToolPose, current_state: JointState,
        use_implicit_goal: bool = True, max_attempts: int = 1,
        success_ratio: float = 1.0, enable_graph_attempt: int = 0,
        finetune_attempts: int = 1, initial_iters=None,
        time_optimal_iters=None, finetune_iters=None,
        finetune_dt_scale: float = 0.55,
    ):
        del (
            success_ratio, finetune_attempts, initial_iters,
            time_optimal_iters, finetune_iters, finetune_dt_scale,
        )
        return super().plan_pose(
            goal_tool_poses, current_state, use_implicit_goal,
            max_attempts, enable_graph_attempt,
        )

    def plan_cspace(
        self, goal_states: JointState, current_state: JointState,
        max_attempts: int = 1, success_ratio: float = 1.0,
        enable_graph_attempt: int = 0,
    ):
        result = super().plan_cspace(
            goal_states, current_state, max_attempts, enable_graph_attempt
        )
        if result.success.numel() and (
            result.success.float().mean().item() + 1e-12 < success_ratio
        ):
            result.success = torch.zeros_like(result.success)
        return result


__all__ = ["BatchMotionPlanner"]
