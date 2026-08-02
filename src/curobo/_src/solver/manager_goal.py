"""Portable goal-buffer lifecycle manager."""

from __future__ import annotations

from typing import Optional

from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


class GoalManager:
    def __init__(self, device_cfg: DeviceCfg):
        self.device_cfg = device_cfg
        self._goal_buffer: Optional[GoalRegistry] = None
        self._solve_state = None
        self._batch_helper = 1

    def create_goal_buffer(
        self, solve_state, goal_tool_poses: Optional[GoalToolPose] = None,
        goal_js: Optional[JointState] = None,
        current_js: Optional[JointState] = None,
        seed_goal_js: Optional[JointState] = None,
        current_state_dt=None,
    ):
        self._solve_state = solve_state
        self._goal_buffer = GoalRegistry(
            batch_size=solve_state.batch_size,
            num_goalset=solve_state.num_goalset,
            num_seeds=solve_state.num_seeds or 1,
            goal_js=goal_js, seed_goal_js=seed_goal_js,
            link_goal_poses=goal_tool_poses, current_js=current_js,
            current_state_dt=current_state_dt,
        )
        self._goal_buffer = self._goal_buffer.create_index_buffers(
            solve_state.batch_size, solve_state.multi_env,
            solve_state.num_seeds or 1, self.device_cfg,
        )
        return self._goal_buffer

    def update_goal_buffer(self, solve_state, **kwargs):
        use_implicit_goal = kwargs.pop("use_implicit_goal", False)
        del use_implicit_goal
        if self._goal_buffer is None:
            return self.create_goal_buffer(solve_state, **kwargs)
        candidate = self.create_goal_buffer(solve_state, **kwargs)
        self._goal_buffer = candidate
        return candidate

    def update_from_goal_registry(self, solve_state, goal):
        self._solve_state = solve_state
        self._goal_buffer = goal.clone()
        return self._goal_buffer

    def update_batch_helper(self, batch_size: int): self._batch_helper = batch_size
    def update_goal_tool_poses(self, goal_tool_poses): self._goal_buffer.link_goal_poses = goal_tool_poses
    def update_current_state(self, current_state): self._goal_buffer.current_js = current_state
    def update_goal_state(self, goal_state): self._goal_buffer.goal_js = goal_state
    goal_buffer = property(lambda self: self._goal_buffer)
    solve_state = property(lambda self: self._solve_state)
    batch_helper = property(lambda self: self._batch_helper)
    def get_batch_size(self): return 0 if self._solve_state is None else self._solve_state.get_batch_size()
    def get_ik_batch_size(self): return 0 if self._solve_state is None else self._solve_state.get_ik_batch_size()
    def get_trajopt_batch_size(self): return 0 if self._solve_state is None else self._solve_state.get_trajopt_batch_size()


__all__ = ["GoalManager"]
