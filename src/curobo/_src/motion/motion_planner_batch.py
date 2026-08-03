"""Portable batched motion planning over the CPU/MPS solver stack.

``BatchMotionPlanner`` deliberately differs from ``MotionPlanner``: every
attempt evaluates all planning problems, and the first successful result for
each problem is retained.  This is the observable V2 contract that makes
batched scheduling useful even when an individual problem needs a retry.  The
implementation is ordinary PyTorch rather than a CUDA graph; ``use_cuda_graph``
therefore remains a configuration-compatible, persistent-state hint.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Dict, List, Optional, Union

import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.collision_scene import SceneCollision, create_scene_collision
from curobo._src.geom.types import SceneCfg
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

from .motion_planner import MotionPlanner, _axis_string_to_vector


class BatchMotionPlanner(MotionPlanner):
    """Solve independent c-space or tool-pose problems in one batch.

    The configured ``max_batch_size`` is retained for V2 configuration
    compatibility, but unlike a statically captured CUDA graph it is not a
    hard capacity on eager CPU/MPS.  Calls may use any non-empty batch.  Per-problem collision worlds (``multi_env=True``) are supported
    through the normal collision checker, but PRM graph seeds are intentionally
    disabled because a shared roadmap is not valid across different worlds.
    """

    def _initialize_components(self):
        # Reuse the production portable composition and only remove the graph
        # cache in the pinned multi-environment case.  The parent factory also
        # accepts task-style graph config values and compiles them lazily.
        super()._initialize_components()
        if self.config.ik_solver_config.multi_env and self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner = None

    @property
    def batch_size(self) -> int:
        return int(self.config.ik_solver_config.max_batch_size)

    @staticmethod
    def _validate_ratio(success_ratio: float) -> None:
        if not isinstance(success_ratio, (int, float)) or not 0.0 <= success_ratio <= 1.0:
            raise ValueError("success_ratio must be a number in [0, 1]")

    def _validate_batch_state(self, state: JointState, name: str) -> int:
        self._validate_state(state, name)
        if state.position.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch, dof] for BatchMotionPlanner; "
                f"got {tuple(state.position.shape)}"
            )
        batch = int(state.position.shape[0])
        if batch < 1:
            raise ValueError(f"{name} batch size must be positive")
        return batch

    @staticmethod
    def _require_attempts(max_attempts: int, enable_graph_attempt: int) -> None:
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if not isinstance(enable_graph_attempt, int) or enable_graph_attempt < 0:
            raise ValueError("enable_graph_attempt must be a non-negative integer")

    @staticmethod
    def _success_by_problem(result) -> torch.Tensor:
        success = result.success
        return success if success.ndim == 1 else success.any(dim=-1)

    @staticmethod
    def _copy_joint_state_rows(destination: JointState, source: JointState, mask: torch.Tensor) -> None:
        for name in ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt"):
            left, right = getattr(destination, name, None), getattr(source, name, None)
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.shape[:1] == right.shape[:1]:
                left[mask] = right[mask]

    @classmethod
    def _copy_result_rows(cls, destination, source, mask: torch.Tensor) -> None:
        """Copy a first-success subset without losing trajectories/metadata.

        The low-level result helper intentionally copies a conservative set of
        tensor fields.  Batched planning additionally promises that selected
        ``js_solution`` and interpolation rows match the selected success row,
        so keep all batch-shaped tensors and joint-state payloads together.
        """
        for item in fields(destination):
            name = item.name
            left, right = getattr(destination, name), getattr(source, name)
            if left is None and right is not None:
                # A stage which first succeeds on a later retry may materialize
                # optional payloads (interpolation/debug trajectories) that
                # were absent from the first result.  Clone the candidate
                # container once, then select its newly successful rows below.
                clone = getattr(right, "clone", None)
                setattr(destination, name, clone() if callable(clone) else right)
                left = getattr(destination, name)
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                if left.ndim and right.ndim and left.shape[:1] == right.shape[:1] == mask.shape:
                    left[mask] = right[mask]
            elif isinstance(left, JointState) and isinstance(right, JointState):
                cls._copy_joint_state_rows(left, right, mask)

    def _merge_attempt(self, best, candidate, solved: torch.Tensor):
        candidate_solved = self._success_by_problem(candidate)
        if best is None:
            return candidate.clone(), candidate_solved.clone()
        newly_solved = candidate_solved & ~solved
        if bool(newly_solved.any().item()):
            self._copy_result_rows(best, candidate, newly_solved)
        return best, solved | candidate_solved

    def warmup(self, enable_graph: bool = True, num_warmup_iterations: int = 5):
        self._assert_live()
        if not isinstance(num_warmup_iterations, int) or num_warmup_iterations < 1:
            raise ValueError("num_warmup_iterations must be a positive integer")
        original_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False
        try:
            current = self.default_joint_state.position.unsqueeze(0).repeat(self.batch_size, 1)
            current_state = JointState.from_position(current, self.joint_names)
            for _ in range(num_warmup_iterations):
                goal_state = current_state.clone()
                goal_state.position[..., 0] += 0.2
                self.plan_cspace(goal_state, current_state)
                tool_poses = self.compute_kinematics(goal_state).tool_poses
                self.plan_pose(
                    GoalToolPose(
                        list(self.tool_frames), tool_poses.position.unsqueeze(3),
                        tool_poses.quaternion.unsqueeze(3),
                    ),
                    current_state,
                )
                self.reset_seed()
            if enable_graph and self.graph_planner is not None:
                self.graph_planner.warmup(num_warmup_iterations=num_warmup_iterations)
        finally:
            self.ik_solver.config.exit_early = original_exit_early
        return True

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 1,
        success_ratio: float = 1.0,
        enable_graph_attempt: int = 0,
        finetune_attempts: int = 1,
        initial_iters=None,
        time_optimal_iters=None,
        finetune_iters=None,
        finetune_dt_scale: float = 0.55,
    ):
        self._assert_live()
        batch = self._validate_batch_state(current_state, "current_state")
        if not isinstance(goal_tool_poses, GoalToolPose):
            raise TypeError("goal_tool_poses must be a GoalToolPose")
        if goal_tool_poses.batch_size != batch:
            raise ValueError("goal_tool_poses and current_state batch sizes must match")
        if goal_tool_poses.device != current_state.device:
            raise ValueError("goal_tool_poses and current_state must be on the same device")
        self._require_attempts(max_attempts, enable_graph_attempt)
        self._validate_ratio(success_ratio)

        # Portable IK has no captured fixed-shape buffers.  Keep the public
        # configuration value intact while allowing the eager batch facade to
        # grow its internal validation capacity for this invocation.
        self.ik_solver.config.max_batch_size = max(self.ik_solver.config.max_batch_size, batch)
        pose_ik = getattr(self.trajopt_solver, "_pose_ik", None)
        if pose_ik is not None:
            pose_ik.config.max_batch_size = max(pose_ik.config.max_batch_size, batch)

        best = None
        solved = torch.zeros(batch, dtype=torch.bool, device=current_state.device)
        total_time = 0.0
        num_seeds = self.trajopt_solver.config.num_seeds
        for attempt in range(max_attempts):
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses, current_state=current_state, return_seeds=num_seeds
            )
            total_time += ik_result.total_time
            if not bool(ik_result.success.any().item()):
                self.reset_seed()
                continue
            seed_traj = None
            if self.graph_planner is not None and attempt >= enable_graph_attempt:
                seed_traj = self._get_graph_seed_trajectories(current_state, ik_result.solution)
            # The upstream implicit-goal route reads the selected IK endpoint
            # from a CUDA rollout buffer.  Portable TrajOpt has no such
            # captured buffer, but it has exactly the same joint-space target
            # here: the deterministic first selected seed from this batch's
            # preceding IK solve.  Supplying it explicitly avoids a redundant
            # second IK solve and keeps the complete batch on CPU/MPS.
            goal_state = JointState.from_position(
                ik_result.solution[:, 0], self.joint_names
            )
            candidate = self.trajopt_solver.solve_pose(
                goal_tool_poses,
                current_state,
                seed_config=ik_result.solution,
                seed_traj=seed_traj,
                # ``use_implicit_goal`` remains accepted for source
                # compatibility.  Its observable endpoint is represented by
                # ``goal_state`` above rather than CUDA-only rollout state.
                use_implicit_goal=False,
                goal_state=goal_state,
                finetune_attempts=finetune_attempts,
                initial_iters=initial_iters,
                time_optimal_iters=time_optimal_iters,
                finetune_iters=finetune_iters,
                finetune_dt_scale=finetune_dt_scale,
            )
            candidate.goalset_index = ik_result.goalset_index
            candidate.debug_info["ik_result"] = ik_result
            candidate.debug_info["attempt"] = attempt + 1
            total_time += candidate.total_time
            candidate.total_time = total_time
            candidate = self._finish_trajectory(candidate)
            best, solved = self._merge_attempt(best, candidate, solved)
            if float(solved.float().mean().item()) >= success_ratio:
                break
            self.reset_seed()
        if best is not None:
            best.total_time = total_time
        return best

    def plan_cspace(
        self,
        goal_states: JointState,
        current_state: JointState,
        max_attempts: int = 1,
        success_ratio: float = 1.0,
        enable_graph_attempt: int = 0,
    ):
        self._assert_live()
        batch = self._validate_batch_state(current_state, "current_state")
        goal_batch = self._validate_batch_state(goal_states, "goal_states")
        if goal_batch != batch:
            raise ValueError("goal_states and current_state batch sizes must match")
        if goal_states.device != current_state.device:
            raise ValueError("goal_states and current_state must be on the same device")
        if goal_states.position.shape[-1] != self.action_dim:
            raise ValueError("goal_states must end in the planner action dimension")
        self._require_attempts(max_attempts, enable_graph_attempt)
        self._validate_ratio(success_ratio)

        # The eager portable solver has no captured fixed-capacity buffers.
        # Keep the configured capacity as a planning hint while allowing this
        # batched facade to pass the actual request through to TrajOpt.
        self.trajopt_solver.config.max_batch_size = max(
            self.trajopt_solver.config.max_batch_size, batch
        )

        best = None
        solved = torch.zeros(batch, dtype=torch.bool, device=current_state.device)
        total_time = 0.0
        for attempt in range(max_attempts):
            seed_traj = None
            if self.graph_planner is not None and attempt >= enable_graph_attempt:
                endpoints = goal_states.position[:, None].expand(
                    -1, self.trajopt_solver.config.num_seeds, -1
                )
                seed_traj = self._get_graph_seed_trajectories(current_state, endpoints)
            candidate = self.trajopt_solver.solve_cspace(
                goal_states, current_state, seed_traj=seed_traj
            )
            candidate.debug_info["attempt"] = attempt + 1
            total_time += candidate.total_time
            candidate.total_time = total_time
            candidate = self._finish_trajectory(candidate)
            best, solved = self._merge_attempt(best, candidate, solved)
            if float(solved.float().mean().item()) >= success_ratio:
                break
            self.reset_seed()
        if best is not None:
            best.total_time = total_time
        return best

    def _get_graph_seed_trajectories(self, current_state: JointState, seed_config):
        """Build graph seeds for every ``batch × seed`` pair in one query."""
        if self.graph_planner is None:
            return None
        values = seed_config.position if isinstance(seed_config, JointState) else seed_config
        if not isinstance(values, torch.Tensor) or values.ndim != 3:
            raise ValueError("seed_config must have shape [batch, seed, dof]")
        batch, num_seeds, dof = values.shape
        if batch != current_state.position.shape[0] or dof != self.action_dim:
            raise ValueError("seed_config must match current_state batch and planner dof")
        starts = current_state.position[:, None].expand(-1, num_seeds, -1).reshape(-1, dof)
        graph_result = self.graph_planner.find_path(
            starts, values.reshape(-1, dof), interpolate_waypoints=True,
            interpolation_steps=self.trajopt_solver.action_horizon,
            validate_interpolated_trajectory=False,
        )
        if graph_result.interpolated_waypoints is None or not bool(graph_result.success.any().item()):
            return None
        return graph_result.interpolated_waypoints.reshape(
            batch, num_seeds, -1, dof
        )

    @staticmethod
    def _axis_offset(axis: str, offset: float, pose: Pose) -> Pose:
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        vector = pose.position.new_zeros(3)
        vector[("x", "y", "z").index(axis)] = float(offset)
        return Pose(vector, pose.position.new_tensor([1.0, 0.0, 0.0, 0.0]))

    def _extract_per_problem_grasp(
        self, grasp_poses: GoalToolPose, goalset_index: torch.Tensor, goalset_ok: torch.Tensor
    ) -> Dict[str, Pose]:
        index = goalset_index[:, 0, 0] if goalset_index.ndim == 3 else goalset_index.reshape(-1)
        index = index.to(device=grasp_poses.device, dtype=torch.long).clone()
        index[~goalset_ok] = 0
        batch_index = torch.arange(grasp_poses.batch_size, device=grasp_poses.device)
        selected: Dict[str, Pose] = {}
        for link_index, frame in enumerate(grasp_poses.tool_frames):
            selected[frame] = Pose(
                grasp_poses.position[:, 0, link_index][batch_index, index],
                grasp_poses.quaternion[:, 0, link_index][batch_index, index],
            )
        return selected

    def _substitute_fallback_goal(
        self, goal_tool_pose: GoalToolPose, current_state: JointState, failed: torch.Tensor
    ) -> None:
        if not bool(failed.any().item()):
            return
        tool_poses = self.compute_kinematics(current_state).tool_poses
        for link_index, frame in enumerate(goal_tool_pose.tool_frames):
            pose = tool_poses.get_link_pose(frame)
            position = pose.position.reshape(current_state.position.shape[0], -1, 3)[:, 0]
            quaternion = pose.quaternion.reshape(current_state.position.shape[0], -1, 4)[:, 0]
            goal_tool_pose.position[failed, :, link_index, :, :] = position[failed, None, None, :]
            goal_tool_pose.quaternion[failed, :, link_index, :, :] = quaternion[failed, None, None, :]

    def plan_grasp(
        self,
        grasp_poses: GoalToolPose,
        current_state: JointState,
        grasp_approach_axis: str = "z", grasp_approach_offset: float = -0.15,
        grasp_approach_in_tool_frame: bool = True,
        grasp_lift_axis: str = "z", grasp_lift_offset: float = -0.15,
        grasp_lift_in_tool_frame: bool = True,
        plan_approach_to_grasp: bool = True, plan_grasp_to_lift: bool = True,
        disable_collision_links: List[str] = None,
    ) -> GraspPlanResult:
        self._assert_live()
        batch = self._validate_batch_state(current_state, "current_state")
        if not isinstance(grasp_poses, GoalToolPose) or grasp_poses.batch_size != batch:
            raise ValueError("grasp_poses must be a GoalToolPose matching current_state batch")
        for axis in (grasp_approach_axis, grasp_lift_axis):
            if axis not in ("x", "y", "z"):
                raise ValueError("grasp approach/lift axes must be 'x', 'y', or 'z'")
        links = list(disable_collision_links or [])
        empty = torch.zeros(batch, dtype=torch.bool, device=current_state.device)
        output = GraspPlanResult(success=empty.clone(), approach_success=empty.clone(),
                                 grasp_success=empty.clone(), lift_success=empty.clone())

        self.disable_link_collision(links)
        try:
            goalset_result = self.plan_pose(grasp_poses, current_state)
        finally:
            self.enable_link_collision(links)
        output.goalset_result = goalset_result
        if goalset_result is None:
            output.status = "Goalset planning returned None."
            return output
        goalset_ok = self._success_by_problem(goalset_result)
        if not bool(goalset_ok.any().item()):
            output.status = "No grasp in goal set was reachable."
            return output
        output.goalset_index = goalset_result.goalset_index.clone()
        selected = self._extract_per_problem_grasp(grasp_poses, output.goalset_index, goalset_ok)

        approach = {}
        for frame, pose in selected.items():
            offset = self._axis_offset(grasp_approach_axis, grasp_approach_offset, pose)
            approach[frame] = pose.multiply(offset) if grasp_approach_in_tool_frame else offset.multiply(pose)
        approach_goal = GoalToolPose.from_poses(approach, self.tool_frames)
        approach_result = self.plan_pose(approach_goal, current_state)
        output.approach_result = approach_result
        if approach_result is None:
            output.status = "Planning to approach pose failed."
            return output
        approach_ok = goalset_ok & self._success_by_problem(approach_result)
        output.approach_success = approach_ok
        output.approach_trajectory, output.approach_trajectory_dt = approach_result.js_solution, approach_result.js_solution.dt
        output.approach_interpolated_trajectory = approach_result.interpolated_trajectory
        output.approach_interpolated_last_tstep = approach_result.interpolated_last_tstep
        if not plan_approach_to_grasp:
            output.success, output.status = approach_ok, "Planning to approach pose completed."
            return output

        approach_end = JointState.from_position(approach_result.js_solution.position[:, 0, -1], self.joint_names)
        grasp_goal = GoalToolPose.from_poses(selected, self.tool_frames)
        self._substitute_fallback_goal(grasp_goal, approach_end, ~approach_ok)
        self.disable_link_collision(links)
        try:
            grasp_result = self.plan_pose(grasp_goal, approach_end)
        finally:
            self.enable_link_collision(links)
        output.grasp_result = grasp_result
        if grasp_result is None:
            output.status = "Planning to grasp pose failed."
            return output
        grasp_ok = approach_ok & self._success_by_problem(grasp_result)
        output.grasp_success = grasp_ok
        output.grasp_trajectory, output.grasp_trajectory_dt = grasp_result.js_solution, grasp_result.js_solution.dt
        output.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory
        output.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep
        if not plan_grasp_to_lift:
            output.success, output.status = grasp_ok, "Planning to grasp pose completed."
            return output

        lift_start = JointState.from_position(grasp_result.js_solution.position[:, 0, -1], self.joint_names)
        lift = {}
        for frame, pose in selected.items():
            offset = self._axis_offset(grasp_lift_axis, grasp_lift_offset, pose)
            lift[frame] = pose.multiply(offset) if grasp_lift_in_tool_frame else offset.multiply(pose)
        lift_goal = GoalToolPose.from_poses(lift, self.tool_frames)
        self._substitute_fallback_goal(lift_goal, lift_start, ~grasp_ok)
        self.disable_link_collision(links)
        try:
            lift_result = self.plan_pose(lift_goal, lift_start)
        finally:
            self.enable_link_collision(links)
        output.lift_result = lift_result
        if lift_result is None:
            output.status = "Planning to lift pose failed."
            return output
        output.lift_success = grasp_ok & self._success_by_problem(lift_result)
        output.success, output.status = output.lift_success.clone(), "Grasp planning completed."
        output.lift_trajectory, output.lift_trajectory_dt = lift_result.js_solution, lift_result.js_solution.dt
        output.lift_interpolated_trajectory = lift_result.interpolated_trajectory
        output.lift_interpolated_last_tstep = lift_result.interpolated_last_tstep
        return output

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]) -> None:
        # Preserve the parent validation and retained public lifecycle state.
        super().update_tool_pose_criteria(tool_pose_criteria)


__all__ = ["BatchMotionPlanner"]
