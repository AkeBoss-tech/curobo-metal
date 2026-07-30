"""Portable high-level planner composed from IK and trajectory optimization."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose

from .motion_planner_cfg import MotionPlannerCfg
from .motion_planner_result import GraspPlanResult


class MotionPlanner:
    def __init__(self, config: MotionPlannerCfg):
        if not isinstance(config, MotionPlannerCfg):
            raise TypeError("config must be MotionPlannerCfg")
        self.config = config
        self._initialize_components()

    def _initialize_components(self):
        self.ik_solver = IKSolver(self.config.ik_solver_config)
        self.trajopt_solver = TrajOptSolver(self.config.trajopt_solver_config)
        self.graph_planner = None
        self._tool_pose_criteria: Dict[str, ToolPoseCriteria] = {}

    def destroy(self):
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def attachment_manager(self): return None
    @property
    def joint_names(self): return self.ik_solver.joint_names
    @property
    def action_dim(self): return self.ik_solver.action_dim
    @property
    def tool_frames(self): return self.ik_solver.tool_frames
    @property
    def default_joint_state(self): return self.ik_solver.default_joint_state
    @property
    def kinematics(self): return self.ik_solver.kinematics

    def compute_kinematics(self, state: JointState):
        return self.ik_solver.compute_kinematics(state)

    def warmup(
        self, enable_graph: bool = True, warmup_joint_index: int = 0,
        warmup_joint_delta: float = 0.2, num_warmup_iterations: int = 10,
    ):
        del enable_graph, num_warmup_iterations
        current = self.default_joint_state
        goal = current.clone()
        goal.position = goal.position.clone()
        goal.position[..., warmup_joint_index] += warmup_joint_delta
        return self.plan_cspace(goal, current, max_attempts=1)

    def plan_pose(
        self, goal_tool_poses: GoalToolPose, current_state: JointState,
        use_implicit_goal: bool = True, max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ):
        del use_implicit_goal, enable_graph_attempt
        last = None
        for _ in range(max_attempts):
            ik = self.ik_solver.solve_pose(
                goal_tool_poses, current_state=current_state, return_seeds=1
            )
            last = ik
            if bool(ik.success.all().item()):
                goal = JointState.from_position(
                    ik.solution[:, 0], self.joint_names
                )
                result = self.plan_cspace(
                    goal, current_state, max_attempts=1, enable_graph_attempt=0
                )
                result.debug_info["ik_result"] = ik
                return result
            self.ik_solver.reset_seed()
        return last

    def _plan_pose_single(self, goal_tool_poses, current_state, max_attempts, enable_graph_attempt):
        return self.plan_pose(
            goal_tool_poses, current_state, max_attempts=max_attempts,
            enable_graph_attempt=enable_graph_attempt,
        )

    def _plan_pose_goalset(
        self, goal_tool_poses, current_state, use_implicit_goal=True, max_attempts=10,
    ):
        if goal_tool_poses.num_goalset != 1:
            raise NotImplementedError(
                "portable goalset planning requires selecting one goal before planning"
            )
        return self.plan_pose(
            goal_tool_poses, current_state, use_implicit_goal, max_attempts
        )

    def plan_cspace(
        self, goal_state: JointState, current_state: JointState,
        max_attempts: int = 5, enable_graph_attempt: int = 1,
    ):
        del enable_graph_attempt
        result = None
        for attempt in range(max_attempts):
            result = self.trajopt_solver.solve_cspace(
                goal_state, current_state,
                initial_iters=self.config.trajopt_solver_config.max_iterations,
            )
            result.debug_info["attempt"] = attempt + 1
            if bool(result.success.all().item()):
                optimized = result.js_solution
                if optimized.position.ndim == 4:
                    optimized = optimized[:, 0]
                optimized.dt = optimized.position.new_full(
                    (optimized.position.shape[0],),
                    self.config.trajopt_solver_config.interpolation_dt,
                )
                interpolated, last = self.trajopt_solver.get_interpolated_trajectory(
                    optimized
                )
                result.interpolated_trajectory = interpolated
                result.interpolated_last_tstep = last
                return result
            self.trajopt_solver.reset_seed()
        return result

    def _get_graph_seed_trajectories(self, current_state, seed_config):
        del current_state, seed_config
        raise NotImplementedError(
            "direct portable PRM seed extraction is exposed through PRMGraphPlanner"
        )

    def plan_grasp(
        self, grasp_poses: GoalToolPose, current_state: JointState,
        grasp_approach_axis: str = "z", grasp_approach_offset: float = -0.15,
        grasp_approach_in_tool_frame: bool = True,
        grasp_lift_axis: str = "z", grasp_lift_offset: float = -0.15,
        grasp_lift_in_tool_frame: bool = True,
        plan_approach_to_grasp: bool = True, plan_grasp_to_lift: bool = True,
        disable_collision_links: List[str] = None,
    ):
        del (
            grasp_approach_axis, grasp_approach_offset,
            grasp_approach_in_tool_frame, grasp_lift_axis, grasp_lift_offset,
            grasp_lift_in_tool_frame, disable_collision_links,
        )
        started = time.monotonic()
        planned = self.plan_pose(grasp_poses, current_state)
        ok = planned.success
        trajectory = planned.js_solution
        interpolated = planned.interpolated_trajectory
        return GraspPlanResult(
            success=ok,
            approach_success=ok if plan_approach_to_grasp else None,
            grasp_success=ok,
            lift_success=ok if plan_grasp_to_lift else None,
            approach_trajectory=trajectory if plan_approach_to_grasp else None,
            approach_interpolated_trajectory=interpolated if plan_approach_to_grasp else None,
            grasp_trajectory=trajectory,
            grasp_interpolated_trajectory=interpolated,
            lift_trajectory=trajectory if plan_grasp_to_lift else None,
            lift_interpolated_trajectory=interpolated if plan_grasp_to_lift else None,
            status="success" if bool(ok.all().item()) else "failed",
            planning_time=time.monotonic() - started,
            goalset_index=getattr(planned, "goalset_index", None),
        )

    def enable_link_collision(self, enable_collision_links: List[str]):
        raise NotImplementedError(
            f"runtime collision-link mutation is unavailable: {enable_collision_links}"
        )

    def disable_link_collision(self, disable_collision_links: List[str]):
        raise NotImplementedError(
            f"runtime collision-link mutation is unavailable: {disable_collision_links}"
        )

    def update_world(self, scene_cfg):
        raise NotImplementedError(
            "MotionPlanner world mutation requires a portable SceneCollision adapter"
        )

    def clear_scene_cache(self): return None
    def reset_seed(self):
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()

    def update_link_inertial(
        self, link_name: str, mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None, inertia: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            f"runtime inertial mutation is unavailable for {link_name}"
        )

    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)

    def update_tool_pose_criteria(self, tool_pose_criteria):
        self._tool_pose_criteria = dict(tool_pose_criteria)
        self.ik_solver.config.tool_pose_criteria = self._tool_pose_criteria


__all__ = ["MotionPlanner"]
