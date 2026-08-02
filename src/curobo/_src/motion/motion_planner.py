"""Portable high-level planner composed from IK and trajectory optimization."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
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
        self._scene_collision = self._make_scene_collision(self.config.scene_collision_cfg)
        self.ik_solver = IKSolver(
            self.config.ik_solver_config, self._scene_collision
        )
        self.trajopt_solver = TrajOptSolver(
            self.config.trajopt_solver_config, self._scene_collision
        )
        self.graph_planner = self._make_graph_planner()
        self._tool_pose_criteria: Dict[str, ToolPoseCriteria] = {}
        self._attachment_manager = AttachmentManager(
            self.ik_solver.kinematics, self._scene_collision, self.config.device_cfg
        )

    def _make_graph_planner(self):
        """Create the portable PRM only when the configuration requests one.

        ``MotionPlannerCfg.create`` deliberately accepts either the pinned
        graph-planner task value or a compiled :class:`PRMGraphPlannerCfg`.
        The portable PRM does not parse CUDA rollout task YAMLs, but it can
        compile the former against the same robot bounds and use it for real
        c-space seed trajectories.
        """
        graph_cfg = self.config.graph_planner_config
        if graph_cfg is None:
            return None
        if not isinstance(graph_cfg, PRMGraphPlannerCfg):
            graph_cfg = PRMGraphPlannerCfg.create(
                self.config.ik_solver_config.robot_config,
                graph_planner_config=graph_cfg,
                device_cfg=self.config.device_cfg,
                use_cuda_graph_for_rollout=False,
            )
            self.config.graph_planner_config = graph_cfg
        return PRMGraphPlanner(graph_cfg, self._scene_collision)

    def _make_scene_collision(self, scene):
        """Build the portable scene adapter when a concrete scene is supplied."""
        if scene is None:
            return None
        if isinstance(scene, SceneCollision):
            return scene
        if isinstance(scene, SceneCollisionCfg):
            return SceneCollision.from_config(scene)
        if isinstance(scene, SceneCfg) or (
            isinstance(scene, list) and all(isinstance(value, SceneCfg) for value in scene)
        ):
            return SceneCollision(SceneCollisionCfg(
                self.config.device_cfg, scene,
                len(scene) if isinstance(scene, list) else 1,
            ))
        # Upstream configuration names may refer to YAML/Isaac scene assets.
        # Those are intentionally not parsed implicitly on Metal.
        raise NotImplementedError(
            "portable MotionPlanner scene_model must be SceneCfg, a list of SceneCfg, "
            "or SceneCollisionCfg; YAML/USD scene asset loading is unavailable"
        )

    def destroy(self):
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()

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
    def attachment_manager(self) -> AttachmentManager: return self._attachment_manager
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
        if not 0 <= warmup_joint_index < self.action_dim:
            raise ValueError(
                f"warmup_joint_index must be in [0, {self.action_dim}), got "
                f"{warmup_joint_index}"
            )
        if num_warmup_iterations < 1:
            raise ValueError("num_warmup_iterations must be positive")
        current = self.default_joint_state.unsqueeze(0)
        for _ in range(num_warmup_iterations):
            goal = current.clone()
            goal.position[..., warmup_joint_index] += warmup_joint_delta
            result = self.plan_cspace(goal, current, max_attempts=1)
            if result is None or not bool(result.success.any().item()):
                return False
            self.reset_seed()
        # Graph capture has no Metal equivalent.  PRM warmup fills only
        # ordinary portable state and is safe to request through the familiar
        # upstream switch.
        if enable_graph and self.graph_planner is not None:
            self.graph_planner.warmup(num_warmup_iterations=num_warmup_iterations)
        return True

    @staticmethod
    def _validate_state(state: JointState, name: str) -> None:
        if not isinstance(state, JointState):
            raise TypeError(f"{name} must be a JointState")
        if state.ndim > 2:
            raise ValueError(
                f"{name} must contain [dof] or [batch, dof], got {state.shape}"
            )

    @staticmethod
    def _any_success(result) -> bool:
        return result is not None and bool(result.success.any().item())

    def _finish_trajectory(self, result):
        """Attach dense interpolation consistently to successful or failed solves."""
        if result is None or result.js_solution is None:
            return result
        # Preserve the pinned [batch, seed, horizon, dof] ``js_solution``
        # payload.  Interpolation is intentionally computed from the selected
        # seed view without collapsing that public result rank.
        optimized = result.js_solution
        if optimized.position.ndim == 4:
            optimized = JointState.from_position(
                optimized.position[:, 0], self.joint_names
            ).finite_difference(self.config.trajopt_solver_config.interpolation_dt)
        optimized.dt = optimized.position.new_full(
            (optimized.position.shape[0],),
            self.config.trajopt_solver_config.interpolation_dt,
        )
        interpolated, last = self.trajopt_solver.get_interpolated_trajectory(optimized)
        result.interpolated_trajectory = interpolated
        result.interpolated_last_tstep = last
        return result

    def plan_pose(
        self, goal_tool_poses: GoalToolPose, current_state: JointState,
        use_implicit_goal: bool = True, max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ):
        self._validate_state(current_state, "current_state")
        if not isinstance(goal_tool_poses, GoalToolPose):
            raise TypeError("goal_tool_poses must be a GoalToolPose")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if goal_tool_poses.num_goalset > 1:
            return self._plan_pose_goalset(
                goal_tool_poses, current_state, use_implicit_goal, max_attempts
            )
        return self._plan_pose_single(
            goal_tool_poses, current_state, max_attempts, enable_graph_attempt
        )

    def _plan_pose_single(self, goal_tool_poses, current_state, max_attempts, enable_graph_attempt):
        total_time = solve_time = 0.0
        last = None
        original = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds
        for attempt in range(max_attempts):
            current = original.clone()
            ik = self.ik_solver.solve_pose(
                goal_tool_poses, current_state=current, return_seeds=num_seeds
            )
            total_time += ik.total_time
            solve_time += ik.solve_time
            if not bool(ik.success.any().item()):
                last = ik
                self.ik_solver.reset_seed()
                continue
            seed_config = ik.solution
            seed_traj = None
            if attempt >= enable_graph_attempt and self.graph_planner is not None:
                seed_traj = self._get_graph_seed_trajectories(current, seed_config)
                # PRM failure is a seed failure, not a reason to discard the
                # direct differentiable trajectory optimizer.
            result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current,
                seed_config=seed_config, seed_traj=seed_traj,
                return_seeds=1,
                use_implicit_goal=True,
            )
            result.debug_info["ik_result"] = ik
            result.debug_info["attempt"] = attempt + 1
            result.goalset_index = ik.goalset_index
            total_time += result.total_time
            solve_time += result.solve_time
            result.total_time, result.solve_time = total_time, solve_time
            last = self._finish_trajectory(result)
            if self._any_success(last):
                return last
            self.reset_seed()
        return last

    def _plan_pose_goalset(
        self, goal_tool_poses, current_state, use_implicit_goal=True, max_attempts=10,
    ):
        # IK evaluates the complete goalset and records its deterministic
        # selected goal index.  TrajOpt then follows that joint endpoint.
        return self._plan_pose_single(
            goal_tool_poses, current_state, max_attempts, enable_graph_attempt=max_attempts
        )

    def plan_cspace(
        self, goal_state: JointState, current_state: JointState,
        max_attempts: int = 5, enable_graph_attempt: int = 1,
    ):
        self._validate_state(goal_state, "goal_state")
        self._validate_state(current_state, "current_state")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        result = None
        total_time = solve_time = 0.0
        original = current_state.clone()
        for attempt in range(max_attempts):
            current = original.clone()
            seed_traj = None
            if attempt >= enable_graph_attempt and self.graph_planner is not None:
                goal_position = goal_state.position
                if goal_position.ndim == 1:
                    goal_position = goal_position.unsqueeze(0)
                seed_config = goal_position[:, None].expand(
                    -1, self.trajopt_solver.config.num_seeds, -1
                )
                seed_traj = self._get_graph_seed_trajectories(current, seed_config)
            result = self.trajopt_solver.solve_cspace(
                goal_state, current,
                seed_traj=seed_traj,
                initial_iters=self.config.trajopt_solver_config.max_iterations,
            )
            result.debug_info["attempt"] = attempt + 1
            total_time += result.total_time
            solve_time += result.solve_time
            result.total_time, result.solve_time = total_time, solve_time
            result = self._finish_trajectory(result)
            if self._any_success(result):
                return result
            self.reset_seed()
        return result

    def _get_graph_seed_trajectories(self, current_state, seed_config):
        if self.graph_planner is None:
            return None
        values = seed_config.position if isinstance(seed_config, JointState) else seed_config
        if not isinstance(values, torch.Tensor) or values.shape[-1] != self.action_dim:
            raise ValueError("seed_config must end in the planner action dimension")
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[0] != 1:
            # Single MotionPlanner intentionally has one planning problem.
            return None
        starts = current_state.position
        if starts.ndim == 1:
            starts = starts.unsqueeze(0)
        starts = starts[:1].expand(values.shape[1], -1)
        paths = self.graph_planner.find_path(
            starts, values[0], interpolate_waypoints=True,
            interpolation_steps=self.trajopt_solver.action_horizon,
            validate_interpolated_trajectory=False,
        )
        if not bool(paths.success.any().item()):
            return None
        return paths.interpolated_waypoints[paths.success].unsqueeze(0)

    def plan_grasp(
        self, grasp_poses: GoalToolPose, current_state: JointState,
        grasp_approach_axis: str = "z", grasp_approach_offset: float = -0.15,
        grasp_approach_in_tool_frame: bool = True,
        grasp_lift_axis: str = "z", grasp_lift_offset: float = -0.15,
        grasp_lift_in_tool_frame: bool = True,
        plan_approach_to_grasp: bool = True, plan_grasp_to_lift: bool = True,
        disable_collision_links: List[str] = None,
    ):
        started = time.monotonic()
        self._validate_state(current_state, "current_state")
        if not isinstance(grasp_poses, GoalToolPose):
            raise TypeError("grasp_poses must be a GoalToolPose")
        for label, axis in (("grasp_approach_axis", grasp_approach_axis),
                            ("grasp_lift_axis", grasp_lift_axis)):
            if axis not in ("x", "y", "z"):
                raise ValueError(f"{label} must be 'x', 'y', or 'z'")

        def empty(status: str) -> GraspPlanResult:
            failed = torch.zeros(
                (max(1, current_state.position.shape[0] if current_state.ndim == 2 else 1), 1),
                dtype=torch.bool, device=current_state.device,
            )
            return GraspPlanResult(
                success=failed, approach_success=failed.clone(),
                grasp_success=failed.clone(), lift_success=failed.clone(),
                status=status, planning_time=time.monotonic() - started,
            )

        # First choose a concrete member of the supplied goalset.  The normal
        # pose planner supplies an IK-selected goal index when available; use
        # index zero as the deterministic portable tie fallback.
        goalset_result = self.plan_pose(grasp_poses, current_state)
        if not self._any_success(goalset_result):
            result = empty("No grasp in goal set was reachable.")
            result.goalset_result = goalset_result
            return result
        chosen = getattr(goalset_result, "goalset_index", None)
        goal_index = 0 if chosen is None else int(chosen.reshape(-1)[0].item())
        goal_index = min(goal_index, grasp_poses.num_goalset - 1)
        goal_poses = {
            frame: Pose(
                grasp_poses.position[:, 0, link_index, goal_index],
                grasp_poses.quaternion[:, 0, link_index, goal_index],
            )
            for link_index, frame in enumerate(grasp_poses.tool_frames)
        }
        grasp_goal = GoalToolPose.from_poses(
            goal_poses, ordered_tool_frames=grasp_poses.tool_frames
        )

        def offset_goal(axis: str, offset: float, in_tool_frame: bool) -> GoalToolPose:
            vector = [0.0, 0.0, 0.0]
            vector[("x", "y", "z").index(axis)] = float(offset)
            offset_pose = Pose.from_list([*vector, 1.0, 0.0, 0.0, 0.0], self.config.device_cfg)
            poses = {
                frame: (pose.multiply(offset_pose) if in_tool_frame else offset_pose.multiply(pose))
                for frame, pose in goal_poses.items()
            }
            return GoalToolPose.from_poses(poses, ordered_tool_frames=grasp_poses.tool_frames)

        contact_links = list(disable_collision_links or [])
        approach_goal = offset_goal(
            grasp_approach_axis, grasp_approach_offset, grasp_approach_in_tool_frame
        )
        approach_result = self.plan_pose(approach_goal, current_state)
        if not self._any_success(approach_result):
            result = empty("Planning to approach pose failed.")
            result.goalset_result, result.approach_result = goalset_result, approach_result
            result.goalset_index = goalset_result.goalset_index
            return result

        result = GraspPlanResult(
            success=approach_result.success.clone(),
            approach_success=approach_result.success.clone(),
            approach_trajectory=approach_result.js_solution,
            approach_trajectory_dt=approach_result.js_solution.dt,
            approach_interpolated_trajectory=approach_result.interpolated_trajectory,
            approach_interpolated_last_tstep=approach_result.interpolated_last_tstep,
            status="Planning to approach pose succeeded.",
            planning_time=0.0,
            goalset_index=goalset_result.goalset_index,
        )
        result.goalset_result, result.approach_result = goalset_result, approach_result
        if not plan_approach_to_grasp:
            result.planning_time = time.monotonic() - started
            return result

        # Use the actual terminal approach configuration, not a copied
        # approach trajectory.  This is a production trajectory composition;
        # only the CUDA-specific linear-motion cost is unavailable.
        approach_end = approach_result.js_solution.position[:, 0, -1]
        approach_state = JointState.from_position(approach_end, self.joint_names)
        self.disable_link_collision(contact_links)
        try:
            grasp_result = self.plan_pose(grasp_goal, approach_state)
        finally:
            self.enable_link_collision(contact_links)
        if not self._any_success(grasp_result):
            result.success = torch.zeros_like(result.success)
            result.grasp_success = torch.zeros_like(result.success)
            result.status = "Planning to grasp pose failed."
            result.grasp_result = grasp_result
            result.planning_time = time.monotonic() - started
            return result
        result.grasp_result = grasp_result
        result.success = grasp_result.success.clone()
        result.grasp_success = grasp_result.success.clone()
        result.grasp_trajectory = grasp_result.js_solution
        result.grasp_trajectory_dt = grasp_result.js_solution.dt
        result.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory
        result.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep
        if not plan_grasp_to_lift:
            result.planning_time = time.monotonic() - started
            return result

        grasp_end = grasp_result.js_solution.position[:, 0, -1]
        lift_state = JointState.from_position(grasp_end, self.joint_names)
        lift_goal = offset_goal(grasp_lift_axis, grasp_lift_offset, grasp_lift_in_tool_frame)
        self.disable_link_collision(contact_links)
        try:
            lift_result = self.plan_pose(lift_goal, lift_state)
        finally:
            self.enable_link_collision(contact_links)
        result.lift_result = lift_result
        if not self._any_success(lift_result):
            result.success = torch.zeros_like(result.success)
            result.lift_success = torch.zeros_like(result.success)
            result.status = "Planning to lift pose failed."
            result.planning_time = time.monotonic() - started
            return result
        result.success = lift_result.success.clone()
        result.lift_success = lift_result.success.clone()
        result.lift_trajectory = lift_result.js_solution
        result.lift_trajectory_dt = lift_result.js_solution.dt
        result.lift_interpolated_trajectory = lift_result.interpolated_trajectory
        result.lift_interpolated_last_tstep = lift_result.interpolated_last_tstep
        result.status = "Planning to lift pose succeeded."
        result.planning_time = time.monotonic() - started
        return result

    def enable_link_collision(self, enable_collision_links: List[str]):
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

    def update_world(self, scene_cfg):
        if self._scene_collision is None:
            self._scene_collision = self._make_scene_collision(scene_cfg)
            self.ik_solver._scene_collision_checker = self._scene_collision
            self.trajopt_solver._scene_collision_checker = self._scene_collision
            self._attachment_manager = AttachmentManager(
                self.ik_solver.kinematics, self._scene_collision, self.config.device_cfg
            )
        else:
            self._scene_collision.load_collision_model(scene_cfg)
        self.config.scene_collision_cfg = scene_cfg
        if self.graph_planner is not None:
            self.graph_planner.scene_collision_checker = self._scene_collision
            self.graph_planner.reset_buffer()

    def clear_scene_cache(self):
        if self._scene_collision is not None:
            self._scene_collision.clear_cache()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
    def reset_seed(self):
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner.reset_seed()

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

    def update_tool_pose_criteria(
        self, tool_pose_criteria: Dict[str, ToolPoseCriteria]
    ) -> None:
        """Forward portable criteria to both planner stages.

        Criteria records are accepted and retained for public lifecycle
        compatibility.  The CUDA-only non-terminal linear rollout weighting
        is not emulated by the underlying portable optimizer.
        """
        if not isinstance(tool_pose_criteria, dict) or not all(
            isinstance(name, str) and isinstance(value, ToolPoseCriteria)
            for name, value in tool_pose_criteria.items()
        ):
            raise TypeError("tool_pose_criteria must map frame names to ToolPoseCriteria")
        unknown = set(tool_pose_criteria) - set(self.tool_frames)
        if unknown:
            raise ValueError(f"unknown tool frame(s): {sorted(unknown)}")
        self._tool_pose_criteria = dict(tool_pose_criteria)
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.trajopt_solver.update_tool_pose_criteria(tool_pose_criteria)

__all__ = ["MotionPlanner"]
