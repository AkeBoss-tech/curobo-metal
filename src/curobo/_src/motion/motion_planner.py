"""Portable high-level planner composed from IK and trajectory optimization."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.collision.collision_scene import (
    SceneCollision,
    SceneCollisionCfg,
    create_scene_collision,
)
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.types.pose import Pose
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.trajectory import TrajInterpolationType
from curobo._src.util.logging import log_and_raise

from .motion_planner_cfg import MotionPlannerCfg
from .motion_planner_result import GraspPlanResult


def _axis_string_to_vector(axis: str) -> List[float]:
    """Return the canonical unit axis used by grasp approach/lift planning.

    The pinned helper is public enough that application code imports it
    directly.  Keeping it dependency-free also lets the batch facade share
    the exact validation semantics without pretending that a Warp vector is
    available on Metal.
    """
    axes = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    try:
        return axes[axis].copy()
    except KeyError as error:
        raise ValueError("axis must be 'x', 'y', or 'z'") from error


class _MotionPlannerPortableMixin:
    @property
    def is_destroyed(self) -> bool:
        return self._destroyed

    @property
    def world_generation(self) -> int:
        return self._world_generation


class MotionPlanner(_MotionPlannerPortableMixin):
    def __init__(self, config: MotionPlannerCfg):
        if not isinstance(config, MotionPlannerCfg):
            raise TypeError("config must be MotionPlannerCfg")
        self.config = config
        self.device_cfg = config.device_cfg
        # Keep the pinned public spelling as an alias.  A few downstream
        # integrations retain this object to mutate their world in place.
        self.scene_collision_checker = None
        self._destroyed = False
        self._world_generation = 0
        self._initialize_components()

    def _initialize_components(self):
        self._scene_collision = self._make_scene_collision(self.config.scene_collision_cfg)
        self.scene_collision_checker = self._scene_collision
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

    def _assert_live(self) -> None:
        if self._destroyed:
            raise RuntimeError("MotionPlanner has been destroyed")

    def _expected_world_environments(self) -> int:
        return (
            int(self.config.ik_solver_config.max_batch_size)
            if self.config.ik_solver_config.multi_env
            else 1
        )

    @staticmethod
    def _as_scene_list(scene: SceneCfg | List[SceneCfg]) -> List[SceneCfg]:
        return scene if isinstance(scene, list) else [scene]

    def _validate_scene_count(self, scenes: List[SceneCfg]) -> None:
        expected = self._expected_world_environments()
        if len(scenes) != expected:
            raise ValueError(
                "updated world environment count must match the planner configuration "
                f"({expected}), got {len(scenes)}"
            )

    @staticmethod
    def _scene_fits_cache(scene: SceneCfg, collision: SceneCollision) -> bool:
        """Preflight a mutation before ``load_collision_model`` clears a cache."""
        return (
            len(scene.cuboid) <= collision._world.primitive_cache.capacity
            and len(scene.mesh) <= collision._world.mesh_cache.capacity
            and len(scene.voxel) <= collision._world.voxel_cache.capacity
        )

    def _record_world_config(self, collision: SceneCollision) -> None:
        """Keep configuration aliases coherent after a portable world swap."""
        record = SceneCollisionCfg(
            device_cfg=self.config.device_cfg,
            scene_model=collision.scene_model,
            num_envs=collision.num_envs,
        )
        self.config.scene_collision_cfg = record
        for solver in (self.ik_solver, self.trajopt_solver):
            core_cfg = getattr(solver.config, "core_cfg", None)
            if core_cfg is not None:
                core_cfg.scene_collision_cfg = record
        pose_ik = getattr(self.trajopt_solver, "_pose_ik", None)
        if pose_ik is not None:
            pose_ik.config.core_cfg.scene_collision_cfg = record
        if self.graph_planner is not None:
            self.graph_planner.config.scene_collision_cfg = record

    def _install_world(self, collision: SceneCollision) -> None:
        """Synchronize every planner stage to one validated world adapter."""
        if collision.device_cfg != self.config.device_cfg:
            raise ValueError("updated world device_cfg must match MotionPlanner.device_cfg")
        if collision.num_envs != self._expected_world_environments():
            raise ValueError(
                "updated world environment count must match the planner configuration"
            )
        previous = self._scene_collision
        # SolverCore owns an IK-side goal/collision lifecycle, so use its
        # public update route instead of changing only a private field.
        self.ik_solver.update_world(collision)
        collision = self.ik_solver.scene_collision_checker
        self.trajopt_solver._scene_collision_checker = collision
        pose_ik = getattr(self.trajopt_solver, "_pose_ik", None)
        if pose_ik is not None:
            pose_ik.update_world(collision)
        self._scene_collision = collision
        self.scene_collision_checker = collision
        if previous is not collision:
            self._attachment_manager = AttachmentManager(
                self.ik_solver.kinematics, collision, self.config.device_cfg
            )
        self._record_world_config(collision)
        if self.graph_planner is not None:
            self.graph_planner.scene_collision_checker = collision
            self.graph_planner.reset_buffer()
        self._world_generation += 1

    def destroy(self):
        """Release planner state exactly once.

        Metal has no CUDA graph executable to destroy, but idempotence is
        important for context-manager and destructor users.  The underlying
        solvers also use this call to invalidate their portable caches.
        """
        if self._destroyed:
            return
        self._destroyed = True
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
    def joint_names(self) -> List[str]: return self.ik_solver.joint_names
    @property
    def action_dim(self) -> int: return self.ik_solver.action_dim
    @property
    def tool_frames(self) -> List[str]: return self.ik_solver.tool_frames
    @property
    def default_joint_state(self) -> JointState: return self.ik_solver.default_joint_state
    @property
    def kinematics(self) -> Kinematics: return self.ik_solver.kinematics

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        self._assert_live()
        return self.ik_solver.compute_kinematics(state)

    def warmup(
        self, enable_graph: bool = True, warmup_joint_index: int = 0,
        warmup_joint_delta: float = 0.2, num_warmup_iterations: int = 10,
    ) -> bool:
        self._assert_live()
        if not 0 <= warmup_joint_index < self.action_dim:
            raise ValueError(
                f"warmup_joint_index must be in [0, {self.action_dim}), got "
                f"{warmup_joint_index}"
            )
        if num_warmup_iterations < 1:
            raise ValueError("num_warmup_iterations must be positive")
        current = self.default_joint_state.unsqueeze(0)
        for _ in range(num_warmup_iterations):
            num_goalset = int(getattr(self.config.ik_solver_config, "max_goalset", 1))
            if num_goalset > 1:
                poses = self._make_warmup_goalset(
                    current, num_goalset, warmup_joint_index, warmup_joint_delta
                )
                goal = GoalToolPose.from_poses(
                    poses, ordered_tool_frames=self.tool_frames, num_goalset=num_goalset
                )
                result = self.plan_pose(goal, current, max_attempts=1)
            else:
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

    def _make_warmup_goalset(
        self, current_state: JointState, num_goalset: int,
        joint_index: int, joint_delta: float,
    ) -> Dict[str, Pose]:
        """Build deterministic, progressively offset FK warmup poses.

        This mirrors the useful part of V2 warmup without CUDA graph capture:
        every tool frame receives ``num_goalset`` poses in the exact flattened
        layout consumed by :meth:`GoalToolPose.from_poses`.
        """
        if num_goalset < 1:
            raise ValueError("num_goalset must be positive")
        positions: Dict[str, list[torch.Tensor]] = {}
        quaternions: Dict[str, list[torch.Tensor]] = {}
        for index in range(num_goalset):
            goal = current_state.clone()
            goal.position[..., joint_index] += joint_delta * (index + 1) / num_goalset
            tool_poses = self.compute_kinematics(goal).tool_poses
            for frame in self.tool_frames:
                pose = tool_poses.get_link_pose(frame)
                positions.setdefault(frame, []).append(pose.position)
                quaternions.setdefault(frame, []).append(pose.quaternion)
        return {
            frame: Pose(torch.cat(positions[frame], dim=0), torch.cat(quaternions[frame], dim=0))
            for frame in positions
        }

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
        # Solver internals optimize only active joints, but V2's public
        # MotionPlanner result restores configured locked joints (Franka's
        # fingers) in both sparse and interpolated trajectories.
        # Pinned V2 expands its 16 B-spline controls to the 81-state rollout
        # horizon before publishing a MotionPlanner result.  Use the portable
        # clamped-uniform B-spline basis rather than treating controls as a
        # piecewise-linear trajectory.
        sparse = result.js_solution
        spline_valid_prefix = None
        if self.config.trajopt_solver_config.interpolation_type == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
            # V2's B-spline CUDA interpolation reports the number of logical
            # spline knots (including its two endpoint samples) as the valid
            # prefix in its fixed interpolation buffer.  The portable solver
            # expands the same controls with composed Torch, so carry this
            # public buffer contract forward instead of exposing its internal
            # 16-control fallback length.
            spline_valid_prefix = ControlSpace.spline_total_knots(
                ControlSpace.BSPLINE_3, sparse.position.shape[-2]
            ) + 1
        public_horizon = 81
        if sparse.position.shape[-2] != public_horizon:
            from curobo_metal.ops.trajectory.dynamics_aware import bspline_matrices
            basis = bspline_matrices(
                sparse.position.shape[-2], public_horizon, degree=3,
                device=sparse.position.device, dtype=sparse.position.dtype,
            ).position
            def resample(value):
                if value is None:
                    return None
                return torch.einsum("hk,...kd->...hd", basis, value)
            sparse = JointState(
                resample(sparse.position), resample(sparse.velocity),
                resample(sparse.acceleration), sparse.joint_names,
                resample(sparse.jerk), dt=sparse.dt,
            )
        result.js_solution = self.trajopt_solver.get_full_js(sparse)
        result.solution = result.js_solution.position
        interpolated = self.trajopt_solver.get_full_js(interpolated)
        # V2 publishes a fixed-capacity interpolation buffer.  Preserve the
        # valid prefix and deterministically hold its final state in unused
        # capacity, while ``interpolated_last_tstep`` identifies the prefix.
        capacity = self.config.trajopt_solver_config.interpolation_buffer_size
        if interpolated.position.shape[-2] < capacity:
            count = capacity - interpolated.position.shape[-2]
            def extend(value):
                if value is None:
                    return None
                return torch.cat((value, value[..., -1:, :].expand(*value.shape[:-2], count, value.shape[-1])), dim=-2)
            interpolated = JointState(
                extend(interpolated.position), extend(interpolated.velocity),
                extend(interpolated.acceleration), interpolated.joint_names,
                extend(interpolated.jerk), dt=interpolated.dt,
            )
        # ``js_solution`` deliberately retains the seed axis used by V2's
        # public MotionPlanner result.  Interpolation above operates on the
        # selected seed to keep the portable solver internals simple; restore
        # that singleton axis before publishing so both result fields have the
        # same [batch, seed, horizon, dof] convention.
        if result.js_solution.position.ndim == 4 and interpolated.position.ndim == 3:
            def with_seed_axis(value):
                # ``dt`` may be scalar while state channels are batched.  A
                # scalar describes every state and must remain scalar rather
                # than acquiring an invalid axis.
                if value is None or value.ndim == 0:
                    return value
                return value.unsqueeze(1)

            interpolated = JointState(
                with_seed_axis(interpolated.position),
                with_seed_axis(interpolated.velocity),
                with_seed_axis(interpolated.acceleration),
                interpolated.joint_names,
                with_seed_axis(interpolated.jerk),
                dt=with_seed_axis(interpolated.dt),
            )
            last = with_seed_axis(last)
        if spline_valid_prefix is not None:
            if spline_valid_prefix > capacity:
                raise ValueError(
                    "interpolation_buffer_size is too small for the configured B-spline prefix"
                )
            last = torch.full_like(last, spline_valid_prefix)
        result.interpolated_trajectory = interpolated
        result.interpolated_last_tstep = last
        return result

    def plan_pose(
        self, goal_tool_poses: GoalToolPose, current_state: JointState,
        use_implicit_goal: bool = True, max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        self._assert_live()
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
            goal_tool_poses, current_state, max_attempts, enable_graph_attempt,
            use_implicit_goal,
        )

    def _plan_pose_single(
        self, goal_tool_poses, current_state, max_attempts, enable_graph_attempt,
        use_implicit_goal=True,
    ):
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
            # Match the V2 retry contract: successful IK seeds repair failed
            # ones so TrajOpt receives a complete seed population.  Indexing
            # with a boolean mask produces a copy in PyTorch, so build the
            # repaired tensor explicitly instead of relying on in-place fancy
            # indexing (which silently did nothing in the old facade).
            if seed_config.ndim == 3 and ik.success.ndim == 2:
                repaired = seed_config.clone()
                for batch in range(repaired.shape[0]):
                    valid = torch.nonzero(ik.success[batch], as_tuple=False).flatten()
                    if valid.numel() and valid.numel() < repaired.shape[1]:
                        reference = repaired[batch, valid[0]].expand_as(repaired[batch])
                        repaired[batch] = torch.where(
                            ik.success[batch, :, None], repaired[batch], reference
                        )
                seed_config = repaired
            seed_traj = None
            if attempt >= enable_graph_attempt and self.graph_planner is not None:
                seed_traj = self._get_graph_seed_trajectories(current, seed_config)
                if seed_traj is None:
                    # Once graph seeding is enabled, a failed PRM query is an
                    # attempt failure.  Retrying from a fresh deterministic
                    # seed mirrors V2 and avoids treating an unvalidated
                    # direct endpoint as a graph-derived motion.
                    last = None
                    self.reset_seed()
                    continue
            result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current,
                seed_config=seed_config, seed_traj=seed_traj,
                return_seeds=1,
                # ``TrajOptSolver`` composes its portable IK route.  Its
                # CUDA rollout-only implicit-goal branch is intentionally not
                # exposed as a fake Metal feature, regardless of the familiar
                # high-level compatibility argument.
                use_implicit_goal=False,
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
            goal_tool_poses, current_state, max_attempts, enable_graph_attempt=max_attempts,
            use_implicit_goal=use_implicit_goal,
        )

    def plan_cspace(
        self, goal_state: JointState, current_state: JointState,
        max_attempts: int = 5, enable_graph_attempt: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        self._assert_live()
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
                finetune_attempts=3 if seed_traj is not None else 1,
                finetune_dt_scale=0.75 if seed_traj is not None else 0.55,
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
    ) -> GraspPlanResult:
        self._assert_live()
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
        self._assert_live()
        if not isinstance(enable_collision_links, list) or not all(
            isinstance(name, str) for name in enable_collision_links
        ):
            raise TypeError("enable_collision_links must be a list of link names")
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        self._assert_live()
        if not isinstance(disable_collision_links, list) or not all(
            isinstance(name, str) for name in disable_collision_links
        ):
            raise TypeError("disable_collision_links must be a list of link names")
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

    def update_world(self, scene_cfg: SceneCfg):
        """Update the collision world without leaving planner stages stale.

        A ``SceneCfg`` (or complete per-environment list) mutates the existing
        cache when it fits, preserving references held by callers and by an
        attachment manager.  An explicit ``SceneCollision`` or config is a
        replacement.  All variants are validated before mutation, and the IK,
        TrajOpt pose-composition IK, graph planner, and attachment manager are
        subsequently pointed at the same object.
        """
        self._assert_live()
        if isinstance(scene_cfg, SceneCollision):
            self._install_world(scene_cfg)
            return
        if isinstance(scene_cfg, SceneCollisionCfg):
            self._install_world(SceneCollision.from_config(scene_cfg))
            return
        if not isinstance(scene_cfg, SceneCfg) and not (
            isinstance(scene_cfg, list) and all(isinstance(value, SceneCfg) for value in scene_cfg)
        ):
            raise NotImplementedError(
                "portable MotionPlanner world updates require SceneCfg, a list of SceneCfg, "
                "SceneCollisionCfg, or SceneCollision; YAML/USD scene assets are unavailable"
            )
        scenes = self._as_scene_list(scene_cfg)
        self._validate_scene_count(scenes)
        # A cache-too-small replacement is built first, so a rejected update
        # cannot clear a caller-visible existing world halfway through.
        if self._scene_collision is None or self._scene_collision.num_envs != len(scenes) or not all(
            self._scene_fits_cache(value, self._scene_collision) for value in scenes
        ):
            self._install_world(self._make_scene_collision(scene_cfg))
            return
        for environment, scene in enumerate(scenes):
            self._scene_collision.load_collision_model(scene, environment)
        self._install_world(self._scene_collision)

    def clear_scene_cache(self):
        self._assert_live()
        if self._scene_collision is not None:
            self._scene_collision.clear_cache()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
        self._world_generation += 1

    def reset_seed(self):
        self._assert_live()
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner.reset_seed()

    def update_link_inertial(
        self, link_name: str, mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None, inertia: Optional[torch.Tensor] = None,
    ) -> None:
        self._assert_live()
        # The production whole-body backend deliberately exposes immutable
        # inertial parameters today.  Delegate rather than inventing a local
        # mutation cache so callers receive the same explicit backend boundary
        # from both planner stages.
        self.ik_solver.update_link_inertial(link_name, mass, com, inertia)
        self.trajopt_solver.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]]
    ) -> None:
        self._assert_live()
        if not isinstance(link_properties, dict):
            raise TypeError("link_properties must map link names to property mappings")
        for name, values in link_properties.items():
            if not isinstance(name, str) or not isinstance(values, dict):
                raise TypeError("link_properties must map link names to property mappings")
            self.update_link_inertial(name, **values)

    def update_tool_pose_criteria(
        self, tool_pose_criteria: Dict[str, ToolPoseCriteria]
    ):
        """Forward portable criteria to both planner stages.

        Criteria records are accepted and retained for public lifecycle
        compatibility.  The CUDA-only non-terminal linear rollout weighting
        is not emulated by the underlying portable optimizer.
        """
        self._assert_live()
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

__all__ = ["MotionPlanner", "_axis_string_to_vector"]
