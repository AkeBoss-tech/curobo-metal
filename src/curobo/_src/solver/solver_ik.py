"""Portable implementation of cuRoboV2's inverse-kinematics facade."""

from __future__ import annotations

import time
from typing import Optional

import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

from .solver_ik_cfg import IKSolverCfg
from .solver_ik_result import IKSolverResult


class IKSolver:
    def __init__(self, config: IKSolverCfg, scene_collision_checker=None):
        if not isinstance(config, IKSolverCfg):
            raise TypeError("config must be IKSolverCfg")
        self.config = config
        # Retain the supplied portable SceneCollision for lifecycle parity.  IK
        # collision costs remain a documented future composition, rather than
        # silently treating a supplied scene as active.
        self._scene_collision_checker = scene_collision_checker
        robot = config.robot_config.kinematics
        kin_cfg = KinematicsCfg(
            config.device_cfg, list(robot.tool_frames), KinematicsParams(robot)
        )
        self._kinematics = Kinematics(kin_cfg)

    @property
    def kinematics(self): return self._kinematics
    @property
    def joint_names(self): return self._kinematics.joint_names
    @property
    def tool_frames(self): return self._kinematics.tool_frames
    @property
    def device_cfg(self): return self.config.device_cfg
    @property
    def action_dim(self): return self._kinematics.dof
    @property
    def action_horizon(self): return 1
    @property
    def default_joint_position(self):
        return self.config.robot_config.kinematics.cspace.default_joint_position
    @property
    def default_joint_state(self):
        return JointState(self.device_cfg.to_device(self.default_joint_position), joint_names=self.joint_names)

    optimizer = property(lambda self: None)
    metrics_rollout = property(lambda self: None)
    auxiliary_rollout = property(lambda self: None)
    transition_model = property(lambda self: None)
    solve_state = property(lambda self: None)
    seed_manager = property(lambda self: None)
    goal_registry_manager = property(lambda self: None)
    scene_collision_checker = property(lambda self: self._scene_collision_checker)
    problem_batch_size = property(lambda self: self.config.max_batch_size)

    def compute_kinematics(self, state: JointState):
        return self._kinematics.compute_kinematics(state)
    def get_active_js(self, full_js): return self._kinematics.get_active_js(full_js)
    def get_full_js(self, active_js): return active_js
    def reset_seed(self): return None
    def reset_shape(self): return None
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")
    def destroy(self): return None

    def get_all_rollout_instances(self, **kwargs):
        del kwargs
        return []
    def prepare_action_seeds(self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None):
        del seed_config
        if seed_traj is not None:
            return seed_traj.position if isinstance(seed_traj, JointState) else seed_traj
        if current_state is not None:
            position = current_state.position
            if position.ndim == 1:
                position = position.unsqueeze(0)
            return position[:, None, None, :].expand(batch_size, num_seeds, 1, -1).clone()
        return self.sample_configs(batch_size * num_seeds).reshape(batch_size, num_seeds, 1, -1)
    def prepare_trajectory_seeds(self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None):
        return self.prepare_action_seeds(batch_size, num_seeds, seed_config, current_state, seed_traj)
    def enable_tool_pose_tracking(self, tool_frames=None):
        del tool_frames
        return None
    def disable_tool_pose_tracking(self, tool_frames=None):
        del tool_frames
        return None
    def enable_joint_position_tracking(self): return None
    def disable_joint_position_tracking(self): return None
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self.config.tool_pose_criteria = dict(tool_pose_criteria)
    def update_world(self, scene_cfg):
        """Replace the active portable world collision scene.

        A supplied :class:`SceneCollision` is mutated in place, matching the
        lifecycle expected by consumers that retain its obstacle cache.  When
        IK was constructed standalone, a concrete portable scene creates the
        same adapter lazily.  YAML/USD scene assets remain intentionally out
        of scope rather than being interpreted as an empty world.
        """
        from curobo._src.geom.collision.collision_scene import (
            SceneCollision,
            SceneCollisionCfg,
        )
        from curobo._src.geom.types import SceneCfg

        if isinstance(scene_cfg, SceneCollision):
            self._scene_collision_checker = scene_cfg
        elif isinstance(scene_cfg, SceneCfg) or (
            isinstance(scene_cfg, list) and all(isinstance(value, SceneCfg) for value in scene_cfg)
        ):
            if self._scene_collision_checker is None:
                self._scene_collision_checker = SceneCollision(SceneCollisionCfg(
                    device_cfg=self.device_cfg,
                    scene_model=scene_cfg,
                    num_envs=len(scene_cfg) if isinstance(scene_cfg, list) else 1,
                ))
            else:
                if isinstance(scene_cfg, list):
                    if len(scene_cfg) != self._scene_collision_checker.num_envs:
                        raise ValueError(
                            "updated world environment count must match the existing SceneCollision"
                        )
                    for env_idx, value in enumerate(scene_cfg):
                        self._scene_collision_checker.load_collision_model(value, env_idx)
                else:
                    self._scene_collision_checker.load_collision_model(scene_cfg)
        else:
            raise NotImplementedError(
                "portable IK world updates require SceneCfg, a list of SceneCfg, or SceneCollision; "
                "YAML/USD scene assets are unavailable"
            )
        self.config.core_cfg.scene_collision_cfg = scene_cfg
    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        del mass, com, inertia
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)
    def debug_dump(self, *args, **kwargs):
        del args, kwargs
        return {"backend": "portable", "cuda_graph": False}

    def sample_configs(self, num_samples: int, rejection_ratio: int = 10):
        del rejection_ratio
        lower, upper = self._joint_limits()
        generator = torch.Generator(device="cpu").manual_seed(self.config.random_seed)
        unit = torch.rand((num_samples, self.action_dim), generator=generator, dtype=self.device_cfg.dtype)
        return lower + unit.to(self.device_cfg.device) * (upper - lower)

    def _joint_limits(self):
        robot = self.config.robot_config.kinematics
        by_name = {joint.name: joint.limits for joint in robot.joints}
        lower = [by_name[name].lower for name in self.joint_names]
        upper = [by_name[name].upper for name in self.joint_names]
        lower_t = self.device_cfg.to_device(lower)
        upper_t = self.device_cfg.to_device(upper)
        lower_t = torch.where(torch.isfinite(lower_t), lower_t, torch.full_like(lower_t, -torch.pi))
        upper_t = torch.where(torch.isfinite(upper_t), upper_t, torch.full_like(upper_t, torch.pi))
        return lower_t, upper_t

    def _prepare_pose_goal(self, goal_tool_poses: GoalToolPose) -> GoalToolPose:
        if not isinstance(goal_tool_poses, GoalToolPose):
            if isinstance(goal_tool_poses, Pose):
                goal_tool_poses = GoalToolPose.from_poses(
                    {self.tool_frames[-1]: goal_tool_poses},
                    ordered_tool_frames=[self.tool_frames[-1]],
                )
            else:
                raise TypeError("goal_tool_poses must be GoalToolPose or Pose")
        if goal_tool_poses.horizon != 1:
            raise NotImplementedError(
                "portable IK solves one target timestep; trajectory pose goals belong to TrajOpt"
            )
        if goal_tool_poses.batch_size < 1:
            raise ValueError("goal_tool_poses batch size must be positive")
        if goal_tool_poses.batch_size > self.config.max_batch_size:
            raise ValueError(
                f"goal batch size {goal_tool_poses.batch_size} exceeds config.max_batch_size "
                f"{self.config.max_batch_size}"
            )
        if goal_tool_poses.num_goalset > self.config.max_goalset:
            raise ValueError(
                f"goal set size {goal_tool_poses.num_goalset} exceeds config.max_goalset "
                f"{self.config.max_goalset}"
            )
        # PyTorch may represent the same MPS device as ``mps`` or ``mps:0``.
        # Unlike CUDA, MPS has no independently addressable device indices.
        same_device = goal_tool_poses.device == self.device_cfg.device or (
            goal_tool_poses.device.type == "mps" and self.device_cfg.device.type == "mps"
        )
        if not same_device:
            raise ValueError(
                f"goal poses are on {goal_tool_poses.device}, expected {self.device_cfg.device}"
            )
        if goal_tool_poses.position.dtype != self.device_cfg.dtype:
            raise ValueError(
                f"goal poses use {goal_tool_poses.position.dtype}, expected {self.device_cfg.dtype}"
            )
        return goal_tool_poses.reorder_links(self.tool_frames)

    def _prepare_seeds(
        self,
        batch: int,
        num_seeds: int,
        current_state: Optional[JointState],
        seed_config,
    ) -> torch.Tensor:
        """Normalize the public seed ranks to ``[batch, seed, dof]``."""
        seeds = self.sample_configs(batch * num_seeds).reshape(batch, num_seeds, self.action_dim)
        if seed_config is None and current_state is not None:
            seed_config = current_state
        if seed_config is None:
            return seeds
        supplied = seed_config.position if isinstance(seed_config, JointState) else seed_config
        if not isinstance(supplied, torch.Tensor):
            supplied = self.device_cfg.to_device(supplied)
        same_device = supplied.device == self.device_cfg.device or (
            supplied.device.type == "mps" and self.device_cfg.device.type == "mps"
        )
        if not same_device or supplied.dtype != self.device_cfg.dtype:
            raise ValueError("seed_config must use the solver device and dtype")
        if supplied.shape[-1] != self.action_dim:
            raise ValueError(f"seed_config final dimension must be {self.action_dim}")
        if supplied.ndim == 1:
            supplied = supplied.reshape(1, 1, -1)
        elif supplied.ndim == 2:
            supplied = supplied.unsqueeze(1)
        elif supplied.ndim != 3:
            raise ValueError("seed_config must have shape [dof], [batch,dof], or [batch,seeds,dof]")
        if supplied.shape[0] != batch:
            raise ValueError(
                f"seed_config batch dimension {supplied.shape[0]} must match goal batch {batch}"
            )
        count = min(supplied.shape[1], num_seeds)
        seeds[:, :count] = supplied[:, :count]
        return seeds

    def _world_clearance(self, state, batch: int, num_seeds: int) -> torch.Tensor:
        """Return minimum signed world clearance for each IK seed.

        This deliberately calls the public SceneCollision query rather than a
        private Metal cache.  It keeps world updates visible to IK and leaves
        exact Warp/BVH behaviour as an explicit unsupported boundary.
        """
        scene = self._scene_collision_checker
        spheres = state.robot_spheres
        if scene is None or spheres is None or spheres.shape[-2] == 0:
            return spheres.new_full((batch, num_seeds), torch.inf) if spheres is not None else (
                self.device_cfg.to_device(torch.full((batch, num_seeds), torch.inf))
            )
        if self.config.multi_env:
            if scene.num_envs != batch:
                raise ValueError(
                    "multi_env IK requires one SceneCollision environment per goal batch"
                )
            env_indices = torch.arange(batch, device=spheres.device).repeat_interleave(num_seeds)
        else:
            env_indices = None
        buffer = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
        distance = scene.get_sphere_distance_raw(
            spheres,
            buffer,
            torch.ones((), device=spheres.device, dtype=spheres.dtype),
            torch.as_tensor(
                self.config.optimizer_collision_activation_distance,
                device=spheres.device,
                dtype=spheres.dtype,
            ),
            env_indices,
        )
        return distance.amin(dim=(-1, -2)).reshape(batch, num_seeds)

    def _evaluate_pose_seeds(
        self,
        q: torch.Tensor,
        goal_tool_poses: GoalToolPose,
        include_collision_cost: bool,
    ):
        """Evaluate every seed against every goal-set candidate deterministically."""
        batch, num_seeds, _ = q.shape
        state = self.compute_kinematics(
            JointState(q.reshape(-1, self.action_dim), joint_names=self.joint_names)
        )
        position = state.tool_poses.position[:, 0].reshape(
            batch, num_seeds, len(self.tool_frames), 3
        )
        quaternion = state.tool_poses.quaternion[:, 0].reshape(
            batch, num_seeds, len(self.tool_frames), 4
        )
        target_p = goal_tool_poses.position[:, 0]
        target_q = goal_tool_poses.quaternion[:, 0]
        position_error = torch.linalg.vector_norm(
            position[:, :, :, None, :] - target_p[:, None], dim=-1
        )
        dot = (quaternion[:, :, :, None, :] * target_q[:, None]).sum(-1).abs().clamp(max=1)
        rotation_error = 2 * torch.acos(dot)
        per_goal_cost = (position_error.square() + 4 * (1 - dot.square())).sum(dim=2)
        pose_cost, goal_index = per_goal_cost.min(dim=-1)
        gather = goal_index[:, :, None, None].expand(-1, -1, len(self.tool_frames), 1)
        selected_position_error = position_error.gather(-1, gather).squeeze(-1)
        selected_rotation_error = rotation_error.gather(-1, gather).squeeze(-1)
        clearance = self._world_clearance(state, batch, num_seeds)
        cost = pose_cost
        if include_collision_cost and self._scene_collision_checker is not None:
            activation = torch.as_tensor(
                self.config.optimizer_collision_activation_distance,
                device=q.device,
                dtype=q.dtype,
            )
            cost = cost + 10.0 * (activation - clearance).clamp_min(0).square()
        return cost, selected_position_error, selected_rotation_error, goal_index, clearance

    def _compute_solution_velocity(
        self, result: IKSolverResult, current_state: Optional[JointState]
    ) -> None:
        if (
            self.config.optimization_dt is None
            or current_state is None
            or result.js_solution is None
        ):
            return
        current = self.get_active_js(current_state).position
        if current.ndim == 1:
            current = current.unsqueeze(0)
        if current.shape[0] != result.solution.shape[0]:
            raise ValueError("current_state batch dimension must match goal batch")
        dt = current.new_full((current.shape[0], 1, 1), self.config.optimization_dt)
        result.js_solution.velocity = (result.solution - current[:, None]) / dt

    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config=None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
    ):
        goal_tool_poses = self._prepare_pose_goal(goal_tool_poses)
        if return_seeds < 1:
            raise ValueError("return_seeds must be positive")
        batch = goal_tool_poses.batch_size
        num_seeds = max(self.config.num_seeds, return_seeds)
        seeds = self._prepare_seeds(batch, num_seeds, current_state, seed_config)
        lower, upper = self._joint_limits()
        q = seeds.clone()
        first = torch.zeros_like(q)
        second = torch.zeros_like(q)
        started = time.monotonic()
        iterations = 200
        if run_optimizer:
            for step in range(1, iterations + 1):
                q = q.requires_grad_(True)
                cost, _, _, _, _ = self._evaluate_pose_seeds(q, goal_tool_poses, True)
                grad = torch.autograd.grad(cost.sum(), q)[0]
                # Exact quaternion alignment and degenerate rotational Jacobians can
                # produce a non-finite intermediate gradient on eager MPS/CPU.
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                first = 0.9 * first + 0.1 * grad
                second = 0.999 * second + 0.001 * grad.square()
                q = (q - 0.05 * first / (1 - 0.9**step) / (
                    (second / (1 - 0.999**step)).sqrt() + 1e-8
                )).clamp(lower, upper)
                q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                    lower, upper
                ).detach()
        cost, position_error_by_link, rotation_error_by_link, goal_index, clearance = (
            self._evaluate_pose_seeds(q, goal_tool_poses, run_optimizer)
        )
        position_error = position_error_by_link.amax(dim=-1)
        rotation_error = rotation_error_by_link.amax(dim=-1)
        rank = cost.argsort(dim=1)
        take = rank[:, :return_seeds]
        solution = torch.gather(q, 1, take[..., None].expand(-1, -1, self.action_dim))
        pe = torch.gather(position_error, 1, take)
        re = torch.gather(rotation_error, 1, take)
        feasible = torch.gather(clearance >= 0, 1, take)
        success = (pe <= self.config.position_tolerance) & (re <= self.config.orientation_tolerance)
        if not self.config.success_requires_convergence:
            success = feasible
        selected_goal = torch.gather(goal_index, 1, take)
        result = IKSolverResult(
            success, solution, JointState(solution, joint_names=self.joint_names),
            pe, re, solve_time=time.monotonic() - started, total_time=time.monotonic() - started,
            optimized_seeds=q, seed_rank=take, seed_cost=torch.gather(cost, 1, take),
            goalset_index=selected_goal[..., None].expand(-1, -1, len(self.tool_frames)),
            batch_size=batch, num_seeds=num_seeds, feasible=feasible,
            metrics={
                "position_error_per_link": torch.gather(
                    position_error_by_link, 1, take[..., None].expand(-1, -1, len(self.tool_frames))
                ),
                "rotation_error_per_link": torch.gather(
                    rotation_error_by_link, 1, take[..., None].expand(-1, -1, len(self.tool_frames))
                ),
                "world_clearance": torch.gather(clearance, 1, take),
            },
        )
        self._compute_solution_velocity(result, current_state)
        return result


__all__ = ["IKSolver"]
