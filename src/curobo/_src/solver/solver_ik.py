"""Portable implementation of cuRoboV2's inverse-kinematics facade."""

from __future__ import annotations

import time
from typing import Optional

import torch

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
        if scene_collision_checker is not None:
            raise NotImplementedError("external SceneCollision injection is not yet portable")
        self.config = config
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

    def compute_kinematics(self, state: JointState):
        return self._kinematics.compute_kinematics(state)
    def get_active_js(self, full_js): return self._kinematics.get_active_js(full_js)
    def get_full_js(self, active_js): return active_js
    def reset_seed(self): return None
    def reset_shape(self): return None
    def reset_cuda_graph(self): return None
    def destroy(self): return None

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

    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config=None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
    ):
        del current_state
        if not run_optimizer:
            raise NotImplementedError("run_optimizer=False metrics-only mode is not implemented")
        if not isinstance(goal_tool_poses, GoalToolPose):
            if isinstance(goal_tool_poses, Pose):
                goal_tool_poses = GoalToolPose.from_poses(
                    {self.tool_frames[-1]: goal_tool_poses},
                    ordered_tool_frames=[self.tool_frames[-1]],
                )
            else:
                raise TypeError("goal_tool_poses must be GoalToolPose or Pose")
        if goal_tool_poses.num_links != 1 or goal_tool_poses.num_goalset != 1:
            raise NotImplementedError("portable IK currently supports one tool and one goal per batch")
        batch = goal_tool_poses.batch_size
        seeds = self.sample_configs(batch * self.config.num_seeds).reshape(
            batch, self.config.num_seeds, self.action_dim
        )
        if seed_config is not None:
            supplied = seed_config.position if isinstance(seed_config, JointState) else seed_config
            supplied = supplied.reshape(batch, -1, self.action_dim)
            count = min(supplied.shape[1], seeds.shape[1])
            seeds[:, :count] = supplied[:, :count]
        lower, upper = self._joint_limits()
        q = seeds.clone()
        first = torch.zeros_like(q)
        second = torch.zeros_like(q)
        target_p = goal_tool_poses.position[:, 0, 0, 0]
        target_q = goal_tool_poses.quaternion[:, 0, 0, 0]
        started = time.monotonic()
        iterations = 200
        for step in range(1, iterations + 1):
            q = q.requires_grad_(True)
            state = self.compute_kinematics(
                JointState(q.reshape(-1, self.action_dim), joint_names=self.joint_names)
            )
            position = state.tool_poses.position[:, 0, -1].reshape(batch, -1, 3)
            quaternion = state.tool_poses.quaternion[:, 0, -1].reshape(batch, -1, 4)
            pos_error = torch.linalg.vector_norm(position - target_p[:, None], dim=-1)
            dot = (quaternion * target_q[:, None]).sum(-1).abs().clamp(max=1)
            rot_error = 2 * torch.acos(dot)
            cost = pos_error.square() + 4 * (1 - dot.square())
            grad = torch.autograd.grad(cost.sum(), q)[0]
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad.square()
            q = (q - 0.05 * first / (1 - 0.9**step) / (
                (second / (1 - 0.999**step)).sqrt() + 1e-8
            )).clamp(lower, upper).detach()
        state = self.compute_kinematics(JointState(q.reshape(-1, self.action_dim), joint_names=self.joint_names))
        position = state.tool_poses.position[:, 0, -1].reshape(batch, -1, 3)
        quaternion = state.tool_poses.quaternion[:, 0, -1].reshape(batch, -1, 4)
        position_error = torch.linalg.vector_norm(position - target_p[:, None], dim=-1)
        rotation_error = 2 * torch.acos(
            (quaternion * target_q[:, None]).sum(-1).abs().clamp(max=1)
        )
        cost = position_error.square() + rotation_error.square()
        rank = cost.argsort(dim=1)
        take = rank[:, :return_seeds]
        solution = torch.gather(q, 1, take[..., None].expand(-1, -1, self.action_dim))
        pe = torch.gather(position_error, 1, take)
        re = torch.gather(rotation_error, 1, take)
        success = (pe <= self.config.position_tolerance) & (re <= self.config.orientation_tolerance)
        if not self.config.success_requires_convergence:
            success = torch.ones_like(success)
        return IKSolverResult(
            success, solution, JointState(solution, joint_names=self.joint_names),
            pe, re, solve_time=time.monotonic() - started, total_time=time.monotonic() - started,
            optimized_seeds=q, seed_rank=take, seed_cost=torch.gather(cost, 1, take),
            batch_size=batch, num_seeds=self.config.num_seeds, feasible=torch.ones_like(success),
        )


__all__ = ["IKSolver"]
