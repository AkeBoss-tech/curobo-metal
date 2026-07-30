"""Portable trajectory optimizer facade over production curobo-metal operations."""

from __future__ import annotations

import time

import torch

from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.util.trajectory import (
    TrajInterpolationType, get_batch_interpolated_trajectory,
)
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator
from curobo_metal.ops.trajectory import TrajectoryProblem, optimize_trajectory


class TrajOptSolver:
    def __init__(self, config: TrajOptSolverCfg, scene_collision_checker=None):
        if not isinstance(config, TrajOptSolverCfg):
            raise TypeError("config must be TrajOptSolverCfg")
        self.config = config
        self.scene_collision_checker = scene_collision_checker
        robot = config.robot_config.kinematics
        self._chain = robot.to_kinematic_chain(
            device=config.device_cfg.device, dtype=config.device_cfg.dtype
        )
        joints = [
            joint for joint in robot.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        self._lower = config.device_cfg.to_device([joint.limits.lower for joint in joints])
        self._upper = config.device_cfg.to_device([joint.limits.upper for joint in joints])
        self._seed_generator = TrajectorySeedGenerator(
            config.action_horizon, self._chain.dof, config.device_cfg
        )

    def prepare_action_seeds(
        self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None,
    ):
        del seed_config
        if seed_traj is not None:
            return seed_traj.position if isinstance(seed_traj, JointState) else seed_traj
        if current_state is None:
            current_state = self.default_joint_state.unsqueeze(0).repeat((batch_size, 1))
        return self._seed_generator.generate_constant_seeds(current_state.position, num_seeds)

    def prepare_trajectory_seeds(
        self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None,
    ):
        return self.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )

    def solve_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        seed_traj=None,
        return_seeds: int = 1,
        num_seeds=None,
        dt=None,
        finetune_attempts: int = 1,
        initial_iters=None,
        time_optimal_iters=None,
        finetune_iters=None,
        finetune_dt_scale: float = 0.55,
    ):
        del finetune_attempts, time_optimal_iters, finetune_iters, finetune_dt_scale
        count = self.config.num_seeds if num_seeds is None else num_seeds
        start = current_state.position
        goal = goal_state.position
        if start.ndim == 1:
            start, goal = start[None], goal[None]
        seeds = None
        if seed_traj is not None:
            seeds = seed_traj.position if isinstance(seed_traj, JointState) else seed_traj
            if seeds.ndim == 3:
                seeds = seeds[:, None]
        else:
            seeds = self._seed_generator.generate_interpolated_seeds(
                start, goal[:, None].expand(-1, count, -1), count
            )
        begin = time.perf_counter()
        result = optimize_trajectory(TrajectoryProblem(
            self._chain, start, goal, self._lower, self._upper,
            self.config.action_horizon, float(dt or self.config.interpolation_dt),
            seeds=seeds,
            max_iterations=self.config.max_iterations if initial_iters is None else initial_iters,
            endpoint_tolerance=self.config.position_tolerance,
            optimizer=self.config.optimizer_name,
        ))
        trajectory = result.trajectories
        if trajectory.ndim == 3:
            trajectory = trajectory[None]
        chosen = trajectory[:, :return_seeds]
        state = JointState.from_position(
            chosen, self.joint_names
        ).finite_difference(float(dt or self.config.interpolation_dt))
        success = result.success
        if success.ndim == 1:
            success = success[None]
        return TrajOptSolverResult(
            success=success[:, :return_seeds],
            solution=chosen,
            js_solution=state,
            cspace_error=result.endpoint_error[..., :return_seeds],
            solve_time=time.perf_counter() - begin,
            total_time=time.perf_counter() - begin,
            optimized_seeds=trajectory,
            seed_cost=result.objective,
            batch_size=start.shape[0],
            num_seeds=count,
            feasible=result.maximum_limit_violation <= 0,
            maximum_trajectory_dt=start.new_full(
                (start.shape[0],), float(dt or self.config.interpolation_dt)
            ),
            minimum_trajectory_dt=start.new_full(
                (start.shape[0],), float(dt or self.config.interpolation_dt)
            ),
        )

    def solve_pose(self, *args, **kwargs):
        goal_state = kwargs.pop("goal_state", None)
        current_state = kwargs.pop("current_state", args[1] if len(args) > 1 else None)
        if goal_state is None:
            raise NotImplementedError(
                "portable solve_pose requires goal_state; pose-to-joint IK belongs to IKSolver"
            )
        kwargs.pop("goal_tool_poses", None)
        return self.solve_cspace(goal_state, current_state, **kwargs)

    def get_interpolated_trajectory(self, js_optimized: JointState):
        kind = self.config.interpolation_type
        if kind == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
            kind = TrajInterpolationType.LINEAR_CUDA
        return get_batch_interpolated_trajectory(
            js_optimized, js_optimized.position.new_tensor(self.config.interpolation_dt),
            kind, device_cfg=self.config.device_cfg,
        )

    def compute_trajectory_dt(self, trajectory: JointState, epsilon: float = 0.001, scale_dt: bool = True):
        del epsilon, scale_dt
        return trajectory.position.new_full(
            trajectory.position.shape[:-2], self.config.interpolation_dt
        )

    def reset_seed(self): return None
    def reset_shape(self): return None
    def destroy(self): return None
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")

    @property
    def action_dim(self): return self._chain.dof
    @property
    def action_horizon(self): return self.config.action_horizon
    @property
    def horizon(self): return self.config.action_horizon
    @property
    def opt_dim(self): return self.action_dim
    @property
    def joint_names(self): return list(self.config.robot_config.kinematics.joint_names)
    @property
    def default_joint_position(self): return self.config.robot_config.kinematics.retract_config
    @property
    def default_joint_state(self):
        return JointState.from_position(self.default_joint_position, self.joint_names)
    @property
    def device_cfg(self): return self.config.device_cfg
    @property
    def problem_batch_size(self): return self.config.max_batch_size
    @property
    def interpolation_steps(self): return self.config.interpolation_buffer_size


__all__ = ["TrajOptSolver"]
