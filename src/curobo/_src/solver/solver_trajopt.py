"""Portable trajectory optimizer facade over production curobo-metal operations."""

from __future__ import annotations

import time

import torch

from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
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
        self._scene_collision_checker = scene_collision_checker
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
        # Pose planning is a composition of the portable IK and c-space
        # trajectory solvers.  Keeping it private avoids changing the existing
        # configuration model while making the upstream ``solve_pose`` entry
        # point usable instead of requiring a nonstandard ``goal_state``.
        self._pose_ik = IKSolver(IKSolverCfg.create(
            config.robot_config,
            device_cfg=config.device_cfg,
            num_seeds=max(config.num_seeds, 1),
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=False,
            random_seed=config.random_seed,
            self_collision_check=config.self_collision_check,
        ), scene_collision_checker=scene_collision_checker)

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
            start = start[None]
        if goal.ndim == 1:
            goal = goal[None]
        if start.shape[-1] != goal.shape[-1]:
            raise ValueError("current_state and goal_state must have the same dof")
        if start.shape[0] != goal.shape[0]:
            if start.shape[0] == 1:
                start = start.expand(goal.shape[0], -1)
            elif goal.shape[0] == 1:
                goal = goal.expand(start.shape[0], -1)
            else:
                raise ValueError("current_state and goal_state batch sizes must match")
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

    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        seed_config=None,
        seed_traj=None,
        return_seeds: int = 1,
        num_seeds=None,
        dt=None,
        use_implicit_goal: bool = False,
        finetune_attempts: int = 1,
        goal_state: JointState | None = None,
        initial_iters=None,
        time_optimal_iters=None,
        finetune_iters=None,
        finetune_dt_scale: float = 0.55,
    ) -> TrajOptSolverResult:
        """Solve a pose request through portable IK followed by trajectory optimization.

        Supplying ``goal_state`` preserves the upstream override path.  Otherwise
        the best IK seed becomes the c-space endpoint.  CUDA graph execution is
        deliberately not emulated; both stages retain normal CPU/MPS autograd.
        """
        del use_implicit_goal
        if goal_state is None:
            ik_result = self._pose_ik.solve_pose(
                goal_tool_poses,
                current_state=current_state,
                seed_config=seed_config,
                return_seeds=1,
            )
            goal_state = JointState.from_position(
                ik_result.solution[:, 0], self.joint_names
            )
        else:
            ik_result = None
        result = self.solve_cspace(
            goal_state,
            current_state,
            seed_traj=seed_traj,
            return_seeds=return_seeds,
            num_seeds=num_seeds,
            dt=dt,
            finetune_attempts=finetune_attempts,
            initial_iters=initial_iters,
            time_optimal_iters=time_optimal_iters,
            finetune_iters=finetune_iters,
            finetune_dt_scale=finetune_dt_scale,
        )
        if ik_result is not None:
            result.debug_info["ik_result"] = ik_result
            # An infeasible pose must not be reported as a successful pose plan
            # merely because the c-space rollout reached its IK endpoint.
            result.success = result.success & ik_result.success[..., :result.success.shape[-1]]
        return result

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
    def destroy(self):
        self._pose_ik.destroy()
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
    def tool_frames(self): return list(self.config.robot_config.kinematics.tool_frames)
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
    optimizer = property(lambda self: self)
    optimizer_rollouts = property(lambda self: [])
    metrics_rollout = property(lambda self: None)
    auxiliary_rollout = property(lambda self: None)
    additional_metrics_rollouts = property(lambda self: [])
    transition_model = property(lambda self: None)
    scene_collision_checker = property(lambda self: self._scene_collision_checker)
    goal_registry_manager = property(lambda self: None)
    seed_manager = property(lambda self: self._seed_generator)
    solve_state = property(lambda self: None)
    kinematics = property(lambda self: self._chain)

    def compute_kinematics(self, state):
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.robot.types.kinematics_params import KinematicsParams
        params = KinematicsParams(self.config.robot_config.kinematics)
        return Kinematics(KinematicsCfg(self.device_cfg, self.config.robot_config.kinematics.tool_frames, params)).compute_kinematics(state)
    def get_all_rollout_instances(self, **kwargs):
        del kwargs
        return []
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
    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        del mass, com, inertia
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)
    def debug_dump(self, *args, **kwargs):
        del args, kwargs
        return {"backend": "portable", "cuda_graph": False}


__all__ = ["TrajOptSolver"]
