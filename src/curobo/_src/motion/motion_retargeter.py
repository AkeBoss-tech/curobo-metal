from __future__ import annotations

import torch

from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose

from .motion_retargeter_cfg import MotionRetargeterCfg
from .motion_retargeter_result import RetargetResult


class MotionRetargeter:
    def __init__(self, config: MotionRetargeterCfg):
        self._config = config
        self._global_ik_solver = IKSolver(IKSolverCfg.create(
            config.robot, device_cfg=config.device_cfg,
            num_seeds=config.num_seeds_global,
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=False,
        ))
        self._local_ik_solver = IKSolver(IKSolverCfg.create(
            config.robot, device_cfg=config.device_cfg,
            num_seeds=config.num_seeds_local,
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=False,
        ))
        self._mpc_solver = MPCSolver(MPCSolverCfg.create(
            config.robot, device_cfg=config.device_cfg,
            optimization_dt=config.optimization_dt,
            interpolation_steps=max(1, config.steps_per_target),
            warm_start_optimization_num_iters=config.mpc_warm_start_num_iters,
            cold_start_optimization_num_iters=config.mpc_cold_start_num_iters,
            use_cuda_graph=False,
        )) if config.use_mpc else None
        self._prev_solution = None
        self._mpc_state = None

    joint_names = property(lambda self: self._global_ik_solver.joint_names)
    action_dim = property(lambda self: self._global_ik_solver.action_dim)
    tool_frames = property(lambda self: self._global_ik_solver.tool_frames)
    kinematics = property(lambda self: self._global_ik_solver.kinematics)
    default_joint_state = property(lambda self: self._global_ik_solver.default_joint_state)
    num_dof = property(lambda self: self.action_dim)
    config = property(lambda self: self._config)

    def reset(self):
        self._prev_solution = None
        self._mpc_state = None

    def solve_frame(self, goal_tool_poses):
        if goal_tool_poses.batch_size != self._config.num_envs:
            raise ValueError("goal batch size must match num_envs")
        solver = self._global_ik_solver if self._prev_solution is None else self._local_ik_solver
        seed = None if self._prev_solution is None else self._prev_solution[:, None]
        current = None if self._prev_solution is None else JointState.from_position(
            self._prev_solution, self.joint_names
        )
        ik = solver.solve_pose(
            goal_tool_poses, current_state=current, seed_config=seed, return_seeds=1
        )
        solution = ik.solution[:, 0]
        self._prev_solution = solution.detach().clone()
        joint_state = JointState.from_position(solution, self.joint_names)
        trajectory = None
        if self._mpc_solver is not None:
            if self._mpc_state is None:
                self._mpc_state = joint_state.clone()
                self._mpc_solver.setup(self._mpc_state)
            self._mpc_solver.update_goal_state(joint_state)
            mpc = self._mpc_solver.optimize_action_sequence(self._mpc_state)
            trajectory = mpc.action_sequence
            self._mpc_state = mpc.next_action
            joint_state = mpc.next_action
            self._prev_solution = joint_state.position.detach().clone()
        return RetargetResult(joint_state, trajectory)

    def solve_sequence(self, tool_poses: SequenceGoalToolPose):
        if tool_poses.num_envs != self._config.num_envs:
            raise ValueError("sequence num_envs must match config")
        self.reset()
        rows, trajectories = [], []
        for index in range(tool_poses.num_frames):
            result = self.solve_frame(tool_poses.get_frame(index))
            rows.append(result.joint_state)
            if result.trajectory is not None:
                trajectories.append(result.trajectory)
        joint_state = JointState.from_position(
            torch.stack([row.position for row in rows], dim=1),
            self.joint_names,
        )
        trajectory = None
        if trajectories:
            trajectory = JointState.from_position(
                torch.cat([row.position for row in trajectories], dim=-2),
                self.joint_names,
            )
        return RetargetResult(joint_state, trajectory)


__all__ = ["MotionRetargeter"]
