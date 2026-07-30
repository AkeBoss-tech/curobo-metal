"""Portable receding-horizon MPC over the production trajectory optimizer."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose

from .solver_mpc_cfg import MPCSolverCfg
from .solver_mpc_result import MPCSolverResult


class MPCSolver:
    def __init__(self, config: MPCSolverCfg, scene_collision_checker=None):
        if not isinstance(config, MPCSolverCfg):
            raise TypeError("config must be MPCSolverCfg")
        if scene_collision_checker is not None:
            raise NotImplementedError("external SceneCollision injection is not portable")
        self.config = config
        self._ik = IKSolver(IKSolverCfg.create(
            config.robot_config, device_cfg=config.device_cfg,
            num_seeds=max(config.num_seeds, 4),
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=False, random_seed=config.random_seed,
        ))
        traj_cfg = TrajOptSolverCfg.create(
            config.robot_config, device_cfg=config.device_cfg,
            num_seeds=max(config.num_seeds, 1),
            interpolation_dt=config.optimization_dt,
            use_cuda_graph=False,
            override_optimizer_num_iters={
                "lbfgs": config.warm_start_optimization_num_iters
            },
        )
        traj_cfg.action_horizon = max(2, config.interpolation_steps + 1)
        self._trajopt = TrajOptSolver(traj_cfg)
        self._goal_state: Optional[JointState] = None
        self._current_state: Optional[JointState] = None
        self._seed_trajectory = None
        self._tool_pose_criteria: Dict[str, ToolPoseCriteria] = {}

    optimizer = property(lambda self: self._trajopt)
    metrics_rollout = property(lambda self: None)
    auxiliary_rollout = property(lambda self: None)
    additional_metrics_rollouts = property(lambda self: [])
    kinematics = property(lambda self: self._ik.kinematics)
    transition_model = property(lambda self: None)
    action_dim = property(lambda self: self._ik.action_dim)
    action_horizon = property(lambda self: self._trajopt.action_horizon)
    joint_names = property(lambda self: self._ik.joint_names)
    tool_frames = property(lambda self: self._ik.tool_frames)
    default_joint_position = property(lambda self: self._ik.default_joint_position)
    default_joint_state = property(lambda self: self._ik.default_joint_state)
    device_cfg = property(lambda self: self.config.device_cfg)
    scene_collision_checker = property(lambda self: None)
    goal_registry_manager = property(lambda self: None)
    solve_state = property(lambda self: None)
    seed_manager = property(lambda self: None)
    problem_batch_size = property(
        lambda self: 0 if self._current_state is None else self._current_state.position.shape[0]
        if self._current_state.position.ndim > 1 else 1
    )

    def get_all_rollout_instances(self, **kwargs):
        del kwargs
        return [self._trajopt]

    def compute_kinematics(self, state): return self._ik.compute_kinematics(state)
    def get_active_js(self, full_js): return self._ik.get_active_js(full_js)
    def get_full_js(self, active_js): return self._ik.get_full_js(active_js)
    def sample_configs(self, num_samples, rejection_ratio=10):
        return self._ik.sample_configs(num_samples, rejection_ratio)
    def prepare_action_seeds(self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None):
        return self._trajopt.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )
    def prepare_trajectory_seeds(self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None):
        return self._trajopt.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_config, seed_traj
        )

    def setup(self, current_state: JointState, tool_frames: Optional[List[str]] = None, dt=None):
        del dt
        if tool_frames is not None and list(tool_frames) != list(self.tool_frames):
            raise NotImplementedError("dynamic MPC tool-frame sets are unavailable")
        self._current_state = current_state
        if self._goal_state is None:
            self._goal_state = current_state.clone()
        return True

    def update_goal_tool_poses(
        self, goal_tool_poses: GoalToolPose, robot_ids=None, run_ik=True,
        use_ik_goal=True, use_best_effort_ik=False,
    ):
        del robot_ids, use_ik_goal
        if not run_ik:
            raise NotImplementedError("MPC pose goals require portable IK")
        result = self._ik.solve_pose(goal_tool_poses, current_state=self._current_state)
        if not bool(result.success.all().item()) and not use_best_effort_ik:
            return result
        self._goal_state = JointState.from_position(
            result.solution[:, 0], self.joint_names
        )
        return result

    def update_goal_state(self, goal_state: JointState, robot_ids=None):
        del robot_ids
        self._goal_state = goal_state
        return True

    def update_current_state(self, current_state: JointState):
        self._current_state = current_state

    def update_seed_trajectory(self, seed_trajectory):
        self._seed_trajectory = seed_trajectory

    def update_seed_trajectory_from_goal_state(self, goal_joint_state):
        self._goal_state = goal_joint_state
        self._seed_trajectory = None

    def _solve_impl(self, current_state: JointState, optimization_niters: int):
        if self._goal_state is None:
            raise RuntimeError("call setup and set a goal before optimizing")
        result = self._trajopt.solve_cspace(
            self._goal_state, current_state, seed_traj=self._seed_trajectory,
            initial_iters=optimization_niters,
            dt=self.config.optimization_dt,
        )
        sequence = result.js_solution
        if sequence.position.ndim == 4:
            sequence = sequence[:, 0]
        next_index = min(1, sequence.position.shape[-2] - 1)
        next_action = sequence.position[..., next_index, :]
        if next_action.ndim > 2:
            next_action = next_action[:, 0]
        return MPCSolverResult(
            success=result.success,
            solution=result.solution,
            js_solution=sequence,
            solve_time=result.solve_time,
            total_time=result.total_time,
            debug_info={"trajectory_result": result},
            feasible=result.feasible,
            next_action=JointState.from_position(next_action, self.joint_names),
            action_sequence=sequence,
            full_action_sequence=sequence,
            action_buffer=sequence.position,
            action_dt=self.config.optimization_dt,
        )

    def optimize_next_action(self, current_state):
        return self.warm_start_solve(current_state)
    def optimize_action_sequence(self, current_state):
        return self.warm_start_solve(current_state)
    def cold_start_solve(self, current_state):
        return self._solve_impl(current_state, self.config.cold_start_optimization_num_iters)
    def warm_start_solve(self, current_state):
        return self._solve_impl(current_state, self.config.warm_start_optimization_num_iters)

    def set_default_goal_from_current_state(self, current_state, robot_ids=None):
        del robot_ids
        self._goal_state = current_state.clone()
    def reset_robot(self, current_state):
        self._current_state = current_state
        self._goal_state = current_state.clone()
    def reset_robot_id(self, current_state, robot_ids):
        del robot_ids
        self.reset_robot(current_state)
    def reset_shape(self): return None
    def reset_seed(self): return self._trajopt.reset_seed()
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")
    def destroy(self):
        self._ik.destroy()
        self._trajopt.destroy()
    def enable_tool_pose_tracking(self, tool_frames=None): return None
    def disable_tool_pose_tracking(self, tool_frames=None): return None
    def enable_joint_position_tracking(self): return None
    def disable_joint_position_tracking(self): return None
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self._tool_pose_criteria = dict(tool_pose_criteria)
    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)


__all__ = ["MPCSolver"]
