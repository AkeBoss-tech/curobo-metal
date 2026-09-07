"""Portable trajectory optimizer facade over production curobo-metal operations."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

import curobo._src.runtime as curobo_runtime
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.util.trajectory import (
    TrajInterpolationType, calculate_dt_no_clamp, get_batch_interpolated_trajectory,
)
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator
from curobo_metal.optim import ExecutionCache
from curobo_metal.ops.trajectory import TrajectoryProblem, optimize_trajectory
from curobo_metal.ops.kinematics import KinematicChain as _KinematicChain
from curobo_metal.reference.forward_kinematics import SerialRobot as _SerialRobot


def _trajectory_dimension_chain(robot, *, device, dtype) -> _KinematicChain:
    """Build the trajectory optimizer's joint-dimension carrier.

    A serial robot can use its actual chain.  Whole-body retargeting robots
    are trees with several tracked effectors, while this optimizer's composed
    objective is purely joint-space unless a collision model is explicitly
    supplied.  For that case, use a deterministic zero-origin serial carrier
    containing every active joint.  Cartesian IK still runs through the real
    tree kinematics; this carrier supplies only the DOF/device metadata used
    by the joint-space trajectory objective.
    """
    if len(robot.tool_frames) == 1:
        return robot.to_kinematic_chain(device=device, dtype=dtype)
    joints = [
        joint for joint in robot.joints
        if joint.kind != "fixed" and joint.mimic_joint is None
    ]
    proxy = _SerialRobot.from_dict({
        "name": f"{robot.name}_trajectory_coordinates",
        "joints": [
            {
                "name": joint.name,
                "type": "revolute",
                "axis": [0.0, 0.0, 1.0],
                "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            }
            for joint in joints
        ],
    })
    return _KinematicChain(proxy, device=device, dtype=dtype)


class TrajOptSolver:
    def __init__(
        self, config: TrajOptSolverCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ):
        if not isinstance(config, TrajOptSolverCfg):
            raise TypeError("config must be TrajOptSolverCfg")
        self.config = config
        self._scene_collision_checker = scene_collision_checker
        robot = config.robot_config.kinematics
        self._chain = _trajectory_dimension_chain(
            robot, device=config.device_cfg.device, dtype=config.device_cfg.dtype
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
        self._trajectory_seed_generator = self._seed_generator
        self._tool_pose_tracking = True
        self._joint_position_tracking = True
        self._sample_generator = torch.Generator(device="cpu")
        self._sample_generator.manual_seed(config.random_seed)
        # Shape-keyed optimizer state remains ordinary CPU/MPS tensor state;
        # unlike an upstream CUDA graph it never captures a device stream.
        self._execution_cache = ExecutionCache()
        self._solve_state: Optional[SolveState] = None
        self._goal_buffer: Optional[GoalRegistry] = None
        self._last_trace: Dict[str, object] = {}
        self._destroyed = False
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
            max_batch_size=config.max_batch_size,
            max_goalset=config.max_goalset,
        ), scene_collision_checker=scene_collision_checker)

    def prepare_action_seeds(
        self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None,
    ):
        if isinstance(num_seeds, bool) or not isinstance(num_seeds, int) or num_seeds < 1:
            raise ValueError("num_seeds must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if seed_traj is not None:
            value = seed_traj.position if isinstance(seed_traj, JointState) else seed_traj
            return self._normalize_seed_trajectory(value, batch_size, num_seeds)
        if seed_config is not None:
            value = seed_config.position if isinstance(seed_config, JointState) else seed_config
            if not isinstance(value, torch.Tensor):
                raise TypeError("seed_config must be a JointState or tensor")
            if value.ndim == 2:
                if value.shape != (batch_size, self.action_dim):
                    raise ValueError("seed_config must have shape [batch, dof]")
                return self._seed_generator.generate_constant_seeds(value, num_seeds)
            if value.ndim == 3:
                if value.shape[0] != batch_size or value.shape[-1] != self.action_dim:
                    raise ValueError("seed_config must have shape [batch, seed, dof]")
                if value.shape[1] < num_seeds:
                    repeats = math.ceil(num_seeds / value.shape[1])
                    value = value.repeat(1, repeats, 1)
                return self._seed_generator.generate_interpolated_seeds(
                    value[:, 0], value[:, :num_seeds], num_seeds
                )
            raise ValueError("seed_config must have shape [batch, dof] or [batch, seed, dof]")
        if current_state is None:
            current_state = self.default_joint_state.unsqueeze(0).repeat((batch_size, 1))
        state = self.get_active_js(current_state)
        if state.position.ndim == 1:
            state = state.unsqueeze(0)
        if state.position.shape != (batch_size, self.action_dim):
            raise ValueError("current_state must have shape [batch, dof]")
        return self._seed_generator.generate_constant_seeds(state.position, num_seeds)

    def prepare_trajectory_seeds(
        self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None,
    ):
        return self.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )

    @profiler.record_function("trajopt_solver/solve_cspace")
    def solve_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        seed_traj=None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        dt=None,
        finetune_attempts: int = 1,
        initial_iters: Optional[int] = None,
        time_optimal_iters: Optional[int] = None,
        finetune_iters: Optional[int] = None,
        finetune_dt_scale: float = 0.55,
    ) -> TrajOptSolverResult:
        self._assert_live()
        self._validate_solve_options(
            finetune_attempts, initial_iters, time_optimal_iters,
            finetune_iters, finetune_dt_scale,
        )
        count = self.config.num_seeds if num_seeds is None else num_seeds
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("num_seeds must be a positive integer")
        if isinstance(return_seeds, bool) or not isinstance(return_seeds, int) or return_seeds < 1:
            raise ValueError("return_seeds must be a positive integer")
        # V2 increases the seed count rather than silently returning fewer
        # trajectories when a caller asks for more ranked plans.
        count = max(count, return_seeds)
        current_state, goal_state = self.get_active_js(current_state), self.get_active_js(goal_state)
        override = self.__dict__.get("_solve_impl")
        # pytest/monkeypatch (and some integrations) restore a bound default
        # method into the instance dictionary.  Only an actual override should
        # intercept the normal portable implementation.
        if (
            override is not None
            and getattr(override, "__func__", None) is not TrajOptSolver._solve_impl
        ):
            batch = 1 if current_state.position.ndim == 1 else current_state.position.shape[0]
            solve_state = SolveState(
                SolveMode.BATCH if batch > 1 else SolveMode.SINGLE,
                batch, 1, num_goalset=1, num_trajopt_seeds=count,
                tool_frames=list(self.tool_frames),
            )
            return override(
                solve_state=solve_state,
                current_state=current_state,
                seed_goal_js=goal_state.unsqueeze(1),
                seed_traj=seed_traj,
                return_seeds=return_seeds,
                num_seeds=count,
                use_implicit_goal=True,
            )
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
        if start.shape[0] > self.config.max_batch_size:
            raise ValueError(
                f"solve_cspace batch_size={start.shape[0]} exceeds config.max_batch_size="
                f"{self.config.max_batch_size}"
            )
        self._set_solve_state(start.shape[0], count, goalset=1)
        solve_dt = self._resolve_dt(dt, start.shape[0], count, start)
        if seed_traj is not None:
            seeds = seed_traj.position if isinstance(seed_traj, JointState) else seed_traj
            seeds = self._normalize_seed_trajectory(seeds, start.shape[0], count)
        else:
            seeds = self._seed_generator.generate_interpolated_seeds(
                start, goal[:, None].expand(-1, count, -1), count
            )
        begin = time.perf_counter()
        iteration_count = self.config.max_iterations if initial_iters is None else initial_iters
        result = self._optimize(start, goal, seeds, solve_dt, iteration_count)
        trajectory = result.trajectories
        if trajectory.ndim == 3:
            trajectory = trajectory[None]
        objective = result.objective[None] if result.objective.ndim == 1 else result.objective
        success = result.success[None] if result.success.ndim == 1 else result.success
        endpoint_error = result.endpoint_error[None] if result.endpoint_error.ndim == 1 else result.endpoint_error
        violation = (result.maximum_limit_violation[None]
                     if result.maximum_limit_violation.ndim == 1
                     else result.maximum_limit_violation)

        # Match V2's time-optimal lifecycle with portable, ordinary PyTorch
        # solves.  A candidate replaces a seed only when it remains feasible,
        # so a faster invalid seed can never displace a working plan.
        trace = [{"attempt": 0, "dt": solve_dt, "iterations": iteration_count,
                  "accepted": int(success.sum().item())}]
        best_dt = solve_dt
        for attempt in range(1, finetune_attempts + 1):
            candidate_dt = min(
                self.config.maximum_trajectory_dt,
                max(self.config.minimum_trajectory_dt, best_dt * finetune_dt_scale),
            )
            if candidate_dt >= best_dt:
                break
            candidate_iterations = (
                time_optimal_iters if attempt == 1 and time_optimal_iters is not None
                else finetune_iters if attempt > 1 and finetune_iters is not None
                else self.config.max_iterations
            )
            candidate = self._optimize(start, goal, trajectory, candidate_dt, candidate_iterations)
            candidate_trajectory = (
                candidate.trajectories[None]
                if candidate.trajectories.ndim == 3 else candidate.trajectories
            )
            candidate_objective = (
                candidate.objective[None] if candidate.objective.ndim == 1 else candidate.objective
            )
            candidate_success = (
                candidate.success[None] if candidate.success.ndim == 1 else candidate.success
            )
            candidate_endpoint = (
                candidate.endpoint_error[None]
                if candidate.endpoint_error.ndim == 1 else candidate.endpoint_error
            )
            candidate_violation = (
                candidate.maximum_limit_violation[None]
                if candidate.maximum_limit_violation.ndim == 1
                else candidate.maximum_limit_violation
            )
            replace = candidate_success & (~success | (candidate_dt < best_dt))
            trajectory = torch.where(replace[..., None, None], candidate_trajectory, trajectory)
            objective = torch.where(replace, candidate_objective, objective)
            success = torch.where(replace, candidate_success, success)
            endpoint_error = torch.where(replace, candidate_endpoint, endpoint_error)
            violation = torch.where(replace, candidate_violation, violation)
            accepted = int(replace.sum().item())
            trace.append({"attempt": attempt, "dt": candidate_dt,
                          "iterations": candidate_iterations, "accepted": accepted})
            if accepted:
                best_dt = candidate_dt
            else:
                break
        rank = objective.argsort(dim=-1)
        chosen_index = rank[:, :return_seeds]
        chosen = self._gather_seed_tensor(trajectory, chosen_index)
        chosen_success = self._gather_seed_tensor(success, chosen_index)
        chosen_error = self._gather_seed_tensor(endpoint_error, chosen_index)
        chosen_violation = self._gather_seed_tensor(violation, chosen_index)
        selected_dt = best_dt
        state = JointState.from_position(
            chosen, self.joint_names
        ).finite_difference(selected_dt)
        state.dt = chosen.new_full(chosen.shape[:2], selected_dt)
        state.knot = chosen
        state.knot_dt = state.dt
        dense_state, last_tstep = self.get_interpolated_trajectory(
            JointState.from_position(
                chosen.reshape(-1, chosen.shape[-2], chosen.shape[-1]), self.joint_names
            ).finite_difference(selected_dt)
        )
        interpolated = dense_state.position.reshape(
            chosen.shape[0], chosen.shape[1], dense_state.position.shape[-2], chosen.shape[-1]
        )
        wall = time.perf_counter() - begin
        def reshape_dense(value):
            if value is None:
                return None
            if value.ndim == 0:
                return value.expand(chosen.shape[:2])
            return value.reshape(chosen.shape[0], chosen.shape[1], *value.shape[1:])

        dense_result = JointState(
            interpolated,
            reshape_dense(dense_state.velocity),
            reshape_dense(dense_state.acceleration),
            self.joint_names,
            reshape_dense(dense_state.jerk),
            dt=reshape_dense(dense_state.dt),
        )
        output = TrajOptSolverResult(
            success=chosen_success,
            solution=chosen,
            js_solution=state,
            cspace_error=chosen_error,
            solve_time=wall,
            total_time=wall,
            optimized_seeds=trajectory,
            seed_cost=objective,
            seed_rank=rank,
            batch_size=start.shape[0],
            num_seeds=count,
            feasible=chosen_violation <= 0,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            total_cost_reshaped=objective,
            interpolated_trajectory=dense_result,
            interpolated_last_tstep=last_tstep.view(chosen.shape[:2]),
            maximum_trajectory_dt=start.new_full((start.shape[0],), selected_dt),
            minimum_trajectory_dt=start.new_full((start.shape[0],), selected_dt),
        )
        output.goalset_index = torch.zeros_like(chosen_index)
        output.debug_info.update({
            "backend": "portable",
            "optimizer": self.config.optimizer_name,
            "finetune": trace,
            "execution_cache": {
                "generation": self._execution_cache.generation,
                "hits": self._execution_cache.hits,
                "misses": self._execution_cache.misses,
                "size": self._execution_cache.size,
            },
        })
        output.process_metrics_and_rank_seeds()
        self._last_trace = output.debug_info
        return output

    @profiler.record_function("trajopt_solver/solve_pose")
    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        seed_config=None,
        seed_traj=None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        dt=None,
        use_implicit_goal: bool = False,
        finetune_attempts: int = 1,
        goal_state: Optional[JointState] = None,
        initial_iters: Optional[int] = None,
        time_optimal_iters: Optional[int] = None,
        finetune_iters: Optional[int] = None,
        finetune_dt_scale: float = 0.55,
    ) -> TrajOptSolverResult:
        """Solve a pose request through portable IK followed by trajectory optimization.

        Supplying ``goal_state`` preserves the upstream override path.  Otherwise
        the best IK seed becomes the c-space endpoint.  CUDA graph execution is
        deliberately not emulated; both stages retain normal CPU/MPS autograd.
        """
        self._assert_live()
        if not isinstance(current_state, JointState):
            raise TypeError("current_state must be a JointState")
        if use_implicit_goal and goal_state is None:
            raise NotImplementedError(
                "implicit pose goals require cuRobo CUDA rollout buffers; provide goal_state "
                "or use the portable IK-composed pose path"
            )
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
        if seed_config is not None and seed_traj is None:
            seed_traj = self._trajectory_from_seed_config(
                current_state, seed_config,
                self.config.num_seeds if num_seeds is None else max(num_seeds, return_seeds),
            )
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

    @profiler.record_function("trajopt_solver/get_interpolated_trajectory")
    def get_interpolated_trajectory(self, js_optimized: JointState) -> Tuple[JointState, torch.Tensor, bool]:
        kind = self.config.interpolation_type
        if kind == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
            kind = TrajInterpolationType.LINEAR_CUDA
        if not isinstance(js_optimized, JointState) or js_optimized.position.ndim != 3:
            raise ValueError("js_optimized must be a JointState with shape [batch, horizon, dof]")
        source_dt = js_optimized.dt
        if source_dt is None or source_dt.ndim == 0:
            source_dt = js_optimized.position.new_full(
                (js_optimized.position.shape[0],),
                self.config.interpolation_dt if source_dt is None else float(source_dt.item()),
            )
            js_optimized = js_optimized.clone()
            js_optimized.dt = source_dt
        elif source_dt.numel() == 1:
            js_optimized = js_optimized.clone()
            js_optimized.dt = source_dt.expand(js_optimized.position.shape[0])
        elif source_dt.shape != (js_optimized.position.shape[0],):
            raise ValueError("js_optimized.dt must be scalar or [batch]")
        state, last_tstep = get_batch_interpolated_trajectory(
            js_optimized, js_optimized.position.new_tensor(self.config.interpolation_dt),
            kind, device_cfg=self.config.device_cfg,
        )
        if state.position.shape[-2] > self.config.interpolation_buffer_size:
            raise ValueError(
                f"Interpolated trajectory requires {state.position.shape[-2]} waypoints, exceeding "
                f"interpolation_buffer_size={self.config.interpolation_buffer_size}. Recreate the "
                "configuration with a larger interpolation_buffer_size or interpolation_dt. "
                "A larger interpolation_dt produces fewer waypoints. Increasing "
                "interpolation_buffer_size can significantly increase GPU memory usage."
            )
        # The portable motion-planner facade consumes the historical two-value
        # form.  Buffer reallocation is not observable here because composed
        # Torch interpolation allocates its output directly.
        return state, last_tstep

    @profiler.record_function("trajopt_solver/compute_trajectory_dt")
    def compute_trajectory_dt(self, trajectory: JointState, epsilon: float = 0.001, scale_dt: bool = True) -> torch.Tensor:
        if not isinstance(trajectory, JointState):
            raise TypeError("trajectory must be a JointState")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        state = trajectory if all(
            value is not None for value in (trajectory.velocity, trajectory.acceleration, trajectory.jerk)
        ) else trajectory.finite_difference(self.config.interpolation_dt)
        limits = self.config.robot_config.kinematics.cspace
        def limit(name, default):
            value = getattr(limits, name, None)
            if value is None:
                value = default
            return torch.as_tensor(value, device=state.position.device, dtype=state.position.dtype).expand(self.action_dim)
        velocity = limit("max_velocity", torch.inf)
        acceleration = limit("max_acceleration", torch.inf)
        jerk = limit("max_jerk", torch.inf)
        def maximum(value):
            return value.abs().amax(dim=-2)
        ratio = torch.maximum(
            (maximum(state.velocity) / velocity.clamp_min(epsilon)).amax(dim=-1),
            torch.sqrt((maximum(state.acceleration) / acceleration.clamp_min(epsilon)).amax(dim=-1)),
        )
        ratio = torch.maximum(
            ratio, (maximum(state.jerk) / jerk.clamp_min(epsilon)).amax(dim=-1).pow(1 / 3)
        )
        dt = ratio * (1 + epsilon)
        if scale_dt and state.dt is not None:
            dt = dt * torch.as_tensor(state.dt, device=dt.device, dtype=dt.dtype)
        return dt.clamp(self.config.minimum_trajectory_dt, self.config.maximum_trajectory_dt)

    def reset_seed(self):
        self._sample_generator.manual_seed(self.config.random_seed)
        self._seed_generator = TrajectorySeedGenerator(
            self.config.action_horizon, self._chain.dof, self.config.device_cfg
        )
        return None
    def reset_shape(self):
        self._execution_cache.reset()
        self._solve_state = None
        self._goal_buffer = None
        return None
    def destroy(self):
        self._pose_ik.destroy()
        self.reset_shape()
        self._destroyed = True
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")

    @property
    def action_dim(self) -> int: return self._chain.dof
    @property
    def action_horizon(self) -> int: return self.config.action_horizon
    @property
    def horizon(self) -> int: return self.config.action_horizon
    @property
    def opt_dim(self) -> int: return self.action_dim * self.action_horizon
    @property
    def joint_names(self): return list(self.config.robot_config.kinematics.joint_names)
    @property
    def tool_frames(self): return list(self.config.robot_config.kinematics.tool_frames)
    @property
    def default_joint_position(self):
        # Robot configuration files are host-side data.  Materialize their
        # retract state on the solver device rather than leaking a CPU tensor
        # into an otherwise MPS planning request.
        return self.config.device_cfg.to_device(
            self.config.robot_config.kinematics.retract_config
        )
    @property
    def default_joint_state(self):
        return JointState.from_position(self.default_joint_position, self.joint_names)
    @property
    def device_cfg(self): return self.config.device_cfg
    @property
    def problem_batch_size(self) -> int: return self.config.max_batch_size
    @property
    def interpolation_steps(self) -> int: return 4

    # These CUDA-rollout ownership accessors are part of the pinned solver
    # facade.  Portable trajectory optimization has no graph-backed rollout
    # objects, so unavailable handles are represented explicitly as ``None``.
    @property
    def optimizer(self):
        return None

    @property
    def optimizer_rollouts(self):
        return None

    @property
    def metrics_rollout(self):
        return None

    @property
    def additional_metrics_rollouts(self):
        return None

    @property
    def auxiliary_rollout(self):
        return None

    @property
    def transition_model(self):
        return None

    @property
    def scene_collision_checker(self):
        return self._scene_collision_checker

    @property
    def goal_registry_manager(self):
        return None

    @property
    def seed_manager(self):
        return None

    @property
    def kinematics(self):
        return self._chain

    @property
    def solve_state(self) -> SolveState:
        return self._solve_state
    optimizer = property(lambda self: self)
    optimizer_rollouts = property(lambda self: [])
    metrics_rollout = property(lambda self: None)
    auxiliary_rollout = property(lambda self: None)
    additional_metrics_rollouts = property(lambda self: {"interpolated_rollout": self})
    transition_model = property(lambda self: None)
    scene_collision_checker = property(lambda self: self._scene_collision_checker)
    goal_registry_manager = property(lambda self: self._pose_ik.goal_registry_manager)
    seed_manager = property(lambda self: self._seed_generator)
    solve_state = property(lambda self: self._solve_state)
    kinematics = property(lambda self: self._pose_ik.kinematics)

    def _solve_impl(self, **kwargs):
        """Compatibility interception point used by graph-backed integrations."""
        raise NotImplementedError("direct rollout-buffer solves are unavailable on CPU/MPS")

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
        self._validate_tool_frames(tool_frames)
        self._tool_pose_tracking = True
        return None
    def disable_tool_pose_tracking(self, tool_frames=None):
        self._validate_tool_frames(tool_frames)
        self._tool_pose_tracking = False
        return None
    def enable_joint_position_tracking(self):
        self._joint_position_tracking = True
        return None
    def disable_joint_position_tracking(self):
        self._joint_position_tracking = False
        return None
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self.config.tool_pose_criteria = dict(tool_pose_criteria)
    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        del mass, com, inertia
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)
    def get_active_js(self, full_js):
        if not isinstance(full_js, JointState):
            raise TypeError("full_js must be a JointState")
        if full_js.joint_names is None or full_js.joint_names == self.joint_names:
            if full_js.position.shape[-1] != self.action_dim:
                raise ValueError("joint state must contain the active trajectory joints")
            return full_js
        return full_js.reorder(self.joint_names)

    def get_full_js(self, active_js):
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.robot.types.kinematics_params import KinematicsParams
        params = KinematicsParams(self.config.robot_config.kinematics)
        model = Kinematics(KinematicsCfg(
            self.device_cfg, self.config.robot_config.kinematics.tool_frames, params
        ))
        return model.get_full_js(active_js)

    def sample_configs(self, num_samples, rejection_ratio=10):
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if isinstance(rejection_ratio, bool) or not isinstance(rejection_ratio, int) or rejection_ratio < 1:
            raise ValueError("rejection_ratio must be a positive integer")
        values = torch.rand((num_samples, self.action_dim), generator=self._sample_generator,
                            dtype=self._lower.dtype, device="cpu")
        return (self._lower.cpu() + values * (self._upper.cpu() - self._lower.cpu())).to(self._lower.device)

    def debug_dump(self, file_path):
        result = {
            "backend": "portable",
            "cuda_graph": False,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "optimizer": self.config.optimizer_name,
            "tool_pose_tracking": self._tool_pose_tracking,
            "joint_position_tracking": self._joint_position_tracking,
        }
        if file_path is not None:
            Path(file_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    def _resolve_dt(self, dt, batch, count, reference):
        if dt is None:
            return float(self.config.interpolation_dt)
        value = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)
        if value.numel() != 1:
            if value.shape not in {(batch,), (batch, count)}:
                raise ValueError("dt must be scalar, [batch], or [batch, num_seeds]")
            if not torch.allclose(value, value.reshape(-1)[0]):
                raise NotImplementedError("per-seed trajectory dt is unavailable on the portable backend")
            value = value.reshape(-1)[0]
        scalar = float(value.item())
        if not math.isfinite(scalar) or scalar <= 0:
            raise ValueError("dt must be finite and positive")
        return min(max(scalar, self.config.minimum_trajectory_dt), self.config.maximum_trajectory_dt)

    def _normalize_seed_trajectory(self, value, batch, count):
        if not isinstance(value, torch.Tensor):
            raise TypeError("seed trajectory must be a torch.Tensor")
        if value.ndim == 3:
            value = value[:, None]
        if value.ndim != 4 or value.shape[0] != batch or value.shape[2:] != (
            self.action_horizon, self.action_dim
        ):
            raise ValueError("seed trajectory must have shape [batch, seed, horizon, dof]")
        if value.shape[1] < count:
            repeats = math.ceil(count / value.shape[1])
            value = value.repeat(1, repeats, 1, 1)
        return value[:, :count].to(device=self._lower.device, dtype=self._lower.dtype)

    @staticmethod
    def _gather_seed_tensor(value, index):
        while index.ndim < value.ndim:
            index = index.unsqueeze(-1)
        return value.gather(1, index.expand(*index.shape[:2], *value.shape[2:]))

    def _assert_live(self) -> None:
        if self._destroyed:
            raise RuntimeError("TrajOptSolver has been destroyed")

    def _set_solve_state(self, batch_size: int, num_seeds: int, *, goalset: int) -> None:
        if batch_size == 1:
            mode = SolveMode.SINGLE
        elif self.config.multi_env:
            mode = SolveMode.MULTI_ENV
        else:
            mode = SolveMode.BATCH
        solve_state = SolveState(
            mode, batch_size, batch_size if self.config.multi_env else 1,
            num_goalset=goalset, num_trajopt_seeds=num_seeds,
            tool_frames=self.tool_frames,
        )
        structural = self._solve_state is None or any(
            getattr(self._solve_state, name, None) != getattr(solve_state, name, None)
            for name in ("solve_type", "batch_size", "num_envs", "num_goalset", "num_trajopt_seeds")
        )
        self._solve_state = solve_state
        if structural:
            self._execution_cache.reset()

    def _optimize(self, start, goal, seeds, dt, max_iterations):
        return optimize_trajectory(TrajectoryProblem(
            self._chain, start, goal, self._lower, self._upper,
            self.config.action_horizon, dt,
            seeds=seeds,
            max_iterations=max_iterations,
            endpoint_tolerance=self.config.position_tolerance,
            optimizer=self.config.optimizer_name,
            optimizer_cache=self._execution_cache,
            warm_start=True,
        ))

    @staticmethod
    def _validate_solve_options(
        finetune_attempts, initial_iters, time_optimal_iters, finetune_iters, finetune_dt_scale,
    ) -> None:
        if (isinstance(finetune_attempts, bool) or not isinstance(finetune_attempts, int)
                or finetune_attempts < 0):
            raise ValueError("finetune_attempts must be a nonnegative integer")
        for name, value in (("initial_iters", initial_iters),
                            ("time_optimal_iters", time_optimal_iters),
                            ("finetune_iters", finetune_iters)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(finetune_dt_scale, (float, int)) or isinstance(finetune_dt_scale, bool):
            raise TypeError("finetune_dt_scale must be a finite float in (0, 1]")
        if not math.isfinite(float(finetune_dt_scale)) or not 0 < float(finetune_dt_scale) <= 1:
            raise ValueError("finetune_dt_scale must be finite and in (0, 1]")

    def _trajectory_from_seed_config(self, current_state, seed_config, count):
        value = seed_config.position if isinstance(seed_config, JointState) else seed_config
        if not isinstance(value, torch.Tensor):
            raise TypeError("seed_config must be a JointState or tensor")
        current = self.get_active_js(current_state).position
        if current.ndim == 1:
            current = current[None]
        if value.ndim == 2:
            if value.shape != current.shape:
                raise ValueError("seed_config must have shape [batch, dof] or [batch, seed, dof]")
            value = value[:, None]
        if value.ndim != 3 or value.shape[0] != current.shape[0] or value.shape[-1] != self.action_dim:
            raise ValueError("seed_config must have shape [batch, dof] or [batch, seed, dof]")
        if value.shape[1] < count:
            value = value.repeat(1, math.ceil(count / value.shape[1]), 1)
        return self._seed_generator.generate_interpolated_seeds(
            current, value[:, :count].to(device=current.device, dtype=current.dtype), count
        )

    def _validate_tool_frames(self, tool_frames) -> None:
        if tool_frames is None:
            return
        invalid = set(tool_frames).difference(self.tool_frames)
        if invalid:
            raise ValueError(f"unknown tool frame(s): {sorted(invalid)}")

# Retain the portable no-argument inspection convenience without changing the
# pinned declaration shape used by the static compatibility inventory.
TrajOptSolver.debug_dump.__defaults__ = (None,)


__all__ = ["TrajOptSolver"]
