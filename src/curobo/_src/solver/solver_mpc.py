"""Portable receding-horizon MPC over the production trajectory optimizer."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
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
        self.config = config
        self._scene_collision_checker = self._coerce_scene_collision(
            scene_collision_checker if scene_collision_checker is not None else config.scene_collision_cfg
        )
        self._ik = IKSolver(IKSolverCfg.create(
            config.robot_config, device_cfg=config.device_cfg,
            num_seeds=max(config.num_seeds, 4),
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=False, random_seed=config.random_seed,
            self_collision_check=config.self_collision_check,
            max_batch_size=config.max_batch_size,
            multi_env=config.multi_env,
            max_goalset=config.max_goalset,
        ), scene_collision_checker=self._scene_collision_checker)
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
        self._trajopt = TrajOptSolver(
            traj_cfg, scene_collision_checker=self._scene_collision_checker
        )
        self._goal_state: Optional[JointState] = None
        self._current_state: Optional[JointState] = None
        self._seed_trajectory = None
        # ``TrajectoryExecutionManager`` in the CUDA implementation owns a
        # graph-resident action ring buffer.  Keep the semantic part of that
        # contract locally: a regular tensor buffer and cursor.  In
        # particular, ``optimize_next_action`` must not re-run a full solve
        # merely because the caller consumes the next command.
        self._action_buffer: Optional[torch.Tensor] = None
        self._action_cursor = 0
        self._solve_count = 0
        self._tool_pose_criteria: Dict[str, ToolPoseCriteria] = {}
        self._goal_manager = GoalManager(config.device_cfg)
        self._solve_state: Optional[SolveState] = None
        self._goal_tool_poses: Optional[GoalToolPose] = None
        self._last_result: Optional[MPCSolverResult] = None
        self._setup_complete = False
        self._warm_start_available = False
        self._tool_pose_tracking = True
        self._joint_position_tracking = False

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
    scene_collision_checker = property(lambda self: self._scene_collision_checker)
    goal_registry_manager = property(lambda self: self._goal_manager)
    # The V2 spelling is useful to integrations which introspect MPC to reuse
    # its pose solver.  Keep the old private name as the implementation detail.
    ik_solver = property(lambda self: self._ik)
    solve_state = property(lambda self: self._solve_state)
    seed_manager = property(lambda self: self._trajopt.seed_manager)
    problem_batch_size = property(
        lambda self: 0 if self._current_state is None else self._current_state.position.shape[0]
        if self._current_state.position.ndim > 1 else 1
    )

    def get_all_rollout_instances(self, **kwargs):
        del kwargs
        return [self._trajopt, self._ik]

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
        """Initialise a fixed-shape portable MPC problem.

        The implementation retains normal PyTorch allocations instead of CUDA
        graph capture.  A setup therefore fixes only the public batch/tool
        shape, not a raw CUDA buffer address.
        """
        current_state = self._normalise_state(current_state, "current_state")
        batch_size = current_state.position.shape[0]
        if batch_size != self.config.max_batch_size:
            raise ValueError(
                f"current_state batch size {batch_size} must equal config.max_batch_size "
                f"{self.config.max_batch_size}"
            )
        frames = list(self.tool_frames) if tool_frames is None else list(tool_frames)
        if not frames:
            raise ValueError("tool_frames must not be empty")
        if not set(frames).issubset(self.tool_frames):
            raise ValueError(f"tool_frames must be a subset of {self.tool_frames}")
        if len(frames) != 1:
            raise NotImplementedError(
                "portable MPC supports one tracked tool frame; multi-tool tracking is unavailable"
            )
        current_state.dt = self._normalise_dt(dt, current_state, batch_size)
        mode = (
            SolveMode.MULTI_ENV if self.config.multi_env else
            (SolveMode.BATCH if batch_size > 1 else SolveMode.SINGLE)
        )
        self._solve_state = SolveState(
            mode, batch_size, batch_size if self.config.multi_env else 1,
            num_goalset=1, num_trajopt_seeds=max(self.config.num_seeds, 1),
            tool_frames=frames,
        )
        current_kinematics = self.compute_kinematics(current_state)
        self._goal_tool_poses = self._as_goal_tool_pose(
            current_kinematics.tool_poses, frames
        )
        self._current_state = current_state.clone()
        self._goal_state = current_state.clone()
        self._goal_manager.create_goal_buffer(
            self._solve_state, goal_tool_poses=self._goal_tool_poses,
            goal_js=self._goal_state, current_js=self._current_state,
            seed_goal_js=self._goal_state.unsqueeze(1),
            current_state_dt=current_state.dt,
        )
        self._seed_trajectory = self.prepare_trajectory_seeds(
            batch_size, max(self.config.num_seeds, 1), current_state
        )
        self._action_buffer = None
        self._action_cursor = 0
        self._solve_count = 0
        self._setup_complete = True
        self._warm_start_available = False
        return True

    def update_goal_tool_poses(
        self, goal_tool_poses: GoalToolPose, robot_ids=None, run_ik=True,
        use_ik_goal=True, use_best_effort_ik=False,
    ):
        if not self._setup_complete:
            raise RuntimeError("MPC problem not setup, call setup first")
        goal_tool_poses = self._normalise_goal_pose(goal_tool_poses)
        if robot_ids is not None:
            if self._goal_tool_poses is None:
                raise RuntimeError("no stored goal pose; call setup first")
            robot_ids = self._normalise_robot_ids(robot_ids)
            if goal_tool_poses.batch_size != self.problem_batch_size:
                raise ValueError("partial MPC pose updates require a full-batch goal pose")
            merged = self._goal_tool_poses.clone()
            merged.position[robot_ids] = goal_tool_poses.position[robot_ids]
            merged.quaternion[robot_ids] = goal_tool_poses.quaternion[robot_ids]
            goal_tool_poses = merged
        self._goal_tool_poses = goal_tool_poses
        self._goal_manager.update_goal_tool_poses(goal_tool_poses)
        if not run_ik:
            # The pose is retained for observability, but portable MPC needs a
            # joint-space endpoint to optimise; it intentionally does not
            # pretend to provide a CUDA rollout-only Cartesian controller.
            self._joint_position_tracking = False
            return True
        result = self._ik.solve_pose(goal_tool_poses, current_state=self._current_state)
        self._last_result = result
        if not bool(result.success.all().item()) and not use_best_effort_ik:
            return False
        if use_ik_goal:
            self.update_goal_state(
                JointState.from_position(result.solution[:, 0], self.joint_names),
                robot_ids=None,
            )
            self._joint_position_tracking = True
        return True

    def update_goal_state(self, goal_state: JointState, robot_ids=None):
        if not self._setup_complete:
            raise RuntimeError("MPC problem not setup, call setup first")
        goal_state = self._normalise_state(goal_state, "goal_state")
        if goal_state.position.shape[0] != self.problem_batch_size:
            raise ValueError("goal_state batch size must match the configured MPC batch")
        if robot_ids is not None:
            ids = self._normalise_robot_ids(robot_ids)
            updated = self._goal_state.clone()
            updated[ids] = goal_state[ids]
            goal_state = updated
        self._goal_state = goal_state
        self._goal_manager.update_goal_state(goal_state)
        return True

    def update_current_state(self, current_state: JointState):
        current_state = self._normalise_state(current_state, "current_state")
        if self._setup_complete and current_state.position.shape[0] != self.problem_batch_size:
            raise ValueError("current_state batch size must match the configured MPC batch")
        if current_state.dt is None:
            current_state.dt = self._normalise_dt(None, current_state, current_state.position.shape[0])
        self._current_state = current_state
        if self._setup_complete:
            self._goal_manager.update_current_state(current_state)

    def update_seed_trajectory(self, seed_trajectory):
        if not self._setup_complete:
            raise RuntimeError("MPC problem not setup, call setup first")
        values = seed_trajectory.position if isinstance(seed_trajectory, JointState) else seed_trajectory
        if not isinstance(values, torch.Tensor):
            raise TypeError("seed_trajectory must be a JointState or tensor")
        if values.ndim not in (3, 4):
            raise ValueError(
                "seed_trajectory must have shape [batch,horizon,dof] or "
                "[batch,seed,horizon,dof]"
            )
        expected_tail = (self.action_horizon, self.action_dim)
        if values.shape[0] != self.problem_batch_size or values.shape[-2:] != expected_tail:
            raise ValueError(
                "seed_trajectory shape must equal "
                f"[{self.problem_batch_size},{self.action_horizon},{self.action_dim}] "
                "or [batch,seed,horizon,dof]"
            )
        if values.ndim == 4 and values.shape[1] < 1:
            raise ValueError("seed_trajectory seed dimension must be non-empty")
        self._require_solver_tensor(values, "seed_trajectory")
        self._seed_trajectory = values.clone()
        self._action_buffer = values[:, 0].clone() if values.ndim == 4 else values.clone()
        self._action_cursor = 0
        self._warm_start_available = True

    def update_seed_trajectory_from_goal_state(self, goal_joint_state):
        self.update_goal_state(goal_joint_state)
        self._seed_trajectory = self.prepare_trajectory_seeds(
            self.problem_batch_size, max(self.config.num_seeds, 1), self._current_state,
            seed_config=self._goal_state.position[:, None],
        )
        self._warm_start_available = True

    def _solve_impl(self, current_state: JointState, optimization_niters: int):
        if not self._setup_complete or self._goal_state is None:
            raise RuntimeError("call setup and set a goal before optimizing")
        self.update_current_state(current_state)
        # Feed the last plan back through TrajOpt as an ordinary PyTorch seed.
        # Once its command cursor has advanced, shift the remainder forward and
        # pad with the terminal pose; this is the portable counterpart to the
        # CUDA action-buffer shift.
        seed = self._seed_trajectory
        if self._warm_start_available and self._action_buffer is not None:
            shift = min(max(self._action_cursor + 1, 1), self.action_horizon - 1)
            tail = self._action_buffer[:, -1:].expand(-1, shift, -1)
            shifted = torch.cat((self._action_buffer[:, shift:], tail), dim=1)
            seed = shifted
        result = self._trajopt.solve_cspace(
            self._goal_state, self._current_state, seed_traj=seed,
            initial_iters=optimization_niters,
            dt=self.config.optimization_dt,
        )
        sequence = result.js_solution
        if sequence.position.ndim == 4:
            sequence = sequence[:, 0]
        # ``next_action`` in a freshly optimized sequence is its first command.
        # ``optimize_next_action`` consumes that command through the buffer
        # cursor below; keeping this at zero avoids silently skipping a step.
        next_index = 0
        next_action = sequence.position[..., next_index, :]
        if next_action.ndim > 2:
            next_action = next_action[:, 0]
        sequence_values = sequence.position
        # An infeasible rollout still needs a physically conservative command
        # for the robot.  Preserve the solver's failure status but replace its
        # command stream with an explicitly documented deceleration/hold plan.
        success_flat = result.success.reshape(self.problem_batch_size, -1).all(dim=-1)
        if not bool(success_flat.all().item()) and self.config.use_deceleration_on_failure:
            safe = self.prepare_safe_deceleration_trajectory(
                self._current_state, ~success_flat
            )
            sequence_values = sequence_values.clone()
            sequence_values[~success_flat] = safe[~success_flat]
            sequence = JointState.from_position(sequence_values, self.joint_names)
        self._seed_trajectory = sequence_values.detach().clone()
        self._action_buffer = sequence_values.detach().clone()
        self._action_cursor = 0
        self._solve_count += 1
        mpc_result = MPCSolverResult(
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
            action_buffer=sequence_values,
            action_dt=self.config.optimization_dt,
            cspace_error=result.cspace_error,
            optimized_seeds=result.optimized_seeds,
            seed_cost=result.seed_cost,
            batch_size=self.problem_batch_size,
            num_seeds=result.num_seeds,
            metrics={
                "endpoint_error": result.cspace_error,
                "feasible": result.feasible,
                "warm_start": self._warm_start_available,
            },
        )
        self._last_result = mpc_result
        self._warm_start_available = True
        return mpc_result

    def optimize_next_action(self, current_state):
        if not self._setup_complete:
            raise RuntimeError("MPC problem not setup, call setup first")
        self.update_current_state(current_state)
        # A cold plan initializes the portable buffer.  Subsequent calls
        # consume it exactly once per command; re-plan only after the final
        # command has been consumed, matching the upstream execution manager.
        if self._action_buffer is None or self._action_cursor >= self.action_horizon:
            if not self._warm_start_available:
                self.cold_start_solve(current_state)
            else:
                self.warm_start_solve(current_state)
        return self._result_from_action_buffer()
    def optimize_action_sequence(self, current_state):
        if not self._setup_complete:
            raise RuntimeError("MPC problem not setup, call setup first")
        return self.warm_start_solve(current_state)
    def cold_start_solve(self, current_state):
        return self._solve_impl(current_state, self.config.cold_start_optimization_num_iters)
    def warm_start_solve(self, current_state):
        return self._solve_impl(current_state, self.config.warm_start_optimization_num_iters)

    def set_default_goal_from_current_state(self, current_state, robot_ids=None):
        self.update_goal_state(current_state, robot_ids)
    def reset_robot(self, current_state):
        self.update_current_state(current_state)
        self._goal_state = self._current_state.clone()
        self._goal_manager.update_goal_state(self._goal_state)
        self._seed_trajectory = self.prepare_trajectory_seeds(
            self.problem_batch_size, max(self.config.num_seeds, 1), self._current_state
        )
        self._warm_start_available = False
        self._action_buffer = None
        self._action_cursor = 0
    def reset_robot_id(self, current_state, robot_ids):
        self.update_current_state(current_state)
        ids = self._normalise_robot_ids(robot_ids)
        goal = self._goal_state.clone()
        goal[ids] = self._current_state[ids]
        self.update_goal_state(goal)
        self._warm_start_available = False
        self._action_buffer = None
        self._action_cursor = 0
    def reset_shape(self):
        self._solve_state = None
        self._goal_tool_poses = None
        self._goal_state = None
        self._current_state = None
        self._seed_trajectory = None
        self._action_buffer = None
        self._action_cursor = 0
        self._solve_count = 0
        self._setup_complete = False
        self._warm_start_available = False
        self._goal_manager = GoalManager(self.config.device_cfg)
    def reset_seed(self):
        self._warm_start_available = False
        self._action_buffer = None
        self._action_cursor = 0
        return self._trajopt.reset_seed()
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")
    def destroy(self):
        self._ik.destroy()
        self._trajopt.destroy()
        self.reset_shape()
    def enable_tool_pose_tracking(self, tool_frames=None):
        self._validate_tracking_frames(tool_frames)
        self._tool_pose_tracking = True
    def disable_tool_pose_tracking(self, tool_frames=None):
        self._validate_tracking_frames(tool_frames)
        self._tool_pose_tracking = False
    def enable_joint_position_tracking(self): self._joint_position_tracking = True
    def disable_joint_position_tracking(self): self._joint_position_tracking = False
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self._tool_pose_criteria = dict(tool_pose_criteria)
        self._ik.update_tool_pose_criteria(tool_pose_criteria)
        self._trajopt.update_tool_pose_criteria(tool_pose_criteria)
    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)

    def update_world(self, scene_cfg) -> None:
        """Replace or mutate the portable :class:`SceneCollision` world.

        Supported values are ``SceneCfg``, a list of ``SceneCfg`` for the
        configured environments, or a prebuilt portable ``SceneCollision``.
        Asset paths and CUDA/Warp collision instances intentionally fail rather
        than being treated as an empty world.
        """
        scene = self._coerce_scene_collision(scene_cfg, replace_existing=True)
        self._scene_collision_checker = scene
        self._ik._scene_collision_checker = scene
        self._trajopt._scene_collision_checker = scene
        self._trajopt._pose_ik._scene_collision_checker = scene
        self.config.core_cfg.scene_collision_cfg = scene_cfg

    def debug_dump(self, file_path=None):
        del file_path
        return {
            "backend": "portable",
            "cuda_graph": False,
            "setup": self._setup_complete,
            "warm_start": self._warm_start_available,
            "command_cursor": self._action_cursor,
            "command_buffer_valid": self._action_buffer is not None and self._action_cursor < self.action_horizon,
            "solve_count": self._solve_count,
            "batch_size": self.problem_batch_size,
            "scene_collision": self._scene_collision_checker is not None,
        }

    def _coerce_scene_collision(self, value, replace_existing: bool = False):
        if value is None:
            return None
        if isinstance(value, SceneCollision):
            return value
        is_scene_list = isinstance(value, list) and all(isinstance(item, SceneCfg) for item in value)
        if not isinstance(value, SceneCfg) and not is_scene_list:
            raise NotImplementedError(
                "portable MPC worlds require SceneCfg, a list of SceneCfg, or SceneCollision; "
                "CUDA/Warp and file-backed world assets are unavailable"
            )
        expected_envs = self.config.max_batch_size if self.config.multi_env else 1
        if is_scene_list and len(value) != expected_envs:
            raise ValueError(
                f"multi_env MPC requires {expected_envs} SceneCfg environments, got {len(value)}"
            )
        if not is_scene_list and self.config.multi_env:
            raise ValueError("multi_env MPC requires a list of SceneCfg worlds")
        if replace_existing and self._scene_collision_checker is not None:
            current = self._scene_collision_checker
            if current.num_envs == (len(value) if is_scene_list else 1):
                if is_scene_list:
                    for index, scene in enumerate(value):
                        current.load_collision_model(scene, index)
                else:
                    current.load_collision_model(value)
                return current
        return SceneCollision(SceneCollisionCfg(
            device_cfg=self.config.device_cfg, scene_model=value,
            num_envs=len(value) if is_scene_list else 1,
        ))

    def _result_from_action_buffer(self) -> MPCSolverResult:
        """Return a command from the existing MPC plan without re-solving."""
        if self._action_buffer is None or self._last_result is None:
            raise RuntimeError("no portable MPC action buffer is available")
        index = min(self._action_cursor, self.action_horizon - 1)
        next_action = self._action_buffer[:, index].clone()
        result = self._last_result.clone()
        result.next_action = JointState.from_position(next_action, self.joint_names)
        result.action_sequence = JointState.from_position(self._action_buffer.clone(), self.joint_names)
        result.full_action_sequence = result.action_sequence.clone()
        result.action_buffer = self._action_buffer.clone()
        result.action_dt = self.config.optimization_dt
        result.metrics = dict(result.metrics or {})
        result.metrics.update({
            "command_index": index,
            "reoptimized": False,
            "solve_count": self._solve_count,
        })
        self._action_cursor += 1
        return result

    def prepare_safe_deceleration_trajectory(
        self,
        current_state: JointState,
        failed_mask: torch.Tensor,
        deceleration_time: Optional[float] = None,
        deceleration_profile: Optional[str] = None,
    ) -> torch.Tensor:
        """Construct a bounded, differentiable fallback plan for failed MPC rows.

        The CUDA implementation delegates this to its seed manager.  This
        version integrates the measured joint velocity over the fixed horizon
        with linear, cosine, or exponential velocity decay.  Rows outside
        ``failed_mask`` are holds, so callers can safely merge the result into
        a mixed feasible/infeasible batch.
        """
        current_state = self._normalise_state(current_state, "current_state")
        mask = torch.as_tensor(failed_mask, device=current_state.position.device, dtype=torch.bool)
        if mask.shape != (self.problem_batch_size,):
            raise ValueError(f"failed_mask must have shape [{self.problem_batch_size}]")
        profile = self.config.deceleration_profile if deceleration_profile is None else deceleration_profile
        if profile not in {"linear", "cosine", "exponential"}:
            raise ValueError("deceleration_profile must be linear, cosine, or exponential")
        duration = self.config.deceleration_time if deceleration_time is None else deceleration_time
        if duration is None:
            duration = self.config.optimization_dt * max(self.action_horizon - 1, 1)
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            raise ValueError("deceleration_time must be a positive finite number")
        duration = min(float(duration), self.config.max_deceleration_time)
        if not torch.isfinite(current_state.position).all():
            raise ValueError("current_state position must be finite")
        velocity = current_state.velocity
        if velocity is None:
            velocity = torch.zeros_like(current_state.position)
        elif velocity.shape != current_state.position.shape:
            raise ValueError("current_state velocity must match position shape")
        steps = torch.arange(self.action_horizon, device=velocity.device, dtype=velocity.dtype)
        phase = (steps * self.config.optimization_dt / duration).clamp(0.0, 1.0)
        if profile == "linear":
            scale = 1.0 - phase
        elif profile == "cosine":
            scale = 0.5 * (1.0 + torch.cos(torch.pi * phase))
        else:
            # Reach exactly zero at the requested duration rather than leaving
            # an exponential residual at the terminal command.
            scale = (torch.exp(-5.0 * phase) - torch.exp(torch.tensor(-5.0, device=phase.device, dtype=phase.dtype))) / (1.0 - torch.exp(torch.tensor(-5.0, device=phase.device, dtype=phase.dtype)))
        incremental_velocity = velocity[:, None, :] * scale[None, :, None]
        positions = current_state.position[:, None, :] + torch.cumsum(
            incremental_velocity * self.config.optimization_dt, dim=1
        )
        lower, upper = self._trajopt._lower, self._trajopt._upper
        positions = positions.clamp(lower[None, None], upper[None, None])
        hold = current_state.position[:, None, :].expand_as(positions)
        return torch.where(mask[:, None, None], positions, hold)

    def _normalise_state(self, state: JointState, name: str) -> JointState:
        if not isinstance(state, JointState):
            raise TypeError(f"{name} must be JointState")
        self._require_solver_tensor(state.position, name)
        if state.position.shape[-1] != self.action_dim:
            raise ValueError(f"{name} must have final dimension {self.action_dim}")
        if state.position.ndim == 1:
            state = state.unsqueeze(0)
        elif state.position.ndim != 2:
            raise NotImplementedError(
                f"portable MPC {name} must have shape [dof] or [batch,dof]; trajectory state ranks are unavailable"
            )
        return state

    def _normalise_dt(self, dt, state: JointState, batch_size: int) -> torch.Tensor:
        if dt is None:
            return state.position.new_full((batch_size,), self.config.optimization_dt)
        if not isinstance(dt, torch.Tensor):
            dt = state.position.new_tensor(dt)
        self._require_solver_tensor(dt, "dt")
        if dt.ndim == 0:
            return dt.expand(batch_size).clone()
        if dt.shape != (batch_size,):
            raise ValueError(f"dt must be scalar or shape [{batch_size}]")
        return dt

    def _normalise_goal_pose(self, goal: GoalToolPose) -> GoalToolPose:
        if not isinstance(goal, GoalToolPose):
            raise TypeError("goal_tool_poses must be GoalToolPose")
        self._require_solver_tensor(goal.position, "goal_tool_poses")
        if goal.horizon != 1:
            raise NotImplementedError("portable MPC supports a single-time pose goal")
        if goal.num_goalset != 1:
            raise NotImplementedError("portable MPC supports one pose goal per robot")
        if goal.num_links != 1:
            raise NotImplementedError("portable MPC supports one tracked tool frame")
        if goal.tool_frames != self._solve_state.tool_frames:
            raise ValueError(f"goal tool frame must be {self._solve_state.tool_frames}")
        if goal.batch_size != self.problem_batch_size:
            raise ValueError("goal pose batch size must match the configured MPC batch")
        if goal.quaternion.dtype != self.config.device_cfg.dtype:
            raise ValueError("goal pose quaternion must use the solver dtype")
        return goal

    @staticmethod
    def _as_goal_tool_pose(tool_pose, frames: List[str]) -> GoalToolPose:
        """Adapt either the public or backend FK ToolPose value type."""
        available = list(tool_pose.tool_frames)
        indices = [available.index(frame) for frame in frames]
        return GoalToolPose(
            list(frames), tool_pose.position[:, :, indices, :].unsqueeze(3),
            tool_pose.quaternion[:, :, indices, :].unsqueeze(3),
        )

    def _normalise_robot_ids(self, robot_ids) -> torch.Tensor:
        ids = torch.as_tensor(robot_ids, device=self.config.device_cfg.device, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("robot_ids must be a non-empty 1D index tensor")
        if bool(((ids < 0) | (ids >= self.problem_batch_size)).any().item()):
            raise ValueError("robot_ids contains an out-of-range MPC batch index")
        return ids

    def _require_solver_tensor(self, value: torch.Tensor, name: str) -> None:
        if value.dtype != self.config.device_cfg.dtype:
            raise ValueError(f"{name} must use solver dtype {self.config.device_cfg.dtype}")
        device = self.config.device_cfg.device
        if value.device != device and not (value.device.type == device.type == "mps"):
            raise ValueError(f"{name} must be on solver device {device}")

    def _validate_tracking_frames(self, frames) -> None:
        if frames is not None and not set(frames).issubset(self.tool_frames):
            raise ValueError(f"tool frames must be a subset of {self.tool_frames}")


__all__ = ["MPCSolver"]
