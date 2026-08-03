from __future__ import annotations

from typing import List, Optional

import torch

from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo._src.types.tool_pose import GoalToolPose

from .motion_retargeter_cfg import MotionRetargeterCfg
from .motion_retargeter_result import RetargetResult


class MotionRetargeter:
    """Stateful global-IK / local-IK-or-MPC motion retargeter.

    This is a CPU/MPS implementation of the V2 lifecycle rather than a
    CUDA-graph shim.  The first frame is always solved globally.  Later
    frames are warm-started locally, or advance a portable MPC problem when
    requested.  Raw CUDA graph, Warp and Isaac retargeting internals remain
    unavailable by design.
    """

    def __init__(self, config: MotionRetargeterCfg):
        if not isinstance(config, MotionRetargeterCfg):
            raise TypeError("config must be MotionRetargeterCfg")
        self._config = config
        self._num_envs = config.num_envs
        self._tool_pose_criteria = dict(config.tool_pose_criteria)
        if config.use_mpc and len(config.tool_frames) != 1:
            raise NotImplementedError(
                "portable MPC retargeting currently supports one tracked tool frame; "
                "use warm-started IK for multi-link retargeting"
            )
        self._global_ik_solver = self._build_global_ik_solver()
        self._joint_names = list(self._global_ik_solver.joint_names)
        self._action_dim = self._global_ik_solver.action_dim
        self._local_ik_solver = None if config.use_mpc else self._build_local_ik_solver()
        self._mpc_solver = self._build_mpc_solver() if config.use_mpc else None
        self._prev_solution: Optional[torch.Tensor] = None
        self._prev_velocity: Optional[torch.Tensor] = None
        self._mpc_state: Optional[JointState] = None
        self._last_result: Optional[RetargetResult] = None
        self._last_ik_result = None
        self._last_mpc_result = None
        self._last_failure_mask: Optional[torch.Tensor] = None
        self._world_generation = 0
        self._destroyed = False

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def tool_frames(self) -> List[str]:
        return self._global_ik_solver.tool_frames

    @property
    def kinematics(self):
        return self._global_ik_solver.kinematics

    @property
    def default_joint_state(self) -> JointState:
        return self._global_ik_solver.default_joint_state

    @property
    def num_dof(self) -> int:
        return self.action_dim

    @property
    def config(self) -> MotionRetargeterCfg:
        return self._config

    @property
    def scene_collision_checker(self):
        """Portable collision adapter shared by every active retargeting stage."""
        return self._global_ik_solver.scene_collision_checker

    @property
    def world_generation(self) -> int:
        """Monotonic generation incremented after each successful world swap."""
        return self._world_generation

    @property
    def is_destroyed(self) -> bool:
        """Whether reusable solver state has been released."""
        return self._destroyed

    @property
    def last_result(self) -> Optional[RetargetResult]:
        """Most recently materialized output, or ``None`` after reset/destroy."""
        return self._last_result

    @property
    def last_failure_mask(self) -> Optional[torch.Tensor]:
        """Per-environment failure mask from the last IK/MPC stage.

        The mask is detached diagnostic state.  Retargeting still returns a
        continuous joint stream: failed warm-started IK rows retain their last
        known-safe position while successful rows advance independently.
        """
        return None if self._last_failure_mask is None else self._last_failure_mask.clone()

    def _assert_live(self) -> None:
        if self._destroyed:
            raise RuntimeError("MotionRetargeter has been destroyed")

    def reset(self) -> None:
        self._assert_live()
        self._prev_solution = None
        self._prev_velocity = None
        self._mpc_state = None
        self._last_result = None
        self._last_ik_result = None
        self._last_mpc_result = None
        self._last_failure_mask = None

    def destroy(self) -> None:
        """Release solver caches and reject further mutable operations.

        This is deliberately idempotent.  CPU/MPS execution owns ordinary
        PyTorch state rather than CUDA graph executables, but applications
        still need a reliable lifecycle boundary when changing robot clips or
        worlds.
        """
        if self._destroyed:
            return
        self._destroyed = True
        self._global_ik_solver.destroy()
        if self._local_ik_solver is not None:
            self._local_ik_solver.destroy()
        if self._mpc_solver is not None:
            self._mpc_solver.destroy()
        self._prev_solution = None
        self._prev_velocity = None
        self._mpc_state = None
        self._last_result = None
        self._last_ik_result = None
        self._last_mpc_result = None
        self._last_failure_mask = None

    def update_world(self, scene_cfg) -> None:
        """Install one concrete portable collision world across all stages.

        The world is first validated by global IK, then that same adapter is
        supplied to local IK or MPC.  Sharing one object preserves in-place
        obstacle cache mutations and prevents global/local frames from seeing
        different worlds.  A world swap invalidates the previous warm start,
        which may have crossed an obstacle that did not exist when it was
        generated.

        ``SceneCfg``, ``SceneCollisionCfg``, and ``SceneCollision`` are
        supported.  YAML/USD/Warp assets remain an explicit boundary in the
        underlying portable solvers.
        """
        self._assert_live()
        if not self._config.load_collision_spheres:
            raise RuntimeError(
                "update_world requires load_collision_spheres=True at retargeter construction; "
                "a collision-free solver cannot acquire robot sphere geometry after setup"
            )
        self._global_ik_solver.update_world(scene_cfg)
        world = self._global_ik_solver.scene_collision_checker
        if self._local_ik_solver is not None:
            self._local_ik_solver.update_world(world)
        if self._mpc_solver is not None:
            self._mpc_solver.update_world(world)
        self._config = self._config.with_scene_model(world)
        self._world_generation += 1
        self.reset()

    def update_tool_pose_criteria(self, tool_pose_criteria) -> None:
        """Atomically update same-topology criteria and invalidate warm starts.

        Link dimensions determine solver cost layouts.  Replacing weights for
        the existing ordered tracked links is safe; adding/removing/reordering
        links requires constructing a new retargeter rather than silently
        changing an active problem's shape.
        """
        self._assert_live()
        replacement = self._config.with_tool_pose_criteria(tool_pose_criteria)
        if replacement.tool_frames != self.tool_frames:
            raise ValueError(
                "updated tool_pose_criteria must preserve the configured ordered tool frames"
            )
        self._config = replacement
        self._tool_pose_criteria = dict(replacement.tool_pose_criteria)
        self._global_ik_solver.update_tool_pose_criteria(self._tool_pose_criteria)
        if self._local_ik_solver is not None:
            self._local_ik_solver.update_tool_pose_criteria(self._tool_pose_criteria)
        if self._mpc_solver is not None:
            self._mpc_solver.update_tool_pose_criteria(self._tool_pose_criteria)
        self.reset()

    def solve_frame(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        self._assert_live()
        if not isinstance(goal_tool_poses, GoalToolPose):
            raise TypeError("goal_tool_poses must be GoalToolPose")
        if goal_tool_poses.batch_size != self._num_envs:
            raise ValueError(
                f"goal batch size ({goal_tool_poses.batch_size}) must match num_envs ({self._num_envs})"
            )
        if goal_tool_poses.horizon != 1:
            raise ValueError(
                "solve_frame accepts exactly one target frame; use solve_sequence "
                "for a [frame, environment, ...] target clip"
            )
        if self._prev_solution is None:
            return self._solve_global_ik(goal_tool_poses)
        if self._config.use_mpc:
            return self._solve_mpc_frame(goal_tool_poses)
        return self._solve_local_ik(goal_tool_poses)

    def solve_sequence(self, tool_poses: SequenceGoalToolPose) -> RetargetResult:
        self._assert_live()
        if not isinstance(tool_poses, SequenceGoalToolPose):
            raise TypeError("tool_poses must be SequenceGoalToolPose")
        if tool_poses.num_envs != self._num_envs:
            raise ValueError(
                f"sequence num_envs ({tool_poses.num_envs}) must match config ({self._num_envs})"
            )
        if tool_poses.num_frames < 1:
            raise ValueError("tool_poses must contain at least one frame")
        self.reset()
        rows, trajectories = [], []
        for index in range(tool_poses.num_frames):
            result = self.solve_frame(tool_poses.get_frame(index))
            rows.append(result.joint_state)
            if result.trajectory is not None:
                trajectories.append(result.trajectory)
        joint_state = self._stack_states(rows, torch.stack, dim=1)
        trajectory = (
            self._stack_states(trajectories, torch.cat, dim=1)
            if trajectories else None
        )
        result = RetargetResult(joint_state, trajectory)
        self._last_result = result
        return result

    def _solve_global_ik(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        result = self._global_ik_solver.solve_pose(goal_tool_poses, return_seeds=1)
        self._last_ik_result = result
        solution = result.solution[:, 0]
        self._last_failure_mask = ~self._success_by_environment(result, solution.shape[0])
        self._prev_solution = solution.detach().clone()
        self._prev_velocity = None
        joint_state = JointState.from_position(solution, self.joint_names)
        if self._mpc_solver is not None:
            self._mpc_state = joint_state.clone()
            self._mpc_solver.setup(self._mpc_state)
        output = RetargetResult(joint_state)
        self._last_result = output
        return output

    def _solve_local_ik(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        assert self._local_ik_solver is not None
        current = JointState.from_position(self._prev_solution.clone(), self.joint_names)
        if self._prev_velocity is not None:
            current.velocity = self._prev_velocity.clone()
        result = self._local_ik_solver.solve_pose(
            goal_tool_poses, current_state=current,
            seed_config=self._prev_solution[:, None], return_seeds=1,
        )
        self._last_ik_result = result
        candidate = result.solution[:, 0]
        success = self._success_by_environment(result, candidate.shape[0])
        self._last_failure_mask = ~success
        # Retaining the prior row for a failed local solve keeps the
        # retargeting stream continuous without disguising the failure: the
        # public diagnostic mask identifies every held environment.
        solution = torch.where(success[:, None], candidate, self._prev_solution)
        if result.js_solution is not None and result.js_solution.velocity is not None:
            velocity = result.js_solution.velocity[:, 0]
            if self._prev_velocity is None:
                previous_velocity = torch.zeros_like(velocity)
            else:
                previous_velocity = self._prev_velocity
            self._prev_velocity = torch.where(success[:, None], velocity, previous_velocity).detach().clone()
        self._prev_solution = solution.detach().clone()
        output = RetargetResult(JointState.from_position(solution, self.joint_names))
        self._last_result = output
        return output

    def _solve_mpc_frame(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        assert self._mpc_solver is not None and self._mpc_state is not None
        # The portable MPC routes Cartesian targets through its real IK front
        # end and uses the result as a joint-space horizon endpoint.  It is not
        # a fake CUDA rollout controller, and ``use_best_effort_ik`` preserves
        # the V2 retargeter's continuous output behavior for difficult clips.
        if not self._mpc_solver.update_goal_tool_poses(
            goal_tool_poses, run_ik=True, use_best_effort_ik=True
        ):
            raise RuntimeError("portable MPC could not update the retargeting goal")
        endpoints = []
        for _ in range(self._config.steps_per_target):
            result = self._mpc_solver.optimize_action_sequence(self._mpc_state)
            self._last_mpc_result = result
            if result.action_sequence is None or result.action_sequence.position.shape[1] < 1:
                raise RuntimeError("portable MPC returned an empty action sequence")
            self._mpc_state = self._endpoint_state(result.action_sequence)
            endpoints.append(self._mpc_state)
        self._prev_solution = self._mpc_state.position.detach().clone()
        success = getattr(self._last_mpc_result, "success", None)
        self._last_failure_mask = self._success_mask_from_tensor(success, self._num_envs)
        output = RetargetResult(
            joint_state=self._mpc_state.clone(),
            trajectory=self._stack_states(endpoints, torch.stack, dim=1),
        )
        self._last_result = output
        return output

    @staticmethod
    def _success_mask_from_tensor(success, batch_size: int) -> Optional[torch.Tensor]:
        if not isinstance(success, torch.Tensor):
            return None
        if success.ndim < 1 or success.shape[0] != batch_size:
            return None
        # Result layouts may include seeds or horizons.  A row is successful
        # only when every reported subproblem is successful, which keeps a
        # mixed result conservative for the next warm-started frame.
        return ~success.reshape(batch_size, -1).all(dim=-1).detach()

    def _success_by_environment(self, result, batch_size: int) -> torch.Tensor:
        failure = self._success_mask_from_tensor(getattr(result, "success", None), batch_size)
        if failure is None:
            return torch.ones(batch_size, device=self._prev_solution.device if self._prev_solution is not None else result.solution.device, dtype=torch.bool)
        return ~failure

    def _endpoint_state(self, action_sequence: JointState) -> JointState:
        def endpoint(value):
            return None if value is None else value[:, -1, :].clone()
        return JointState(
            endpoint(action_sequence.position), endpoint(action_sequence.velocity),
            endpoint(action_sequence.acceleration), self.joint_names,
            endpoint(action_sequence.jerk),
        )

    def _stack_states(self, states, operation, *, dim: int) -> JointState:
        if not states:
            raise ValueError("cannot combine an empty state sequence")
        def combine(field):
            values = [getattr(state, field) for state in states]
            return None if any(value is None for value in values) else operation(values, dim=dim)
        return JointState(
            combine("position"), combine("velocity"), combine("acceleration"),
            self.joint_names, combine("jerk"),
        )

    def _build_global_ik_solver(self) -> IKSolver:
        cfg = self._config
        solver = IKSolver(IKSolverCfg.create(
            cfg.robot, optimizer_configs=cfg.ik_optimizer_configs,
            scene_model=cfg.scene_model, self_collision_check=cfg.self_collision_check,
            device_cfg=cfg.device_cfg, num_seeds=cfg.num_seeds_global,
            position_tolerance=cfg.position_tolerance,
            orientation_tolerance=cfg.orientation_tolerance,
            use_cuda_graph=False,
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
            override_optimizer_num_iters={"lbfgs": cfg.global_ik_num_iters},
            optimization_dt=None, load_collision_spheres=cfg.load_collision_spheres,
            max_batch_size=cfg.num_envs,
        ))
        solver.update_tool_pose_criteria(self._tool_pose_criteria)
        return solver

    def _build_local_ik_solver(self) -> IKSolver:
        cfg = self._config
        solver = IKSolver(IKSolverCfg.create(
            cfg.robot, optimizer_configs=cfg.ik_optimizer_configs,
            scene_model=cfg.scene_model, self_collision_check=cfg.self_collision_check,
            device_cfg=cfg.device_cfg, num_seeds=cfg.num_seeds_local,
            position_tolerance=cfg.position_tolerance,
            orientation_tolerance=cfg.orientation_tolerance,
            use_cuda_graph=False,
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
            override_optimizer_num_iters={"lbfgs": cfg.local_ik_num_iters},
            optimization_dt=cfg.optimization_dt, load_collision_spheres=cfg.load_collision_spheres,
            velocity_regularization_weight=cfg.velocity_regularization_weight,
            acceleration_regularization_weight=cfg.acceleration_regularization_weight,
            max_batch_size=cfg.num_envs,
        ))
        solver.update_tool_pose_criteria(self._tool_pose_criteria)
        solver.config.use_lm_seed = False
        solver.config.exit_early = False
        return solver

    def _build_mpc_solver(self) -> MPCSolver:
        cfg = self._config
        solver = MPCSolver(MPCSolverCfg.create(
            cfg.robot, optimizer_configs=cfg.mpc_optimizer_configs,
            scene_model=cfg.scene_model, self_collision_check=cfg.self_collision_check,
            device_cfg=cfg.device_cfg, optimization_dt=cfg.optimization_dt,
            interpolation_steps=max(1, cfg.steps_per_target),
            load_collision_spheres=cfg.load_collision_spheres,
            num_control_points=cfg.num_control_points,
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
            warm_start_optimization_num_iters=cfg.mpc_warm_start_num_iters,
            cold_start_optimization_num_iters=cfg.mpc_cold_start_num_iters,
            max_batch_size=cfg.num_envs,
        ), scene_collision_checker=self._global_ik_solver.scene_collision_checker)
        solver.update_tool_pose_criteria(self._tool_pose_criteria)
        # MPCSolver correctly retains the requested batch capacity, but its
        # portable TrajOpt child is deliberately constructed lazily with the
        # standalone default.  A retargeter has a fixed ``num_envs`` lifetime,
        # so carry that declared capacity into the child before the first
        # rollout.  This makes ``num_envs > 1`` MPC retargeting genuinely
        # batched rather than accepting the configuration and failing on the
        # second (MPC) frame.  No CUDA graph buffer is resized here.
        solver._trajopt.config.max_batch_size = cfg.num_envs
        return solver


__all__ = ["MotionRetargeter"]
