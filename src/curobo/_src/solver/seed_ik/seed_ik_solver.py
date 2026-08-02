"""Deterministic portable Levenberg--Marquardt seeded IK.

This is a real small-batch seed solver, rather than an alias to the later
particle/L-BFGS IK stage.  Its numerical operations are standard PyTorch
CPU/MPS tensors.  CUDA graph capture, Warp kernels and their packed ABI remain
explicitly outside this portable implementation.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose

from .seed_ik_error_calculator import SeedIKErrorCalculator
from .seed_ik_solver_cfg import SeedIKSolverCfg
from .seed_ik_state import SeedIKState
from .seed_iteration_state_manager import SeedIterationStateManager


class SeedIKSolver:
    """Generate deterministic approximate IK seeds with damped Gauss--Newton.

    The public result layout follows :class:`IKSolverResult`: solutions are
    ``[batch, return_seeds, dof]``.  For a goal set, every candidate is solved
    independently and top seeds are selected globally with stable ties by
    goalset index then seed index.
    """

    def __init__(self, config: SeedIKSolverCfg):
        if not isinstance(config, SeedIKSolverCfg):
            raise TypeError("config must be SeedIKSolverCfg")
        self.config = config
        self.device_cfg = config.device_cfg
        robot = config.robot_config.kinematics
        kin_cfg = KinematicsCfg(
            config.device_cfg, list(robot.tool_frames), KinematicsParams(robot)
        )
        self._robot_model = Kinematics(kin_cfg, compute_jacobian=True, compute_spheres=False)
        self._aux_robot_model = Kinematics(kin_cfg, compute_jacobian=False, compute_spheres=True)
        self.dof = self._robot_model.get_dof()
        self.joint_names = self._robot_model.joint_names
        self.tool_frames = self._robot_model.tool_frames
        self.num_links = len(self.tool_frames)
        self.default_joint_position = self._robot_model.default_joint_position
        limits = self._robot_model.get_joint_limits().position
        margin = (limits[1] - limits[0]) * config.joint_limit_margin
        self.action_min = limits[0] + margin
        self.action_max = limits[1] - margin
        self.action_step_max = config.max_step_size * (self.action_max - self.action_min).abs()
        self.error_calculator = SeedIKErrorCalculator(
            self._robot_model, config, self.action_min, self.action_max, config.device_cfg
        )
        self._iteration_state_manager = SeedIterationStateManager(
            self.action_min, self.action_max, config.rho_min, config.lambda_factor,
            config.lambda_min, config.lambda_max,
            config.convergence_position_tolerance,
            config.convergence_orientation_tolerance,
            config.convergence_joint_limit_weight,
        )
        self._batch_size = self._num_seeds = self._num_problems = -1
        self._idxs_goal: Optional[torch.Tensor] = None
        self._velocity_current_position = None
        self._velocity_current_velocity = None
        self._velocity_dt = None
        self._velocity_clamping_active = False
        self._generator = torch.Generator(device="cpu").manual_seed(config.sampler_seed)
        self._n_residuals: Optional[int] = None

    def _calculate_n_residuals(self, num_links: int, joint_limit_weight: float):
        result = 6 * num_links
        if joint_limit_weight > 0:
            result += self.dof
        if self.config.velocity_weight > 0:
            result += self.dof
        if self.config.acceleration_weight > 0:
            result += self.dof
        return result

    @classmethod
    def _setup_lm_step(cls, num_links: int, dof: int, n_residuals: int, tile_threads: int = 64):
        del cls, num_links, dof, n_residuals, tile_threads
        raise NotImplementedError(
            "the packed CUDA/warp LevenbergMarquardtStep ABI is unavailable; "
            "SeedIKSolver uses torch.linalg.solve directly"
        )

    @property
    def n_residuals(self):
        if self._n_residuals is None:
            self._n_residuals = self._calculate_n_residuals(
                self.num_links, self.config.joint_limit_weight
            )
        return self._n_residuals

    def _setup_batch_size(self, batch_size: int, num_seeds: int = 1):
        if batch_size <= 0 or num_seeds <= 0:
            raise ValueError("batch_size and num_seeds must be positive")
        self._batch_size, self._num_seeds = batch_size, num_seeds
        self._num_problems = batch_size * num_seeds
        self.error_calculator.setup_batch_tensors(batch_size, num_seeds)
        self._idxs_goal = torch.arange(batch_size, device=self.device_cfg.device).repeat_interleave(num_seeds)
        self._velocity_current_position = torch.zeros(
            self._num_problems, self.dof, **self.device_cfg.as_torch_dict()
        )
        self._velocity_current_velocity = torch.zeros_like(self._velocity_current_position)
        self._velocity_dt = torch.ones(self._num_problems, **self.device_cfg.as_torch_dict())

    def _setup_error_calculator(self):
        """Recreate the calculator after intentional mutable config replacement."""
        self.error_calculator = SeedIKErrorCalculator(
            self._robot_model, self.config, self.action_min, self.action_max, self.device_cfg
        )

    def _compute_pose_error_and_jacobian(
        self, joint_position: torch.Tensor, goal_tool_poses: GoalToolPose
    ) -> SeedIKState:
        if self._idxs_goal is None:
            raise RuntimeError("call _setup_batch_size before evaluating seed IK")
        result = self.error_calculator.compute_error_and_jacobian(
            joint_position, goal_tool_poses, self._idxs_goal,
            self._velocity_current_position, self._velocity_current_velocity,
            self._velocity_dt, self._velocity_clamping_active,
        )
        return SeedIKState(
            joint_position=result.joint_position, error_norm=result.error_norm,
            jTerror=result.jTerror, jacobian=result.jacobian,
            position_errors=result.position_errors,
            orientation_errors=result.orientation_errors,
        )

    def _levenberg_marquardt_step_impl(
        self, current_iteration_state: SeedIKState, goal_tool_poses: GoalToolPose
    ) -> SeedIKState:
        jacobian = current_iteration_state.jacobian
        hessian = torch.einsum("brd,bre->bde", jacobian, jacobian)
        eye = torch.eye(self.dof, device=jacobian.device, dtype=jacobian.dtype).expand_as(hessian)
        damping = current_iteration_state.lambda_damping.reshape(-1, 1, 1)
        system = hessian + damping * eye
        gradient = current_iteration_state.jTerror.unsqueeze(-1)
        try:
            step = torch.linalg.solve(system, gradient).squeeze(-1)
        except RuntimeError:
            # Singular kinematic trees can arise at an exact joint limit.  The
            # pseudo-inverse is deterministic and keeps the portable solver
            # useful without pretending to match the CUDA tile solver.
            step = torch.matmul(torch.linalg.pinv(system), gradient).squeeze(-1)
        if self.config.max_step_size > 0:
            step = torch.maximum(torch.minimum(step, self.action_step_max), -self.action_step_max)
        candidate_q = (current_iteration_state.joint_position - step).clamp(
            self.action_min, self.action_max
        ).detach()
        candidate = self._compute_pose_error_and_jacobian(candidate_q, goal_tool_poses)
        candidate.lambda_damping = current_iteration_state.lambda_damping
        predicted_reduction = (current_iteration_state.jTerror * step).sum(-1).clamp_min(1e-8)
        return self._iteration_state_manager.update_iteration_state(
            current_iteration_state, candidate, predicted_reduction, candidate_q.shape[0]
        )

    def _levenberg_marquardt_step_inner_iterations(
        self, iteration_state: SeedIKState, goal_tool_poses: GoalToolPose
    ) -> SeedIKState:
        for _ in range(self.config.inner_iterations):
            iteration_state = self._levenberg_marquardt_step_impl(iteration_state, goal_tool_poses)
        return iteration_state

    def _levenberg_marquardt_step_inner_iterations_impl(self, iteration_state, goal_tool_poses):
        return self._levenberg_marquardt_step_inner_iterations(iteration_state, goal_tool_poses)

    def _check_convergence(self, iteration_state: SeedIKState, batch_size: int):
        if iteration_state.joint_position.shape[0] != batch_size:
            raise ValueError("batch_size must match flattened iteration state")
        pose = (iteration_state.position_errors <= self.config.position_tolerance) & (
            iteration_state.orientation_errors <= self.config.orientation_tolerance
        )
        return pose & ((iteration_state.joint_position >= self.action_min) & (
            iteration_state.joint_position <= self.action_max
        )).all(-1)

    def _pad_goal_tool_poses(self, goal_tool_poses: GoalToolPose) -> GoalToolPose:
        """Portable execution has no static graph shape, so no padding is needed."""
        return goal_tool_poses

    def _compute_initial_iteration_state(self, joint_position, goal_tool_poses):
        return self._compute_pose_error_and_jacobian(joint_position.detach(), goal_tool_poses)

    def _calculate_exit_condition(self, success, success_num_seeds, batch_success_threshold, batch_size):
        success = success.reshape(batch_size, -1)
        return bool((success.sum(-1) >= success_num_seeds).sum().item() >= batch_success_threshold * batch_size)

    def _normalize_seed_config(self, batch_size: int, seed_config) -> Optional[torch.Tensor]:
        if seed_config is None:
            return None
        seed = seed_config.position if isinstance(seed_config, JointState) else seed_config
        if not isinstance(seed, torch.Tensor):
            seed = self.device_cfg.to_device(seed)
        if seed.device.type != self.device_cfg.device.type or seed.dtype != self.device_cfg.dtype:
            raise ValueError("seed_config must use the solver device and dtype")
        if seed.shape[-1] != self.dof:
            raise ValueError(f"seed_config final dimension must be {self.dof}")
        if seed.ndim == 1:
            seed = seed.reshape(1, 1, self.dof)
        elif seed.ndim == 2:
            seed = seed[:, None, :]
        elif seed.ndim != 3:
            raise ValueError("seed_config must have rank 1, 2, or 3")
        if seed.shape[0] != batch_size:
            raise ValueError("seed_config batch dimension must match the goal batch")
        return seed

    def _generate_seed_configs(self, batch_size: int, seed_config: Optional[torch.Tensor] = None):
        supplied = self._normalize_seed_config(batch_size, seed_config)
        unit = torch.rand(
            (batch_size, self.config.num_seeds, self.dof), generator=self._generator,
            dtype=self.device_cfg.dtype, device="cpu",
        ).to(self.device_cfg.device)
        seeds = self.action_min + unit * (self.action_max - self.action_min)
        seeds[:, -1] = self.default_joint_position
        if supplied is not None:
            count = min(supplied.shape[1], self.config.num_seeds)
            seeds[:, :count] = supplied[:, :count]
        return seeds

    def _select_top_solutions(
        self, solutions, successes, position_errors, orientation_errors,
        start_joint_position, batch_size, num_seeds, return_seeds,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if solutions.shape != (batch_size, num_seeds, self.dof):
            raise ValueError("solutions must have shape [batch, num_seeds, dof]")
        if return_seeds < 1 or return_seeds > num_seeds:
            raise ValueError("return_seeds must be in [1, num_seeds]")
        cost = position_errors + orientation_errors
        if self.config.start_cspace_dist_weight > 0 and start_joint_position is not None:
            cost = cost + self.config.start_cspace_dist_weight * torch.linalg.vector_norm(
                solutions - start_joint_position[:, None], dim=-1
            )
        cost = cost + (~successes).to(cost.dtype) * 1e10
        rank = torch.argsort(cost, dim=-1, stable=True)[:, :return_seeds]
        return (
            successes.gather(1, rank),
            solutions.gather(1, rank[..., None].expand(-1, -1, self.dof)),
            position_errors.gather(1, rank), orientation_errors.gather(1, rank),
        )

    def _prepare_velocity_buffers(self, current_state, batch_size, num_seeds):
        self._velocity_current_position.zero_()
        self._velocity_current_velocity.zero_()
        self._velocity_dt.fill_(1.0)
        self._velocity_clamping_active = False
        if current_state is None or current_state.dt is None:
            return
        position = current_state.position
        if position.ndim == 1:
            position = position[None]
        if position.shape != (batch_size, self.dof):
            raise ValueError("current_state position must be [goal batch, dof]")
        dt = current_state.dt.reshape(batch_size, -1)[:, 0]
        self._velocity_current_position.copy_(position.repeat_interleave(num_seeds, dim=0))
        self._velocity_dt.copy_(dt.repeat_interleave(num_seeds))
        if current_state.velocity is not None:
            velocity = current_state.velocity
            if velocity.ndim == 1:
                velocity = velocity[None]
            self._velocity_current_velocity.copy_(velocity.repeat_interleave(num_seeds, dim=0))
        self._velocity_clamping_active = True

    def _optimize(self, initial_config, goal_tool_poses, success_num_seeds: int = 1):
        if initial_config.ndim != 3 or initial_config.shape[-1] != self.dof:
            raise ValueError("initial_config must have shape [batch, seed, dof]")
        batch, seeds, _ = initial_config.shape
        self._setup_batch_size(batch, seeds)
        state = self._compute_initial_iteration_state(initial_config.reshape(-1, self.dof), goal_tool_poses)
        state.lambda_damping = torch.full(
            (batch * seeds, 1, 1), self.config.lambda_initial, **self.device_cfg.as_torch_dict()
        )
        iterations = 0
        groups = self.config.max_iterations // self.config.inner_iterations
        for _ in range(groups):
            state = self._levenberg_marquardt_step_inner_iterations(state, goal_tool_poses)
            iterations += self.config.inner_iterations
            success = self._check_convergence(state, batch * seeds).reshape(batch, seeds)
            if self._calculate_exit_condition(success, success_num_seeds, self.config.batch_success_threshold, batch):
                break
        q = state.joint_position.reshape(batch, seeds, self.dof)
        success = self._check_convergence(state, batch * seeds).reshape(batch, seeds)
        return q, success, state.position_errors.reshape(batch, seeds), state.orientation_errors.reshape(batch, seeds), iterations

    def _solve_impl(
        self, goal_tool_poses: GoalToolPose, current_state: Optional[JointState] = None,
        seed_config=None, return_seeds: int = 1, batch_size: int = 1,
    ) -> IKSolverResult:
        started = time.monotonic()
        if goal_tool_poses.batch_size != batch_size:
            raise ValueError(f"expects batch size {batch_size}, got {goal_tool_poses.batch_size}")
        if return_seeds < 1 or return_seeds > self.config.num_seeds:
            raise ValueError("return_seeds must be in [1, config.num_seeds]")
        goal = goal_tool_poses.reorder_links(self.tool_frames)
        if goal.horizon != 1:
            raise NotImplementedError("seed IK solves a single target timestep")
        # Solve every goal-set item as an independent batch, then rank globally.
        goal_count = goal.num_goalset
        expanded_goal = GoalToolPose(
            goal.tool_frames,
            goal.position.permute(0, 3, 1, 2, 4).reshape(batch_size * goal_count, 1, self.num_links, 1, 3),
            goal.quaternion.permute(0, 3, 1, 2, 4).reshape(batch_size * goal_count, 1, self.num_links, 1, 4),
        )
        normalized_seed = self._normalize_seed_config(batch_size, seed_config)
        if normalized_seed is None and current_state is not None:
            normalized_seed = self._normalize_seed_config(batch_size, current_state)
        seeds = self._generate_seed_configs(batch_size, normalized_seed)
        seeds = seeds[:, None].expand(-1, goal_count, -1, -1).reshape(batch_size * goal_count, self.config.num_seeds, self.dof).clone()
        if current_state is not None:
            current_state = current_state.clone()
            def repeat_field(value):
                if value is None:
                    return None
                if value.ndim == 1:
                    value = value[None]
                return value[:, None].expand(-1, goal_count, *value.shape[1:]).reshape(batch_size * goal_count, *value.shape[1:])
            current_state.position = repeat_field(current_state.position)
            current_state.velocity = repeat_field(current_state.velocity)
            current_state.acceleration = repeat_field(current_state.acceleration)
            current_state.dt = repeat_field(current_state.dt)
        self._setup_batch_size(batch_size * goal_count, self.config.num_seeds)
        self._prepare_velocity_buffers(current_state, batch_size * goal_count, self.config.num_seeds)
        solutions, success, position_error, orientation_error, iterations = self._optimize(seeds, expanded_goal)
        solutions = solutions.reshape(batch_size, goal_count * self.config.num_seeds, self.dof)
        success = success.reshape(batch_size, goal_count * self.config.num_seeds)
        position_error = position_error.reshape(batch_size, goal_count * self.config.num_seeds)
        orientation_error = orientation_error.reshape(batch_size, goal_count * self.config.num_seeds)
        source_goal = torch.arange(goal_count, device=solutions.device).repeat_interleave(self.config.num_seeds)
        cost = position_error + orientation_error + (~success).to(position_error.dtype) * 1e10
        if current_state is not None and self.config.start_cspace_dist_weight > 0:
            start = current_state.position.reshape(batch_size, goal_count, self.dof)[:, 0]
            cost = cost + self.config.start_cspace_dist_weight * torch.linalg.vector_norm(
                solutions - start[:, None], dim=-1
            )
        rank = torch.argsort(cost, dim=-1, stable=True)[:, :return_seeds]
        top_solution = solutions.gather(1, rank[..., None].expand(-1, -1, self.dof))
        top_success = success.gather(1, rank)
        top_position_error = position_error.gather(1, rank)
        top_orientation_error = orientation_error.gather(1, rank)
        selected_goal = source_goal[rank]
        elapsed = time.monotonic() - started
        return IKSolverResult(
            success=top_success, solution=top_solution,
            js_solution=JointState.from_position(top_solution, self.joint_names),
            position_error=top_position_error, rotation_error=top_orientation_error,
            goalset_index=selected_goal[..., None].expand(-1, -1, self.num_links),
            optimized_seeds=solutions, seed_rank=rank, seed_cost=cost.gather(1, rank),
            batch_size=batch_size, num_seeds=goal_count * self.config.num_seeds,
            solve_time=elapsed, total_time=elapsed,
            metrics={"iterations": iterations, "backend": "torch-lm", "cuda_graph": False},
        )

    def solve_single(self, goal_tool_poses, current_state=None, seed_config=None, return_seeds=1):
        if goal_tool_poses.batch_size != 1:
            raise ValueError("solve_single requires batch size 1")
        return self._solve_impl(goal_tool_poses, current_state, seed_config, return_seeds, 1)

    def solve_batch(self, goal_tool_poses, current_state=None, seed_config=None, return_seeds=1):
        return self._solve_impl(
            goal_tool_poses, current_state, seed_config, return_seeds, goal_tool_poses.batch_size
        )

    @property
    def joint_limits(self):
        return self._robot_model.get_joint_limits()

    def get_default_joint_position(self):
        return self.default_joint_position.clone()

    @property
    def kinematics(self):
        return self._aux_robot_model

    def compute_kinematics(self, joint_position):
        state = joint_position if isinstance(joint_position, JointState) else JointState.from_position(
            joint_position, self.joint_names
        )
        return self.kinematics.compute_kinematics(state)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, object]):
        self.error_calculator.update_tool_pose_criteria(tool_pose_criteria)

    def reset_seed(self):
        self._generator.manual_seed(self.config.sampler_seed)

    def destroy(self):
        self._idxs_goal = None
        self._velocity_current_position = self._velocity_current_velocity = self._velocity_dt = None


__all__ = ["SeedIKSolver"]
