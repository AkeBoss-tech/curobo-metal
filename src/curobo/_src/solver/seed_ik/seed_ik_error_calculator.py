"""Portable residual and Jacobian evaluation for seeded IK.

The upstream implementation uses CUDA streams and a packed Warp/Autograd
cost kernel.  This version deliberately keeps the mathematical pieces visible
as ordinary PyTorch operations so it can run deterministically on CPU and MPS
with ``PYTORCH_ENABLE_MPS_FALLBACK=0``.  It is not a CUDA graph or raw-kernel
ABI replacement.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.transform import quaternion_rate_to_axis_angle_rate
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class ErrorJacobianResult:
    """Per-problem seed-IK residual summary and stacked geometric Jacobian."""

    position_errors: torch.Tensor
    orientation_errors: torch.Tensor
    jTerror: torch.Tensor
    jacobian: torch.Tensor
    error_norm: torch.Tensor
    joint_position: torch.Tensor


def _quaternion_residual(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a smooth small-angle world-frame residual for ``current-target``.

    Quaternions use cuRobo's ``wxyz`` convention.  Canonicalizing the target
    hemisphere makes ``q`` and ``-q`` identical targets, including at the
    deterministic dot==0 tie.  The scaled vector component agrees with axis
    angle in the small-angle region used by the LM step and remains finite near
    180 degrees.
    """

    dot = (current * target).sum(dim=-1, keepdim=True)
    target = torch.where(dot < 0, -target, target)
    cw, cx, cy, cz = current.unbind(dim=-1)
    tw, tx, ty, tz = target.unbind(dim=-1)
    # current * conjugate(target), wxyz Hamilton product.
    vector = torch.stack(
        (
            -cw * tx + tw * cx - cy * tz + cz * ty,
            -cw * ty + tw * cy - cz * tx + cx * tz,
            -cw * tz + tw * cz - cx * ty + cy * tx,
        ),
        dim=-1,
    )
    return 2.0 * vector


class _SeedIKErrorCalculatorPortable:
    """Evaluate pose, bound, velocity and acceleration LM residuals.

    ``idxs_goal`` maps flattened ``[batch, seed]`` rows to their goal batch.
    Goal-set selection stays deterministic: this low-level calculator uses the
    first goal candidate; :class:`SeedIKSolver` evaluates every goal candidate
    before ranking returned seeds.
    """

    def __init__(self, robot_model, config, action_min, action_max, device_cfg):
        self.robot_model = robot_model
        self.config = config
        self.action_min = action_min
        self.action_max = action_max
        self.device_cfg = device_cfg
        self.velocity_limits = robot_model.get_joint_limits().velocity
        self.num_links = len(robot_model.tool_frames)
        self.dof = robot_model.get_dof()
        self._batch_size = -1
        self._num_seeds = -1
        self._num_problems = -1
        self._cost_shape = None
        self._cost_shape_cache: Dict[Tuple[int, torch.dtype, torch.device], torch.Tensor] = {}
        # Keeping an actual portable ToolPoseCost makes this low-level class
        # honour the public criteria-update lifecycle.  The LM residual below
        # intentionally remains geometric (rather than invoking a custom
        # CUDA backward), but reads the same stacked criteria state.
        self.pose_cost = self._setup_cost_function()
        self._criteria: Dict[str, ToolPoseCriteria] = self.pose_cost.config.tool_pose_criteria
        self._streams = {}
        self._events = {}
        for stream_name in (
            "pose_residual",
            "joint_limit_residual",
            "velocity_residual",
            "acceleration_residual",
        ):
            self._streams[stream_name], self._events[stream_name] = create_cuda_stream_pair(
                self.device_cfg.device
            )

    def setup_batch_tensors(self, batch_size: int, num_seeds: int = 1):
        if batch_size <= 0 or num_seeds <= 0:
            raise ValueError("batch_size and num_seeds must be positive")
        if batch_size == self._batch_size and num_seeds == self._num_seeds:
            return
        self._batch_size, self._num_seeds = batch_size, num_seeds
        self._num_problems = batch_size * num_seeds
        key = (self._num_problems, self.device_cfg.dtype, self.device_cfg.device)
        self._cost_shape = self._cost_shape_cache.get(key)
        if self._cost_shape is None:
            self._cost_shape = torch.ones(
                (self._num_problems, 1, 2 * self.num_links),
                **self.device_cfg.as_torch_dict(),
            )
            self._cost_shape_cache[key] = self._cost_shape
        self.pose_cost.setup_batch_tensors(self._num_problems, 1)

    def _validate_problem_tensor(self, value: torch.Tensor, name: str, problems: int) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (problems, self.dof):
            raise ValueError(f"{name} must have shape [{problems}, {self.dof}]")
        if not self.device_cfg.is_same_torch_device(value.device):
            raise ValueError(f"{name} must use the configured device")
        if value.dtype != self.device_cfg.dtype:
            raise ValueError(f"{name} must use the configured dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """Return the goal-frame rotation for projected pose criteria."""
        quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
        qw, qx, qy, qz = quaternion.unbind(-1)
        return torch.stack(
            (
                1 - 2 * (qy.square() + qz.square()),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx.square() + qz.square()),
                2 * (qy * qz - qx * qw),
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx.square() + qy.square()),
            ),
            dim=-1,
        ).reshape(*quaternion.shape[:-1], 3, 3)

    def _pose_axes(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get terminal per-link axis weights and goal-frame projection flags."""
        criteria = self.pose_cost._stacked_tool_pose_criteria
        axes = criteria.terminal_pose_axes_weight_factor.to(value)
        project = criteria.project_distance_to_goal.to(device=value.device).bool().reshape(-1)
        return axes, project

    def _compute_pose_errors(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_problem_tensor(joint_position, "joint_position", joint_position.shape[0])
        if not isinstance(goal_poses, GoalToolPose):
            raise TypeError("goal_poses must be a GoalToolPose")
        if goal_poses.horizon != 1:
            raise NotImplementedError("portable seeded IK evaluates a single target timestep")
        if not self.device_cfg.is_same_torch_device(goal_poses.position.device) or not self.device_cfg.is_same_torch_device(goal_poses.quaternion.device):
            raise ValueError("goal_poses must use the configured device")
        if goal_poses.position.dtype != joint_position.dtype or goal_poses.quaternion.dtype != joint_position.dtype:
            raise ValueError("goal_poses must use the joint-position dtype")
        if not bool(torch.isfinite(goal_poses.position).all().item()) or not bool(torch.isfinite(goal_poses.quaternion).all().item()):
            raise ValueError("goal_poses must contain only finite values")
        state = self.robot_model.compute_kinematics(
            JointState.from_position(joint_position, self.robot_model.joint_names)
        )
        if state.tool_jacobians is None:
            raise RuntimeError("SeedIKErrorCalculator requires a kinematics model with Jacobians")
        current_p = state.tool_poses.position[:, 0]
        current_q = state.tool_poses.quaternion[:, 0]
        idxs_goal = idxs_goal.reshape(-1).to(dtype=torch.long, device=joint_position.device)
        if idxs_goal.numel() != joint_position.shape[0]:
            raise ValueError("idxs_goal must contain one goal index per joint-position row")
        if bool(((idxs_goal < 0) | (idxs_goal >= goal_poses.batch_size)).any().item()):
            raise ValueError("idxs_goal contains a goal index outside the goal batch")
        goal = goal_poses.reorder_links(self.robot_model.tool_frames)
        target_p = goal.position.index_select(0, idxs_goal)[:, 0, :, 0]
        target_q = goal.quaternion.index_select(0, idxs_goal)[:, 0, :, 0]
        target_q_norm = torch.linalg.vector_norm(target_q, dim=-1)
        if bool((target_q_norm <= torch.finfo(target_q.dtype).eps).any().item()):
            raise ValueError("goal_poses quaternion must have nonzero norm")
        position_residual = current_p - target_p
        axes, project = self._pose_axes(joint_position)
        if bool(project.any().item()):
            projected = torch.matmul(
                position_residual.unsqueeze(-2), self._quaternion_to_matrix(target_q)
            ).squeeze(-2)
            position_residual = torch.where(project.reshape(1, -1, 1), projected, position_residual)
        orientation_residual = _quaternion_residual(
            torch.nn.functional.normalize(current_q, dim=-1),
            torch.nn.functional.normalize(target_q, dim=-1),
        )
        residual = torch.cat((position_residual, orientation_residual), dim=-1)
        jacobian = state.tool_jacobians[:, 0].reshape(joint_position.shape[0], -1, self.dof)
        component_weight = axes.clamp_min(0).sqrt().reshape(1, self.num_links, 6)
        component_weight = component_weight * residual.new_tensor(
            [self.config.position_weight**0.5] * 3 + [self.config.orientation_weight**0.5] * 3
        ).reshape(1, 1, 6)
        weighted_residual = residual * component_weight
        weighted_jacobian = jacobian * component_weight.reshape(1, -1, 1)
        flat_residual = weighted_residual.reshape(joint_position.shape[0], -1)
        jterror = torch.einsum("brd,br->bd", weighted_jacobian, flat_residual)
        return (
            jterror,
            weighted_jacobian,
            torch.linalg.vector_norm(weighted_residual[..., :3], dim=-1).amax(dim=-1),
            torch.linalg.vector_norm(weighted_residual[..., 3:], dim=-1).amax(dim=-1),
            flat_residual.square().sum(dim=-1),
        )

    def _compute_joint_limit_errors(
        self,
        joint_position: torch.Tensor,
        batch_size: int,
        current_position: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lower, upper = self.action_min, self.action_max
        if velocity_clamping_active and current_position is not None and dt is not None:
            if current_position.shape != joint_position.shape or dt.reshape(-1).shape[0] != batch_size:
                raise ValueError("velocity clamp buffers must match the flattened problem batch")
            delta = dt.reshape(-1, 1)
            lower = torch.maximum(lower, current_position + self.velocity_limits[0] * delta)
            upper = torch.minimum(upper, current_position + self.velocity_limits[1] * delta)
        lower_violation = (lower - joint_position).clamp_min(0)
        upper_violation = (joint_position - upper).clamp_min(0)
        residual = self.config.joint_limit_weight * (lower_violation + upper_violation)
        derivative = self.config.joint_limit_weight * (
            (upper_violation > 0).to(joint_position.dtype)
            - (lower_violation > 0).to(joint_position.dtype)
        )
        jacobian = torch.diag_embed(derivative)
        # This is intentionally the sum of per-joint residuals, matching the
        # upstream LM acceptance scalar; pose/velocity/acceleration terms are
        # already squared costs by construction.
        return derivative * residual, jacobian, residual.sum(dim=-1)

    def _compute_velocity_errors(
        self, joint_position, current_position, dt, batch_size
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current_position is None or dt is None:
            zero = torch.zeros_like(joint_position)
            return zero, torch.diag_embed(zero), zero.sum(dim=-1)
        if current_position.shape != joint_position.shape or dt.reshape(-1).numel() != batch_size:
            raise ValueError("velocity buffers must match the flattened problem batch")
        inv_dt = dt.reshape(-1, 1).clamp_min(1e-10).reciprocal()
        derivative = (self.config.velocity_weight * dt.reshape(-1, 1).clamp_min(0)).sqrt() * inv_dt
        residual = derivative * (joint_position - current_position)
        return derivative * residual, torch.diag_embed(derivative.expand_as(joint_position)), residual.square().sum(-1)

    def _compute_acceleration_errors(
        self, joint_position, current_position, current_velocity, dt, batch_size
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current_position is None or current_velocity is None or dt is None:
            zero = torch.zeros_like(joint_position)
            return zero, torch.diag_embed(zero), zero.sum(dim=-1)
        if current_position.shape != joint_position.shape or current_velocity.shape != joint_position.shape:
            raise ValueError("acceleration buffers must match the flattened problem batch")
        inv_dt = dt.reshape(-1, 1).clamp_min(1e-10).reciprocal()
        derivative = joint_position.new_tensor(self.config.acceleration_weight**0.5) * inv_dt
        residual = derivative * (joint_position - current_position) - (
            self.config.acceleration_weight**0.5 * current_velocity
        )
        return derivative * residual, torch.diag_embed(derivative.expand_as(joint_position)), residual.square().sum(-1)

    def _combine_errors(
        self,
        pose_jterror, pose_jacobian, pose_error_norm,
        joint_limit_jterror, joint_limit_jacobian, joint_limit_error,
        vel_jterror=None, vel_jacobian=None, vel_error_norm=None,
        accel_jterror=None, accel_jacobian=None, accel_error_norm=None,
    ):
        jterror = pose_jterror + joint_limit_jterror
        jacobians = [pose_jacobian, joint_limit_jacobian]
        error_norm = pose_error_norm + joint_limit_error
        if vel_jterror is not None:
            jterror = jterror + vel_jterror
            jacobians.append(vel_jacobian)
            error_norm = error_norm + vel_error_norm
        if accel_jterror is not None:
            jterror = jterror + accel_jterror
            jacobians.append(accel_jacobian)
            error_norm = error_norm + accel_error_norm
        return jterror, torch.cat(jacobians, dim=1), error_norm

    def compute_error_and_jacobian(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        current_velocity: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ) -> ErrorJacobianResult:
        if joint_position.ndim != 2 or joint_position.shape[-1] != self.dof:
            raise ValueError(f"joint_position must have shape [problems, {self.dof}]")
        problems = joint_position.shape[0]
        if self._num_problems >= 0 and problems != self._num_problems:
            raise ValueError(f"num_problems size mismatch: {problems} != {self._num_problems}")
        with self.stream_context("pose_residual"):
            pose = self._compute_pose_errors(joint_position, goal_poses, idxs_goal)
        limits = (None, None, None)
        if self.config.joint_limit_weight > 0:
            with self.stream_context("joint_limit_residual"):
                limits = self._compute_joint_limit_errors(
                    joint_position, problems, current_position, dt, velocity_clamping_active
                )
        velocity = acceleration = None
        if self.config.velocity_weight > 0:
            with self.stream_context("velocity_residual"):
                velocity = self._compute_velocity_errors(joint_position, current_position, dt, problems)
        if self.config.acceleration_weight > 0:
            with self.stream_context("acceleration_residual"):
                acceleration = self._compute_acceleration_errors(
                    joint_position, current_position, current_velocity, dt, problems
                )
        synchronize_cuda_streams(self._events, self.device_cfg.device)
        velocity_parts = velocity if velocity is not None else (None, None, None)
        acceleration_parts = acceleration if acceleration is not None else (None, None, None)
        if limits[0] is None:
            jterror, jacobian, error_norm = pose[0], pose[1], pose[4]
            if velocity is not None:
                jterror, jacobian, error_norm = self._combine_errors(
                    jterror, jacobian, error_norm,
                    velocity[0], velocity[1], velocity[2], *acceleration_parts,
                )
            elif acceleration is not None:
                jterror = jterror + acceleration[0]
                jacobian = torch.cat((jacobian, acceleration[1]), dim=1)
                error_norm = error_norm + acceleration[2]
        else:
            jterror, jacobian, error_norm = self._combine_errors(
                pose[0], pose[1], pose[4], *limits, *velocity_parts, *acceleration_parts
            )
        return ErrorJacobianResult(
            pose[2], pose[3], jterror, jacobian, error_norm, joint_position.detach()
        )

    def _reduce_pose_errors(self, position_errors, orientation_errors, cost, batch_size):
        return (
            position_errors.reshape(batch_size, -1).amax(-1),
            orientation_errors.reshape(batch_size, -1).amax(-1),
            cost.reshape(batch_size, -1).sum(-1),
        )

    def _compute_analytical_pose_jTerror(self, current_poses, jacobian, batch_size):
        if not isinstance(current_poses, ToolPose):
            raise TypeError("current_poses must be a ToolPose")
        if current_poses.position.grad is None or current_poses.quaternion.grad is None:
            raise RuntimeError("current pose gradients must be populated before analytical reduction")
        position_residual = current_poses.position.grad.reshape(batch_size, self.num_links, 3)
        quaternion_residual = current_poses.quaternion.grad.reshape(batch_size, self.num_links, 4)
        quaternion = current_poses.quaternion.detach().reshape(batch_size, self.num_links, 4)
        angular = quaternion_rate_to_axis_angle_rate(quaternion_residual, quaternion)
        residual = torch.cat((position_residual, angular), dim=-1).reshape(batch_size, -1)
        return torch.matmul(jacobian.transpose(-2, -1), residual.unsqueeze(-1)).squeeze(-1)

    def _setup_cost_function(self):
        criteria = {
            name: ToolPoseCriteria(
                terminal_pose_convergence_tolerance=[0.0, 0.0],
                terminal_pose_axes_weight_factor=[1.0] * 6,
                device_cfg=self.device_cfg,
            )
            for name in self.robot_model.tool_frames
        }
        return ToolPoseCost(ToolPoseCostCfg(
            weight=[self.config.position_weight, self.config.orientation_weight],
            tool_frames=list(self.robot_model.tool_frames),
            tool_pose_criteria=criteria,
            device_cfg=self.device_cfg,
            use_lie_group=False,
        ))

    def update_tool_pose_criteria(self, tool_pose_criteria):
        self.pose_cost.update_tool_pose_criteria(tool_pose_criteria)
        self._criteria = self.pose_cost.config.tool_pose_criteria

    def stream_context(self, stream_name: str):
        if stream_name not in {"pose_residual", "joint_limit_residual", "velocity_residual", "acceleration_residual", "default", "kinematics", "cost"}:
            raise ValueError(f"unknown portable stream: {stream_name}")
        if stream_name == "default":
            return nullcontext()
        return cuda_stream_context(stream_name, self._streams, self._events, self.device_cfg.device)


class SeedIKErrorCalculator:
    """Unified calculator for IK error types and jacobians (pose + joint limits)."""

    def __init__(self, robot_model, config, action_min, action_max, device_cfg):
        raise NotImplementedError

    def setup_batch_tensors(self, batch_size: int, num_seeds: int = 1):
        raise NotImplementedError

    def _setup_cost_function(self):
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/compute_all_errors")
    def compute_error_and_jacobian(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        current_velocity: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ) -> ErrorJacobianResult:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/compute_pose_errors")
    def _compute_pose_errors(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/reduce_pose_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _reduce_pose_errors(
        self,
        position_errors: torch.Tensor,
        orientation_errors: torch.Tensor,
        cost: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/compute_analytical_pose_jTerror")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _compute_analytical_pose_jTerror(
        self,
        current_poses: ToolPose,
        jacobian: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/add_joint_limit_errors")
    def _compute_joint_limit_errors(
        self,
        joint_position: torch.Tensor,
        batch_size: int,
        current_position: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        velocity_clamping_active: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/compute_velocity_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _compute_velocity_errors(
        self,
        joint_position: torch.Tensor,
        current_position: torch.Tensor,
        dt: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/compute_acceleration_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _compute_acceleration_errors(
        self,
        joint_position: torch.Tensor,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        dt: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @profiler.record_function("seed_ik_error_calculator/combine_errors")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _combine_errors(
        self,
        pose_jTerror,
        pose_jacobian,
        pose_error_norm,
        joint_limit_jTerror,
        joint_limit_jacobian,
        joint_limit_error,
        vel_jTerror=None,
        vel_jacobian=None,
        vel_error_norm=None,
        accel_jTerror=None,
        accel_jacobian=None,
        accel_error_norm=None,
    ):
        raise NotImplementedError

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        raise NotImplementedError

    def stream_context(self, stream_name: str):
        raise NotImplementedError


if not TYPE_CHECKING:
    # Runtime retains portable CPU/MPS residual evaluation and stream handling.
    SeedIKErrorCalculator = _SeedIKErrorCalculatorPortable


__all__ = ["ErrorJacobianResult", "SeedIKErrorCalculator"]
