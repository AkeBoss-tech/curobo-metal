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
from typing import Dict, Optional, Tuple

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


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


class SeedIKErrorCalculator:
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
        self._criteria: Dict[str, object] = {}

    def setup_batch_tensors(self, batch_size: int, num_seeds: int = 1):
        if batch_size <= 0 or num_seeds <= 0:
            raise ValueError("batch_size and num_seeds must be positive")
        self._batch_size, self._num_seeds = batch_size, num_seeds
        self._num_problems = batch_size * num_seeds

    def _compute_pose_errors(
        self,
        joint_position: torch.Tensor,
        goal_poses: GoalToolPose,
        idxs_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        position_residual = current_p - target_p
        orientation_residual = _quaternion_residual(current_q, target_q)
        residual = torch.cat((position_residual, orientation_residual), dim=-1)
        jacobian = state.tool_jacobians[:, 0].reshape(joint_position.shape[0], -1, self.dof)
        weighted_residual = residual.clone()
        weighted_residual[..., :3] *= self.config.position_weight**0.5
        weighted_residual[..., 3:] *= self.config.orientation_weight**0.5
        weighted_jacobian = jacobian.clone()
        weighted_jacobian[:, 0::6] *= self.config.position_weight**0.5
        weighted_jacobian[:, 1::6] *= self.config.position_weight**0.5
        weighted_jacobian[:, 2::6] *= self.config.position_weight**0.5
        weighted_jacobian[:, 3::6] *= self.config.orientation_weight**0.5
        weighted_jacobian[:, 4::6] *= self.config.orientation_weight**0.5
        weighted_jacobian[:, 5::6] *= self.config.orientation_weight**0.5
        flat_residual = weighted_residual.reshape(joint_position.shape[0], -1)
        jterror = torch.einsum("brd,br->bd", weighted_jacobian, flat_residual)
        return (
            jterror,
            weighted_jacobian,
            torch.linalg.vector_norm(position_residual, dim=-1).amax(dim=-1),
            2 * torch.acos((current_q * target_q).sum(dim=-1).abs().clamp(max=1)).amax(dim=-1),
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
        return derivative * residual, jacobian, residual.square().sum(dim=-1)

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
        pose = self._compute_pose_errors(joint_position, goal_poses, idxs_goal)
        limits = self._compute_joint_limit_errors(
            joint_position, problems, current_position, dt, velocity_clamping_active
        )
        velocity = acceleration = None
        if self.config.velocity_weight > 0:
            velocity = self._compute_velocity_errors(joint_position, current_position, dt, problems)
        if self.config.acceleration_weight > 0:
            acceleration = self._compute_acceleration_errors(
                joint_position, current_position, current_velocity, dt, problems
            )
        velocity_parts = velocity if velocity is not None else (None, None, None)
        acceleration_parts = acceleration if acceleration is not None else (None, None, None)
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
        del current_poses, jacobian, batch_size
        raise NotImplementedError(
            "portable SeedIK uses the geometric Jacobian directly; raw pose-gradient buffers are CUDA-only"
        )

    def _setup_cost_function(self):
        """CUDA ToolPoseCost is intentionally not constructed by this backend."""
        return None

    def update_tool_pose_criteria(self, tool_pose_criteria):
        self._criteria = dict(tool_pose_criteria)

    def stream_context(self, stream_name: str):
        if stream_name not in {"pose_residual", "joint_limit_residual", "velocity_residual", "acceleration_residual", "default", "kinematics", "cost"}:
            raise ValueError(f"unknown portable stream: {stream_name}")
        return nullcontext()


__all__ = ["ErrorJacobianResult", "SeedIKErrorCalculator"]
