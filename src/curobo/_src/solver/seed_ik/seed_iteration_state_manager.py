"""Portable trust-region state updates for seeded IK."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .seed_ik_state import SeedIKState


class SeedIterationStateManager:
    """Portable per-candidate Levenberg--Marquardt lifecycle manager.

    The state transition is ordinary differentiable PyTorch on CPU/MPS.  It
    intentionally has no CUDA graph capture or Warp LM kernel ownership.
    """

    EPSILON_DIVISION_SAFETY = 1e-8

    def __init__(
        self, action_min, action_max, rho_min, lambda_factor, lambda_min,
        lambda_max, convergence_position_tolerance,
        convergence_orientation_tolerance, convergence_joint_limit_weight,
    ):
        if not isinstance(action_min, torch.Tensor) or not isinstance(action_max, torch.Tensor):
            raise TypeError("action_min and action_max must be torch.Tensor values")
        if action_min.ndim != 1 or action_min.shape != action_max.shape:
            raise ValueError("action_min and action_max must be matching rank-1 tensors")
        if action_min.device != action_max.device or action_min.dtype != action_max.dtype:
            raise ValueError("action_min and action_max must share device and dtype")
        if not action_min.is_floating_point() or not bool(torch.isfinite(action_min).all().item()) or not bool(torch.isfinite(action_max).all().item()):
            raise ValueError("action limits must be finite real tensors")
        if bool((action_min >= action_max).any().item()):
            raise ValueError("each action_min value must be smaller than action_max")
        numeric = {
            "rho_min": rho_min, "lambda_factor": lambda_factor,
            "lambda_min": lambda_min, "lambda_max": lambda_max,
            "convergence_position_tolerance": convergence_position_tolerance,
            "convergence_orientation_tolerance": convergence_orientation_tolerance,
            "convergence_joint_limit_weight": convergence_joint_limit_weight,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be a finite real number")
        if lambda_factor <= 1.0:
            raise ValueError("lambda_factor must be greater than one")
        if lambda_min <= 0.0 or lambda_max < lambda_min:
            raise ValueError("lambda_min must be positive and lambda_max must be >= lambda_min")
        if convergence_position_tolerance < 0.0 or convergence_orientation_tolerance < 0.0:
            raise ValueError("convergence tolerances must be non-negative")
        self.action_min, self.action_max = action_min, action_max
        self.rho_min, self.lambda_factor = float(rho_min), float(lambda_factor)
        self.lambda_min, self.lambda_max = float(lambda_min), float(lambda_max)
        self.convergence_position_tolerance = float(convergence_position_tolerance)
        self.convergence_orientation_tolerance = float(convergence_orientation_tolerance)
        self.convergence_joint_limit_weight = float(convergence_joint_limit_weight)
        # Historical aliases remain public for clients that adopted the early
        # portable facade before the pinned V2 spellings were restored.
        self.position_tolerance = self.convergence_position_tolerance
        self.orientation_tolerance = self.convergence_orientation_tolerance
        self.joint_limit_weight = self.convergence_joint_limit_weight

    @staticmethod
    def _require_batch(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("batch_size must be a positive integer")
        return value

    def _validate_state(self, state: SeedIKState, name: str, batch_size: int) -> None:
        if not isinstance(state, SeedIKState):
            raise TypeError(f"{name} must be a SeedIKState")
        fields = ("joint_position", "error_norm", "jTerror", "jacobian", "lambda_damping", "position_errors", "orientation_errors")
        for field in fields:
            value = getattr(state, field)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name}.{field} must be a torch.Tensor")
            if value.shape[0] != batch_size:
                raise ValueError(f"{name}.{field} batch dimension must match batch_size")
            if value.device != self.action_min.device:
                raise ValueError(f"{name}.{field} must be on the action-limit device")
        if state.joint_position.ndim != 2 or state.joint_position.shape[1] != self.action_min.numel():
            raise ValueError(f"{name}.joint_position must have shape [batch, dof]")
        if state.jTerror.shape != state.joint_position.shape:
            raise ValueError(f"{name}.jTerror must match joint_position")
        if state.jacobian.ndim != 3 or state.jacobian.shape[0] != batch_size or state.jacobian.shape[-1] != self.action_min.numel():
            raise ValueError(f"{name}.jacobian must have shape [batch, residuals, dof]")
        if state.lambda_damping.ndim != 3 or state.lambda_damping.shape[1:] != (1, 1):
            raise ValueError(f"{name}.lambda_damping must have shape [batch, 1, 1]")
        for field in ("error_norm", "position_errors", "orientation_errors"):
            if getattr(state, field).numel() != batch_size:
                raise ValueError(f"{name}.{field} must contain one value per batch item")

    def update_iteration_state(
        self, current_state: SeedIKState, candidate_state: SeedIKState,
        predicted_reduction: torch.Tensor, batch_size: int,
    ):
        batch_size = self._require_batch(batch_size)
        self._validate_state(current_state, "current_state", batch_size)
        self._validate_state(candidate_state, "candidate_state", batch_size)
        if not isinstance(predicted_reduction, torch.Tensor) or predicted_reduction.numel() != batch_size:
            raise ValueError("predicted_reduction must contain one value per batch item")
        if predicted_reduction.device != self.action_min.device:
            raise ValueError("predicted_reduction must be on the action-limit device")
        ratio = self._calculate_trust_region_ratio(
            current_state.error_norm, candidate_state.error_norm,
            predicted_reduction, batch_size,
        )
        accepted = self._determine_step_acceptance(ratio, batch_size)
        selected = self._select_state_values(current_state, candidate_state, accepted)
        error_norm = torch.where(accepted, candidate_state.error_norm, current_state.error_norm)
        damping = self._update_damping_parameter(
            current_state.lambda_damping, accepted, batch_size
        )
        success = self._check_convergence(
            selected.joint_position, selected.position_errors, selected.orientation_errors
        )
        return SeedIKState(
            success=success, improvement=accepted, joint_position=selected.joint_position,
            error_norm=error_norm, jTerror=selected.jTerror, jacobian=selected.jacobian,
            lambda_damping=damping, position_errors=selected.position_errors,
            orientation_errors=selected.orientation_errors,
        )

    def _calculate_trust_region_ratio(self, old_error_norm, new_error_norm, predicted_reduction, batch_size):
        batch_size = self._require_batch(batch_size)
        predicted = predicted_reduction.reshape(batch_size)
        safe_predicted = torch.where(
            predicted.abs() < self.EPSILON_DIVISION_SAFETY,
            torch.full_like(predicted, self.EPSILON_DIVISION_SAFETY),
            predicted,
        )
        return (old_error_norm.reshape(batch_size) - new_error_norm.reshape(batch_size)) / safe_predicted
    def _determine_step_acceptance(self, trust_ratio, batch_size):
        return trust_ratio.reshape(self._require_batch(batch_size)) >= self.rho_min
    def _update_damping_parameter(self, current_damping, step_accepted, batch_size):
        batch_size = self._require_batch(batch_size)
        if not isinstance(current_damping, torch.Tensor) or current_damping.shape != (batch_size, 1, 1):
            raise ValueError("current_damping must have shape [batch, 1, 1]")
        if not isinstance(step_accepted, torch.Tensor) or step_accepted.numel() != batch_size:
            raise ValueError("step_accepted must contain one value per batch item")
        factor = torch.where(
            step_accepted.reshape(batch_size, 1, 1),
            current_damping.new_tensor(1 / self.lambda_factor),
            current_damping.new_tensor(self.lambda_factor),
        )
        return (current_damping * factor).clamp(self.lambda_min, self.lambda_max)
    def _check_convergence(self, joint_position, position_errors, orientation_errors):
        pose = self._check_pose_convergence(position_errors, orientation_errors)
        if self.convergence_joint_limit_weight <= 0.0:
            return pose
        return self._check_joint_limit_satisfaction(joint_position) & pose
    def _check_pose_convergence(self, position_errors, orientation_errors):
        return (position_errors < self.convergence_position_tolerance) & (
            orientation_errors < self.convergence_orientation_tolerance
        )
    def _check_joint_limit_satisfaction(self, joint_position):
        return ((joint_position > self.action_min) & (joint_position < self.action_max)).all(-1)

    @dataclass
    class SelectedStateValues:
        joint_position: torch.Tensor
        jTerror: torch.Tensor
        jacobian: torch.Tensor
        position_errors: torch.Tensor
        orientation_errors: torch.Tensor

    def _select_state_values(
        self, current_state: SeedIKState, candidate_state: SeedIKState, accepted: torch.Tensor
    ) -> "SeedIterationStateManager.SelectedStateValues":
        mask_1d = accepted.reshape(-1)
        mask_2d = mask_1d[:, None]
        mask_3d = mask_1d[:, None, None]
        return self.SelectedStateValues(
            torch.where(mask_2d, candidate_state.joint_position, current_state.joint_position),
            torch.where(mask_2d, candidate_state.jTerror, current_state.jTerror),
            torch.where(mask_3d, candidate_state.jacobian, current_state.jacobian),
            torch.where(mask_1d, candidate_state.position_errors, current_state.position_errors),
            torch.where(mask_1d, candidate_state.orientation_errors, current_state.orientation_errors),
        )


__all__ = ["SeedIterationStateManager"]
