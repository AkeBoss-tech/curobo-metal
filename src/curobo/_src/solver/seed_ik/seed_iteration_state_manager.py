"""Portable trust-region state updates for seeded IK."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .seed_ik_state import SeedIKState


class SeedIterationStateManager:
    def __init__(
        self, action_min, action_max, rho_min, lambda_factor, lambda_min,
        lambda_max, convergence_position_tolerance,
        convergence_orientation_tolerance, convergence_joint_limit_weight,
    ):
        self.action_min, self.action_max = action_min, action_max
        self.rho_min, self.lambda_factor = rho_min, lambda_factor
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.position_tolerance = convergence_position_tolerance
        self.orientation_tolerance = convergence_orientation_tolerance
        self.joint_limit_weight = convergence_joint_limit_weight

    def update_iteration_state(
        self, current_state: SeedIKState, candidate_state: SeedIKState,
        predicted_reduction: torch.Tensor, batch_size: int,
    ):
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
        return (old_error_norm - new_error_norm) / (
            predicted_reduction.reshape(batch_size).clamp_min(1e-8)
        )
    def _determine_step_acceptance(self, trust_ratio, batch_size):
        return trust_ratio.reshape(batch_size) >= self.rho_min
    def _update_damping_parameter(self, current_damping, step_accepted, batch_size):
        factor = torch.where(
            step_accepted.reshape(batch_size, 1, 1),
            current_damping.new_tensor(1 / self.lambda_factor),
            current_damping.new_tensor(self.lambda_factor),
        )
        return (current_damping * factor).clamp(self.lambda_min, self.lambda_max)
    def _check_convergence(self, joint_position, position_errors, orientation_errors):
        inside = ((joint_position >= self.action_min) & (joint_position <= self.action_max)).all(-1)
        return inside & self._check_pose_convergence(position_errors, orientation_errors)
    def _check_pose_convergence(self, position_errors, orientation_errors):
        return (position_errors <= self.position_tolerance) & (
            orientation_errors <= self.orientation_tolerance
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
