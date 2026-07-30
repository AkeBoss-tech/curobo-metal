"""Portable trust-region state updates for seeded IK."""

from __future__ import annotations

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
        result = current_state.clone()
        for name in (
            "joint_position", "error_norm", "jTerror", "jacobian",
            "position_errors", "orientation_errors",
        ):
            old, new = getattr(result, name), getattr(candidate_state, name)
            if old is not None and new is not None:
                mask = accepted.reshape(accepted.shape + (1,) * (new.ndim - accepted.ndim))
                setattr(result, name, torch.where(mask, new, old))
        result.lambda_damping = self._update_damping_parameter(
            current_state.lambda_damping, accepted, batch_size
        )
        result.success = self._check_convergence(
            result.joint_position, result.position_errors, result.orientation_errors
        )
        return result

    def _calculate_trust_region_ratio(self, old_error_norm, new_error_norm, predicted_reduction, batch_size):
        del batch_size
        return (old_error_norm - new_error_norm) / predicted_reduction.clamp_min(1e-12)
    def _determine_step_acceptance(self, trust_ratio, batch_size):
        del batch_size
        return trust_ratio > self.rho_min
    def _update_damping_parameter(self, current_damping, step_accepted, batch_size):
        del batch_size
        factor = torch.where(
            step_accepted,
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
        return ((joint_position >= self.action_min) & (joint_position <= self.action_max)).all(-1)


__all__ = ["SeedIterationStateManager"]
