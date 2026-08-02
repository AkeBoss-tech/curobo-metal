"""Portable, validated configuration for c-space bound costs.

This module deliberately owns the configuration contract while the cost
evaluation remains in :mod:`curobo._src.cost.portable`.  Keeping the checks at
the public configuration boundary avoids a class of late failures in a solver
iteration (and prevents an unsupported retiming request from being silently
ignored on CPU/MPS).
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch

from .portable import (
    CSpaceCostCfg as _PortableCSpaceCostCfg,
    CSpaceCostType,
    PositionCSpaceCost,
    StateCSpaceCost,
    UnsupportedCostFeature,
)


class CSpaceCostCfg(_PortableCSpaceCostCfg):
    """Validated public configuration for ``PositionCSpaceCost``/``StateCSpaceCost``.

    The common CPU/MPS position and state bound/target costs are implemented
    by composed PyTorch.  A missing ``cost_type`` retains the early portable
    shim's scalar-position convenience for existing applications, but new
    callers should always choose :class:`CSpaceCostType` explicitly as the
    pinned CUDA API requires.  CUDA/Warp weight-retiming kernels are not part
    of the portable evaluator and requests for them fail at construction.
    """

    def __post_init__(self) -> None:
        # ``bool`` is an ``int`` in Python and accepting it as a degree of
        # freedom previously produced a confusing one-DOF target buffer.
        if isinstance(self.dof, bool) or not isinstance(self.dof, Integral):
            raise TypeError("dof must be a non-negative integer")
        if self.dof < 0:
            raise ValueError("dof must be non-negative")
        if self.retime_weights or self.retime_regularization_weights:
            raise UnsupportedCostFeature(
                "CSpaceCostCfg retime_weights requires the CUDA/Warp rollout "
                "kernels and is not available in the portable CPU/MPS backend"
            )
        super().__post_init__()
        # The upstream public contract represents scalar target controls as
        # one-element vectors.  ``torch.as_tensor(float)`` produces a scalar,
        # so canonicalize it after the common portable dataclass conversion.
        for name in ("cspace_target_weight", "cspace_non_terminal_weight_factor"):
            value = getattr(self, name)
            if value.ndim == 0:
                setattr(self, name, value.reshape(1))
        if self.cspace_target_dof_weight.ndim == 0:
            self.cspace_target_dof_weight = self.cspace_target_dof_weight.reshape(1)
        self._validate_finite_nonnegative("weight", self.weight)
        self._validate_finite_nonnegative("activation_distance", self.activation_distance)
        self._validate_finite_nonnegative(
            "squared_l2_regularization_weight", self.squared_l2_regularization_weight
        )
        self._validate_finite_nonnegative("cspace_target_weight", self.cspace_target_weight)
        self._validate_finite_nonnegative(
            "cspace_non_terminal_weight_factor", self.cspace_non_terminal_weight_factor
        )
        self._validate_finite_nonnegative(
            "cspace_target_dof_weight", self.cspace_target_dof_weight
        )
        if self.joint_limits is not None:
            self._validate_joint_limits(self.joint_limits)

    @staticmethod
    def _validate_finite_nonnegative(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")
        if bool((value < 0).any().item()):
            raise ValueError(f"{name} must be non-negative")

    def _validate_joint_limits(self, bounds: Any) -> None:
        """Validate the limit fields consumed by the portable evaluator.

        The high-level evaluator consumes the standard ``JointLimits`` shape
        rather than CUDA's packed struct.  Accept compatible duck-typed
        records, while checking all state fields before a trajectory solve.
        """
        dof = self.dof
        required = ("position",)
        if self.cost_type is CSpaceCostType.STATE:
            required = ("position", "velocity", "acceleration", "jerk", "effort")
        elif getattr(bounds, "effort", None) is not None:
            # POSITION only evaluates this term when torque is supplied, but
            # validate an optional effort range now if a caller provides it.
            required = (*required, "effort")
        for name in required:
            value = getattr(bounds, name, None)
            if value is None:
                raise ValueError(f"joint_limits.{name} is required for {self.cost_type.name} cost")
            tensor = torch.as_tensor(value)
            if tensor.ndim != 2 or tensor.shape[0] != 2:
                raise ValueError(f"joint_limits.{name} must have shape [2, dof]")
            if dof and tensor.shape[1] != dof:
                raise ValueError(
                    f"joint_limits.{name} has {tensor.shape[1]} dof; expected {dof}"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"joint_limits.{name} must contain finite values")
            if bool((tensor[0] >= tensor[1]).any().item()):
                raise ValueError(f"joint_limits.{name} lower bounds must be below upper bounds")

        # CUDA's STATE cost has no meaningful derivative-bound term if one of
        # these ranges is zero.  Reject that configuration deterministically
        # instead of deferring the failure to a CUDA tensor check.
        if self.cost_type is CSpaceCostType.STATE:
            for name in ("velocity", "acceleration", "jerk"):
                value = torch.as_tensor(getattr(bounds, name))
                if bool(torch.max(value[1] - value[0]).eq(0).item()):
                    raise ValueError(f"joint {name} limits must have non-zero range")

    def set_bounds(self, bounds: Any, teleport_mode: bool = False):
        if bounds is None:
            raise TypeError("bounds must be a JointLimits-compatible record")
        # Validate before cloning so malformed custom records fail without
        # invoking arbitrary ``clone`` implementations.
        self._validate_joint_limits(bounds)
        result = super().set_bounds(bounds, teleport_mode=teleport_mode)
        # Teleport mode converts state cost to POSITION and intentionally no
        # longer requires velocity/acceleration/jerk/effort limits.
        self._validate_joint_limits(self.joint_limits)
        return result

    def initialize_from_transition_model(self, transition_model: Any):
        action_dim = getattr(transition_model, "action_dim", None)
        if isinstance(action_dim, bool) or not isinstance(action_dim, Integral) or action_dim < 0:
            raise TypeError("transition_model.action_dim must be a non-negative integer")
        get_bounds = getattr(transition_model, "get_state_bounds", None)
        if not callable(get_bounds):
            raise TypeError("transition_model must provide get_state_bounds()")
        self.update_dof(action_dim)
        return self.set_bounds(
            get_bounds(), teleport_mode=bool(getattr(transition_model, "teleport_mode", False))
        )

    def update_dof(self, dof: int):
        if isinstance(dof, bool) or not isinstance(dof, Integral) or dof < 0:
            raise TypeError("dof must be a non-negative integer")
        return super().update_dof(int(dof))


__all__ = ["CSpaceCostCfg", "CSpaceCostType", "PositionCSpaceCost", "StateCSpaceCost"]
