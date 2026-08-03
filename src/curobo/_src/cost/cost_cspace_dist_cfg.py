"""Validated lifecycle configuration for portable C-space distance costs."""

from __future__ import annotations

from numbers import Integral
from typing import Any, List, Type, Union

import torch

from curobo._src.transition.robot_state_transition import RobotStateTransition

from .cost_base_cfg import BaseCostCfg
from .cost_cspace_dist import CSpaceDistCost
from .portable import CSpaceDistCostCfg as _PortableCSpaceDistCostCfg


class CSpaceDistCostCfg(_PortableCSpaceDistCostCfg):
    """Device-safe terminal/running C-space distance configuration.

    Weight changes are copied into existing buffers when their shape is
    stable.  This mirrors the useful optimizer lifecycle property of the V2
    class without requiring the CUDA graph buffers used by the upstream
    implementation.
    """

    def __post_init__(self) -> None:
        if isinstance(self.dof, bool) or not isinstance(self.dof, Integral) or self.dof < 0:
            raise TypeError("dof must be a non-negative integer")
        requested_dof = int(self.dof)
        super().__post_init__()
        self.dof = int(self.dof)
        if requested_dof and self.dof != requested_dof:
            raise ValueError("provided dof weights do not match configured dof")
        if self.only_terminal_cost and self.non_terminal_dof_weight is not None:
            self.non_terminal_dof_weight.zero_()
        self._validate_scalar_weight()
        self._validate_weight_pair()
        self.class_type = CSpaceDistCost

    def _validate_scalar_weight(self) -> None:
        if self.weight.ndim != 1 or self.weight.numel() != 1:
            raise ValueError("CSpaceDistCostCfg weight must be scalar")
        if not bool(torch.isfinite(self.weight).all().item()) or bool((self.weight < 0).any().item()):
            raise ValueError("weight must be finite and non-negative")

    def _validate_dof_weight(self, name: str, value: torch.Tensor | None) -> None:
        if value is None:
            return
        if value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 dof-weight tensor")
        if self.dof and value.numel() != self.dof:
            raise ValueError(f"{name} must contain {self.dof} values")
        if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
            raise ValueError(f"{name} must be finite and non-negative")

    def _validate_weight_pair(self) -> None:
        self._validate_dof_weight("terminal_dof_weight", self.terminal_dof_weight)
        self._validate_dof_weight("non_terminal_dof_weight", self.non_terminal_dof_weight)
        if self.terminal_dof_weight is not None and self.non_terminal_dof_weight is not None:
            if self.terminal_dof_weight.shape != self.non_terminal_dof_weight.shape:
                raise ValueError("terminal_dof_weight and non_terminal_dof_weight must have matching shapes")

    def update_dof(self, dof: int):
        if isinstance(dof, bool) or not isinstance(dof, Integral) or dof < 0:
            raise TypeError("dof must be a non-negative integer")
        result = super().update_dof(int(dof))
        self._validate_weight_pair()
        return result

    def _normalize_update_weight(self, name: str, value: Union[torch.Tensor, List[float]]) -> torch.Tensor:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a rank-1 floating tensor or sequence")
        converted = self.device_cfg.to_device(value)
        if converted.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 dof-weight tensor")
        if self.dof and converted.numel() != self.dof:
            raise ValueError(f"{name} must contain {self.dof} values")
        if not bool(torch.isfinite(converted).all().item()) or bool((converted < 0).any().item()):
            raise ValueError(f"{name} must be finite and non-negative")
        return converted

    def update_terminal_dof_weight(self, dof_weight: Union[torch.Tensor, List[float]]):
        value = self._normalize_update_weight("terminal_dof_weight", dof_weight)
        if self.dof == 0:
            self.dof = int(value.numel())
        if self.terminal_dof_weight is not None and self.terminal_dof_weight.shape == value.shape:
            self.terminal_dof_weight.copy_(value)
        else:
            self.terminal_dof_weight = value.clone()
        self._validate_weight_pair()
        return self

    def update_non_terminal_dof_weight(
        self, non_terminal_dof_weight: Union[torch.Tensor, List[float]]
    ):
        value = self._normalize_update_weight("non_terminal_dof_weight", non_terminal_dof_weight)
        if self.dof == 0:
            self.dof = int(value.numel())
        if self.only_terminal_cost:
            value = torch.zeros_like(value)
        if self.non_terminal_dof_weight is not None and self.non_terminal_dof_weight.shape == value.shape:
            self.non_terminal_dof_weight.copy_(value)
        else:
            self.non_terminal_dof_weight = value.clone()
        self._validate_weight_pair()
        return self

    def initialize_from_transition_model(self, transition_model: Any):
        action_dim = getattr(transition_model, "action_dim", None)
        if isinstance(action_dim, bool) or not isinstance(action_dim, Integral) or action_dim < 0:
            raise TypeError("transition_model.action_dim must be a non-negative integer")
        attr = "null_space_weight" if self.use_null_space else "cspace_distance_weight"
        if not hasattr(transition_model, attr):
            raise TypeError(f"transition_model must provide {attr}")
        self.update_dof(int(action_dim))
        weight = getattr(transition_model, attr)
        self.update_terminal_dof_weight(weight)
        self.update_non_terminal_dof_weight(weight)
        return self


__all__ = [
    "BaseCostCfg",
    "CSpaceDistCost",
    "CSpaceDistCostCfg",
    "List",
    "RobotStateTransition",
    "Type",
    "Union",
]
