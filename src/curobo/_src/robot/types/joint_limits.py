"""Tensor-valued joint limits compatible with cuRoboV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class JointLimits:
    joint_names: List[str]
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    effort: Optional[torch.Tensor] = None
    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self) -> None:
        self.validate_shape(len(self.joint_names))
        for name in ("position", "velocity", "acceleration", "jerk", "effort"):
            value = getattr(self, name)
            if value is not None and bool((value[0] >= value[1]).any().item()):
                raise ValueError(f"lower {name} limits must be less than upper limits")

    @staticmethod
    def from_data_dict(data: Dict, device_cfg: DeviceCfg = DeviceCfg()) -> "JointLimits":
        tensor = device_cfg.to_device
        return JointLimits(
            list(data["joint_names"]), tensor(data["position"]), tensor(data["velocity"]),
            tensor(data["acceleration"]), tensor(data["jerk"]),
            None if data.get("effort") is None else tensor(data["effort"]), device_cfg,
        )

    def clone(self) -> "JointLimits":
        return JointLimits(
            self.joint_names.copy(), self.position.clone(), self.velocity.clone(),
            self.acceleration.clone(), self.jerk.clone(),
            None if self.effort is None else self.effort.clone(), self.device_cfg,
        )

    def copy_(self, new_jl: "JointLimits") -> "JointLimits":
        if self.position.shape != new_jl.position.shape:
            raise ValueError("joint limit shapes must match for copy_")
        self.joint_names = new_jl.joint_names.copy()
        for name in ("position", "velocity", "acceleration", "jerk"):
            getattr(self, name).copy_(getattr(new_jl, name))
        self.effort = (
            None if new_jl.effort is None else
            new_jl.effort.clone() if self.effort is None else self.effort.copy_(new_jl.effort)
        )
        return self

    def validate_shape(self, dof: int, check_effort: bool = True) -> None:
        for name in ("position", "velocity", "acceleration", "jerk"):
            if getattr(self, name).shape != (2, dof):
                raise ValueError(f"{name} shape does not match dof: expected {(2, dof)}")
        if check_effort and self.effort is not None and self.effort.shape != (2, dof):
            raise ValueError(f"effort shape does not match dof: expected {(2, dof)}")

    @property
    def position_lower_limits(self) -> torch.Tensor:
        return self.position[0]

    @property
    def position_upper_limits(self) -> torch.Tensor:
        return self.position[1]


__all__ = ["JointLimits"]
