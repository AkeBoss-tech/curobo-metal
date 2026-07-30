"""Portable joint state for the pinned internal import path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from curobo_metal.types.state import JointState as _MetalJointState

from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class JointState(_MetalJointState):
    device_cfg: DeviceCfg = DeviceCfg()
    control_space: Optional[ControlSpace] = None

    def __post_init__(self) -> None:
        if isinstance(self.position, torch.Tensor):
            self.device_cfg = DeviceCfg(self.position.device, self.position.dtype)
        else:
            self.position = self.device_cfg.to_device(self.position)
        for field in ("velocity", "acceleration", "jerk"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, torch.Tensor):
                setattr(self, field, self.device_cfg.to_device(value))
        if self.joint_names is not None:
            self.joint_names = list(self.joint_names)
            if len(self.joint_names) != self.position.shape[-1]:
                raise ValueError("joint_names must match the final position dimension")

    @staticmethod
    def from_state_tensor(
        state_tensor: torch.Tensor, joint_names: Optional[list[str]] = None, dof: int = 7
    ) -> "JointState":
        return JointState(
            state_tensor[..., :dof].contiguous(),
            state_tensor[..., dof : 2 * dof].contiguous(),
            state_tensor[..., 2 * dof : 3 * dof].contiguous(),
            joint_names=joint_names,
            jerk=state_tensor[..., 3 * dof : 4 * dof].contiguous(),
        )

    @staticmethod
    def from_list(
        position: list[float],
        velocity: list[float],
        acceleration: list[float],
        device_cfg: DeviceCfg,
    ) -> "JointState":
        return JointState(position, velocity, acceleration, device_cfg=device_cfg)

    @classmethod
    def zeros(
        cls,
        size: tuple[int, ...],
        device_cfg: DeviceCfg,
        joint_names: Optional[list[str]] = None,
    ) -> "JointState":
        value = torch.zeros(size, **device_cfg.as_torch_dict())
        return cls(
            value,
            value.clone(),
            value.clone(),
            joint_names,
            value.clone(),
            device_cfg,
            dt=torch.ones(size[0], **device_cfg.as_torch_dict()),
        )

    def data_ptr(self) -> int:
        return self.position.data_ptr()

    def to(self, device_cfg: DeviceCfg) -> "JointState":
        return self._map(device_cfg.to_device)

    def detach(self) -> "JointState":
        for field in ("position", "velocity", "acceleration", "jerk", "dt"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.detach())
        return self

    def copy_reference(self, in_joint_state: "JointState") -> "JointState":
        for field in (
            "position", "velocity", "acceleration", "jerk", "dt",
            "joint_names", "knot", "knot_dt",
        ):
            setattr(self, field, getattr(in_joint_state, field))
        return self

    def copy_(self, in_joint_state: "JointState", allow_clone: bool = True) -> "JointState":
        same = all(
            getattr(in_joint_state, field) is None
            or (
                getattr(self, field) is not None
                and getattr(self, field).shape == getattr(in_joint_state, field).shape
            )
            for field in ("position", "velocity", "acceleration", "jerk", "dt")
        )
        if not same:
            if not allow_clone:
                raise ValueError(
                    f"current state has shape: {self.position.shape} while new shape is "
                    f"{in_joint_state.position.shape}"
                )
            return self.copy_reference(in_joint_state.clone())
        for field in ("position", "velocity", "acceleration", "jerk", "dt"):
            source, target = getattr(in_joint_state, field), getattr(self, field)
            if source is not None:
                target.copy_(source)
        if in_joint_state.joint_names is not None:
            self.joint_names = in_joint_state.joint_names
        return self

    def __setitem__(self, index: int | torch.Tensor, value: "JointState") -> None:
        for field in ("position", "velocity", "acceleration", "jerk"):
            target, source = getattr(self, field), getattr(value, field)
            if target is not None and source is not None:
                target[index] = source
        if self.dt is not None and value.dt is not None:
            self.dt[index] = value.dt


__all__ = ["JointState"]
