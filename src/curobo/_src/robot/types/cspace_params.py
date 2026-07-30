"""Configuration-space parameters for portable planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch

from .joint_limits import JointLimits
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class CSpaceParams:
    joint_names: List[str]
    default_joint_position: object | None = None
    cspace_distance_weight: object | None = None
    null_space_weight: object | None = None
    null_space_maximum_distance: object | None = None
    device_cfg: DeviceCfg = DeviceCfg()
    max_acceleration: Union[float, List[float]] = 10.0
    max_jerk: Union[float, List[float]] = 500.0
    velocity_scale: Union[float, List[float]] = 1.0
    acceleration_scale: Union[float, List[float]] = 1.0
    jerk_scale: Union[float, List[float]] = 1.0
    position_limit_clip: Union[float, List[float]] = 0.0

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        for name in (
            "default_joint_position", "cspace_distance_weight", "null_space_weight",
            "null_space_maximum_distance",
        ):
            value = getattr(self, name)
            if value is not None:
                value = self.device_cfg.to_device(value).reshape(-1)
                if value.numel() != n:
                    raise ValueError(f"{name} must contain {n} values")
                setattr(self, name, value)
        for name in ("max_acceleration", "max_jerk", "velocity_scale", "acceleration_scale", "jerk_scale"):
            value = getattr(self, name)
            if isinstance(value, (float, int)):
                value = [float(value)] * n
            value = self.device_cfg.to_device(value).reshape(-1)
            if value.numel() == 1 and n > 1:
                value = value.expand(n).clone()
            if value.numel() != n:
                raise ValueError(f"{name} must contain {n} values")
            setattr(self, name, value)
        if isinstance(self.position_limit_clip, list):
            self.position_limit_clip = self.device_cfg.to_device(self.position_limit_clip)
        if self.null_space_maximum_distance is None and self.null_space_weight is not None:
            self.null_space_maximum_distance = self.device_cfg.to_device([0.1] * n)

    def inplace_reindex(self, joint_names: List[str]) -> None:
        index = torch.tensor(
            [self.joint_names.index(name) for name in joint_names],
            device=self.device_cfg.device,
        )
        for name in (
            "default_joint_position", "cspace_distance_weight", "null_space_weight",
            "null_space_maximum_distance", "max_acceleration", "max_jerk",
            "velocity_scale", "acceleration_scale", "jerk_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.index_select(0, index).clone())
        self.joint_names = list(joint_names)

    def copy_(self, new_config: "CSpaceParams") -> "CSpaceParams":
        replacement = new_config.clone()
        self.__dict__.update(replacement.__dict__)
        return self

    def clone(self) -> "CSpaceParams":
        kwargs = {}
        for name, value in self.__dict__.items():
            kwargs[name] = value.clone() if isinstance(value, torch.Tensor) else (
                value.copy() if isinstance(value, list) else value
            )
        return CSpaceParams(**kwargs)

    def scale_joint_limits(self, joint_limits: JointLimits) -> JointLimits:
        result = joint_limits.clone()
        result.velocity *= self.velocity_scale
        result.acceleration *= self.acceleration_scale
        result.jerk *= self.jerk_scale
        clip = self.position_limit_clip
        if isinstance(clip, torch.Tensor) or float(clip) != 0.0:
            clip_tensor = torch.as_tensor(clip, **self.device_cfg.as_torch_dict())
            result.position[0] += clip_tensor
            result.position[1] -= clip_tensor
        return result

    @staticmethod
    def load_from_joint_limits(
        joint_position_upper: torch.Tensor,
        joint_position_lower: torch.Tensor,
        joint_names: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "CSpaceParams":
        middle = ((joint_position_upper + joint_position_lower) / 2).flatten()
        ones = torch.ones_like(middle)
        return CSpaceParams(
            joint_names, middle, ones, ones, ones, device_cfg=device_cfg
        )


__all__ = ["CSpaceParams"]
