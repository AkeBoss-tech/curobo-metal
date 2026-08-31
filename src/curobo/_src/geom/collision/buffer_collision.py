"""Pinned collision query buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class CollisionBuffer:
    distance: torch.Tensor
    gradient: torch.Tensor
    shape: Optional[torch.Size] = None
    device_cfg: DeviceCfg = None

    def __post_init__(self) -> None:
        if self.shape is None:
            self.shape = self.distance.shape
        if self.device_cfg is None:
            self.device_cfg = DeviceCfg()

    @classmethod
    def from_shape(cls, shape: torch.Size, device_cfg: DeviceCfg) -> CollisionBuffer:
        output = torch.Size(shape[:3])
        return cls(
            torch.zeros(output, device=device_cfg.device, dtype=device_cfg.collision_distance_dtype),
            torch.zeros((*output, 4), device=device_cfg.device, dtype=device_cfg.collision_gradient_dtype),
            output, device_cfg,
        )

    def zero_(self) -> None:
        self.distance.zero_()
        self.gradient.zero_()

    def resize(self, shape: Union[torch.Size, List[int]], device_cfg: DeviceCfg) -> None:
        new_shape = torch.Size(shape[:3])
        if self.shape != new_shape:
            replacement = type(self).from_shape(torch.Size((*new_shape, 4)), device_cfg)
            self.distance, self.gradient, self.shape, self.device_cfg = (
                replacement.distance, replacement.gradient, replacement.shape, replacement.device_cfg
            )

    def clone(self) -> CollisionBuffer:
        return type(self)(self.distance.clone(), self.gradient.clone(), self.shape, self.device_cfg)

    def __mul__(self, scalar: float) -> CollisionBuffer:
        self.distance *= scalar
        self.gradient *= scalar
        return self

    def __add__(self, other: CollisionBuffer) -> CollisionBuffer:
        self.distance += other.distance
        self.gradient += other.gradient
        return self
