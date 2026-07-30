"""Dynamics configuration backed by portable kinematics parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from curobo._src.robot.types import KinematicsParams
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class DynamicsCfg:
    kinematics_config: KinematicsParams
    device_cfg: DeviceCfg
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    def get_gravity_spatial(self) -> torch.Tensor:
        gravity = self.device_cfg.to_device(self.gravity)
        if gravity.shape != (3,):
            raise ValueError("gravity must contain three values")
        return torch.cat((gravity.new_zeros(3), gravity))


__all__ = ["DynamicsCfg"]
