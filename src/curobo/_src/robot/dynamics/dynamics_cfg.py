"""Dynamics configuration backed by portable kinematics parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from curobo._src.robot.types import KinematicsParams
from curobo._src.types.device_cfg import DeviceCfg


class _DynamicsCfgPortableMixin:
    def get_gravity(self) -> torch.Tensor:
        value = torch.as_tensor(self.gravity, dtype=torch.float64).reshape(-1)
        if value.numel() != 3 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("gravity must contain three finite values")
        return self.device_cfg.to_device(value)


@dataclass
class DynamicsCfg(_DynamicsCfgPortableMixin):
    kinematics_config: KinematicsParams
    device_cfg: DeviceCfg
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    def __post_init__(self) -> None:
        """Normalize the small mutable record before a dynamics model owns it.

        The CUDA implementation defers most malformed-input failures until a
        kernel launch.  That is especially unhelpful on a portable CPU/MPS
        backend, where a configuration can live for a long time before its
        first rollout.  Keep the public list representation (callers commonly
        mutate ``cfg.gravity``), but reject a device split or non-finite world
        gravity at construction.
        """
        if not isinstance(self.kinematics_config, KinematicsParams):
            raise TypeError("kinematics_config must be KinematicsParams")
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be DeviceCfg")
        if self.kinematics_config.device_cfg != self.device_cfg:
            raise ValueError("kinematics_config.device_cfg must match device_cfg")
        value = torch.as_tensor(self.gravity, dtype=torch.float64).reshape(-1)
        if value.numel() != 3:
            raise ValueError("gravity must contain three values")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("gravity must contain only finite values")
        self.gravity = [float(item) for item in value.tolist()]

    def get_gravity_spatial(self) -> torch.Tensor:
        """Return Featherstone's base spatial acceleration convention.

        The RNEA kernels store ``[angular, linear]`` acceleration and encode
        gravity as an upward base acceleration.  Keeping this conversion on
        the portable configuration makes callers which consume this public
        helper see the same convention even though the composed PyTorch RNEA
        backend accepts a conventional world gravity vector.
        """
        gravity = self.get_gravity()
        return torch.cat((gravity.new_zeros(3), -gravity))


__all__ = ["DynamicsCfg"]
