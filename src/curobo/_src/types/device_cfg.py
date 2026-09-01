"""Tensor device configuration without eager CUDA dependencies."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _default_device() -> torch.device:
    """Use Apple's accelerator when present and retain CPU portability elsewhere."""
    return torch.device("mps", 0) if torch.backends.mps.is_available() else torch.device("cpu")


class _DeviceCfgPortableMixin:
    def to_int32_device(self, data_tensor: Any) -> torch.Tensor:
        return torch.as_tensor(data_tensor, device=self.device, dtype=torch.int32)

    def to_int64_device(self, data_tensor: Any) -> torch.Tensor:
        return torch.as_tensor(data_tensor, device=self.device, dtype=torch.int64)

    def to_bool_device(self, data_tensor: Any) -> torch.Tensor:
        return torch.as_tensor(data_tensor, device=self.device, dtype=torch.bool)

    def clone(self) -> "DeviceCfg":
        return type(self)(
            self.device,
            self.dtype,
            self.collision_geometry_dtype,
            self.collision_gradient_dtype,
            self.collision_distance_dtype,
        )


@dataclass(frozen=True)
class DeviceCfg(_DeviceCfgPortableMixin):
    """Pinned field layout with the adapted Apple accelerator default."""

    device: torch.device = _default_device()
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if isinstance(self.device, str):
            object.__setattr__(self, "device", torch.device(self.device))

    @staticmethod
    def from_basic(device: str, dev_id: int):
        return DeviceCfg(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return torch.as_tensor(data_tensor, device=self.device, dtype=torch.int8)

    def cpu(self):
        return type(self)(
            device=torch.device("cpu"),
            dtype=self.dtype,
            collision_geometry_dtype=self.collision_geometry_dtype,
            collision_gradient_dtype=self.collision_gradient_dtype,
            collision_distance_dtype=self.collision_distance_dtype,
        )

    def as_torch_dict(self):
        return {"device": self.device, "dtype": self.dtype}

    def is_same_torch_device(self, other: torch.device) -> bool:
        other = torch.device(other)
        if self.device == other:
            return True
        left = self.device.index if self.device.index is not None else 0
        right = other.index if other.index is not None else 0
        return self.device.type == other.type and left == right


__all__ = ["DeviceCfg"]
