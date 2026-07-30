"""Tensor device configuration without eager CUDA dependencies."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceCfg:
    """Pinned field layout with an explicitly portable CPU default."""

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            object.__setattr__(self, "device", torch.device(self.device))

    @staticmethod
    def from_basic(device: str, dev_id: int) -> "DeviceCfg":
        return DeviceCfg(torch.device(device, dev_id))

    def to_device(self, data_tensor: Any) -> torch.Tensor:
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor: torch.Tensor) -> torch.Tensor:
        return data_tensor.to(device=self.device, dtype=torch.int8)

    def cpu(self) -> "DeviceCfg":
        return DeviceCfg(device=torch.device("cpu"), dtype=self.dtype)

    def as_torch_dict(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}

    def is_same_torch_device(self, other: torch.device) -> bool:
        if self.device == other:
            return True
        left = self.device.index if self.device.index is not None else 0
        right = other.index if other.index is not None else 0
        return self.device.type == other.type and left == right


__all__ = ["DeviceCfg"]
