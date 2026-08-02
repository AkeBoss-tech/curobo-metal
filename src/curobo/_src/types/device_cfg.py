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
        # MPS does not offer the same dtype coverage as CUDA.  Rejecting this
        # at construction keeps a later tensor conversion from silently taking
        # a CPU path (or failing far from the caller's configuration error).
        if self.device.type == "mps" and self.dtype != torch.float32:
            raise TypeError("MPS compatibility types support only float32")

    @staticmethod
    def from_basic(device: str, dev_id: int) -> "DeviceCfg":
        index = None if device in {"cpu", "mps"} else dev_id
        return DeviceCfg(torch.device(device, index))

    def to_device(self, data_tensor: Any) -> torch.Tensor:
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor: Any) -> torch.Tensor:
        return torch.as_tensor(data_tensor, device=self.device, dtype=torch.int8)

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

    def cpu(self) -> "DeviceCfg":
        return type(self)(
            device=torch.device("cpu"),
            dtype=self.dtype,
            collision_geometry_dtype=self.collision_geometry_dtype,
            collision_gradient_dtype=self.collision_gradient_dtype,
            collision_distance_dtype=self.collision_distance_dtype,
        )

    def as_torch_dict(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}

    def is_same_torch_device(self, other: torch.device | str) -> bool:
        other = torch.device(other)
        if self.device == other:
            return True
        left = self.device.index if self.device.index is not None else 0
        right = other.index if other.index is not None else 0
        return self.device.type == other.type and left == right


__all__ = ["DeviceCfg"]
