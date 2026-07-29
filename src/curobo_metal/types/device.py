"""Tensor device configuration without CUDA or simulator assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceCfg:
    """Subset-compatible replacement for cuRoboV2 ``DeviceCfg``."""

    device: torch.device | str = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        if self.device.type == "mps" and self.dtype != torch.float32:
            raise TypeError("MPS compatibility types support only float32")

    @classmethod
    def from_basic(cls, device: str, dev_id: int = 0) -> "DeviceCfg":
        index = None if device in {"cpu", "mps"} else dev_id
        return cls(torch.device(device, index))

    def to_device(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(value), device=self.device, dtype=self.dtype)

    def to_int8_device(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=torch.int8)

    def to_int32_device(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=torch.int32)

    def to_int64_device(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=torch.int64)

    def to_bool_device(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=torch.bool)

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
            "cpu",
            self.dtype,
            self.collision_geometry_dtype,
            self.collision_gradient_dtype,
            self.collision_distance_dtype,
        )

    def as_torch_dict(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}

    def is_same_torch_device(self, other: torch.device | str) -> bool:
        candidate = torch.device(other)
        left = self.device.index if self.device.index is not None else 0
        right = candidate.index if candidate.index is not None else 0
        return self.device.type == candidate.type and left == right


TensorDeviceType = DeviceCfg
