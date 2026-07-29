"""Portable joint-space state with cuRoboV2-shaped tensor operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .device import DeviceCfg


@dataclass
class JointState:
    position: torch.Tensor | Sequence[float]
    velocity: torch.Tensor | Sequence[float] | None = None
    acceleration: torch.Tensor | Sequence[float] | None = None
    joint_names: list[str] | tuple[str, ...] | None = None
    jerk: torch.Tensor | Sequence[float] | None = None
    device_cfg: DeviceCfg | None = None
    dt: torch.Tensor | None = None
    aux_data: dict[str, Any] = field(default_factory=dict)
    knot: torch.Tensor | None = None
    knot_dt: torch.Tensor | None = None
    control_space: object | None = None

    def __post_init__(self) -> None:
        cfg = self.device_cfg or (
            DeviceCfg(self.position.device, self.position.dtype)
            if isinstance(self.position, torch.Tensor)
            else DeviceCfg()
        )
        self.device_cfg = cfg
        self.position = cfg.to_device(self.position)
        for name in ("velocity", "acceleration", "jerk"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                setattr(self, name, cfg.to_device(value))
        if self.joint_names is not None:
            self.joint_names = list(self.joint_names)
            if len(self.joint_names) != self.position.shape[-1]:
                raise ValueError("joint_names must match the final position dimension")

    @classmethod
    def from_position(
        cls, position: torch.Tensor, joint_names: Sequence[str] | None = None
    ) -> "JointState":
        zero = torch.zeros_like(position)
        return cls(position, zero, zero, list(joint_names) if joint_names else None, zero)

    @classmethod
    def from_numpy(
        cls,
        joint_names: Sequence[str],
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
        jerk: np.ndarray | None = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "JointState":
        position_t = device_cfg.to_device(position)
        zero = torch.zeros_like(position_t)
        return cls(
            position_t,
            zero if velocity is None else device_cfg.to_device(velocity),
            zero if acceleration is None else device_cfg.to_device(acceleration),
            list(joint_names),
            zero if jerk is None else device_cfg.to_device(jerk),
            device_cfg,
        )

    @classmethod
    def zeros(
        cls, size: Sequence[int], device_cfg: DeviceCfg, joint_names: Sequence[str] | None = None
    ) -> "JointState":
        value = torch.zeros(tuple(size), **device_cfg.as_torch_dict())
        return cls.from_position(value, joint_names)

    @property
    def device(self) -> torch.device:
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    @property
    def shape(self) -> torch.Size:
        return self.position.shape

    @property
    def ndim(self) -> int:
        return self.position.ndim

    def _map(self, function: Any) -> "JointState":
        def apply(value: Any) -> Any:
            return None if value is None else function(value)
        return type(self)(
            apply(self.position), apply(self.velocity), apply(self.acceleration),
            None if self.joint_names is None else self.joint_names.copy(),
            apply(self.jerk), dt=apply(self.dt), knot=apply(self.knot),
            knot_dt=apply(self.knot_dt), aux_data=dict(self.aux_data),
            control_space=self.control_space,
        )

    def to(
        self,
        device_cfg: DeviceCfg | torch.device | str | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "JointState":
        if isinstance(device_cfg, DeviceCfg):
            cfg = device_cfg
        else:
            target = device if device is not None else device_cfg
            cfg = DeviceCfg(self.device if target is None else target, dtype or self.dtype)
        return self._map(lambda value: value.to(**cfg.as_torch_dict()))

    def clone(self) -> "JointState":
        return self._map(torch.Tensor.clone)

    def detach(self) -> "JointState":
        return self._map(torch.Tensor.detach)

    def cpu(self) -> "JointState":
        return self.to(device="cpu")

    def unsqueeze(self, dim: int) -> "JointState":
        return self._shape_map(lambda value: value.unsqueeze(dim))

    def squeeze(self, dim: int = 0) -> "JointState":
        return self._shape_map(lambda value: value.squeeze(dim))

    def view(self, *shape: int) -> "JointState":
        return self._shape_map(lambda value: value.view(*shape))

    def repeat(self, repeats: Sequence[int]) -> "JointState":
        return self._shape_map(lambda value: value.repeat(*repeats))

    def __getitem__(self, index: Any) -> "JointState":
        result = self._shape_map(lambda value: value[index])
        if self.dt is not None:
            result.dt = self.dt[index]
        return result

    def __len__(self) -> int:
        return len(self.position)

    def get_state_tensor(self) -> torch.Tensor:
        values = (self.position, self.velocity, self.acceleration, self.jerk)
        return torch.cat(tuple(value for value in values if value is not None), dim=-1)

    def _shape_map(self, function: Any) -> "JointState":
        """Transform DOF tensors while retaining trajectory-only metadata."""
        def apply(value: Any) -> Any:
            return None if value is None else function(value)
        return type(self)(
            apply(self.position), apply(self.velocity), apply(self.acceleration),
            None if self.joint_names is None else self.joint_names.copy(),
            apply(self.jerk), dt=self.dt, knot=apply(self.knot),
            knot_dt=self.knot_dt, aux_data=dict(self.aux_data),
            control_space=self.control_space,
        )

    def reorder(self, joint_names: Sequence[str]) -> "JointState":
        if self.joint_names is None:
            raise ValueError("cannot reorder a JointState without joint_names")
        try:
            indices = [self.joint_names.index(name) for name in joint_names]
        except ValueError as error:
            raise ValueError("requested joint is absent from JointState") from error
        index = torch.tensor(indices, device=self.device)
        output = self._map(lambda value: value.index_select(-1, index))
        output.joint_names = list(joint_names)
        return output
