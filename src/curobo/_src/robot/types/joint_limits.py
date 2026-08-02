"""Tensor-valued joint limits compatible with cuRoboV2.

The upstream record is intentionally a compact collection of ``[lower,
upper]`` tensors.  This portable implementation retains that public layout,
but makes device materialization and name-indexed lifecycle operations
explicit so a caller cannot accidentally mix CPU and Metal limit buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.tensor_util import copy_or_clone


@dataclass
class JointLimits:
    """Lower/upper position, velocity, acceleration, jerk, and effort limits.

    Every present tensor has shape ``[2, dof]`` and is materialized using
    ``device_cfg``.  Infinite bounds are accepted for unbounded joints, while
    NaNs and inverted ranges are rejected because they make downstream
    clipping and trajectory costs undefined.
    """

    joint_names: List[str]
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    effort: Optional[torch.Tensor] = None
    device_cfg: DeviceCfg = DeviceCfg()

    _LIMIT_FIELDS = ("position", "velocity", "acceleration", "jerk")
    _ALL_LIMIT_FIELDS = (*_LIMIT_FIELDS, "effort")

    def __post_init__(self) -> None:
        self.joint_names = list(self.joint_names)
        self._validate_joint_names()
        for name in self._ALL_LIMIT_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, self.device_cfg.to_device(value).contiguous())
        self.validate_shape(len(self.joint_names))
        self._validate_ranges()

    def _validate_joint_names(self) -> None:
        if not self.joint_names:
            raise ValueError("joint_names must not be empty")
        if any(not isinstance(name, str) or not name for name in self.joint_names):
            raise TypeError("joint_names must contain non-empty strings")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")

    def _validate_ranges(self) -> None:
        for name in self._ALL_LIMIT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if bool(torch.isnan(value).any().item()):
                raise ValueError(f"{name} limits must not contain NaN")
            if bool((value[0] >= value[1]).any().item()):
                label = "position" if name == "position" else name
                raise ValueError(f"lower {label} limits must be less than upper {label} limits")

    @staticmethod
    def from_data_dict(data: Dict, device_cfg: DeviceCfg = DeviceCfg()) -> "JointLimits":
        """Create limits from a serialized record on ``device_cfg``'s device."""
        if not isinstance(data, dict):
            raise TypeError("joint limit data must be a dictionary")
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        required = ("joint_names", *JointLimits._LIMIT_FIELDS)
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"joint limit data is missing required fields: {missing}")
        tensor = device_cfg.to_device
        return JointLimits(
            list(data["joint_names"]),
            tensor(data["position"]),
            tensor(data["velocity"]),
            tensor(data["acceleration"]),
            tensor(data["jerk"]),
            None if data.get("effort") is None else tensor(data["effort"]),
            device_cfg,
        )

    def clone(self) -> "JointLimits":
        """Return an independent record preserving device and dtype policy."""
        return JointLimits(
            self.joint_names.copy(),
            self.position.clone(),
            self.velocity.clone(),
            self.acceleration.clone(),
            self.jerk.clone(),
            None if self.effort is None else self.effort.clone(),
            self.device_cfg,
        )

    def to(self, device_cfg: DeviceCfg) -> "JointLimits":
        """Materialize an independent limits record under a new device policy."""
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        return JointLimits(
            self.joint_names.copy(),
            self.position, self.velocity, self.acceleration, self.jerk, self.effort, device_cfg,
        )

    def copy_(self, new_jl: "JointLimits") -> "JointLimits":
        """Copy compatible values while retaining caller-owned tensor buffers.

        This follows the upstream buffer convention: an absent source effort
        limit does not discard an already allocated optional effort buffer.
        """
        if not isinstance(new_jl, JointLimits):
            raise TypeError("new_jl must be a JointLimits")
        if self.device_cfg != new_jl.device_cfg:
            raise ValueError("copy_ requires matching device_cfg values")
        new_jl.validate_shape(len(new_jl.joint_names))
        self.joint_names = new_jl.joint_names.copy()
        for name in self._LIMIT_FIELDS:
            setattr(self, name, copy_or_clone(getattr(new_jl, name), getattr(self, name)))
        self.effort = copy_or_clone(new_jl.effort, self.effort)
        return self

    def validate_shape(self, dof: int, check_effort: bool = True) -> None:
        """Validate the fixed ``[2, dof]`` limit layout."""
        if not isinstance(dof, int) or dof < 0:
            raise ValueError("dof must be a non-negative integer")
        errors = []
        for name in self._LIMIT_FIELDS:
            value = getattr(self, name)
            if value.shape != (2, dof):
                errors.append(f"{name} shape does not match dof: {tuple(value.shape)} != {(2, dof)}")
        if check_effort and self.effort is not None and self.effort.shape != (2, dof):
            errors.append(
                f"effort shape does not match dof: {tuple(self.effort.shape)} != {(2, dof)}"
            )
        if errors:
            raise ValueError("Joint limits validation failed:\n  " + "\n  ".join(errors))

    def reindex(self, joint_names: Sequence[str]) -> "JointLimits":
        """Return a cloned, ordered subset selected by joint name."""
        requested = self._validate_requested_names(joint_names)
        index = torch.tensor(
            [self.joint_names.index(name) for name in requested],
            dtype=torch.long,
            device=self.device_cfg.device,
        )
        selected = {
            name: getattr(self, name).index_select(1, index).clone()
            for name in self._LIMIT_FIELDS
        }
        effort = None if self.effort is None else self.effort.index_select(1, index).clone()
        return JointLimits(requested, effort=effort, device_cfg=self.device_cfg, **selected)

    def inplace_reindex(self, joint_names: Sequence[str]) -> "JointLimits":
        """Select and reorder limits atomically while preserving compatible buffers."""
        selected = self.reindex(joint_names)
        return self.copy_(selected)

    @property
    def dof(self) -> int:
        """Number of named degrees of freedom represented by this record."""
        return len(self.joint_names)

    def as_dict(self, *, clone: bool = False) -> Dict[str, object]:
        """Return the public serialized layout without moving tensors to CPU.

        ``clone=True`` is appropriate for mutable configuration snapshots;
        the default intentionally retains tensor references for efficient
        read-only solver/config plumbing on CPU and MPS.
        """
        values: Dict[str, object] = {"joint_names": self.joint_names.copy()}
        for name in self._ALL_LIMIT_FIELDS:
            value = getattr(self, name)
            values[name] = None if value is None else (value.clone() if clone else value)
        return values

    def lower_limits(self, limit_type: str = "position") -> torch.Tensor:
        """Return the lower row for a named limit family."""
        return self._limit_tensor(limit_type)[0]

    def upper_limits(self, limit_type: str = "position") -> torch.Tensor:
        """Return the upper row for a named limit family."""
        return self._limit_tensor(limit_type)[1]

    def clamp_position(self, position: torch.Tensor) -> torch.Tensor:
        """Differentiably clamp positions along the final named-DOF axis."""
        if not isinstance(position, torch.Tensor):
            raise TypeError("position must be a torch.Tensor")
        if position.ndim == 0 or position.shape[-1] != self.dof:
            raise ValueError(f"position must end in the {self.dof} configured joint values")
        if not self.device_cfg.is_same_torch_device(position.device):
            raise ValueError("position must be on device_cfg.device")
        if not position.dtype.is_floating_point:
            raise TypeError("position must have a floating dtype")
        lower = self.position[0].to(dtype=position.dtype)
        upper = self.position[1].to(dtype=position.dtype)
        return torch.maximum(torch.minimum(position, upper), lower)

    def with_position_margin(self, margin: float | Sequence[float] | torch.Tensor) -> "JointLimits":
        """Return a safety-shrunk copy of position bounds.

        The margin may be scalar or per-DOF.  Other limit families remain
        unchanged, so this is safe to use for collision clearance policies
        without accidentally rescaling dynamics constraints.
        """
        value = self.device_cfg.to_device(margin)
        if value.ndim == 0:
            value = value.expand(self.dof)
        if value.ndim != 1 or value.numel() != self.dof:
            raise ValueError("position margin must be scalar or contain one value per joint")
        if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
            raise ValueError("position margin must be finite and non-negative")
        if bool((2 * value >= self.position[1] - self.position[0]).any().item()):
            raise ValueError("position margin collapses one or more joint ranges")
        result = self.clone()
        result.position = torch.stack((self.position[0] + value, self.position[1] - value))
        return result

    def detach(self) -> "JointLimits":
        """Return an independent record detached from any autograd graph."""
        return JointLimits(
            self.joint_names.copy(), self.position.detach().clone(), self.velocity.detach().clone(),
            self.acceleration.detach().clone(), self.jerk.detach().clone(),
            None if self.effort is None else self.effort.detach().clone(), self.device_cfg,
        )

    def merge(self, other: "JointLimits", *, overwrite: bool = False) -> "JointLimits":
        """Return the ordered union of two named limit records.

        Existing joints remain first.  Overlapping joints must have identical
        limits unless ``overwrite=True`` asks for ``other``'s values.  This is
        useful when attaching an independently configured gripper to an arm,
        while preventing accidental silent replacement of robot safety bounds.
        """
        if not isinstance(other, JointLimits):
            raise TypeError("other must be a JointLimits")
        if self.device_cfg != other.device_cfg:
            raise ValueError("merge requires matching device_cfg values")
        names = self.joint_names.copy()
        values = {name: self._column_for_name(name) for name in self.joint_names}
        for name in other.joint_names:
            incoming = other._column_for_name(name)
            if name in values and not overwrite and not self._columns_equal(values[name], incoming):
                raise ValueError(f"conflicting limits for joint {name!r}; pass overwrite=True to replace")
            if name not in values:
                names.append(name)
            values[name] = incoming
        return self._from_columns(names, values)

    def _validate_requested_names(self, joint_names: Sequence[str]) -> List[str]:
        requested = list(joint_names)
        if not requested:
            raise ValueError("joint_names for reindex must not be empty")
        if any(not isinstance(name, str) or not name for name in requested):
            raise TypeError("joint_names for reindex must contain non-empty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("joint_names for reindex must be unique")
        unknown = [name for name in requested if name not in self.joint_names]
        if unknown:
            raise ValueError(f"joint_names for reindex contain unknown joints: {unknown}")
        return requested

    def _limit_tensor(self, limit_type: str) -> torch.Tensor:
        if limit_type not in self._ALL_LIMIT_FIELDS:
            allowed = ", ".join(self._ALL_LIMIT_FIELDS)
            raise ValueError(f"unknown limit_type {limit_type!r}; expected one of {allowed}")
        value = getattr(self, limit_type)
        if value is None:
            raise ValueError(f"{limit_type} limits are not configured")
        return value

    def _column_for_name(self, name: str) -> Dict[str, Optional[torch.Tensor]]:
        index = self.joint_names.index(name)
        return {
            field: None if getattr(self, field) is None else getattr(self, field)[:, index : index + 1]
            for field in self._ALL_LIMIT_FIELDS
        }

    @staticmethod
    def _columns_equal(
        left: Dict[str, Optional[torch.Tensor]], right: Dict[str, Optional[torch.Tensor]]
    ) -> bool:
        return all(
            (a is None and b is None)
            or (a is not None and b is not None and bool(torch.equal(a, b)))
            for a, b in ((left[field], right[field]) for field in JointLimits._ALL_LIMIT_FIELDS)
        )

    def _from_columns(
        self, names: Iterable[str], columns: Dict[str, Dict[str, Optional[torch.Tensor]]]
    ) -> "JointLimits":
        names = list(names)
        result = {}
        for field in self._LIMIT_FIELDS:
            result[field] = torch.cat([columns[name][field] for name in names], dim=1)
        effort_columns = [columns[name]["effort"] for name in names]
        if any(column is None for column in effort_columns):
            if not all(column is None for column in effort_columns):
                raise ValueError("cannot merge limits when effort is defined for only some joints")
            effort = None
        else:
            effort = torch.cat(effort_columns, dim=1)
        return JointLimits(names, effort=effort, device_cfg=self.device_cfg, **result)

    @property
    def position_lower_limits(self) -> torch.Tensor:
        return self.position[0]

    @property
    def position_upper_limits(self) -> torch.Tensor:
        return self.position[1]

    @property
    def velocity_lower_limits(self) -> torch.Tensor:
        return self.velocity[0]

    @property
    def velocity_upper_limits(self) -> torch.Tensor:
        return self.velocity[1]

    @property
    def acceleration_lower_limits(self) -> torch.Tensor:
        return self.acceleration[0]

    @property
    def acceleration_upper_limits(self) -> torch.Tensor:
        return self.acceleration[1]

    @property
    def jerk_lower_limits(self) -> torch.Tensor:
        return self.jerk[0]

    @property
    def jerk_upper_limits(self) -> torch.Tensor:
        return self.jerk[1]


__all__ = ["JointLimits"]
