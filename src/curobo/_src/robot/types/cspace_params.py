"""Configuration-space parameters for portable robot planning.

The upstream value is deliberately a small tensor record.  This implementation
keeps that layout while making the per-DOF and device invariants explicit: a
configuration-space parameter is never a batched trajectory tensor and all
materialized values live on its :class:`DeviceCfg` device.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import List, Sequence, Union

import torch

from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tensor import T_DOF
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import clone_if_not_none, copy_or_clone


@dataclass
class CSpaceParams:
    """Per-DOF planning parameters for one robot configuration space.

    The tensors in this record are length ``len(joint_names)``.  In particular,
    batch and horizon dimensions belong to a :class:`JointState` or planner
    input, not to ``CSpaceParams``.  Scalars for limits and scales are expanded
    to one value per degree of freedom.
    """

    joint_names: List[str]
    default_joint_position: T_DOF | None = None
    cspace_distance_weight: T_DOF | None = None
    null_space_weight: T_DOF | None = None
    null_space_maximum_distance: T_DOF | None = None
    device_cfg: DeviceCfg = DeviceCfg()
    max_acceleration: Union[float, List[float]] = 10.0
    max_jerk: Union[float, List[float]] = 500.0
    velocity_scale: Union[float, List[float]] = 1.0
    acceleration_scale: Union[float, List[float]] = 1.0
    jerk_scale: Union[float, List[float]] = 1.0
    position_limit_clip: Union[float, List[float]] = 0.0

    _VECTOR_FIELDS = (
        "default_joint_position",
        "cspace_distance_weight",
        "null_space_weight",
        "null_space_maximum_distance",
    )
    _SCALE_FIELDS = ("velocity_scale", "acceleration_scale", "jerk_scale")
    _PER_DOF_FIELDS = (
        "default_joint_position",
        "cspace_distance_weight",
        "null_space_weight",
        "null_space_maximum_distance",
        "max_acceleration",
        "max_jerk",
        "velocity_scale",
        "acceleration_scale",
        "jerk_scale",
    )

    def __post_init__(self) -> None:
        self.joint_names = list(self.joint_names)
        self._validate_joint_names()
        dof = len(self.joint_names)

        for name in self._VECTOR_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, self._per_dof_tensor(name, value, dof))

        for name in ("max_acceleration", "max_jerk", *self._SCALE_FIELDS):
            setattr(self, name, self._per_dof_tensor(name, getattr(self, name), dof, scalar=True))

        # Keep the upstream scalar-or-vector representation for this field so
        # YAML round-trips retain scalar safety margins.  Vector margins are
        # still normalized onto the configured device and must be per-DOF.
        if isinstance(self.position_limit_clip, torch.Tensor):
            if self.position_limit_clip.ndim == 0:
                self.position_limit_clip = float(self.position_limit_clip.item())
            else:
                self.position_limit_clip = self._per_dof_tensor(
                    "position_limit_clip", self.position_limit_clip, dof
                )
        elif isinstance(self.position_limit_clip, (list, tuple)):
            self.position_limit_clip = self._per_dof_tensor(
                "position_limit_clip", self.position_limit_clip, dof
            )
        elif not isinstance(self.position_limit_clip, Real):
            raise TypeError("position_limit_clip must be a scalar or a per-DOF vector")

        if self.null_space_maximum_distance is None and self.null_space_weight is not None:
            self.null_space_maximum_distance = self.device_cfg.to_device([0.1] * dof)

        self._validate_values()

    def _validate_joint_names(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.joint_names):
            raise TypeError("joint_names must contain non-empty strings")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("duplicate joint names; joint_names must be unique")

    def _per_dof_tensor(
        self, name: str, value: object, dof: int, *, scalar: bool = False
    ) -> torch.Tensor:
        tensor = self.device_cfg.to_device(value)
        if tensor.ndim == 0:
            if not scalar:
                raise ValueError(f"{name} must be a rank-1 per-DOF tensor of length {dof}")
            tensor = tensor.expand(dof).clone()
        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must be rank-1 with length {dof}; batch/horizon dimensions are unsupported"
            )
        if tensor.numel() == 1 and scalar and dof > 1:
            tensor = tensor.expand(dof).clone()
        if tensor.numel() != dof:
            raise ValueError(f"{name} must contain {dof} values")
        return tensor.contiguous()

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor | float) -> None:
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain only finite values")

    def _validate_values(self) -> None:
        for name in self._PER_DOF_FIELDS:
            value = getattr(self, name)
            if value is not None:
                self._require_finite(name, value)
        self._require_finite("position_limit_clip", self.position_limit_clip)

        for name in ("max_acceleration", "max_jerk"):
            if bool((getattr(self, name) <= 0).any().item()):
                raise ValueError(f"{name} must be strictly positive")
        for name in self._SCALE_FIELDS:
            if bool((getattr(self, name) < 0).any().item()):
                raise ValueError(f"{name} must be non-negative")
        for name in ("cspace_distance_weight", "null_space_weight", "null_space_maximum_distance"):
            value = getattr(self, name)
            if value is not None and bool((value < 0).any().item()):
                raise ValueError(f"{name} must be non-negative")
        clip = self.position_limit_clip
        if isinstance(clip, torch.Tensor):
            invalid = bool((clip < 0).any().item())
        else:
            invalid = clip < 0
        if invalid:
            raise ValueError("position_limit_clip must be non-negative")

    def inplace_reindex(self, joint_names: List[str]):
        """Select and reorder per-DOF parameters by joint name.

        ``joint_names`` may be a strict subset (used by reduced kinematic
        trees), but duplicate or unknown names are rejected before mutating any
        tensor.  This gives the operation all-or-nothing semantics.
        """
        requested = list(joint_names)
        if not requested:
            raise ValueError("joint_names for reindex must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("joint_names for reindex must be unique")
        unknown = [name for name in requested if name not in self.joint_names]
        if unknown:
            raise ValueError(f"joint_names for reindex contain unknown joints: {unknown}")
        index = torch.tensor(
            [self.joint_names.index(name) for name in requested],
            dtype=torch.long,
            device=self.device_cfg.device,
        )
        for name in self._PER_DOF_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, index).clone())
        if isinstance(self.position_limit_clip, torch.Tensor):
            self.position_limit_clip = self.position_limit_clip.index_select(0, index).clone()
        self.joint_names = requested

    def copy_(self, new_config: CSpaceParams) -> CSpaceParams:
        """Copy values while retaining compatible caller-held tensor buffers."""
        if not isinstance(new_config, CSpaceParams):
            raise TypeError("new_config must be a CSpaceParams")
        if self.device_cfg != new_config.device_cfg:
            raise ValueError("copy_ requires matching device_cfg values")
        self.joint_names = new_config.joint_names.copy()
        for name in self._PER_DOF_FIELDS:
            setattr(self, name, copy_or_clone(getattr(new_config, name), getattr(self, name)))
        source_clip = new_config.position_limit_clip
        target_clip = self.position_limit_clip
        if isinstance(source_clip, torch.Tensor):
            self.position_limit_clip = copy_or_clone(
                source_clip, target_clip if isinstance(target_clip, torch.Tensor) else None
            )
        else:
            self.position_limit_clip = source_clip
        return self

    def clone(self) -> CSpaceParams:
        """Return an independent configuration with the same device policy."""
        return CSpaceParams(
            joint_names=self.joint_names.copy(),
            default_joint_position=clone_if_not_none(self.default_joint_position),
            cspace_distance_weight=clone_if_not_none(self.cspace_distance_weight),
            null_space_weight=clone_if_not_none(self.null_space_weight),
            null_space_maximum_distance=clone_if_not_none(self.null_space_maximum_distance),
            device_cfg=self.device_cfg,
            max_acceleration=self.max_acceleration.clone(),
            max_jerk=self.max_jerk.clone(),
            velocity_scale=self.velocity_scale.clone(),
            acceleration_scale=self.acceleration_scale.clone(),
            jerk_scale=self.jerk_scale.clone(),
            position_limit_clip=(
                self.position_limit_clip.clone()
                if isinstance(self.position_limit_clip, torch.Tensor)
                else self.position_limit_clip
            ),
        )

    def scale_joint_limits(self, joint_limits: JointLimits) -> JointLimits:
        """Return scaled, safety-clipped joint limits without mutating ``joint_limits``."""
        if joint_limits.joint_names != self.joint_names:
            raise ValueError("joint_limits.joint_names must match CSpaceParams joint_names")
        if joint_limits.device_cfg != self.device_cfg:
            raise ValueError("joint_limits.device_cfg must match CSpaceParams.device_cfg")
        result = joint_limits.clone()
        result.velocity *= self.velocity_scale
        result.acceleration *= self.acceleration_scale
        result.jerk *= self.jerk_scale
        clip = self.position_limit_clip
        if isinstance(clip, torch.Tensor) or float(clip) != 0.0:
            clip_tensor = torch.as_tensor(clip, **self.device_cfg.as_torch_dict())
            if bool((result.position[1] - result.position[0] <= 2 * clip_tensor).any().item()):
                raise ValueError("position_limit_clip collapses one or more joint position ranges")
            result.position[0] += clip_tensor
            result.position[1] -= clip_tensor
        return result

    @staticmethod
    def load_from_joint_limits(
        joint_position_upper: torch.Tensor,
        joint_position_lower: torch.Tensor,
        joint_names: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> CSpaceParams:
        """Create unit-weight C-space parameters centered in valid position limits."""
        names = list(joint_names)
        dof = len(names)
        upper = device_cfg.to_device(joint_position_upper)
        lower = device_cfg.to_device(joint_position_lower)
        if upper.ndim != 1 or lower.ndim != 1 or upper.numel() != dof or lower.numel() != dof:
            raise ValueError("joint position limits must be rank-1 vectors matching joint_names")
        if not bool(torch.isfinite(upper).all().item()) or not bool(torch.isfinite(lower).all().item()):
            raise ValueError("joint position limits must be finite")
        if bool((lower >= upper).any().item()):
            raise ValueError("joint position lower limits must be less than upper limits")
        middle = (upper + lower) / 2
        ones = torch.ones_like(middle)
        return CSpaceParams(
            names,
            default_joint_position=middle,
            cspace_distance_weight=ones,
            null_space_weight=ones,
            null_space_maximum_distance=ones,
            device_cfg=device_cfg,
        )


__all__ = ["CSpaceParams"]
