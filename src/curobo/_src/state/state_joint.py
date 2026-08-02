"""Portable joint state for the pinned internal import path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from curobo_metal.types.state import JointState as _MetalJointState

from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from .filter_coeff import FilterCoeff
from .state_base import State

# The upstream aliases are intentionally tensors in the portable backend.  They
# retain type-level compatibility without implying the CUDA packed-buffer ABI.
T_BDOF = torch.Tensor
T_DOF = torch.Tensor


@dataclass
class JointState(_MetalJointState, State):
    device_cfg: DeviceCfg = DeviceCfg()
    control_space: Optional[ControlSpace] = None

    def __post_init__(self) -> None:
        if isinstance(self.position, torch.Tensor):
            self.device_cfg = DeviceCfg(self.position.device, self.position.dtype)
        else:
            self.position = self.device_cfg.to_device(self.position)
        for field in ("velocity", "acceleration", "jerk"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, torch.Tensor):
                setattr(self, field, self.device_cfg.to_device(value))
        if self.joint_names is not None:
            self.joint_names = list(self.joint_names)
            if len(self.joint_names) != self.position.shape[-1]:
                raise ValueError("joint_names must match the final position dimension")

    @staticmethod
    def from_state_tensor(
        state_tensor: torch.Tensor, joint_names: Optional[list[str]] = None, dof: int = 7
    ) -> "JointState":
        return JointState(
            state_tensor[..., :dof].contiguous(),
            state_tensor[..., dof : 2 * dof].contiguous(),
            state_tensor[..., 2 * dof : 3 * dof].contiguous(),
            joint_names=joint_names,
            jerk=state_tensor[..., 3 * dof : 4 * dof].contiguous(),
        )

    @staticmethod
    def from_numpy(
        joint_names: List[str],
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        jerk: Optional[np.ndarray] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        position_t = device_cfg.to_device(position)
        zero = torch.zeros_like(position_t)
        return JointState(
            position_t,
            zero if velocity is None else device_cfg.to_device(velocity),
            zero if acceleration is None else device_cfg.to_device(acceleration),
            joint_names,
            zero if jerk is None else device_cfg.to_device(jerk),
            device_cfg,
        )

    @staticmethod
    def from_position(position: T_BDOF, joint_names: Optional[List[str]] = None):
        zero = torch.zeros_like(position)
        return JointState(position, zero, zero, joint_names, zero)

    @staticmethod
    def from_list(
        position: list[float],
        velocity: list[float],
        acceleration: list[float],
        device_cfg: DeviceCfg,
    ) -> "JointState":
        return JointState(position, velocity, acceleration, device_cfg=device_cfg)

    @classmethod
    def zeros(
        cls,
        size: tuple[int, ...],
        device_cfg: DeviceCfg,
        joint_names: Optional[list[str]] = None,
    ) -> "JointState":
        value = torch.zeros(size, **device_cfg.as_torch_dict())
        return cls(
            value,
            value.clone(),
            value.clone(),
            joint_names,
            value.clone(),
            device_cfg,
            dt=torch.ones(size[0], **device_cfg.as_torch_dict()),
        )

    def data_ptr(self) -> int:
        return self.position.data_ptr()

    def copy_data(self, in_joint_state: "JointState"):
        """Copy tensor contents while retaining this object's metadata buffers."""
        for field in ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt"):
            source, target = getattr(in_joint_state, field), getattr(self, field)
            if source is not None and target is not None:
                target.copy_(source)
        return self

    def to(self, device_cfg: DeviceCfg) -> "JointState":
        return self._map(device_cfg.to_device)

    def detach(self) -> "JointState":
        for field in ("position", "velocity", "acceleration", "jerk", "dt"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.detach())
        return self

    def copy_reference(self, in_joint_state: "JointState") -> "JointState":
        for field in (
            "position", "velocity", "acceleration", "jerk", "dt",
            "joint_names", "knot", "knot_dt",
        ):
            setattr(self, field, getattr(in_joint_state, field))
        return self

    def copy_(self, in_joint_state: "JointState", allow_clone: bool = True) -> "JointState":
        same = all(
            getattr(in_joint_state, field) is None
            or (
                getattr(self, field) is not None
                and getattr(self, field).shape == getattr(in_joint_state, field).shape
            )
            for field in ("position", "velocity", "acceleration", "jerk", "dt")
        )
        if not same:
            if not allow_clone:
                raise ValueError(
                    f"current state has shape: {self.position.shape} while new shape is "
                    f"{in_joint_state.position.shape}"
                )
            return self.copy_reference(in_joint_state.clone())
        for field in ("position", "velocity", "acceleration", "jerk", "dt"):
            source, target = getattr(in_joint_state, field), getattr(self, field)
            if source is not None:
                target.copy_(source)
        if in_joint_state.joint_names is not None:
            self.joint_names = in_joint_state.joint_names
        return self

    def __setitem__(self, index: int | torch.Tensor, value: "JointState") -> None:
        for field in ("position", "velocity", "acceleration", "jerk"):
            target, source = getattr(self, field), getattr(value, field)
            if target is not None and source is not None:
                target[index] = source
        if self.dt is not None and value.dt is not None:
            self.dt[index] = value.dt

    def reindex(self, joint_names: List[str]):
        value = self.reorder(joint_names)
        self.copy_reference(value)

    def stack(self, new_state: "JointState") -> "JointState":
        return self._combine(new_state, torch.stack)

    def cat(self, other_js: "JointState", dim: int) -> "JointState":
        return self._combine(other_js, lambda values: torch.cat(values, dim=dim))

    def _combine(self, other: "JointState", operation) -> "JointState":
        values = {}
        for field in ("position", "velocity", "acceleration", "jerk"):
            left, right = getattr(self, field), getattr(other, field)
            values[field] = None if left is None else operation((left, right))
        return type(self)(joint_names=self.joint_names, **values)

    def repeat_seeds(self, num_seeds: int) -> "JointState":
        if num_seeds <= 1:
            return self.clone()
        def repeat(value):
            return value.view(value.shape[0], 1, *value.shape[1:]).repeat(
                1, num_seeds, *([1] * (value.ndim - 1))
            ).reshape(value.shape[0] * num_seeds, *value.shape[1:])
        return self._map(repeat)

    def get_state_tensor(self) -> torch.Tensor:
        return torch.cat([x for x in (self.position, self.velocity, self.acceleration, self.jerk)
                          if x is not None], dim=-1)

    def blend(self, coeff: FilterCoeff, new_state: "JointState"):
        from .state_joint_ops import blend_joint_states
        return blend_joint_states(self, new_state, coeff)

    def apply_kernel(self, kernel_mat):
        from .state_joint_ops import apply_kernel_to_joint_state
        return apply_kernel_to_joint_state(self, kernel_mat)

    def scale(self, dt):
        from .state_joint_ops import scale_joint_state
        return scale_joint_state(self, dt)

    def scale_by_dt(self, dt, new_dt):
        from .state_joint_ops import scale_joint_state_by_dt
        return scale_joint_state_by_dt(self, dt, new_dt)

    def scale_time(self, new_dt):
        from .state_joint_ops import scale_joint_state_time
        return scale_joint_state_time(self, new_dt)

    def calculate_fd_from_position(self, dt: Optional[torch.Tensor] = None):
        from .state_joint_ops import calculate_fd_from_position
        return calculate_fd_from_position(self, dt)

    def get_augmented_joint_state(self, joint_names, lock_joints: Optional["JointState"] = None) -> "JointState":
        from .state_joint_ops import augment_joint_state
        return augment_joint_state(self, joint_names, lock_joints)

    def append_joints(self, joint_state):
        from .state_joint_ops import append_joints_to_state
        return append_joints_to_state(self, joint_state)

    def gather_by_seed_index(self, idx: torch.Tensor):
        from .state_joint_trajectory_ops import gather_joint_state_by_seed
        return gather_joint_state_by_seed(self, idx)

    def copy_only_index(self, in_joint_state: "JointState", idx: Union[int, torch.Tensor]):
        from .state_joint_trajectory_ops import copy_joint_state_only_index
        return copy_joint_state_only_index(self, in_joint_state, idx)

    def copy_at_index(self, in_joint_state: "JointState", idx: Union[int, torch.Tensor]):
        from .state_joint_trajectory_ops import copy_joint_state_at_index
        return copy_joint_state_at_index(self, in_joint_state, idx)

    def copy_at_batch_seed_indices(self, in_joint_state: "JointState", batch_idx: torch.Tensor, seed_idx: torch.Tensor):
        from .state_joint_trajectory_ops import copy_joint_state_at_batch_seed_indices
        return copy_joint_state_at_batch_seed_indices(
            self, in_joint_state, batch_idx, seed_idx
        )

    def get_trajectory_at_horizon_index(self, horizon_index: int):
        from .state_joint_trajectory_ops import get_joint_state_at_horizon_index
        return get_joint_state_at_horizon_index(self, horizon_index)

    def trim_trajectory(self, start_idx: int, end_idx: Optional[int] = None):
        from .state_joint_trajectory_ops import trim_joint_state_trajectory
        return trim_joint_state_trajectory(self, start_idx, end_idx)

    def index_dof(self, idx: int):
        from .state_joint_trajectory_ops import index_joint_state_dof
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self.device)
        return index_joint_state_dof(self, idx)


# These reexports mirror the direct-import ergonomics of cuRobo's state
# module.  They use regular PyTorch operations and do not expose CUDA JIT ABI.
from .state_joint_ops import (  # noqa: E402
    append_joints_to_state,
    apply_kernel_to_joint_state,
    augment_joint_state,
    blend_joint_states,
    calculate_fd_from_position,
    cat_joint_states,
    joint_state_to_tensor,
    reindex_joint_state_inplace,
    reorder_joint_state,
    repeat_joint_state,
    repeat_joint_state_seeds,
    scale_joint_state,
    scale_joint_state_by_dt,
    scale_joint_state_time,
    stack_joint_states,
)
from .state_joint_trajectory_ops import (  # noqa: E402
    copy_joint_state_at_batch_seed_indices,
    copy_joint_state_at_index,
    copy_joint_state_only_index,
    gather_joint_state_by_seed,
    get_joint_state_at_horizon_index,
    index_joint_state_dof,
    trim_joint_state_trajectory,
)

__all__ = [
    "JointState", "T_BDOF", "T_DOF", "append_joints_to_state", "apply_kernel_to_joint_state",
    "augment_joint_state", "blend_joint_states", "calculate_fd_from_position", "cat_joint_states",
    "copy_joint_state_at_batch_seed_indices", "copy_joint_state_at_index", "copy_joint_state_only_index",
    "gather_joint_state_by_seed", "get_joint_state_at_horizon_index", "index_joint_state_dof",
    "joint_state_to_tensor", "reindex_joint_state_inplace", "reorder_joint_state", "repeat_joint_state",
    "repeat_joint_state_seeds", "scale_joint_state", "scale_joint_state_by_dt", "scale_joint_state_time",
    "stack_joint_states", "trim_joint_state_trajectory",
]
