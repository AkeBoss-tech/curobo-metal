"""Portable joint state for the pinned internal import path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
            if value is None:
                continue
            value = self.device_cfg.to_device(value)
            # Finite differences intentionally shorten the horizon for each
            # successive derivative, so only the joint/DOF axis is invariant.
            # Rejecting a shorter velocity/acceleration here would make an
            # otherwise valid V2 trajectory impossible to index or clone.
            if value.ndim == 0 or value.shape[-1] != self.position.shape[-1]:
                raise ValueError(f"{field} must end in the position DOF dimension")
            setattr(self, field, value)

        # These trajectory metadata tensors do not necessarily have the same
        # rank as position (for example dt may be [batch] or [batch, horizon]),
        # but they must reside with the state.  Converting them here avoids a
        # hidden CPU tensor surviving inside an otherwise MPS state.
        for field in ("dt", "knot", "knot_dt"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, self.device_cfg.to_device(value))
        if self.joint_names is not None:
            self.joint_names = list(self.joint_names)
            if len(self.joint_names) != self.position.shape[-1]:
                raise ValueError("joint_names must match the final position dimension")

    # These accessors are deliberately declared on the compatibility class
    # rather than inherited invisibly from the portable value implementation.
    # A fair amount of downstream cuRobo code introspects this surface.
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

    @staticmethod
    def from_state_tensor(
        state_tensor, joint_names=None, dof=7
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
        return JointState(
            position_t,
            torch.zeros_like(position_t) if velocity is None else device_cfg.to_device(velocity),
            torch.zeros_like(position_t) if acceleration is None else device_cfg.to_device(acceleration),
            joint_names,
            torch.zeros_like(position_t) if jerk is None else device_cfg.to_device(jerk),
            device_cfg,
        )

    @staticmethod
    def from_position(position: T_BDOF, joint_names: Optional[List[str]] = None):
        # Keep derivative channels independently mutable.  Reusing a single
        # zero tensor aliases velocity/acceleration/jerk and means a later
        # in-place update to one silently changes the others.
        return JointState(
            position,
            torch.zeros_like(position),
            torch.zeros_like(position),
            joint_names,
            torch.zeros_like(position),
        )

    @staticmethod
    def from_list(
        position, velocity, acceleration, device_cfg: DeviceCfg(),
    ):
        return JointState(position, velocity, acceleration, device_cfg=device_cfg)

    @staticmethod
    def zeros(size: Tuple[int], device_cfg: DeviceCfg, joint_names: Optional[List[str]] = None):
        value = torch.zeros(size, **device_cfg.as_torch_dict())
        return JointState(
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

    def __len__(self) -> int:
        return self.position.shape[0]

    def clone(self) -> "JointState":
        """Clone every materialized state channel without breaking autograd.

        The base portable value model deliberately has a minimal clone helper.
        The compatibility type additionally owns timing, knot, and control
        metadata, so copying it explicitly avoids losing state during solver
        buffer lifecycle operations.
        """
        clone = lambda value: None if value is None else value.clone()
        return type(self)(
            clone(self.position), clone(self.velocity), clone(self.acceleration),
            None if self.joint_names is None else self.joint_names.copy(), clone(self.jerk),
            self.device_cfg, clone(self.dt), aux_data=dict(self.aux_data),
            knot=clone(self.knot), knot_dt=clone(self.knot_dt),
            control_space=self.control_space,
        )

    def unsqueeze(self, idx: int):
        return self._shape_result(
            self.position.unsqueeze(idx),
            self._apply_optional(self.velocity, lambda value: value.unsqueeze(idx)),
            self._apply_optional(self.acceleration, lambda value: value.unsqueeze(idx)),
            self._apply_optional(self.jerk, lambda value: value.unsqueeze(idx)),
            knot=self._apply_optional(self.knot, lambda value: value.unsqueeze(idx)),
        )

    def squeeze(self, dim: Optional[int] = 0):
        return self._shape_result(
            self.position.squeeze(dim),
            self._apply_optional(self.velocity, lambda value: value.squeeze(dim)),
            self._apply_optional(self.acceleration, lambda value: value.squeeze(dim)),
            self._apply_optional(self.jerk, lambda value: value.squeeze(dim)),
            knot=self._apply_optional(self.knot, lambda value: value.squeeze(dim)),
        )

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        dt = self.dt
        if dt is not None and len(shape) > 2:
            # cuRobo's trajectory convention stores dt over batch/horizon,
            # not over the final DOF axis.  Preserve it if it cannot be
            # reshaped to that prefix (e.g. a scalar or global schedule).
            prefix = tuple(shape[:2])
            if dt.numel() == int(np.prod(prefix)):
                dt = dt.view(*prefix)
        return self._shape_result(
            self.position.view(*shape),
            self._apply_optional(self.velocity, lambda value: value.view(*shape)),
            self._apply_optional(self.acceleration, lambda value: value.view(*shape)),
            self._apply_optional(self.jerk, lambda value: value.view(*shape)),
            dt=dt,
            knot=self.knot,
            knot_dt=self.knot_dt,
        )

    def repeat(self, repeat_input: List[int]):
        repeats = tuple(repeat_input)
        return self._shape_result(
            self.position.repeat(*repeats),
            self._apply_optional(self.velocity, lambda value: value.repeat(*repeats)),
            self._apply_optional(self.acceleration, lambda value: value.repeat(*repeats)),
            self._apply_optional(self.jerk, lambda value: value.repeat(*repeats)),
            knot=self.knot,
            knot_dt=self.knot_dt,
        )

    @staticmethod
    def _apply_optional(value, function):
        return None if value is None else function(value)

    def _shape_result(
        self,
        position: torch.Tensor,
        velocity: Optional[torch.Tensor],
        acceleration: Optional[torch.Tensor],
        jerk: Optional[torch.Tensor],
        *,
        dt: Optional[torch.Tensor] = None,
        knot: Optional[torch.Tensor] = None,
        knot_dt: Optional[torch.Tensor] = None,
    ) -> "JointState":
        """Build a shaped state without treating scalar trajectory data as DOF data."""
        return type(self)(
            position, velocity, acceleration,
            None if self.joint_names is None else self.joint_names.copy(), jerk,
            self.device_cfg,
            self.dt if dt is None else dt,
            aux_data=dict(self.aux_data),
            knot=self.knot if knot is None else knot,
            knot_dt=self.knot_dt if knot_dt is None else knot_dt,
            control_space=self.control_space,
        )

    def __getitem__(self, index) -> "JointState":
        """Index state tensors while keeping per-batch timing well formed.

        An integer batch index removes the leading position dimension, but V2
        timing metadata remains a one-item batch tensor.  That distinction is
        important to trajectory code which subsequently broadcasts a selected
        state back into a batch.
        """
        if isinstance(index, list):
            index = torch.as_tensor(index, device=self.device, dtype=torch.long)

        def select(value):
            return self._apply_optional(value, lambda tensor: tensor[index])

        batch_index = index[0] if isinstance(index, tuple) and index else index

        def select_batch_metadata(value):
            if value is None or value.ndim == 0 or value.shape[0] != self.position.shape[0]:
                return value
            if isinstance(batch_index, int):
                return value[batch_index].unsqueeze(0)
            if isinstance(batch_index, torch.Tensor) and batch_index.numel() == 1:
                return value[batch_index.reshape(-1)[0]].unsqueeze(0)
            return value[batch_index]

        return type(self)(
            select(self.position), select(self.velocity), select(self.acceleration),
            None if self.joint_names is None else self.joint_names.copy(), select(self.jerk),
            self.device_cfg, select_batch_metadata(self.dt), aux_data=dict(self.aux_data),
            knot=select_batch_metadata(self.knot),
            knot_dt=select_batch_metadata(self.knot_dt),
            control_space=self.control_space,
        )

    def reorder(self, joint_names: List[str]) -> JointState:
        if self.joint_names is None:
            raise ValueError("cannot reorder a JointState without joint_names")
        try:
            indices = [self.joint_names.index(name) for name in joint_names]
        except ValueError as error:
            raise ValueError("requested joint is absent from JointState") from error
        index = torch.tensor(indices, device=self.device)

        def select(value):
            return None if value is None else value.index_select(-1, index)

        # Construct without names first: inherited _map preserves the old names
        # while changing the final DOF dimension, which violates this facade's
        # eager validation before it can replace the names.
        output = type(self)(
            select(self.position), select(self.velocity), select(self.acceleration),
            None, select(self.jerk), self.device_cfg, self.dt,
            aux_data=dict(self.aux_data), knot=select(self.knot), knot_dt=self.knot_dt,
            control_space=self.control_space,
        )
        output.joint_names = list(joint_names)
        return output

    def copy_data(self, in_joint_state: "JointState"):
        """Copy tensor contents while retaining this object's metadata buffers.

        This is a deprecated upstream API, but planner buffer owners still use
        it when a state changes shape.  Match ``copy_``'s useful lifecycle
        guarantee: reuse compatible allocation, otherwise adopt the source
        tensor reference rather than silently dropping the new channel.
        """
        for field in self._tensor_fields():
            source, target = getattr(in_joint_state, field), getattr(self, field)
            if source is None:
                continue
            if (
                target is not None
                and target.shape == source.shape
                and target.device == source.device
                and target.dtype == source.dtype
            ):
                target.copy_(source)
            else:
                setattr(self, field, source)
        return self

    def to(self, device_cfg: DeviceCfg) -> "JointState":
        convert = lambda value: None if value is None else device_cfg.to_device(value)
        return type(self)(
            convert(self.position), convert(self.velocity), convert(self.acceleration),
            None if self.joint_names is None else self.joint_names.copy(), convert(self.jerk),
            device_cfg, convert(self.dt), aux_data=dict(self.aux_data),
            knot=convert(self.knot), knot_dt=convert(self.knot_dt),
            control_space=self.control_space,
        )

    def detach(self) -> "JointState":
        for field in self._tensor_fields():
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, value.detach())
        return self

    def copy_reference(self, in_joint_state: JointState):
        for field in (
            "position", "velocity", "acceleration", "jerk", "dt",
            "joint_names", "knot", "knot_dt",
        ):
            setattr(self, field, getattr(in_joint_state, field))
        self.device_cfg = in_joint_state.device_cfg
        self.aux_data = dict(in_joint_state.aux_data)
        self.control_space = in_joint_state.control_space
        return self

    def copy_(self, in_joint_state: JointState, allow_clone: bool = True):
        if not self._same_shape(in_joint_state):
            if not allow_clone:
                raise ValueError(
                    f"current state has shape: {self.position.shape} while new shape is "
                    f"{in_joint_state.position.shape}"
                )
            return self.copy_reference(in_joint_state.clone())
        for field in self._tensor_fields():
            source, target = getattr(in_joint_state, field), getattr(self, field)
            if source is not None and target is not None:
                target.copy_(source)
        if in_joint_state.joint_names is not None:
            self.joint_names = in_joint_state.joint_names.copy()
        self.aux_data = dict(in_joint_state.aux_data)
        self.control_space = in_joint_state.control_space
        return self

    @staticmethod
    def _tensor_fields() -> tuple[str, ...]:
        return ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt")

    def _same_shape(self, other: "JointState") -> bool:
        """Whether this object can receive ``other`` through in-place copy.

        Optional channels are allowed in the source (matching V2's partial
        state behavior), but each materialized source channel must have a
        corresponding same-shaped target buffer on the same device.
        """
        for field in self._tensor_fields():
            source = getattr(other, field)
            target = getattr(self, field)
            if source is None:
                continue
            if target is None or target.shape != source.shape or target.device != source.device:
                return False
        return True

    def __setitem__(self, index: int | torch.Tensor, value: "JointState") -> None:
        for field in self._tensor_fields():
            target, source = getattr(self, field), getattr(value, field)
            if target is not None and source is not None:
                target[index] = source

    def reindex(self, joint_names: List[str]):
        value = self.reorder(joint_names)
        self.copy_reference(value)

    def stack(self, new_state: JointState):
        # Despite the historical name, pinned cuRobo stacks consecutive
        # waypoints by concatenating the second-to-last (trajectory) axis.
        # Using torch.stack here would invent a seed axis and makes callers
        # pass a rank that the upstream solver never produces.
        from .state_joint_ops import stack_joint_states
        return stack_joint_states(self, new_state)

    def cat(self, other_js: JointState, dim: int):
        dof_dim = dim if dim >= 0 else self.position.ndim + dim
        return self._combine(
            other_js, lambda values: torch.cat(values, dim=dim),
            join_names=dof_dim == self.position.ndim - 1,
        )

    def _combine(self, other: "JointState", operation, *, join_names: bool) -> "JointState":
        values = {}
        for field in ("position", "velocity", "acceleration", "jerk"):
            left, right = getattr(self, field), getattr(other, field)
            values[field] = None if left is None or right is None else operation((left, right))
        if join_names and self.joint_names is not None and other.joint_names is not None:
            joint_names = self.joint_names + other.joint_names
        else:
            joint_names = None if self.joint_names is None else self.joint_names.copy()
        return type(self)(
            joint_names=joint_names, device_cfg=self.device_cfg,
            dt=None if self.dt is None else self.dt.clone(),
            aux_data=dict(self.aux_data), knot=None if self.knot is None else self.knot.clone(),
            knot_dt=None if self.knot_dt is None else self.knot_dt.clone(),
            control_space=self.control_space, **values,
        )

    def repeat_seeds(self, num_seeds: int) -> "JointState":
        if num_seeds <= 1:
            return self.clone()

        batch = self.position.shape[0]

        def repeat(value):
            if value.ndim == 0 or value.shape[0] != batch:
                return value
            return value.unsqueeze(1).expand(
                batch, num_seeds, *value.shape[1:]
            ).reshape(batch * num_seeds, *value.shape[1:])

        return type(self)(
            repeat(self.position), self._apply_optional(self.velocity, repeat),
            self._apply_optional(self.acceleration, repeat),
            None if self.joint_names is None else self.joint_names.copy(),
            self._apply_optional(self.jerk, repeat), self.device_cfg,
            repeat(self.dt) if self.dt is not None else None,
            aux_data=dict(self.aux_data),
            knot=repeat(self.knot) if self.knot is not None else None,
            knot_dt=repeat(self.knot_dt) if self.knot_dt is not None else None,
            control_space=self.control_space,
        )

    def get_state_tensor(self) -> torch.Tensor:
        # The V2 packing ABI always allocates four derivative channels.  A
        # partial state therefore packs absent derivatives as zeros rather
        # than shifting the position/velocity layout based on optional fields.
        zero = torch.zeros_like(self.position)
        return torch.cat(
            tuple(value if value is not None else zero for value in (
                self.position, self.velocity, self.acceleration, self.jerk
            )),
            dim=-1,
        )

    def blend(self, coeff: FilterCoeff, new_state: "JointState"):
        from .state_joint_ops import blend_joint_states
        return blend_joint_states(self, new_state, coeff)

    def apply_kernel(self, kernel_mat):
        from .state_joint_ops import apply_kernel_to_joint_state
        return apply_kernel_to_joint_state(self, kernel_mat)

    def scale(self, dt: Union[float, torch.Tensor]):
        from .state_joint_ops import scale_joint_state
        return scale_joint_state(self, dt)

    def scale_by_dt(self, dt: torch.Tensor, new_dt: torch.Tensor):
        from .state_joint_ops import scale_joint_state_by_dt
        return scale_joint_state_by_dt(self, dt, new_dt)

    def scale_time(self, new_dt: torch.Tensor):
        from .state_joint_ops import scale_joint_state_time
        return scale_joint_state_time(self, new_dt)

    def calculate_fd_from_position(self, dt: Optional[torch.Tensor] = None):
        from .state_joint_ops import calculate_fd_from_position
        return calculate_fd_from_position(self, dt)

    def get_augmented_joint_state(self, joint_names, lock_joints: Optional[JointState] = None) -> JointState:
        from .state_joint_ops import augment_joint_state
        return augment_joint_state(self, joint_names, lock_joints)

    def append_joints(self, joint_state: JointState):
        from .state_joint_ops import append_joints_to_state
        return append_joints_to_state(self, joint_state)

    def gather_by_seed_index(self, idx: torch.Tensor):
        from .state_joint_trajectory_ops import gather_joint_state_by_seed
        return gather_joint_state_by_seed(self, idx)

    def copy_only_index(self, in_joint_state: JointState, idx: Union[int, torch.Tensor]):
        from .state_joint_trajectory_ops import copy_joint_state_only_index
        return copy_joint_state_only_index(self, in_joint_state, idx)

    def copy_at_index(self, in_joint_state: JointState, idx: Union[int, torch.Tensor]):
        from .state_joint_trajectory_ops import copy_joint_state_at_index
        return copy_joint_state_at_index(self, in_joint_state, idx)

    def copy_at_batch_seed_indices(self, in_joint_state: JointState, batch_idx: torch.Tensor, seed_idx: torch.Tensor):
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
