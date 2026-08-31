"""Portable common result lifecycle for cuRobo solver outputs.

The pinned V2 type is shared by IK, trajectory optimisation, and MPC.  CUDA
cuRobo populates it from packed graph-resident buffers, but consumers only
need its materialised tensor/state values.  This implementation intentionally
keeps those values as regular PyTorch data so clone, merge, and device moves
work equally on CPU and Apple MPS.  CUDA graph handles and raw result-buffer
ABI objects are not represented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.runtime import debug as debug_mode
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import copy_joint_state_at_batch_seed_indices
from curobo._src.state.state_robot import RobotState
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.types.device_cfg import DeviceCfg


def _clone_value(value: Any) -> Any:
    """Clone nested materialised result payloads without tensor aliasing."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, torch.dtype, torch.device)):
        return value
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_value(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{item.name: _clone_value(getattr(value, item.name)) for item in fields(value)})
    return value


def _move_value(value: Any, device_cfg: DeviceCfg) -> Any:
    """Move ordinary result values while preserving bool and index dtypes."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, torch.dtype, torch.device)):
        return value
    if isinstance(value, torch.Tensor):
        dtype = device_cfg.dtype if value.is_floating_point() or value.is_complex() else value.dtype
        return value.to(device=device_cfg.device, dtype=dtype)
    if isinstance(value, DeviceCfg):
        return device_cfg
    if isinstance(value, JointState):
        return value.to(device_cfg)
    # ``RobotState`` owns optional CUDA graph/model buffers.  A portable result
    # cannot safely claim to have moved such an opaque object.
    from curobo._src.state.state_robot import RobotState

    if isinstance(value, RobotState):
        if value.cuda_robot_model_state is not None:
            raise NotImplementedError(
                "moving a RobotState with CUDA model buffers is unavailable in the portable backend"
            )
        torque = None if value.joint_torque is None else _move_value(value.joint_torque, device_cfg)
        return type(value)(value.joint_state.to(device_cfg), torque)
    if isinstance(value, Mapping):
        return type(value)((key, _move_value(item, device_cfg)) for key, item in value.items())
    if isinstance(value, list):
        return [_move_value(item, device_cfg) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device_cfg) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{item.name: _move_value(getattr(value, item.name), device_cfg) for item in fields(value)})
    return value


def _detach_value(value: Any) -> Any:
    """Detach a cloned portable result payload without mutating its owner."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, torch.dtype, torch.device)):
        return value
    if isinstance(value, torch.Tensor):
        # Tensor.detach is a view and would let a later in-place update to the
        # live result mutate the supposedly value-like detached result.
        return value.detach().clone()
    if isinstance(value, Mapping):
        return type(value)((key, _detach_value(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_detach_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    clone = getattr(value, "clone", None)
    detach = getattr(value, "detach", None)
    if callable(clone) and callable(detach):
        # JointState.detach is intentionally in-place, so detaching a clone is
        # essential for result objects that still own an autograd graph.
        return clone().detach()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{item.name: _detach_value(getattr(value, item.name)) for item in fields(value)})
    return value


def _select_batch_value(value: Any, indices: torch.Tensor, batch_size: int) -> Any:
    """Select value payloads that actually carry a leading result batch."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[indices] if value.ndim > 0 and value.shape[0] == batch_size else value
    if isinstance(value, JointState):
        return value[indices] if value.position.ndim > 0 and value.position.shape[0] == batch_size else value
    if isinstance(value, Mapping):
        return type(value)((key, _select_batch_value(item, indices, batch_size)) for key, item in value.items())
    if isinstance(value, list):
        return [_select_batch_value(item, indices, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_batch_value(item, indices, batch_size) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{
            item.name: _select_batch_value(getattr(value, item.name), indices, batch_size)
            for item in fields(value)
        })
    return value


def _copy_masked_tensor(
    target: Optional[torch.Tensor], source: Optional[torch.Tensor], mask: torch.Tensor
) -> None:
    """Copy tensor slices selected by a prefix-shaped boolean mask."""
    if target is None and source is None:
        return
    if not isinstance(target, torch.Tensor) or not isinstance(source, torch.Tensor):
        raise ValueError("both result fields must be tensors or both must be None")
    if target.device != source.device:
        raise ValueError("result tensors must share a device")
    if mask.device != target.device:
        raise ValueError("result mask must share the result tensor device")
    if target.shape != source.shape:
        raise ValueError(f"result tensor shapes differ: {tuple(target.shape)} != {tuple(source.shape)}")
    if target.ndim < mask.ndim or tuple(target.shape[: mask.ndim]) != tuple(mask.shape):
        raise ValueError("result tensor does not have the selected batch/seed prefix")
    target[mask] = source[mask]


def _copy_joint_state_masked(target: Optional[JointState], source: Optional[JointState], mask: torch.Tensor) -> None:
    if target is None and source is None:
        return
    if target is None or source is None:
        raise ValueError("both joint solutions must be set or both must be None")
    for name in target._tensor_fields():
        left, right = getattr(target, name), getattr(source, name)
        if left is None and right is None:
            continue
        if left is None or right is None:
            raise ValueError("joint solution channels must be set on both results or neither")
        # dt/knot_dt can be per batch even when the position tensors are
        # per-batch/per-seed.  A successful seed then selects its owning batch
        # row; it must not make an otherwise valid result merge fail merely
        # because the timing value has no seed axis.
        if (
            mask.ndim == 2
            and left.ndim >= 1
            and left.shape[0] == mask.shape[0]
            and (left.ndim < 2 or left.shape[1] != mask.shape[1])
        ):
            _copy_masked_tensor(left, right, mask.any(dim=1))
        else:
            _copy_masked_tensor(left, right, mask)


def _copy_object_masked(target: Any, source: Any, mask: torch.Tensor) -> None:
    """Copy state/metric values using their stable public result protocols."""
    if target is None and source is None:
        return
    if target is None or source is None:
        raise ValueError("both result payloads must be set or both must be None")
    if isinstance(target, (str, bytes, int, float, bool)) and isinstance(
        source, (str, bytes, int, float, bool)
    ):
        # Scalar diagnostic metadata has no batch axis; result merges retain
        # the destination's global metadata exactly as V2 does.
        return
    if isinstance(target, torch.Tensor) or isinstance(source, torch.Tensor):
        _copy_masked_tensor(target, source, mask)
        return
    if isinstance(target, JointState) or isinstance(source, JointState):
        _copy_joint_state_masked(target, source, mask)
        return
    if isinstance(target, Mapping) or isinstance(source, Mapping):
        if not isinstance(target, Mapping) or not isinstance(source, Mapping):
            raise ValueError("result mapping payload types differ")
        if target.keys() != source.keys():
            raise ValueError("result mapping payload keys differ")
        for key in target:
            _copy_object_masked(target[key], source[key], mask)
        return
    if isinstance(target, list) or isinstance(source, list):
        if not isinstance(target, list) or not isinstance(source, list) or len(target) != len(source):
            raise ValueError("result list payloads differ")
        for left, right in zip(target, source):
            _copy_object_masked(left, right, mask)
        return
    if mask.ndim == 2:
        method = getattr(target, "copy_at_batch_seed_indices", None)
        if callable(method):
            batch_idx, seed_idx = mask.nonzero(as_tuple=True)
            method(source, batch_idx, seed_idx)
            return
    if mask.ndim == 1:
        method = getattr(target, "copy_only_index", None)
        if callable(method):
            method(source, mask)
            return
    raise NotImplementedError(
        "copying this result payload requires the public metric/state copy protocol"
    )


@dataclass
class _PortableBaseSolverResult:
    """Shared materialised result data for IK, TrajOpt, and MPC.

    Per-seed fields use the V2 ``[batch, seed, ...]`` convention; MPC also
    permits a one-dimensional ``[batch]`` success tensor.  All methods retain
    rank and dtype, including boolean success/feasibility and integer ranks.
    """

    success: torch.Tensor
    solution: Optional[torch.Tensor] = None
    js_solution: Optional[JointState] = None
    position_error: Optional[torch.Tensor] = None
    rotation_error: Optional[torch.Tensor] = None
    cspace_error: Optional[torch.Tensor] = None
    goalset_index: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    total_time: float = 0.0
    debug_info: Dict = field(default_factory=dict)
    optimized_seeds: Optional[torch.Tensor] = None
    metrics: object = None
    position_tolerance: float = 0.0
    orientation_tolerance: float = 0.0
    seed_rank: Optional[torch.Tensor] = None
    seed_cost: Optional[torch.Tensor] = None
    batch_size: int = 0
    num_seeds: int = 0
    total_cost_reshaped: Optional[torch.Tensor] = None
    solution_state: object = None
    feasible: Optional[torch.Tensor] = None

    @property
    def device(self) -> Optional[torch.device]:
        """Device owning the first materialised tensor or joint-state payload."""
        for name in ("solution", "success", "seed_cost", "position_error", "optimized_seeds"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                return value.device
        return None if self.js_solution is None else self.js_solution.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Floating dtype of materialised output data, when available."""
        for name in ("solution", "seed_cost", "position_error", "optimized_seeds"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                return value.dtype
        return None if self.js_solution is None else self.js_solution.dtype

    @property
    def result_batch_size(self) -> int:
        """Infer the leading batch count without requiring stale metadata."""
        if self.batch_size > 0:
            return int(self.batch_size)
        if isinstance(self.success, torch.Tensor) and self.success.ndim > 0:
            return int(self.success.shape[0])
        return 0

    @property
    def seed_shape(self) -> Optional[torch.Size]:
        """Return the canonical ``[batch, seed]`` shape, if this is a seeded result."""
        # A solver may already have materialised only ``return_seeds`` while
        # retaining a full optimizer cost table for diagnostics.  The latter
        # is the authoritative optimizer seed axis until TrajOpt normalises
        # it, so do not infer stale metadata from the compact success buffer.
        for value in (self.seed_cost, self.seed_rank, self.total_cost_reshaped):
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                return value.shape[:2]
        if isinstance(self.success, torch.Tensor) and self.success.ndim >= 2:
            return self.success.shape[:2]
        return None

    def refresh_batch_metadata(self) -> "BaseSolverResult":
        """Synchronise optional V2 batch/seed metadata with materialised tensors."""
        if not isinstance(self.success, torch.Tensor) or self.success.ndim < 1:
            raise ValueError("success must be a tensor with a leading batch dimension")
        self.batch_size = int(self.success.shape[0])
        shape = self.seed_shape
        self.num_seeds = 0 if shape is None else int(shape[1])
        return self

    def validate(self) -> "BaseSolverResult":
        """Validate public tensor layout/device invariants before result merging."""
        if not isinstance(self.success, torch.Tensor) or self.success.ndim < 1:
            raise ValueError("success must be a tensor with a leading batch dimension")
        if self.success.dtype != torch.bool:
            raise TypeError("success must have torch.bool dtype")
        batch = int(self.success.shape[0])
        if self.batch_size not in (0, batch):
            raise ValueError("batch_size does not match success")
        seeded = self.seed_shape
        if seeded is not None and self.num_seeds not in (0, int(seeded[1])):
            raise ValueError("num_seeds does not match seeded result tensors")
        for name in (
            "solution", "position_error", "rotation_error", "cspace_error", "goalset_index",
            "optimized_seeds", "seed_rank", "seed_cost", "total_cost_reshaped", "feasible",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != batch:
                raise ValueError(f"{name} must have the result batch as its leading dimension")
            if value.device != self.success.device:
                raise ValueError(f"{name} must share the success tensor device")
        if self.feasible is not None and self.feasible.shape != self.success.shape:
            raise ValueError("feasible and success must have identical shapes")
        if self.js_solution is not None:
            if self.js_solution.device != self.success.device:
                raise ValueError("js_solution must share the success tensor device")
            if self.js_solution.position.ndim < 1 or self.js_solution.position.shape[0] != batch:
                raise ValueError("js_solution must have the result batch as its leading dimension")
        return self

    def clone(self) -> "BaseSolverResult":
        """Return a deep clone of every materialised value and nested diagnostic."""
        return type(self)(**{item.name: _clone_value(getattr(self, item.name)) for item in fields(self)})

    def to(self, device_cfg: DeviceCfg) -> "BaseSolverResult":
        """Return a device/dtype-converted value without changing bool/index dtypes."""
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        return type(self)(**{item.name: _move_value(getattr(self, item.name), device_cfg) for item in fields(self)})

    def detach(self) -> "BaseSolverResult":
        """Return a detached value copy suitable for non-differentiable reuse.

        This intentionally keeps the public value semantics of solver results;
        it does not expose or attempt to detach CUDA graph/result-buffer state.
        """
        return type(self)(**{item.name: _detach_value(getattr(self, item.name)) for item in fields(self)})

    def select_batch(self, indices: torch.Tensor | int | list[int]) -> "BaseSolverResult":
        """Select batch entries while retaining a leading batch dimension."""
        batch = self.result_batch_size
        if batch < 1:
            raise ValueError("batch_size must be positive for batch selection")
        device = self.device if self.device is not None else torch.device("cpu")
        tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        if tensor.numel() and (tensor.min() < 0 or tensor.max() >= batch):
            raise IndexError("batch index is outside the available batch range")
        values = {
            item.name: _select_batch_value(getattr(self, item.name), tensor, batch)
            for item in fields(self)
        }
        values["batch_size"] = int(tensor.numel())
        return type(self)(**values)

    def copy_successful_solutions(self, other: "BaseSolverResult") -> None:
        """Merge every successful batch/seed candidate from ``other`` in-place.

        The operation is intentionally local to materialised tensor/state
        payloads.  It cannot transfer opaque CUDA graph/result-buffer objects.
        """
        if not isinstance(other, BaseSolverResult):
            raise TypeError("other must be a BaseSolverResult")
        self.validate()
        other.validate()
        if self.success.shape != other.success.shape or self.success.device != other.success.device:
            raise ValueError("result success tensors must share shape and device")
        mask = other.success
        _copy_masked_tensor(self.success, other.success, mask)
        for name in (
            "solution", "position_error", "rotation_error", "cspace_error", "goalset_index",
            "seed_rank", "seed_cost", "total_cost_reshaped", "optimized_seeds", "feasible",
        ):
            _copy_masked_tensor(getattr(self, name), getattr(other, name), mask)
        _copy_joint_state_masked(self.js_solution, other.js_solution, mask)
        _copy_object_masked(self.solution_state, other.solution_state, mask)
        _copy_object_masked(self.metrics, other.metrics, mask)

    def copy_at_batch_indices(self, other: "BaseSolverResult", mask: torch.Tensor) -> None:
        """Overwrite complete leading batch entries selected by ``mask`` in-place."""
        if not isinstance(other, BaseSolverResult):
            raise TypeError("other must be a BaseSolverResult")
        self.validate()
        other.validate()
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("mask must be a one-dimensional boolean batch tensor")
        if mask.device != self.success.device:
            raise ValueError("mask must share the result tensor device")
        if mask.numel() != self.result_batch_size or other.result_batch_size != self.result_batch_size:
            raise ValueError("mask length must match both result batch sizes")
        _copy_masked_tensor(self.success, other.success, mask)
        for name in (
            "solution", "position_error", "rotation_error", "cspace_error", "goalset_index",
            "seed_rank", "seed_cost", "total_cost_reshaped", "optimized_seeds", "feasible",
        ):
            _copy_masked_tensor(getattr(self, name), getattr(other, name), mask)
        _copy_joint_state_masked(self.js_solution, other.js_solution, mask)
        _copy_object_masked(self.solution_state, other.solution_state, mask)
        _copy_object_masked(self.metrics, other.metrics, mask)


@dataclass
class BaseSolverResult(_PortableBaseSolverResult):
    """Pinned V2 declaration facade for the portable materialised result."""

    success: torch.Tensor
    solution: Optional[torch.Tensor] = None
    js_solution: Optional[JointState] = None
    position_error: Optional[torch.Tensor] = None
    rotation_error: Optional[torch.Tensor] = None
    cspace_error: Optional[torch.Tensor] = None
    goalset_index: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    total_time: float = 0.0
    debug_info: Dict = field(default_factory=dict)
    optimized_seeds: Optional[torch.Tensor] = None
    metrics: Optional[RolloutMetrics] = None
    position_tolerance: float = 0.0
    orientation_tolerance: float = 0.0
    seed_rank: Optional[torch.Tensor] = None
    seed_cost: Optional[torch.Tensor] = None
    batch_size: int = 0
    num_seeds: int = 0
    total_cost_reshaped: Optional[torch.Tensor] = None
    solution_state: Optional[RobotState] = None
    feasible: Optional[torch.Tensor] = None

    @profiler.record_function("solver_base_result/clone")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def clone(self) -> "BaseSolverResult":
        return _PortableBaseSolverResult.clone(self)

    @profiler.record_function("solver_base_result/copy_successful_solutions")
    def copy_successful_solutions(self, other: "BaseSolverResult") -> None:
        _PortableBaseSolverResult.copy_successful_solutions(self, other)

    def copy_at_batch_indices(self, other: "BaseSolverResult", mask: torch.Tensor) -> None:
        _PortableBaseSolverResult.copy_at_batch_indices(self, other, mask)


__all__ = ["BaseSolverResult"]
