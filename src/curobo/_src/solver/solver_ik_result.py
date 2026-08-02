"""Portable result lifecycle helpers for inverse kinematics.

The pinned V2 class deliberately has no extra data fields beyond
``BaseSolverResult``.  The helpers here consequently operate only on that
public layout; they make consuming batched, multi-seed CPU/MPS IK results less
error-prone without claiming the CUDA result-buffer ABI.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg

from .solver_base_result import BaseSolverResult


_TensorIndex = Union[torch.Tensor, Sequence[int], int]


def _clone_value(value: Any) -> Any:
    """Clone tensor-bearing result metadata without modifying caller state."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_value(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value.clone() if hasattr(value, "clone") else value


def _move_value(value: Any, device_cfg: DeviceCfg) -> Any:
    """Move metadata while retaining integer, boolean, and index dtypes."""
    if isinstance(value, torch.Tensor):
        dtype = device_cfg.dtype if value.is_floating_point() or value.is_complex() else value.dtype
        return value.to(device=device_cfg.device, dtype=dtype)
    if isinstance(value, JointState):
        return value.to(device_cfg)
    if isinstance(value, Mapping):
        return type(value)((key, _move_value(item, device_cfg)) for key, item in value.items())
    if isinstance(value, list):
        return [_move_value(item, device_cfg) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device_cfg) for item in value)
    return value


def _gather_seed_value(value: Any, indices: torch.Tensor, shape: torch.Size) -> Any:
    """Gather values which use the public ``[batch, seed, ...]`` layout."""
    if isinstance(value, torch.Tensor):
        if value.ndim < 2 or value.shape[:2] != shape:
            return value
        expanded = indices.reshape(indices.shape + (1,) * (value.ndim - 2))
        return value.gather(1, expanded.expand(indices.shape + value.shape[2:]))
    if isinstance(value, JointState):
        clone = value.clone()
        for name in ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt"):
            item = getattr(clone, name, None)
            if isinstance(item, torch.Tensor) and item.ndim >= 2 and item.shape[:2] == shape:
                setattr(clone, name, _gather_seed_value(item, indices, shape))
        return clone
    if isinstance(value, Mapping):
        return type(value)(
            (key, _gather_seed_value(item, indices, shape)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_gather_seed_value(item, indices, shape) for item in value]
    if isinstance(value, tuple):
        return tuple(_gather_seed_value(item, indices, shape) for item in value)
    return value


def _select_batch_value(value: Any, indices: torch.Tensor, batch_size: int) -> Any:
    """Select only tensors/states with a leading result batch dimension."""
    if isinstance(value, torch.Tensor):
        return value[indices] if value.ndim > 0 and value.shape[0] == batch_size else value
    if isinstance(value, JointState):
        if value.position.ndim > 0 and value.position.shape[0] == batch_size:
            return value[indices]
        return value
    if isinstance(value, Mapping):
        return type(value)(
            (key, _select_batch_value(item, indices, batch_size)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_select_batch_value(item, indices, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_batch_value(item, indices, batch_size) for item in value)
    return value


@dataclass
class IKSolverResult(BaseSolverResult):
    """A batched, multi-seed inverse-kinematics result.

    ``solution`` and all per-seed result tensors use ``[batch, seed, ...]``.
    The portable helper methods preserve that rank even when selecting one
    seed, making results suitable for immediate reuse as deterministic IK
    seeds.  CUDA graph/result-buffer objects are intentionally not represented.
    """

    @property
    def device(self) -> Optional[torch.device]:
        """Device of the first materialized result tensor, if one exists."""
        for name in ("success", "solution", "seed_cost", "position_error", "optimized_seeds"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                return value.device
        if self.js_solution is not None:
            return self.js_solution.device
        return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Floating result dtype, or ``None`` when the result has no tensors."""
        for name in ("solution", "seed_cost", "position_error", "optimized_seeds", "success"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                return value.dtype
        return None if self.js_solution is None else self.js_solution.dtype

    @property
    def seed_shape(self) -> Optional[torch.Size]:
        """The canonical ``[batch, seed]`` shape when present."""
        for value in (self.seed_cost, self.success, self.solution):
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                return value.shape[:2]
        return None

    def clone(self) -> "IKSolverResult":
        """Deep clone tensors, joint solutions, and nested diagnostic metadata."""
        return type(self)(**{item.name: _clone_value(getattr(self, item.name)) for item in fields(self)})

    def to(self, device_cfg: DeviceCfg) -> "IKSolverResult":
        """Return a copy on ``device_cfg`` without converting index/bool tensors."""
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        return type(self)(
            **{item.name: _move_value(getattr(self, item.name), device_cfg) for item in fields(self)}
        )

    def get_topk_seeds(self, topk: int) -> "IKSolverResult":
        """Return the lowest-cost ``topk`` seeds for every batch deterministically."""
        if self.seed_cost is None or self.seed_cost.ndim != 2:
            raise ValueError("seed_cost with shape [batch, seed] is required")
        if topk < 1 or topk > self.seed_cost.shape[1]:
            raise ValueError("topk must be between 1 and the available seed count")
        indices = self.seed_cost.argsort(dim=1, stable=True)[:, :topk]
        return self.select_seed_indices(indices)

    def select_seed_indices(self, indices: _TensorIndex) -> "IKSolverResult":
        """Select seed columns, retaining a ``[batch, selected_seed, ...]`` layout.

        ``indices`` can be a common one-dimensional selection or one row of
        indices for every batch.  The latter is useful for selecting each
        batch's best seed while preserving downstream result rank contracts.
        """
        shape = self.seed_shape
        if shape is None:
            raise ValueError("a [batch, seed] result tensor is required for seed selection")
        batch, seeds = shape
        tensor = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).expand(batch, -1)
        if tensor.ndim != 2 or tensor.shape[0] != batch:
            raise ValueError("indices must have shape [selected_seed] or [batch, selected_seed]")
        if tensor.numel() and (tensor.min() < 0 or tensor.max() >= seeds):
            raise IndexError("seed index is outside the available seed range")
        values = {
            item.name: _gather_seed_value(getattr(self, item.name), tensor, shape)
            for item in fields(self)
        }
        values["num_seeds"] = tensor.shape[1]
        return type(self)(**values)

    def best_seed(self) -> "IKSolverResult":
        """Select one lowest-cost seed per batch while preserving a seed axis."""
        if self.seed_cost is None or self.seed_cost.ndim != 2:
            raise ValueError("seed_cost with shape [batch, seed] is required")
        return self.select_seed_indices(self.seed_cost.argsort(dim=1, stable=True)[:, :1])

    def select_batch(self, indices: _TensorIndex) -> "IKSolverResult":
        """Return selected batch entries, preserving a leading batch axis."""
        batch = self.batch_size or (int(self.success.shape[0]) if self.success.ndim else 0)
        if batch < 1:
            raise ValueError("batch_size must be positive for batch selection")
        tensor = torch.as_tensor(indices, device=self.device, dtype=torch.long)
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
        values["batch_size"] = tensor.numel()
        return type(self)(**values)

    def process_metrics_and_rank_seeds(self) -> None:
        """Update deterministic cost ranking and enforce known feasibility flags."""
        if self.seed_cost is not None:
            if self.seed_cost.ndim != 2:
                raise ValueError("seed_cost must have shape [batch, seed]")
            self.seed_rank = self.seed_cost.argsort(dim=1, stable=True)
        if self.feasible is not None:
            if self.feasible.shape != self.success.shape:
                raise ValueError("feasible and success must have identical shapes")
            self.success = self.success & self.feasible

    def copy_successful_solutions(self, other: "IKSolverResult") -> None:
        """Merge successful entries and copy every materialized joint channel."""
        super().copy_successful_solutions(other)
        if self.js_solution is None or other.js_solution is None:
            return
        mask = other.success
        for name in ("velocity", "acceleration", "jerk", "dt", "knot", "knot_dt"):
            dst, src = getattr(self.js_solution, name, None), getattr(other.js_solution, name, None)
            if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor) and dst.shape == src.shape:
                if dst.ndim >= mask.ndim and dst.shape[: mask.ndim] == mask.shape:
                    dst[mask] = src[mask]


__all__ = ["IKSolverResult"]
