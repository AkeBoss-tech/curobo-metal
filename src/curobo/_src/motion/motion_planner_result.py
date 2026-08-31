"""Pinned high-level motion-planning result models.

The V2 dataclasses deliberately have a very small declared surface.  A real
motion-planning result, however, carries tensors and trajectories produced by
several solvers and is often retained across retries, batched calls, and
device changes.  The helpers in this module keep that useful lifecycle
portable: they operate on ordinary PyTorch tensors and :class:`JointState`
objects rather than CUDA-owned result buffers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, Optional, Sequence, Union

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


_BatchIndex = Union[int, slice, torch.Tensor, Sequence[int]]


def _clone_value(value: Any) -> Any:
    """Clone result payloads while preserving PyTorch's autograd graph."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, JointState):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _detach_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, JointState):
        # ``JointState.detach`` is intentionally in-place in the pinned
        # facade, so first give the result its own trajectory payload.
        return value.clone().detach()
    if isinstance(value, dict):
        return {key: _detach_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    return value


def _to_value(value: Any, device_cfg: DeviceCfg) -> Any:
    if isinstance(value, torch.Tensor):
        # Status/index tensors must retain their dtypes.  DeviceCfg.to_device
        # intentionally casts arbitrary tensors for configuration loading,
        # which is not appropriate for a result payload.
        return value.to(device=device_cfg.device, dtype=(
            device_cfg.dtype if value.is_floating_point() else value.dtype
        ))
    if isinstance(value, JointState):
        return value.to(device_cfg)
    if isinstance(value, dict):
        return {key: _to_value(item, device_cfg) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_value(item, device_cfg) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_value(item, device_cfg) for item in value)
    return value


def _select_value(value: Any, index: Any, batch_size: int) -> Any:
    """Select a leading problem batch where the payload owns one.

    Dynamic fields such as ``approach_result`` are solver result objects.  A
    shallow copy plus recursive attribute selection makes a selected grasp
    result self-contained without assuming a CUDA-specific result ABI.
    """
    if isinstance(value, torch.Tensor):
        return value[index] if value.ndim > 0 and value.shape[0] == batch_size else value.clone()
    if isinstance(value, JointState):
        return value[index] if value.position.ndim > 0 and value.position.shape[0] == batch_size else value.clone()
    if isinstance(value, dict):
        return {key: _select_value(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) == batch_size:
            return _clone_value(value[index])
        return [_select_value(item, index, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_value(item, index, batch_size) for item in value)
    if hasattr(value, "__dict__"):
        selected = copy.copy(value)
        for name, item in value.__dict__.items():
            setattr(selected, name, _select_value(item, index, batch_size))
        return selected
    return value


def _normalise_batch_index(
    index: _BatchIndex, *, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Return a valid rank-one index without dropping the batch dimension."""
    if isinstance(index, slice):
        output = torch.arange(batch_size, device=device, dtype=torch.long)[index]
    elif isinstance(index, int):
        normalised = index + batch_size if index < 0 else index
        if normalised < 0 or normalised >= batch_size:
            raise IndexError("motion planner result batch index is out of range")
        output = torch.tensor([normalised], device=device, dtype=torch.long)
    elif isinstance(index, torch.Tensor):
        output = index.to(device=device)
    elif isinstance(index, (list, tuple)):
        output = torch.as_tensor(index, device=device)
    else:
        raise TypeError("batch index must be an int, slice, sequence, or tensor")

    if output.ndim == 0:
        output = output.reshape(1)
    if output.ndim != 1:
        raise IndexError("motion planner result batch index must be one-dimensional")
    if output.dtype == torch.bool:
        if output.numel() != batch_size:
            raise IndexError("boolean batch index must have batch_size elements")
        return output
    if output.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("batch index tensor must be integer or boolean")
    output = output.to(dtype=torch.long)
    output = torch.where(output < 0, output + batch_size, output)
    if output.numel() and (output.min() < 0 or output.max() >= batch_size):
        raise IndexError("motion planner result batch index is out of range")
    return output


class _PortableResultMixin:
    """Portable lifecycle shared by motion and grasp result containers."""

    def __post_init__(self) -> None:
        success = getattr(self, "success", None)
        if success is not None:
            if not isinstance(success, torch.Tensor):
                raise TypeError("success must be a torch.Tensor or None")
            if success.dtype is not torch.bool:
                raise TypeError("success must have dtype torch.bool")
        status = getattr(self, "status", None)
        if status is not None and not isinstance(status, str):
            raise TypeError("status must be a string or None")

    def validate(self):
        """Validate materialised batch, stage, and device invariants.

        V2 leaves these payloads as a permissive dataclass so partially
        produced plans can be returned on failure.  This opt-in check is for
        applications that retain or combine results and need to detect a
        mixed-device/shape payload before using it.  It only covers ordinary
        tensor and :class:`JointState` values; CUDA graph/result-buffer ABI
        objects remain outside the portable contract.
        """
        success = getattr(self, "success", None)
        if success is not None:
            if not isinstance(success, torch.Tensor):
                raise TypeError("success must be a torch.Tensor or None")
            if success.dtype is not torch.bool:
                raise TypeError("success must have dtype torch.bool")

        status = getattr(self, "status", None)
        if status is not None and not isinstance(status, str):
            raise TypeError("status must be a string or None")

        planning_time = getattr(self, "planning_time", None)
        if planning_time is not None:
            if not isinstance(planning_time, Real):
                raise TypeError("planning_time must be a real scalar")
            if planning_time < 0:
                raise ValueError("planning_time must be non-negative")

        if success is None:
            return self
        batch = self.batch_size
        for name in ("approach_success", "grasp_success", "lift_success"):
            value = getattr(self, name, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.dtype is not torch.bool:
                raise TypeError(f"{name} must be a torch.bool tensor or None")
            if value.device != success.device:
                raise ValueError(f"{name} must share the success tensor device")
            if value.shape != success.shape:
                raise ValueError(f"{name} must have the same shape as success")

        for name in (
            "approach_trajectory", "approach_interpolated_trajectory",
            "grasp_trajectory", "grasp_interpolated_trajectory",
            "lift_trajectory", "lift_interpolated_trajectory",
        ):
            value = getattr(self, name, None)
            if value is None:
                continue
            if not isinstance(value, JointState):
                raise TypeError(f"{name} must be a JointState or None")
            if value.device != success.device:
                raise ValueError(f"{name} must share the success tensor device")
            if batch and (value.position.ndim < 1 or value.position.shape[0] != batch):
                raise ValueError(f"{name} must have the result batch as its leading dimension")

        for name in (
            "approach_trajectory_dt", "grasp_trajectory_dt", "lift_trajectory_dt",
            "approach_interpolated_last_tstep", "grasp_interpolated_last_tstep",
            "lift_interpolated_last_tstep", "goalset_index",
        ):
            value = getattr(self, name, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor or None")
            if value.device != success.device:
                raise ValueError(f"{name} must share the success tensor device")
            if value.ndim and batch and value.shape[0] != batch:
                raise ValueError(f"{name} must have the result batch as its leading dimension")
        return self

    @property
    def batch_size(self) -> int:
        """Number of planning problems represented by this result."""
        success = getattr(self, "success", None)
        return 0 if success is None or success.ndim == 0 else int(success.shape[0])

    @property
    def device(self) -> Optional[torch.device]:
        """Device of the primary result payload, if one is available."""
        success = getattr(self, "success", None)
        if isinstance(success, torch.Tensor):
            return success.device
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor):
                return value.device
            if isinstance(value, JointState):
                return value.device
        return None

    @property
    def success_per_problem(self) -> Optional[torch.Tensor]:
        """Collapse seed rank into one deterministic success value per problem."""
        success = getattr(self, "success", None)
        if success is None or success.ndim <= 1:
            return success
        return success.any(dim=-1)

    @property
    def num_success(self) -> int:
        success = self.success_per_problem
        return 0 if success is None else int(success.sum().item())

    @property
    def success_ratio(self) -> float:
        success = self.success_per_problem
        if success is None or success.numel() == 0:
            return 0.0
        return float(success.to(dtype=torch.float32).mean().item())

    def any_success(self) -> bool:
        success = getattr(self, "success", None)
        return bool(success is not None and success.any().item())

    def all_success(self) -> bool:
        success = self.success_per_problem
        return bool(success is not None and success.numel() > 0 and success.all().item())

    @property
    def num_failures(self) -> int:
        """Number of failed planning problems after collapsing seed rank."""
        success = self.success_per_problem
        return 0 if success is None else int((~success).sum().item())

    @property
    def failure_mask(self) -> Optional[torch.Tensor]:
        """Per-problem boolean failure mask, or ``None`` before planning."""
        success = self.success_per_problem
        return None if success is None else ~success

    def clone(self):
        """Return an independent result, including dynamically attached stages."""
        output = type(self)(**{
            item.name: _clone_value(getattr(self, item.name)) for item in fields(self)
        })
        declared = {item.name for item in fields(self)}
        for name, value in self.__dict__.items():
            if name not in declared:
                setattr(output, name, _clone_value(value))
        return output

    def detach(self):
        """Return a result whose tensor payloads are detached from autograd."""
        output = self.clone()
        for name, value in tuple(output.__dict__.items()):
            setattr(output, name, _detach_value(value))
        return output

    def to(self, device_cfg: DeviceCfg | torch.device | str):
        """Move tensor and trajectory payloads without changing index dtypes."""
        if not isinstance(device_cfg, DeviceCfg):
            dtype = None
            for value in self.__dict__.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    dtype = value.dtype
                    break
                if isinstance(value, JointState):
                    dtype = value.dtype
                    break
            device_cfg = DeviceCfg(torch.device(device_cfg), dtype or torch.float32)
        output = self.clone()
        for name, value in tuple(output.__dict__.items()):
            setattr(output, name, _to_value(value, device_cfg))
        return output

    def select_batch(self, index: _BatchIndex):
        """Select result rows while preserving a leading problem batch axis.

        Unlike ``result[index]``, selecting an integer produces one problem
        with shape ``[1, ...]``.  That makes the selected value immediately
        reusable by batched portable motion-planning APIs.
        """
        if self.batch_size == 0:
            raise IndexError("cannot select a batch from a result without a batched success tensor")
        device = self.device if self.device is not None else torch.device("cpu")
        indices = _normalise_batch_index(index, batch_size=self.batch_size, device=device)
        output = self.clone()
        for name, value in tuple(output.__dict__.items()):
            setattr(output, name, _select_value(value, indices, self.batch_size))
        return output

    def successful(self):
        """Return successful problem rows with a preserved batch dimension."""
        success = self.success_per_problem
        if success is None or success.ndim == 0:
            raise ValueError("a batched success tensor is required to filter successful results")
        return self.select_batch(success)

    def __getitem__(self, index: Any):
        """Select a problem batch while retaining all stage metadata."""
        if self.batch_size == 0:
            raise IndexError("cannot index a result without a batched success tensor")
        output = self.clone()
        for name, value in tuple(output.__dict__.items()):
            setattr(output, name, _select_value(value, index, self.batch_size))
        return output

    def stage_success(self, stage: str) -> Optional[torch.Tensor]:
        if stage not in {"approach", "grasp", "lift"}:
            raise ValueError("stage must be 'approach', 'grasp', or 'lift'")
        return getattr(self, f"{stage}_success")

    def stage_trajectory(self, stage: str, *, interpolated: bool = False) -> Optional[JointState]:
        if stage not in {"approach", "grasp", "lift"}:
            raise ValueError("stage must be 'approach', 'grasp', or 'lift'")
        name = f"{stage}_{'interpolated_' if interpolated else ''}trajectory"
        return getattr(self, name)


@dataclass
class MotionPlannerResult(_PortableResultMixin):
    """Result of a motion-planning operation."""

    success: Optional[torch.Tensor] = None


@dataclass
class GraspPlanResult(_PortableResultMixin):
    """Result of a grasp planning operation."""

    success: Optional[torch.Tensor] = None
    approach_success: Optional[torch.Tensor] = None
    grasp_success: Optional[torch.Tensor] = None
    lift_success: Optional[torch.Tensor] = None
    approach_trajectory: Optional[JointState] = None
    approach_trajectory_dt: Optional[torch.Tensor] = None
    approach_interpolated_trajectory: Optional[JointState] = None
    grasp_trajectory: Optional[JointState] = None
    grasp_trajectory_dt: Optional[torch.Tensor] = None
    grasp_interpolated_trajectory: Optional[JointState] = None
    lift_trajectory: Optional[JointState] = None
    lift_trajectory_dt: Optional[torch.Tensor] = None
    lift_interpolated_trajectory: Optional[JointState] = None
    approach_interpolated_last_tstep: Optional[torch.Tensor] = None
    grasp_interpolated_last_tstep: Optional[torch.Tensor] = None
    lift_interpolated_last_tstep: Optional[torch.Tensor] = None
    status: Optional[str] = None
    planning_time: float = 0.0
    goalset_index: Optional[torch.Tensor] = None

__all__ = ["MotionPlannerResult", "GraspPlanResult"]
