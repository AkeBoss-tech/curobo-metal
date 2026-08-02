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
from typing import Any, Optional

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


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

    def __getitem__(self, index: Any):
        """Select a problem batch while retaining all stage metadata."""
        if self.batch_size == 0:
            raise IndexError("cannot index a result without a batched success tensor")
        output = self.clone()
        for name, value in tuple(output.__dict__.items()):
            setattr(output, name, _select_value(value, index, self.batch_size))
        return output


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

    def stage_success(self, stage: str) -> Optional[torch.Tensor]:
        """Return the requested ``approach``, ``grasp``, or ``lift`` status."""
        if stage not in {"approach", "grasp", "lift"}:
            raise ValueError("stage must be 'approach', 'grasp', or 'lift'")
        return getattr(self, f"{stage}_success")

    def stage_trajectory(self, stage: str, *, interpolated: bool = False) -> Optional[JointState]:
        """Return a stage trajectory without forcing CUDA interpolation buffers."""
        if stage not in {"approach", "grasp", "lift"}:
            raise ValueError("stage must be 'approach', 'grasp', or 'lift'")
        name = f"{stage}_{'interpolated_' if interpolated else ''}trajectory"
        return getattr(self, name)


__all__ = ["MotionPlannerResult", "GraspPlanResult"]
