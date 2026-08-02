"""Pinned graph-planner result model with portable tensor lifecycle helpers.

The pinned V2 declaration is intentionally a small dataclass.  In practice a
graph result is often saved between planner calls, selected out of a batched
query, or handed to a trajectory optimiser.  CUDA's graph buffers are not
part of the result contract, so those useful operations can be implemented
entirely with ordinary PyTorch tensors on CPU or MPS.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

from curobo._src.types.device_cfg import DeviceCfg


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else copy.copy(value)


def _detach_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    return value


def _to_value(value: Any, device_cfg: DeviceCfg) -> Any:
    if isinstance(value, torch.Tensor):
        dtype = device_cfg.dtype if value.is_floating_point() else value.dtype
        return value.to(device=device_cfg.device, dtype=dtype)
    if isinstance(value, dict):
        return {key: _to_value(item, device_cfg) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_value(item, device_cfg) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_value(item, device_cfg) for item in value)
    return value


def _select_value(value: Any, indices: torch.Tensor, batch_size: int) -> Any:
    """Select tensor metadata only when it owns the result's leading batch."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[indices].clone()
        return value.clone()
    if isinstance(value, dict):
        return {key: _select_value(item, indices, batch_size) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) == batch_size:
            return [_clone_value(value[int(item)]) for item in indices.detach().cpu().tolist()]
        return [_select_value(item, indices, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_value(item, indices, batch_size) for item in value)
    if hasattr(value, "__dict__"):
        selected = copy.copy(value)
        for name, item in value.__dict__.items():
            setattr(selected, name, _select_value(item, indices, batch_size))
        return selected
    return value


@dataclass
class GraphPlannerResult:
    """Data class storing graph-planner information for a batch of queries.

    ``plan_waypoints`` deliberately remains a list because every query may
    have a different number of roadmap vertices.  :meth:`as_padded_waypoints`
    gives batched consumers a device-resident tensor representation without
    pretending that a raw CUDA graph/path buffer exists.
    """

    success: torch.Tensor
    plan_waypoints: Optional[List[Union[torch.Tensor, None]]] = None
    interpolated_waypoints: Optional[torch.Tensor] = None
    joint_names: Optional[List[str]] = None
    path_length: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    valid_query: bool = True
    debug_info: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, torch.Tensor):
            raise TypeError("success must be a torch.Tensor")
        if self.success.dtype != torch.bool or self.success.ndim != 1:
            raise ValueError("success must be a one-dimensional bool tensor")
        batch_size = int(self.success.shape[0])
        if self.plan_waypoints is not None:
            if len(self.plan_waypoints) != batch_size:
                raise ValueError("plan_waypoints must have one entry per batch item")
            for path in self.plan_waypoints:
                if path is None:
                    continue
                if not isinstance(path, torch.Tensor) or path.ndim != 2:
                    raise ValueError("each plan waypoint entry must be a 2D tensor or None")
                if path.device != self.success.device:
                    raise ValueError("plan waypoint tensors must be on the success device")
        if self.interpolated_waypoints is not None:
            if not isinstance(self.interpolated_waypoints, torch.Tensor) or self.interpolated_waypoints.ndim != 3:
                raise ValueError("interpolated_waypoints must be a [batch, step, action] tensor")
            if self.interpolated_waypoints.shape[0] != batch_size:
                raise ValueError("interpolated_waypoints must have the same batch size as success")
            if self.interpolated_waypoints.device != self.success.device:
                raise ValueError("interpolated_waypoints must be on the success device")
        if self.path_length is not None:
            if not isinstance(self.path_length, torch.Tensor) or self.path_length.shape != (batch_size,):
                raise ValueError("path_length must be a tensor with shape [batch]")
            if self.path_length.device != self.success.device:
                raise ValueError("path_length must be on the success device")
        if not isinstance(self.valid_query, bool):
            raise TypeError("valid_query must be a bool")

    @property
    def batch_size(self) -> int:
        return int(self.success.shape[0])

    @property
    def device(self) -> torch.device:
        return self.success.device

    @property
    def num_success(self) -> int:
        return int(self.success.sum().item())

    @property
    def success_ratio(self) -> float:
        if self.batch_size == 0:
            return 0.0
        return float(self.success.to(dtype=torch.float32).mean().item())

    def any_success(self) -> bool:
        return bool(self.success.any().item())

    def all_success(self) -> bool:
        return bool(self.batch_size > 0 and self.success.all().item())

    def successful_paths(self) -> List[torch.Tensor]:
        """Return successful variable-length paths in deterministic batch order."""
        if self.plan_waypoints is None:
            return []
        return [
            path for path, succeeded in zip(self.plan_waypoints, self.success.tolist())
            if succeeded and path is not None
        ]

    def as_padded_waypoints(self, pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([batch, max_waypoints, action_dim], valid_mask)``.

        Failed queries and absent paths have an all-false mask.  The padded
        representation intentionally preserves autograd for non-empty paths
        and keeps allocation on the result device.
        """
        paths = self.plan_waypoints or []
        concrete = [path for path in paths if path is not None]
        if concrete:
            action_dim = int(concrete[0].shape[-1])
            if any(path.shape[-1] != action_dim for path in concrete):
                raise ValueError("all plan waypoint tensors must have the same action dimension")
            max_waypoints = max(int(path.shape[0]) for path in concrete)
            dtype = concrete[0].dtype
        elif self.interpolated_waypoints is not None:
            action_dim = int(self.interpolated_waypoints.shape[-1])
            max_waypoints, dtype = 0, self.interpolated_waypoints.dtype
        else:
            action_dim, max_waypoints, dtype = 0, 0, torch.float32
        padded = torch.full(
            (self.batch_size, max_waypoints, action_dim), pad_value,
            device=self.device, dtype=dtype,
        )
        valid = torch.zeros((self.batch_size, max_waypoints), device=self.device, dtype=torch.bool)
        for index, path in enumerate(paths):
            if path is None:
                continue
            count = int(path.shape[0])
            padded[index, :count] = path
            valid[index, :count] = True
        return padded, valid

    def clone(self) -> "GraphPlannerResult":
        return type(self)(
            _clone_value(self.success), _clone_value(self.plan_waypoints),
            _clone_value(self.interpolated_waypoints), _clone_value(self.joint_names),
            _clone_value(self.path_length), self.solve_time, self.valid_query,
            _clone_value(self.debug_info),
        )

    def detach(self) -> "GraphPlannerResult":
        output = self.clone()
        output.success = _detach_value(output.success)
        output.plan_waypoints = _detach_value(output.plan_waypoints)
        output.interpolated_waypoints = _detach_value(output.interpolated_waypoints)
        output.path_length = _detach_value(output.path_length)
        output.debug_info = _detach_value(output.debug_info)
        return output

    def to(self, device_cfg: DeviceCfg | torch.device | str) -> "GraphPlannerResult":
        """Move all tensor payloads while retaining boolean/index dtypes."""
        if not isinstance(device_cfg, DeviceCfg):
            dtype = self.path_length.dtype if self.path_length is not None else torch.float32
            device_cfg = DeviceCfg(torch.device(device_cfg), dtype)
        return type(self)(
            _to_value(self.success, device_cfg), _to_value(self.plan_waypoints, device_cfg),
            _to_value(self.interpolated_waypoints, device_cfg), _clone_value(self.joint_names),
            _to_value(self.path_length, device_cfg), self.solve_time, self.valid_query,
            _to_value(self.debug_info, device_cfg),
        )

    def __getitem__(self, index: int | slice | torch.Tensor) -> "GraphPlannerResult":
        """Select result problems while preserving a one-dimensional batch axis."""
        indices = torch.arange(self.batch_size, device=self.device)[index]
        if indices.ndim == 0:
            indices = indices.reshape(1)
        success = self.success[indices]
        selected_paths = None if self.plan_waypoints is None else [
            _clone_value(self.plan_waypoints[int(item)]) for item in indices.detach().cpu().tolist()
        ]
        interpolated = None if self.interpolated_waypoints is None else self.interpolated_waypoints[indices].clone()
        lengths = None if self.path_length is None else self.path_length[indices].clone()
        output = type(self)(
            success.clone(), selected_paths, interpolated, _clone_value(self.joint_names), lengths,
            self.solve_time, self.valid_query,
            _select_value(self.debug_info, indices, self.batch_size),
        )
        return output


__all__ = ["GraphPlannerResult"]
