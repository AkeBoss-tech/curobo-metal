"""Batched, device-resident linear PRM steering.

The upstream connector launches Warp/CUDA collision queries over a regularly
spaced C-space segment.  Its public behaviour is small and useful without
those backends: sample a weighted straight line and return the sample directly
before the first infeasible one.  This implementation keeps the sampling,
first-collision tie rule and indexed-node layout while delegating feasibility
to the configured production collision/rollout callback.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class LinearConnector:
    """Steer indexed roadmap rows along sampled C-space line segments.

    Input rows use ``[action..., graph_index]``.  The graph index is metadata,
    not a configuration-space coordinate; returned rows carry a zero padding
    index just as pinned cuRobo does, and :class:`GraphConstructor` resolves it
    when registering the candidate in the roadmap.

    Continuous collision detection, Warp kernels, CUDA graph capture, and raw
    preallocated CUDA buffer APIs remain explicitly unavailable.  Feasibility
    is a discrete sampled query at ``cspace_similarity_threshold`` resolution.
    """

    def __init__(self, config: PRMGraphPlannerCfg, device_cfg: Optional[DeviceCfg] = None):
        self.config = config
        self.device_cfg = config.device_cfg if device_cfg is None else device_cfg
        self.cspace_similarity_threshold = float(config.cspace_similarity_threshold)
        if not math.isfinite(self.cspace_similarity_threshold) or self.cspace_similarity_threshold <= 0:
            raise ValueError("cspace_similarity_threshold must be a finite positive value")
        steer_buffer_size = int(config.steer_buffer_size)
        if steer_buffer_size < 2:
            raise ValueError("steer_buffer_size must be at least 2")

        # Keeping these buffers public matches the useful upstream inspection
        # surface.  They are normal tensors, never CUDA graph-owned storage.
        self._preallocated_steer_buffer = torch.arange(
            steer_buffer_size, **self.device_cfg.as_torch_dict()
        )
        self._preallocated_idx_buffer: Optional[torch.Tensor] = None
        self._node_idx_padding_buffer = torch.zeros(
            (1,), **self.device_cfg.as_torch_dict()
        )
        self.action_dim: Optional[int] = None
        self.cspace_distance_weight: Optional[torch.Tensor] = None
        self._check_feasibility_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

        # Historical portable aliases; retain these so code written against an
        # earlier curobo-metal release does not silently use stale state.
        self.distance_weight: Optional[torch.Tensor] = None
        self.check_feasibility_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self.preallocated_idx_buffer: Optional[torch.Tensor] = None

    def set_dependencies(
        self,
        action_dim: int,
        cspace_distance_weight: torch.Tensor,
        check_feasibility_fn,
        preallocated_idx_buffer: torch.Tensor,
    ):
        """Install the rollout/collision callback and C-space metric.

        No data is copied: all dependencies must already be on this
        connector's device.  This prevents a CPU feasibility callback from
        accidentally turning an MPS graph build into a host round-trip.
        """
        if not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if not isinstance(cspace_distance_weight, torch.Tensor):
            raise TypeError("cspace_distance_weight must be a tensor")
        if cspace_distance_weight.ndim != 1 or cspace_distance_weight.numel() != action_dim:
            raise ValueError("cspace_distance_weight must have shape [action_dim]")
        if not self.device_cfg.is_same_torch_device(cspace_distance_weight.device):
            raise ValueError("cspace_distance_weight must be on the configured device")
        if cspace_distance_weight.dtype != self.device_cfg.dtype:
            raise ValueError("cspace_distance_weight must use the configured dtype")
        if not bool(torch.isfinite(cspace_distance_weight).all().item()):
            raise ValueError("cspace_distance_weight must contain finite values")
        if bool((cspace_distance_weight < 0).any().item()):
            raise ValueError("cspace_distance_weight must be nonnegative")
        if not callable(check_feasibility_fn):
            raise TypeError("check_feasibility_fn must be callable")
        if not isinstance(preallocated_idx_buffer, torch.Tensor):
            raise TypeError("preallocated_idx_buffer must be a tensor")
        if preallocated_idx_buffer.ndim != 1:
            raise ValueError("preallocated_idx_buffer must be one-dimensional")
        if not self.device_cfg.is_same_torch_device(preallocated_idx_buffer.device):
            raise ValueError("preallocated_idx_buffer must be on the configured device")
        if preallocated_idx_buffer.numel() < 2:
            raise ValueError("preallocated_idx_buffer must contain at least two entries")

        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self._check_feasibility_fn = check_feasibility_fn
        self._preallocated_idx_buffer = preallocated_idx_buffer
        self.distance_weight = cspace_distance_weight
        self.check_feasibility_fn = check_feasibility_fn
        self.preallocated_idx_buffer = preallocated_idx_buffer

    def _require_dependencies(self) -> tuple[int, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        if self.action_dim is None or self.cspace_distance_weight is None or self._check_feasibility_fn is None:
            raise RuntimeError("set_dependencies must be called before steering")
        return self.action_dim, self.cspace_distance_weight, self._check_feasibility_fn

    def _validate_nodes(self, nodes: torch.Tensor, *, name: str, action_dim: int) -> None:
        if not isinstance(nodes, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if nodes.ndim != 2 or nodes.shape[1] != action_dim + 1:
            raise ValueError(f"{name} must have shape [B, {action_dim + 1}]")
        if not self.device_cfg.is_same_torch_device(nodes.device):
            raise ValueError(f"{name} must be on {self.device_cfg.device}")
        if nodes.dtype != self.device_cfg.dtype:
            raise ValueError(f"{name} must use {self.device_cfg.dtype}")
        if not bool(torch.isfinite(nodes).all().item()):
            raise ValueError(f"{name} must contain finite values")

    def _step_count(self, start_actions: torch.Tensor, desired_actions: torch.Tensor) -> int:
        """Return the common sample count used for every row in a batch."""
        assert self.cspace_distance_weight is not None
        weighted_delta = (desired_actions - start_actions).abs() * self.cspace_distance_weight
        maximum = float(weighted_delta.amax().item()) if weighted_delta.numel() else 0.0
        # Pinned cuRobo uses a shared horizon and includes both endpoints.
        intervals = max(1, int(math.ceil(maximum / self.cspace_similarity_threshold)))
        count = intervals + 1
        if count > self._preallocated_steer_buffer.numel():
            raise ValueError(
                "steering segment exceeds steer_buffer_size; increase steer_buffer_size "
                "or cspace_similarity_threshold"
            )
        return count

    def _compute_steering_line_points(
        self, start_nodes: torch.Tensor, desired_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Return ``[B, H, action_dim]`` endpoint-inclusive samples."""
        action_dim, _, _ = self._require_dependencies()
        self._validate_nodes(start_nodes, name="start_nodes", action_dim=action_dim)
        self._validate_nodes(desired_nodes, name="desired_nodes", action_dim=action_dim)
        if start_nodes.shape[0] != desired_nodes.shape[0]:
            raise ValueError("start_nodes and desired_nodes must have the same batch size")
        if start_nodes.shape[0] == 0:
            return start_nodes.new_empty((0, 0, action_dim))

        starts = start_nodes[:, :action_dim]
        desired = desired_nodes[:, :action_dim]
        count = self._step_count(starts, desired)
        coefficients = self._preallocated_steer_buffer[:count] / (count - 1)
        return starts[:, None, :] + coefficients[None, :, None] * (desired - starts)[:, None, :]

    def _find_last_feasible_point(self, line_vec: torch.Tensor, mask: torch.Tensor, h: int) -> torch.Tensor:
        """Return the sample preceding the first infeasible point per row."""
        action_dim, _, _ = self._require_dependencies()
        if line_vec.ndim != 3 or line_vec.shape[1] != h or line_vec.shape[2] != action_dim:
            raise ValueError("line_vec must have shape [B, H, action_dim]")
        if mask.shape != line_vec.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool shape [B, H]")
        if h < 1:
            return line_vec.new_empty((line_vec.shape[0], action_dim + 1))

        # ``argmax`` is a first-index reduction.  Guard all-feasible rows so
        # their endpoint is selected instead of argmax's default zero.
        first_bad = (~mask).to(torch.int64).argmax(dim=1)
        has_bad = (~mask).any(dim=1)
        indices = torch.where(has_bad, (first_bad - 1).clamp_min(0), first_bad.new_full(first_bad.shape, h - 1))
        batch = torch.arange(line_vec.shape[0], device=line_vec.device)
        actions = line_vec[batch, indices]
        return torch.cat((actions, self._node_idx_padding_buffer.expand(actions.shape[0], 1)), dim=-1)

    def steer_until_infeasible(
        self, start_nodes: torch.Tensor, desired_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Return last discretely feasible row for each paired segment.

        The feasibility callback receives an ``[B * H, action_dim]`` tensor
        and must return boolean shape ``[B * H]`` on the same device.
        """
        action_dim, _, feasibility = self._require_dependencies()
        self._validate_nodes(start_nodes, name="start_nodes", action_dim=action_dim)
        self._validate_nodes(desired_nodes, name="desired_nodes", action_dim=action_dim)
        if start_nodes.shape[0] != desired_nodes.shape[0]:
            raise ValueError("start_nodes and desired_nodes must have the same batch size")
        if start_nodes.shape[0] == 0:
            return start_nodes.new_empty((0, action_dim + 1))
        line = self._compute_steering_line_points(start_nodes, desired_nodes)
        flat = line.reshape(-1, action_dim)
        mask = feasibility(flat)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("check_feasibility_fn must return a tensor")
        if mask.device != flat.device:
            raise ValueError("check_feasibility_fn must preserve the input device")
        if mask.dtype != torch.bool or mask.shape != (flat.shape[0],):
            raise ValueError("check_feasibility_fn must return bool shape [B * H]")
        return self._find_last_feasible_point(line, mask.reshape(line.shape[:2]), line.shape[1])


__all__ = ["LinearConnector"]
