"""Deterministic shortcut pruning for portable PRM paths.

The pinned implementation batches every ordered-forward pair of vertices in a
path, asks the graph constructor to collision-check and register those edges,
and then reruns the graph shortest-path query.  The useful semantics do not
depend on CUDA or Warp, so this module retains that lifecycle for CPU and MPS
tensors.  Collision checking itself remains delegated to the installed graph
constructor callback; this is not an analytic CCD implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, List, Optional, Tuple

import torch

from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg


class PathPruner:
    """Register deterministic all-forward shortcuts and rerun graph search.

    Nodes in a path are integer indices into the supplied indexed roadmap
    buffer.  For a path of length ``N`` the candidate batch contains
    ``N * (N + 1) // 2`` pairs, including self and adjacent pairs, exactly as
    cuRoboV2 does.  Including existing edges is intentional: it makes a
    pruning pass independent of whether those edge records were retained by a
    preceding graph-generation lifecycle.

    Raw CUDA graph capture, Warp edge kernels and analytic continuous
    collision detection are intentionally outside this portable facade.
    """

    def __init__(self, config: PRMGraphPlannerCfg, device_cfg: Optional[DeviceCfg] = None):
        self.config = config
        self.device_cfg = config.device_cfg if device_cfg is None else device_cfg
        self.action_dim: Optional[int] = None
        self.cspace_distance_weight: Optional[torch.Tensor] = None
        self._preallocated_node_buffer: Optional[torch.Tensor] = None
        self._steer_and_register_edges_fn: Optional[Callable[..., Any]] = None
        self._find_path_for_index_pairs_fn: Optional[Callable[..., Any]] = None

        # Retain the earlier portable names for source compatibility.
        self.distance_weight: Optional[torch.Tensor] = None
        self.node_buffer: Optional[torch.Tensor] = None
        self.steer: Optional[Callable[..., Any]] = None
        self.find_paths: Optional[Callable[..., Any]] = None

    def set_dependencies(
        self,
        action_dim: int,
        cspace_distance_weight: torch.Tensor,
        preallocated_node_buffer: torch.Tensor,
        steer_and_register_edges_fn: Callable[..., Any],
        find_path_for_index_pairs_fn: Callable[..., Any],
    ) -> None:
        """Install the roadmap and graph callbacks without copying tensors."""
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if not isinstance(cspace_distance_weight, torch.Tensor):
            raise TypeError("cspace_distance_weight must be a tensor")
        if cspace_distance_weight.ndim != 1 or cspace_distance_weight.shape[0] != action_dim:
            raise ValueError("cspace_distance_weight must have shape [action_dim]")
        if not self.device_cfg.is_same_torch_device(cspace_distance_weight.device):
            raise ValueError("cspace_distance_weight must be on the configured device")
        if cspace_distance_weight.dtype != self.device_cfg.dtype:
            raise ValueError("cspace_distance_weight must use the configured dtype")
        if not bool(torch.isfinite(cspace_distance_weight).all().item()):
            raise ValueError("cspace_distance_weight must contain finite values")
        if bool((cspace_distance_weight < 0).any().item()):
            raise ValueError("cspace_distance_weight must be nonnegative")
        if not isinstance(preallocated_node_buffer, torch.Tensor):
            raise TypeError("preallocated_node_buffer must be a tensor")
        if preallocated_node_buffer.ndim != 2 or preallocated_node_buffer.shape[1] != action_dim + 1:
            raise ValueError("preallocated_node_buffer must have shape [capacity, action_dim + 1]")
        if not self.device_cfg.is_same_torch_device(preallocated_node_buffer.device):
            raise ValueError("preallocated_node_buffer must be on the configured device")
        if preallocated_node_buffer.dtype != self.device_cfg.dtype:
            raise ValueError("preallocated_node_buffer must use the configured dtype")
        if not callable(steer_and_register_edges_fn):
            raise TypeError("steer_and_register_edges_fn must be callable")
        if not callable(find_path_for_index_pairs_fn):
            raise TypeError("find_path_for_index_pairs_fn must be callable")

        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self._preallocated_node_buffer = preallocated_node_buffer
        self._steer_and_register_edges_fn = steer_and_register_edges_fn
        self._find_path_for_index_pairs_fn = find_path_for_index_pairs_fn
        self.distance_weight = cspace_distance_weight
        self.node_buffer = preallocated_node_buffer
        self.steer = steer_and_register_edges_fn
        self.find_paths = find_path_for_index_pairs_fn

    def _require_dependencies(self) -> tuple[int, torch.Tensor, Callable[..., Any], Callable[..., Any]]:
        if (
            self.action_dim is None
            or self.cspace_distance_weight is None
            or self._preallocated_node_buffer is None
            or self._steer_and_register_edges_fn is None
            or self._find_path_for_index_pairs_fn is None
        ):
            raise RuntimeError("set_dependencies must be called before path pruning")
        return (
            self.action_dim,
            self._preallocated_node_buffer,
            self._steer_and_register_edges_fn,
            self._find_path_for_index_pairs_fn,
        )

    def _validate_paths(self, paths: Sequence[Sequence[int]], node_count: int) -> None:
        if not isinstance(paths, Sequence):
            raise TypeError("paths must be a sequence of index sequences")
        for path in paths:
            if not isinstance(path, Sequence):
                raise TypeError("every path must be a sequence of integer node indices")
            for index in path:
                if not isinstance(index, int) or isinstance(index, bool):
                    raise TypeError("path node indices must be integers")
                if index < 0 or index >= node_count:
                    raise ValueError("path node indices must be within the supplied node buffer")

    @staticmethod
    def _validate_terminal_indices(values: Sequence[int], *, name: str, count: int) -> None:
        if not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a sequence of integer node indices")
        if len(values) != count:
            raise ValueError(f"{name} must have one entry per path")
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must contain integer node indices")

    def prune_path_with_shortcuts(
        self, paths: List[List[int]], start_idx: List[int], goal_idx: List[int]
    ) -> Tuple[List[List[int]], List[float]]:
        """Collision-check shortcut candidates then return fresh shortest paths.

        Callback return values are deliberately passed through unchanged.  That
        preserves V2's per-query failure representation (``None`` path plus
        infinity length in the portable network finder) instead of replacing
        a disconnected query with the stale input path.
        """
        _, node_buffer, steer, find_paths = self._require_dependencies()
        self._validate_paths(paths, node_buffer.shape[0])
        self._validate_terminal_indices(start_idx, name="start_idx", count=len(paths))
        self._validate_terminal_indices(goal_idx, name="goal_idx", count=len(paths))
        if not paths:
            return [], []

        edge_set = self._prepare_edges_for_shortcuts(paths)
        # Empty paths have no candidates.  Avoid calling graph-construction
        # callbacks with unsupported rank-zero batches, but still perform the
        # requested re-query for each start/goal pair.
        if edge_set.shape[0]:
            steer(edge_set[:, 0], edge_set[:, 1], add_exact_node=False)
        return find_paths(start_idx, goal_idx, return_length=True)

    def _prepare_edges_for_shortcuts(self, paths: Sequence[Sequence[int]]) -> torch.Tensor:
        """Return V2-shaped ``[E, 2, action_dim + 1]`` ordered-forward pairs."""
        action_dim, node_buffer, _, _ = self._require_dependencies()
        self._validate_paths(paths, node_buffer.shape[0])
        total_edges = sum(len(path) * (len(path) + 1) // 2 for path in paths)
        edge_set = torch.empty(
            (total_edges, 2, action_dim + 1),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        offset = 0
        for path in paths:
            path_length = len(path)
            if path_length == 0:
                continue
            path_indices = torch.as_tensor(path, device=node_buffer.device, dtype=torch.int64)
            path_nodes = node_buffer.index_select(0, path_indices)
            for start in range(path_length):
                count = path_length - start
                edge_set[offset : offset + count, 0] = path_nodes[start]
                edge_set[offset : offset + count, 1] = path_nodes[start:]
                offset += count
        return edge_set


__all__ = ["PathPruner"]
