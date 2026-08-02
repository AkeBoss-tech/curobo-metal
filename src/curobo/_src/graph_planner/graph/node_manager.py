"""Portable, deterministic storage for pinned cuRobo PRM graph vertices.

The upstream implementation uses TorchScript helpers backed by CUDA rollout
buffers.  The observable manager contract is substantially smaller: vertices
are stored as ``[action..., index]`` rows, duplicate candidate vertices map to
the first matching roadmap vertex, and connection records use stable integer
indices.  This module preserves that contract using ordinary CPU/MPS PyTorch
operations.  It deliberately does *not* expose CUDA graph capture, Warp
neighbour kernels, or analytic continuous-collision detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch


def _same_device(left: torch.device, right: torch.device) -> bool:
    """Treat backend-default and explicit index zero as the same device."""
    return left.type == right.type and (left.index or 0) == (right.index or 0)


@dataclass
class ConnectedGraph:
    """A materialized PRM graph suitable for portable inspection/debugging."""

    nodes: torch.Tensor
    edges: torch.Tensor
    connectivity: torch.Tensor
    robot_state_nodes: Optional[Any] = None
    shortest_path_lengths: Optional[torch.Tensor] = None

    # Older metal releases exposed these aliases before matching the upstream
    # field names.  Keeping them avoids an unnecessary source break.
    @property
    def node_set(self) -> torch.Tensor:
        return self.nodes

    @property
    def edge_set(self) -> torch.Tensor:
        return self.edges

    def set_shortest_path_lengths(self, shortest_path_lengths: torch.Tensor) -> None:
        if not isinstance(shortest_path_lengths, torch.Tensor):
            raise TypeError("shortest_path_lengths must be a tensor")
        if shortest_path_lengths.ndim != 1:
            raise ValueError("shortest_path_lengths must have shape [N]")
        if shortest_path_lengths.device != self.nodes.device:
            raise ValueError("shortest_path_lengths must be on the graph device")
        if shortest_path_lengths.dtype != self.nodes.dtype:
            raise ValueError("shortest_path_lengths must use the graph dtype")
        self.shortest_path_lengths = shortest_path_lengths

    def get_node_distance(self) -> Optional[torch.Tensor]:
        """Return action rows augmented with aligned shortest-path lengths."""
        if self.shortest_path_lengths is None:
            return None
        count = min(self.nodes.shape[0], self.shortest_path_lengths.shape[0])
        return torch.cat((self.nodes[:count], self.shortest_path_lengths[:count, None]), dim=-1)


def jit_add_nodes_to_buffer(
    preallocated_node_buffer: torch.Tensor,
    new_nodes: torch.Tensor,
    used_node_count: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Add action rows and their deterministic indices to a preallocated buffer.

    The historical helper is TorchScript/CUDA annotated; using native tensor
    slicing preserves its useful callable contract on both CPU and MPS.
    """
    if preallocated_node_buffer.ndim != 2 or preallocated_node_buffer.shape[1] != action_dim + 1:
        raise ValueError("preallocated_node_buffer must have shape [capacity, action_dim + 1]")
    if new_nodes.ndim != 2 or new_nodes.shape[1] != action_dim:
        raise ValueError("new_nodes must have shape [N, action_dim]")
    if not _same_device(new_nodes.device, device) or new_nodes.dtype != dtype:
        raise ValueError("new_nodes must match the requested device and dtype")
    if not _same_device(preallocated_node_buffer.device, device) or preallocated_node_buffer.dtype != dtype:
        raise ValueError("preallocated_node_buffer must match the requested device and dtype")
    if used_node_count < 0 or used_node_count + new_nodes.shape[0] > preallocated_node_buffer.shape[0]:
        raise ValueError("graph node buffer capacity exceeded")
    end = used_node_count + new_nodes.shape[0]
    preallocated_node_buffer[used_node_count:end, :action_dim] = new_nodes
    preallocated_node_buffer[used_node_count:end, action_dim] = torch.arange(
        used_node_count, end, device=device, dtype=dtype
    )
    return preallocated_node_buffer


def jit_add_all_nodes_to_buffer(
    all_nodes: torch.Tensor,
    preallocated_node_buffer: torch.Tensor,
    used_node_count: int,
    action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    buffer = jit_add_nodes_to_buffer(
        preallocated_node_buffer,
        all_nodes,
        used_node_count,
        action_dim,
        all_nodes.device,
        all_nodes.dtype,
    )
    end = used_node_count + all_nodes.shape[0]
    return buffer, buffer[used_node_count:end], end


class GraphNodeManager:
    """Maintain a bounded, deterministic PRM node set on CPU or MPS."""

    def __init__(
        self,
        config: Any,
        distance_calculator: Any = None,
        graph_path_finder: Any = None,
        auxiliary_rollout: Any = None,
        device_cfg: Any = None,
    ):
        self.config = config
        self.distance_calculator = distance_calculator
        self.graph_path_finder = graph_path_finder
        self.auxiliary_rollout = auxiliary_rollout
        self.device_cfg = device_cfg or config.device_cfg
        self._action_dim = self._infer_action_dim()
        self._node_idx_padding_buffer = torch.zeros(
            (1,), **self.device_cfg.as_torch_dict()
        )
        self._preallocated_idx_buffer = torch.arange(
            int(config.steer_buffer_size), device=self.device_cfg.device, dtype=torch.int64
        )
        self._preallocated_node_buffer = torch.zeros(
            (int(config.max_nodes), self._action_dim + 1), **self.device_cfg.as_torch_dict()
        )
        self._used_node_count = 0

    def _infer_action_dim(self) -> int:
        bounds = getattr(self.config, "action_lower_bounds", None)
        if bounds is not None:
            if not isinstance(bounds, torch.Tensor) or bounds.ndim != 1:
                raise ValueError("action_lower_bounds must be a one-dimensional tensor")
            return int(bounds.numel())
        if self.auxiliary_rollout is not None and hasattr(self.auxiliary_rollout, "action_dim"):
            return int(self.auxiliary_rollout.action_dim)
        raise ValueError("GraphNodeManager requires action_lower_bounds or auxiliary_rollout.action_dim")

    def _validate_nodes(self, nodes: torch.Tensor, *, name: str = "nodes") -> None:
        if not isinstance(nodes, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if nodes.ndim != 2:
            raise ValueError(f"{name} must be a 2D tensor")
        if nodes.shape[1] != self.action_dim:
            raise ValueError(f"{name} must have action_dim={self.action_dim} columns")
        if not self.device_cfg.is_same_torch_device(nodes.device):
            raise ValueError(f"{name} must be on {self.device_cfg.device}")
        if nodes.dtype != self.device_cfg.dtype:
            raise ValueError(f"{name} must use {self.device_cfg.dtype}")
        if not bool(torch.isfinite(nodes).all().item()):
            raise ValueError(f"{name} must contain finite values")

    def _weighted_distance(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if self.distance_calculator is not None:
            return self.distance_calculator.calculate_weighted_distance(lhs, rhs)
        return torch.linalg.vector_norm(lhs - rhs, dim=-1)

    def _stable_unique(self, nodes: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """First-occurrence unique rows plus a mapping for every input row."""
        if nodes.shape[0] == 0:
            return nodes, torch.empty((0,), device=nodes.device, dtype=torch.int64)
        unique_rows: list[torch.Tensor] = []
        mapping: list[int] = []
        tolerance = float(threshold)
        for row in nodes:
            if not unique_rows:
                unique_rows.append(row)
                mapping.append(0)
                continue
            candidates = torch.stack(unique_rows)
            matches = self._weighted_distance(row[None, :], candidates)
            matching = torch.nonzero(matches <= tolerance, as_tuple=False)
            if matching.numel():
                mapping.append(int(matching[0, 0].item()))
            else:
                mapping.append(len(unique_rows))
                unique_rows.append(row)
        return torch.stack(unique_rows), torch.tensor(mapping, device=nodes.device, dtype=torch.int64)

    def add_nodes_to_buffer(self, new_nodes: torch.Tensor) -> None:
        """Append raw, already-selected action rows to the storage buffer."""
        self._validate_nodes(new_nodes, name="new_nodes")
        if self.n_nodes + new_nodes.shape[0] > self._preallocated_node_buffer.shape[0]:
            raise ValueError("graph node buffer capacity exceeded")
        self._preallocated_node_buffer = jit_add_nodes_to_buffer(
            self._preallocated_node_buffer,
            new_nodes,
            self.n_nodes,
            self.action_dim,
            self.device_cfg.device,
            self.device_cfg.dtype,
        )
        self._used_node_count += int(new_nodes.shape[0])

    def _edge_records(self) -> list[tuple[int, int, float]]:
        if self.graph_path_finder is None:
            return []
        if hasattr(self.graph_path_finder, "update_graph"):
            self.graph_path_finder.update_graph()
        if hasattr(self.graph_path_finder, "get_edges"):
            return [(int(a), int(b), float(w)) for a, b, w in self.graph_path_finder.get_edges()]
        graph = getattr(self.graph_path_finder, "_graph", {})
        return [
            (int(a), int(b), float(weight))
            for a, row in graph.items() for b, weight in row.items() if a < b
        ]

    def get_connected_graph(self) -> Optional[ConnectedGraph]:
        if self.n_nodes == 0:
            return None
        records = self._edge_records()
        nodes = self.valid_node_buffer[:, :self.action_dim]
        if records:
            connectivity = torch.tensor(records, device=nodes.device, dtype=nodes.dtype)
            indices = connectivity[:, :2].to(torch.int64)
            edges = torch.stack((nodes[indices[:, 0]], nodes[indices[:, 1]]), dim=1)
        else:
            connectivity = nodes.new_empty((0, 3))
            edges = nodes.new_empty((0, 2, self.action_dim))
        robot_state_nodes = None
        if self.auxiliary_rollout is not None and hasattr(self.auxiliary_rollout, "compute_state_from_action"):
            robot_state_nodes = self.auxiliary_rollout.compute_state_from_action(nodes.unsqueeze(1))
        return ConnectedGraph(nodes, edges, connectivity, robot_state_nodes)

    def add_nodes_to_roadmap(self, nodes: torch.Tensor, add_exact_node: bool = False) -> torch.Tensor:
        """Deduplicate candidate nodes and return each with its roadmap index."""
        self._validate_nodes(nodes)
        if self.n_nodes == 0:
            raise ValueError("used_node_count must not be 0; add initial exact nodes first")
        threshold = 0.0 if add_exact_node else float(self.config.cspace_similarity_threshold)
        if threshold < 0:
            raise ValueError("cspace_similarity_threshold must be nonnegative")
        unique_nodes, inverse = self._stable_unique(nodes, threshold)
        distances = self._weighted_distance(unique_nodes[:, None, :], self.valid_node_buffer[None, :, :self.action_dim])
        closest_distance, closest_idx = distances.min(dim=-1)
        exists = closest_distance <= threshold
        missing = unique_nodes[~exists]
        if self.n_nodes + missing.shape[0] > self._preallocated_node_buffer.shape[0]:
            raise ValueError("graph node buffer capacity exceeded")
        base = self.n_nodes
        if missing.shape[0]:
            self.add_nodes_to_buffer(missing)
        resolved = closest_idx.to(dtype=nodes.dtype)
        resolved[~exists] = torch.arange(
            base, base + missing.shape[0], device=nodes.device, dtype=nodes.dtype
        )
        resolved = resolved[inverse]
        result = torch.cat((nodes, resolved[:, None]), dim=-1)
        if self.graph_path_finder is not None:
            self.graph_path_finder.add_nodes(resolved.to(torch.int64).detach().cpu().tolist())
        return result

    def add_initial_exact_nodes_to_roadmap(self, nodes: torch.Tensor) -> torch.Tensor:
        self._validate_nodes(nodes)
        if self.n_nodes != 0:
            raise ValueError("used_node_count must be 0 before adding initial exact nodes")
        unique_nodes, inverse = self._stable_unique(nodes, 0.0)
        if unique_nodes.shape[0] > self._preallocated_node_buffer.shape[0]:
            raise ValueError("graph node buffer capacity exceeded")
        self.add_nodes_to_buffer(unique_nodes)
        indices = torch.arange(unique_nodes.shape[0], device=nodes.device, dtype=nodes.dtype)[inverse]
        result = torch.cat((nodes, indices[:, None]), dim=-1)
        if self.graph_path_finder is not None:
            self.graph_path_finder.add_nodes(indices.to(torch.int64).detach().cpu().tolist())
        return result

    def register_nodes_and_connections(
        self, node_set: torch.Tensor, start_nodes: torch.Tensor, add_exact_node: bool = False
    ) -> bool:
        """Add candidate vertices and connect each to its paired start vertex."""
        if node_set.ndim != 2 or node_set.shape[1] != self.action_dim + 1:
            raise ValueError("node_set must have shape [N, action_dim + 1]")
        if start_nodes.ndim != 2 or start_nodes.shape != node_set.shape:
            raise ValueError("start_nodes must match node_set shape [N, action_dim + 1]")
        if node_set.device != self.device_cfg.device or start_nodes.device != self.device_cfg.device:
            raise ValueError("node_set and start_nodes must be on the configured device")
        nodes_in_roadmap = self.add_nodes_to_roadmap(node_set[:, :self.action_dim], add_exact_node)
        if self.graph_path_finder is not None:
            weights = self._weighted_distance(start_nodes[:, None, :self.action_dim], nodes_in_roadmap[:, None, :self.action_dim]).reshape(-1)
            edges = [
                [int(start_nodes[index, self.action_dim].item()), int(nodes_in_roadmap[index, self.action_dim].item()), float(weights[index].item())]
                for index in range(nodes_in_roadmap.shape[0])
            ]
            self.graph_path_finder.add_edges(edges)
        return True

    def get_nodes_in_path(self, path_list: List[Union[List[int], torch.Tensor, None]]):
        paths = []
        for path in path_list:
            if path is None:
                paths.append(None)
                continue
            index = torch.as_tensor(path, device=self.device_cfg.device, dtype=torch.int64)
            if index.ndim != 1 or bool(((index < 0) | (index >= self.n_nodes)).any().item()):
                raise ValueError("path must contain valid one-dimensional node indices")
            paths.append(self.valid_node_buffer[index, :self.action_dim])
        return paths

    def reset_buffer(self) -> None:
        self._preallocated_node_buffer.zero_()
        self._used_node_count = 0
        self._default_joint_position_feasible = None
        self.reset_graph_path_finder()

    def reset_graph_path_finder(self) -> None:
        if self.graph_path_finder is not None:
            self.graph_path_finder.reset_graph()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_nodes(self) -> int:
        return self._used_node_count

    @property
    def node_idx_padding_buffer(self) -> torch.Tensor:
        return self._node_idx_padding_buffer

    @property
    def preallocated_node_buffer(self) -> torch.Tensor:
        return self._preallocated_node_buffer

    @property
    def valid_node_buffer(self) -> torch.Tensor:
        return self._preallocated_node_buffer[:self.n_nodes]


__all__ = [
    "ConnectedGraph", "GraphNodeManager", "jit_add_all_nodes_to_buffer", "jit_add_nodes_to_buffer",
]
