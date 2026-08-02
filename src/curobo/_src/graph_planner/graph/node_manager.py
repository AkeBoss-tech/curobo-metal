"""Mutable graph node buffers matching the pinned public lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch


@dataclass
class ConnectedGraph:
    node_set: torch.Tensor
    edge_set: torch.Tensor
    shortest_path_lengths: torch.Tensor | None = None

    def set_shortest_path_lengths(self, shortest_path_lengths: torch.Tensor):
        self.shortest_path_lengths = shortest_path_lengths

    def get_node_distance(self):
        return self.shortest_path_lengths


def jit_add_nodes_to_buffer(
    preallocated_node_buffer: torch.Tensor,
    new_nodes: torch.Tensor,
    used_node_count: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    del action_dim, device, dtype
    end = used_node_count + new_nodes.shape[0]
    preallocated_node_buffer[used_node_count:end] = new_nodes
    return preallocated_node_buffer


def jit_add_all_nodes_to_buffer(
    all_nodes, preallocated_node_buffer, used_node_count: int, action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    result = jit_add_nodes_to_buffer(
        preallocated_node_buffer, all_nodes, used_node_count, action_dim,
        all_nodes.device, all_nodes.dtype,
    )
    count = used_node_count + all_nodes.shape[0]
    return result, torch.arange(used_node_count, count, device=all_nodes.device), count


class GraphNodeManager:
    def __init__(
        self, config, distance_calculator=None, graph_path_finder=None,
        auxiliary_rollout=None, device_cfg=None,
    ):
        self.config = config
        self.distance_calculator = distance_calculator
        self.graph_path_finder = graph_path_finder
        self.auxiliary_rollout = auxiliary_rollout
        self.device_cfg = device_cfg or config.device_cfg
        self._action_dim = (
            int(config.action_lower_bounds.numel())
            if config.action_lower_bounds is not None else 0
        )
        self.reset_buffer()

    def add_nodes_to_buffer(self, new_nodes):
        if self._action_dim == 0:
            self._action_dim = new_nodes.shape[-1]
        available = self.config.max_nodes - self._used
        if new_nodes.shape[0] > available:
            raise ValueError("graph node buffer capacity exceeded")
        self._buffer[self._used:self._used + new_nodes.shape[0]] = new_nodes
        indices = torch.arange(
            self._used, self._used + new_nodes.shape[0], device=new_nodes.device
        )
        self._used += new_nodes.shape[0]
        return indices

    def get_connected_graph(self):
        edges = torch.as_tensor(
            [(a, b) for a, row in self.graph_path_finder._graph.items() for b in row if a < b],
            device=self.device_cfg.device, dtype=torch.int64,
        )
        return ConnectedGraph(self.valid_node_buffer, edges.reshape(-1, 2))

    def add_nodes_to_roadmap(self, nodes: torch.Tensor, add_exact_node: bool = False) -> torch.Tensor:
        del add_exact_node
        indices = self.add_nodes_to_buffer(nodes)
        if self.graph_path_finder is not None:
            self.graph_path_finder.add_nodes(indices.detach().cpu().tolist())
        return torch.cat((nodes, indices[:, None].to(nodes)), dim=-1)

    def add_initial_exact_nodes_to_roadmap(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.add_nodes_to_roadmap(nodes, True)

    def register_nodes_and_connections(
        self, node_set: torch.Tensor, start_nodes: torch.Tensor, add_exact_node=False
    ):
        del start_nodes
        return self.add_nodes_to_roadmap(node_set, add_exact_node)

    def get_nodes_in_path(self, path_list: List[Union[List[int], None]]):
        return [
            None if path is None else self._buffer[
                torch.as_tensor(path, device=self.device_cfg.device)
            ] for path in path_list
        ]

    def reset_buffer(self):
        width = max(self._action_dim, 1)
        self._buffer = torch.zeros(
            (self.config.max_nodes, width), **self.device_cfg.as_torch_dict()
        )
        self._used = 0
        self.reset_graph_path_finder()

    def reset_graph_path_finder(self):
        if self.graph_path_finder is not None:
            self.graph_path_finder.reset_graph()

    @property
    def action_dim(self) -> int: return self._action_dim
    @property
    def n_nodes(self) -> int: return self._used
    @property
    def preallocated_node_buffer(self): return self._buffer
    @property
    def valid_node_buffer(self): return self._buffer[:self._used]
    @property
    def node_idx_padding_buffer(self):
        return torch.arange(self.config.max_nodes, device=self.device_cfg.device)


__all__ = [
    "ConnectedGraph", "GraphNodeManager", "jit_add_all_nodes_to_buffer",
    "jit_add_nodes_to_buffer",
]
