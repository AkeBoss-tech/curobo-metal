"""Deterministic, portable replacement for cuRobo's NetworkX path finder.

The pinned cuRoboV2 object uses :mod:`networkx` only as a small control-plane
weighted graph.  Roadmap vertices and edge candidates themselves commonly
arrive from CPU or MPS tensors, but no CUDA kernel is involved in the final
shortest-path search.  This implementation preserves the useful public
lifecycle without making a NetworkX installation (or a CUDA runtime) part of
the Metal package: node/edge additions are staged until a query, graph resets
are observable, and weighted undirected paths are deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
import heapq
import math
import random
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx
import numpy as np
import torch
from torch import profiler


def _scalar(value: Any, *, name: str) -> Any:
    """Convert a Python/NumPy/Torch scalar to a control-plane Python value."""
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} tensor must contain exactly one value")
        # This is intentionally the control-plane boundary.  Roadmap tensors
        # remain on MPS until their integer graph identifiers are consumed.
        value = value.detach().cpu().item()
    elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except ValueError:
            pass
    return value


def _node(value: Any) -> int:
    value = _scalar(value, name="node")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError("graph node identifiers must be integer scalars")


def _weight(value: Any) -> float:
    value = _scalar(value, name="weight")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("edge weight must be a scalar") from exc
    if not math.isfinite(result):
        raise ValueError("edge weight must be finite")
    if result < 0.0:
        # Both upstream and this implementation use Dijkstra.  Failing at the
        # boundary is clearer than silently returning a non-shortest path.
        raise ValueError("negative edge weights are not supported by Dijkstra search")
    return result


class _PortableGraphView:
    """Small read-only NetworkX-shaped graph inspection facade.

    It intentionally covers the graph inspection used by cuRobo integrations,
    not NetworkX's full algorithm or attribute API.  Mutation belongs on
    :class:`NetworkXPathFinder` so staged additions retain their pinned
    lifecycle.
    """

    def __init__(self, owner: "NetworkXPathFinder") -> None:
        self._owner = owner

    def clear(self) -> None:
        self._owner.reset_graph()

    def has_node(self, node: Any) -> bool:
        self._owner.update_graph()
        return _node(node) in self._owner._graph

    def has_edge(self, start: Any, end: Any) -> bool:
        self._owner.update_graph()
        start_node, end_node = _node(start), _node(end)
        return end_node in self._owner._graph.get(start_node, {})

    def number_of_nodes(self) -> int:
        self._owner.update_graph()
        return len(self._owner._graph)

    def number_of_edges(self) -> int:
        self._owner.update_graph()
        return len(self._owner._edge_order)

    def nodes(self) -> list[int]:
        self._owner.update_graph()
        return list(self._owner._graph)

    def edges(self, data: Optional[str] = None) -> list[tuple[Any, ...]]:
        if data not in (None, "weight"):
            raise ValueError("only the 'weight' edge attribute is available")
        records = self._owner.get_edges()
        return records if data == "weight" else [(start, end) for start, end, _ in records]


class _NetworkXPathFinderPortable:
    """A buffered, deterministic undirected weighted roadmap searcher.

    ``add_node``/``add_nodes`` and ``add_edge``/``add_edges`` only enqueue
    changes, matching the historical NetworkX wrapper.  Every query calls
    :meth:`update_graph`; direct callers can also use it as an explicit commit
    point.  Equal-cost routes use the lexicographically smallest complete node
    sequence, avoiding insertion-order-dependent planning outcomes.
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._graph: dict[int, dict[int, float]] = {}
        self._edge_order: list[tuple[int, int]] = []
        self.node_list: list[int] = []
        self.edge_list: list[list[int | float]] = []
        self.graph = _PortableGraphView(self)

    def reset_graph(self) -> None:
        """Discard materialized and staged roadmap state."""
        self._graph.clear()
        self._edge_order.clear()
        self.node_list.clear()
        self.edge_list.clear()

    def reset_seed(self) -> None:
        """Reset the pinned global random streams used by upstream planning."""
        random.seed(self.seed)
        if np is not None:
            np.random.seed(self.seed)

    def add_node(self, i: Any) -> None:
        self.node_list.append(_node(i))

    def add_nodes(self, node_list: Iterable[Any]) -> None:
        if torch is not None and isinstance(node_list, torch.Tensor):
            node_list = node_list.reshape(-1)
        self.node_list.extend(_node(node) for node in node_list)

    def add_edge(self, start_i: Any, end_i: Any, weight: Any) -> None:
        self.edge_list.append([_node(start_i), _node(end_i), _weight(weight)])

    def add_edges(self, edge_list: Iterable[Iterable[Any]]) -> None:
        if torch is not None and isinstance(edge_list, torch.Tensor):
            if edge_list.ndim != 2 or edge_list.shape[-1] != 3:
                raise ValueError("edge tensor must have shape [N, 3]")
            edge_list = edge_list.unbind(dim=0)
        for edge in edge_list:
            values = tuple(edge)
            if len(values) != 3:
                raise ValueError("each edge must be a (start, end, weight) triple")
            self.add_edge(*values)

    def update_graph(self) -> None:
        """Materialize pending nodes and edges in stable insertion order."""
        for node in self.node_list:
            self._graph.setdefault(node, {})
        self.node_list.clear()
        for start, end, weight in self.edge_list:
            self._graph.setdefault(start, {})
            self._graph.setdefault(end, {})
            edge_key = (start, end) if start <= end else (end, start)
            if edge_key not in self._edge_order:
                self._edge_order.append(edge_key)
            # A repeated NetworkX Graph edge updates the existing edge's weight.
            self._graph[start][end] = weight
            self._graph[end][start] = weight
        self.edge_list.clear()

    def get_edges(self, attribue: str = "weight") -> list[tuple[int, int, float]]:
        """Return one weighted record for every materialized undirected edge."""
        del attribue  # Preserve the historical misspelled, ignored argument.
        return [
            (start, end, self._graph[start][end])
            for start, end in self._edge_order
            if start in self._graph and end in self._graph[start]
        ]

    def path_exists(self, start_node_idx: Any, goal_node_idx: Any) -> bool:
        self.update_graph()
        start, goal = _node(start_node_idx), _node(goal_node_idx)
        if start not in self._graph or goal not in self._graph:
            return False
        return self._shortest_path(start, goal) is not None

    def _shortest_path(self, start: int, goal: int) -> Optional[tuple[list[int], float]]:
        if start not in self._graph or goal not in self._graph:
            return None
        # The path key makes equal-weight choices independent of edge insertion
        # order.  It is cheap because this graph is only PRM control-plane data.
        queue: list[tuple[float, tuple[int, ...], int]] = [(0.0, (start,), start)]
        best: dict[int, tuple[float, tuple[int, ...]]] = {start: (0.0, (start,))}
        while queue:
            cost, path_key, node = heapq.heappop(queue)
            if best.get(node) != (cost, path_key):
                continue
            if node == goal:
                return list(path_key), cost
            for neighbor, weight in self._graph[node].items():
                candidate = (cost + weight, path_key + (neighbor,))
                current = best.get(neighbor)
                if current is None or candidate < current:
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
        return None

    def get_shortest_path(
        self, start_node_idx: Any, goal_node_idx: Any, return_length: bool = False
    ) -> Optional[list[int]] | tuple[Optional[list[int]], float]:
        """Return a stable weighted shortest path, or ``None`` if unavailable.

        Returning ``None`` for absent/disconnected nodes is the historical
        portable facade convention used by batched PRM queries.  It avoids a
        control-flow exception for one failed pair while preserving the upstream
        success result shape; callers needing a boolean can use
        :meth:`path_exists`.
        """
        self.update_graph()
        result = self._shortest_path(_node(start_node_idx), _node(goal_node_idx))
        if result is None:
            return (None, float("inf")) if return_length else None
        path, length = result
        return (path, length) if return_length else path

    def get_path_lengths(self, goal_node_idx: Any) -> list[float]:
        """Return upstream-shaped dense distances from every reachable node.

        Indices without a route retain ``-1.0``.  Roadmap identifiers are
        non-negative integer slots, so this dense result can be directly copied
        into the graph debug buffer on either CPU or MPS.
        """
        self.update_graph()
        goal = _node(goal_node_idx)
        if goal not in self._graph:
            return []
        if not self._graph:
            return []
        if min(self._graph) < 0:
            raise ValueError("get_path_lengths requires non-negative node identifiers")
        queue: list[tuple[float, int]] = [(0.0, goal)]
        distances: dict[int, float] = {goal: 0.0}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distances[node]:
                continue
            for neighbor, weight in self._graph[node].items():
                candidate = cost + weight
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        lengths = [-1.0] * (max(self._graph) + 1)
        for node, distance in distances.items():
            lengths[node] = distance
        return lengths


class NetworkXPathFinder:
    def __init__(self, seed: int = 42):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/reset_graph")
    def reset_graph(self):
        raise NotImplementedError

    def reset_seed(self):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/add_node")
    def add_node(self, i):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/add_edges")
    def add_edges(self, edge_list):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/add_nodes")
    def add_nodes(self, node_list):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/add_edge")
    def add_edge(self, start_i, end_i, weight):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/update_graph")
    def update_graph(self):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/get_edges")
    def get_edges(self, attribue="weight"):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/path_exists")
    def path_exists(self, start_node_idx, goal_node_idx):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/get_shortest_path")
    def get_shortest_path(self, start_node_idx, goal_node_idx, return_length=False):
        raise NotImplementedError

    @profiler.record_function("networkx_path_finder/get_path_lengths")
    def get_path_lengths(self, goal_node_idx):
        raise NotImplementedError


if not TYPE_CHECKING:
    # The declaration above tracks the pinned NetworkX contract; runtime uses
    # the portable, deterministic graph implementation.
    NetworkXPathFinder = _NetworkXPathFinderPortable


__all__ = ["NetworkXPathFinder"]
