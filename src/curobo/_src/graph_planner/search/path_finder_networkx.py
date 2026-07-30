"""Dependency-free implementation of the historical NetworkX path finder."""

from __future__ import annotations

import heapq
import random


class NetworkXPathFinder:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset_graph()
        self.reset_seed()

    def reset_graph(self):
        self._graph: dict[int, dict[int, float]] = {}

    def reset_seed(self):
        random.seed(self.seed)

    def add_node(self, i):
        self._graph.setdefault(int(i), {})

    def add_edges(self, edge_list):
        for edge in edge_list:
            self.add_edge(*edge)

    def add_nodes(self, node_list):
        for node in node_list:
            self.add_node(node)

    def add_edge(self, start_i, end_i, weight):
        start_i, end_i = int(start_i), int(end_i)
        self.add_node(start_i); self.add_node(end_i)
        self._graph[start_i][end_i] = float(weight)
        self._graph[end_i][start_i] = float(weight)

    def update_graph(self):
        return None

    def get_edges(self, attribue="weight"):
        del attribue
        return [
            (a, b, weight) for a, row in self._graph.items()
            for b, weight in row.items() if a < b
        ]

    def path_exists(self, start_node_idx, goal_node_idx):
        return self.get_shortest_path(start_node_idx, goal_node_idx) is not None

    def get_shortest_path(self, start_node_idx, goal_node_idx, return_length=False):
        start, goal = int(start_node_idx), int(goal_node_idx)
        queue = [(0.0, start)]
        distance = {start: 0.0}
        parent: dict[int, int] = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return (path, cost) if return_length else path
            for neighbor, weight in sorted(self._graph.get(node, {}).items()):
                candidate = cost + weight
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    parent[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        return (None, float("inf")) if return_length else None

    def get_path_lengths(self, goal_node_idx):
        return {
            node: self.get_shortest_path(node, goal_node_idx, return_length=True)[1]
            for node in self._graph
        }


__all__ = ["NetworkXPathFinder"]
