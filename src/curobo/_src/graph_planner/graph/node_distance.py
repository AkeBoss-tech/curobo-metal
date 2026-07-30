"""Weighted joint-space distance operations."""

import torch


class DistanceNeighborCalculator:
    def __init__(self, action_dim: int, cspace_distance_weight: torch.Tensor, device_cfg):
        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self.device_cfg = device_cfg

    def calculate_weighted_distance(self, pt, batch_pts):
        return self.jit_calculate_weighted_distance(
            pt, batch_pts, self.cspace_distance_weight
        )

    def find_nearest_neighbors(self, new_nodes, existing_nodes, neighbors_per_node):
        distance = self.jit_calculate_weighted_distance(
            new_nodes[:, None, :], existing_nodes[None, :, :], self.cspace_distance_weight
        )
        return torch.topk(
            distance, min(neighbors_per_node, existing_nodes.shape[0]),
            dim=-1, largest=False, sorted=True,
        ).indices

    def get_unique_nodes(self, nodes, similarity_threshold):
        if nodes.shape[0] == 0:
            return nodes, torch.empty(0, dtype=torch.int64, device=nodes.device)
        distance = torch.cdist(
            nodes * self.cspace_distance_weight, nodes * self.cspace_distance_weight
        )
        keep = torch.ones(nodes.shape[0], dtype=torch.bool, device=nodes.device)
        for index in range(nodes.shape[0]):
            if index:
                keep[index] = not bool((distance[index, :index] < similarity_threshold).any())
        return nodes[keep], torch.nonzero(keep, as_tuple=False).flatten()

    def get_unique_nodes_zero_distance(self, nodes):
        return self.get_unique_nodes(nodes, 0.0)

    @staticmethod
    def jit_calculate_weighted_distance(pt, batch_pts, distance_weight):
        return torch.linalg.vector_norm((pt - batch_pts) * distance_weight, dim=-1)

    jit_find_nearest_neighbors = staticmethod(
        lambda new_nodes, existing_nodes, distance_weight, neighbors_per_node, action_dim:
        torch.topk(
            torch.cdist(new_nodes * distance_weight, existing_nodes * distance_weight),
            min(neighbors_per_node, existing_nodes.shape[0]), largest=False, sorted=True,
        ).indices
    )


__all__ = ["DistanceNeighborCalculator"]
