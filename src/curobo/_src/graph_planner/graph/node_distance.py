"""Weighted joint-space distance operations.

The upstream helpers happen to be TorchScript functions.  These declarations
keep their public calling convention, but deliberately use ordinary tensor
operations so that the exact same functions work on CPU and MPS.
"""

from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.tensor_util import stable_topk
from curobo._src.util.torch_util import get_torch_jit_decorator


class DistanceNeighborCalculator:
    def __init__(
        self, action_dim: int, cspace_distance_weight: torch.Tensor, device_cfg: DeviceCfg
    ):
        self.action_dim = action_dim
        self.cspace_distance_weight = cspace_distance_weight
        self.device_cfg = device_cfg

    def calculate_weighted_distance(self, pt: torch.Tensor, batch_pts: torch.Tensor) -> torch.Tensor:
        return self.jit_calculate_weighted_distance(
            pt, batch_pts, self.cspace_distance_weight
        )

    def find_nearest_neighbors(
        self, new_nodes: torch.Tensor, existing_nodes: torch.Tensor, neighbors_per_node: int
    ) -> torch.Tensor:
        distance = self.jit_calculate_weighted_distance(
            new_nodes[:, None, :], existing_nodes[None, :, :], self.cspace_distance_weight
        )
        return torch.topk(
            distance, min(neighbors_per_node, existing_nodes.shape[0]),
            dim=-1, largest=False, sorted=True,
        ).indices

    def get_unique_nodes(
        self, nodes: torch.Tensor, similarity_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def get_unique_nodes_zero_distance(self, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_unique_nodes(nodes, 0.0)

    @staticmethod
    def jit_calculate_weighted_distance(
        pt: torch.Tensor, batch_pts: torch.Tensor, distance_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.vector_norm((pt - batch_pts) * distance_weight, dim=-1)

    @staticmethod
    def jit_find_nearest_neighbors(
        new_nodes: torch.Tensor,
        existing_nodes: torch.Tensor,
        distance_weight: torch.Tensor,
        neighbors_per_node: int,
        action_dim: int,
    ) -> torch.Tensor:
        del action_dim
        if existing_nodes.shape[0] == 0:
            return torch.empty((new_nodes.shape[0], 0), dtype=torch.int64, device=new_nodes.device)
        return torch.topk(
            torch.cdist(new_nodes * distance_weight, existing_nodes * distance_weight),
            min(neighbors_per_node, existing_nodes.shape[0]), largest=False, sorted=True,
        ).indices

    @staticmethod
    def jit_get_unique_nodes(
        nodes: torch.Tensor,
        distance_weight: torch.Tensor,
        node_similarity_threshold: float,
        action_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return first-occurrence unique roadmap vertices.

        The CUDA implementation exposes this as a JIT helper.  Keeping it as a
        tensor-only static method gives callers the same useful entry point on
        CPU and MPS without pretending to expose a CUDA kernel.
        """
        del action_dim
        calculator = DistanceNeighborCalculator(nodes.shape[-1], distance_weight, None)
        return calculator.get_unique_nodes(nodes, node_similarity_threshold)

    @staticmethod
    def jit_get_unique_nodes_zero_distance(
        nodes: torch.Tensor, distance_weight: torch.Tensor, action_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return DistanceNeighborCalculator.jit_get_unique_nodes(
            nodes, distance_weight, torch.finfo(nodes.dtype).eps, action_dim
        )


__all__ = ["DistanceNeighborCalculator"]
