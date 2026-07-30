"""Self-collision sphere-pair preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

import torch

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class SelfCollisionKinematicsCfg:
    num_spheres: int = 0
    sphere_padding: Optional[torch.Tensor] = None
    collision_pairs: Optional[torch.Tensor] = None
    _num_checks_per_thread_large_collision_pairs: int = 256
    _max_threads_per_block_large_collision_pairs: int = 512
    _max_threads_per_block_small_collision_pairs: int = 64
    _num_checks_per_thread_small_collision_pairs: int = 32

    @property
    def num_checks_per_thread(self) -> int:
        return (
            self._num_checks_per_thread_large_collision_pairs
            if self.num_spheres > 100 else self._num_checks_per_thread_small_collision_pairs
        )

    @property
    def max_threads_per_block(self) -> int:
        return (
            self._max_threads_per_block_large_collision_pairs
            if self.num_spheres > 100 else self._max_threads_per_block_small_collision_pairs
        )

    @property
    def num_blocks_per_batch(self) -> int:
        if self.collision_pairs is None:
            return 0
        return math.ceil(self.collision_pairs.shape[0] / self.max_threads_per_block)

    @staticmethod
    def create_from_sphere_pair_distances(
        sphere_pair_distances: torch.Tensor, sphere_padding: torch.Tensor
    ) -> "SelfCollisionKinematicsCfg":
        if sphere_pair_distances.ndim != 2 or sphere_pair_distances.shape[0] != sphere_pair_distances.shape[1]:
            raise ValueError("sphere_pair_distances must be square")
        pairs = torch.nonzero(torch.triu(torch.isfinite(sphere_pair_distances), diagonal=1))
        return SelfCollisionKinematicsCfg(
            sphere_pair_distances.shape[0], sphere_padding, pairs.to(torch.int32)
        )

    @staticmethod
    def compute_sphere_pair_distance_with_link_pair_ignores(
        collision_link_names: List[str],
        link_name_to_sphere_index: Dict[str, int],
        self_collision_link_pair_ignores: Dict[str, List[str]],
        self_collision_link_padding: Dict[str, float],
        all_link_spheres: torch.Tensor,
        link_index_to_sphere_index: torch.Tensor,
        device_cfg: DeviceCfg,
    ) -> torch.Tensor:
        del link_name_to_sphere_index, self_collision_link_padding
        count = all_link_spheres.shape[-2]
        result = torch.zeros((count, count), **device_cfg.as_torch_dict())
        sphere_links = link_index_to_sphere_index.reshape(-1).tolist()
        for i in range(count):
            for j in range(i + 1, count):
                left = collision_link_names[sphere_links[i]]
                right = collision_link_names[sphere_links[j]]
                ignored = right in self_collision_link_pair_ignores.get(left, [])
                if ignored or left == right:
                    result[i, j] = result[j, i] = float("inf")
        return result

    @staticmethod
    def create_from_link_pairs(
        collision_link_names: List[str],
        link_name_to_sphere_index: Dict[str, int],
        self_collision_link_pair_ignores: Dict[str, List[str]],
        self_collision_link_padding: Dict[str, float],
        all_link_spheres: torch.Tensor,
        link_index_to_sphere_index: torch.Tensor,
        device_cfg: DeviceCfg,
    ) -> "SelfCollisionKinematicsCfg":
        distances = SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores(
            collision_link_names, link_name_to_sphere_index,
            self_collision_link_pair_ignores, self_collision_link_padding,
            all_link_spheres, link_index_to_sphere_index, device_cfg,
        )
        padding = device_cfg.to_device(
            [self_collision_link_padding.get(name, 0.0) for name in collision_link_names]
        )
        return SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(distances, padding)


__all__ = ["SelfCollisionKinematicsCfg"]
