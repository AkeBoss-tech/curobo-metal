"""Portable self-collision sphere-pair preprocessing.

The pinned V2 configuration is a small tensor record consumed by a Warp CUDA
kernel.  The public preprocessing contract is useful independently of that
ABI: pairs whose entry is ``-inf`` are disabled, every other (including
``+inf``) entry represents a valid sphere pair, and link-pair ignores disable
the pair in either direction.  This module preserves that contract while
producing ordinary PyTorch index tensors suitable for the CPU/MPS collision
operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import torch

from curobo._src.types.device_cfg import DeviceCfg


_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


@dataclass
class SelfCollisionKinematicsCfg:
    """Validated sphere-pair configuration for portable self collision.

    ``collision_pairs`` is an ordered, unique ``[P, 2]`` tensor with each
    pair canonicalized as ``left < right``.  Pairs use ``int64`` when created
    by this module because that is the standard PyTorch indexing dtype; the
    CUDA/Warp ``int16`` launch-buffer ABI is intentionally not emulated.
    """

    num_spheres: int = 0
    sphere_padding: Optional[torch.Tensor] = None
    collision_pairs: Optional[torch.Tensor] = None
    _num_checks_per_thread_large_collision_pairs: int = 256
    _max_threads_per_block_large_collision_pairs: int = 512
    _max_threads_per_block_small_collision_pairs: int = 64
    _num_checks_per_thread_small_collision_pairs: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.num_spheres, bool) or not isinstance(self.num_spheres, int):
            raise TypeError("num_spheres must be an integer")
        if self.num_spheres < 0:
            raise ValueError("num_spheres must be non-negative")

        padding = self.sphere_padding
        if padding is not None:
            if not isinstance(padding, torch.Tensor):
                raise TypeError("sphere_padding must be a torch.Tensor or None")
            if padding.ndim != 1 or padding.numel() != self.num_spheres:
                raise ValueError("sphere_padding must have shape [num_spheres]")
            if not padding.dtype.is_floating_point:
                raise TypeError("sphere_padding must have a floating dtype")
            if bool(torch.isnan(padding).any().item()) or bool((padding < 0).any().item()):
                raise ValueError("sphere_padding must contain non-negative non-NaN values")

        pairs = self.collision_pairs
        if pairs is None:
            return
        if not isinstance(pairs, torch.Tensor):
            raise TypeError("collision_pairs must be a torch.Tensor or None")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("collision_pairs must have shape [num_collision_checks, 2]")
        if pairs.dtype not in _INTEGER_DTYPES:
            raise TypeError("collision_pairs must use an integer dtype")
        if padding is not None and pairs.device != padding.device:
            raise ValueError("sphere_padding and collision_pairs must be on the same device")
        if pairs.numel() == 0:
            return
        pairs_long = pairs.to(dtype=torch.int64)
        if bool(((pairs_long < 0) | (pairs_long >= self.num_spheres)).any().item()):
            raise ValueError("collision_pairs contains an out-of-range sphere index")
        if bool((pairs_long[:, 0] >= pairs_long[:, 1]).any().item()):
            raise ValueError("collision_pairs must contain unique canonical pairs with left < right")
        # ``unique`` executes on the configured device, including MPS.  We do
        # this once at configuration time to avoid non-deterministic repeated
        # checks in the collision inner loop.
        if torch.unique(pairs_long, dim=0).shape[0] != pairs.shape[0]:
            raise ValueError("collision_pairs must not contain duplicate pairs")

    @property
    def num_checks_per_thread(self) -> int:
        """Pinned V2 launch heuristic, exposed for configuration parity."""
        pair_count = 0 if self.collision_pairs is None else int(self.collision_pairs.shape[0])
        return (
            self._num_checks_per_thread_large_collision_pairs
            if pair_count > 1000
            else self._num_checks_per_thread_small_collision_pairs
        )

    @property
    def max_threads_per_block(self) -> int:
        """Pinned V2 launch heuristic, exposed for configuration parity."""
        pair_count = 0 if self.collision_pairs is None else int(self.collision_pairs.shape[0])
        return (
            self._max_threads_per_block_large_collision_pairs
            if pair_count > 1000
            else self._max_threads_per_block_small_collision_pairs
        )

    @property
    def num_blocks_per_batch(self) -> int:
        """Number of V2-style reduction blocks needed for the configured pairs."""
        pair_count = 0 if self.collision_pairs is None else int(self.collision_pairs.shape[0])
        if pair_count == 0:
            return 0
        return math.ceil(pair_count / (self.num_checks_per_thread * self.max_threads_per_block))

    @staticmethod
    def create_from_sphere_pair_distances(
        sphere_pair_distances: torch.Tensor, sphere_padding: torch.Tensor
    ) -> "SelfCollisionKinematicsCfg":
        """Compile the V2 ``-inf`` pair-disable matrix into indexed pairs.

        Values in the upper triangle other than ``-inf`` are retained.  In
        particular, ``+inf`` remains a valid pair exactly as it does in V2;
        NaN has no meaningful collision interpretation and is rejected.
        """
        if not isinstance(sphere_pair_distances, torch.Tensor):
            raise TypeError("sphere_pair_distances must be a torch.Tensor")
        if not isinstance(sphere_padding, torch.Tensor):
            raise TypeError("sphere_padding must be a torch.Tensor")
        if sphere_padding.ndim != 1:
            raise ValueError("sphere_padding must have shape [num_spheres]")
        num_spheres = sphere_padding.numel()
        if sphere_pair_distances.ndim != 2 or sphere_pair_distances.shape != (num_spheres, num_spheres):
            raise ValueError(
                "sphere_pair_distances must have shape [num_spheres, num_spheres]"
            )
        if sphere_pair_distances.device != sphere_padding.device:
            raise ValueError("sphere_pair_distances and sphere_padding must be on the same device")
        if not sphere_pair_distances.dtype.is_floating_point:
            raise TypeError("sphere_pair_distances must have a floating dtype")
        if bool(torch.isnan(sphere_pair_distances).any().item()):
            raise ValueError("sphere_pair_distances must not contain NaN")

        upper = torch.triu(
            ~torch.isneginf(sphere_pair_distances), diagonal=1
        )
        pairs = torch.nonzero(upper, as_tuple=False).to(dtype=torch.int64).contiguous()
        return SelfCollisionKinematicsCfg(
            num_spheres=num_spheres,
            sphere_padding=sphere_padding,
            collision_pairs=pairs,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build V2 distance/padding tensors from link-level collision data.

        Sphere entries belonging to the same link and ignored link pairs get
        ``-inf``.  The remaining entries contain the sum of both radii and
        per-link padding, which is the V2 self-collision activation distance.
        Inputs must already be resident on ``device_cfg`` so a caller cannot
        accidentally construct a CPU matrix for an MPS collision pipeline.
        """
        if not isinstance(all_link_spheres, torch.Tensor) or all_link_spheres.ndim != 2:
            raise ValueError("all_link_spheres must have shape [num_spheres, 4]")
        if all_link_spheres.shape[1] != 4:
            raise ValueError("all_link_spheres must contain xyzw-radius values")
        if not all_link_spheres.dtype.is_floating_point:
            raise TypeError("all_link_spheres must have a floating dtype")
        if not isinstance(link_index_to_sphere_index, torch.Tensor):
            raise TypeError("link_index_to_sphere_index must be a torch.Tensor")
        if link_index_to_sphere_index.numel() != all_link_spheres.shape[0]:
            raise ValueError("link_index_to_sphere_index must contain one entry per sphere")
        if link_index_to_sphere_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("link_index_to_sphere_index must use an integer dtype")
        if not device_cfg.is_same_torch_device(all_link_spheres.device):
            raise ValueError("all_link_spheres must be on device_cfg.device")
        if not device_cfg.is_same_torch_device(link_index_to_sphere_index.device):
            raise ValueError("link_index_to_sphere_index must be on device_cfg.device")
        if len(collision_link_names) != len(set(collision_link_names)):
            raise ValueError("collision_link_names must not contain duplicates")
        if set(collision_link_names) != set(link_name_to_sphere_index):
            raise ValueError("link_name_to_sphere_index must map every collision link exactly once")
        if bool(torch.isnan(all_link_spheres).any().item()) or bool((all_link_spheres[:, 3] < 0).any().item()):
            raise ValueError("all_link_spheres must have non-negative non-NaN radii")

        num_spheres = all_link_spheres.shape[0]
        spec = {"device": device_cfg.device, "dtype": device_cfg.dtype}
        padding = torch.zeros((num_spheres,), **spec)
        distance = torch.full((num_spheres, num_spheres), -torch.inf, **spec)
        link_ids = link_index_to_sphere_index.reshape(-1).to(dtype=torch.int64)
        radii = all_link_spheres[:, 3].to(**spec)

        # The original V2 routine iterates both directions then takes the
        # matrix minimum.  Consequently an ignore in either direction disables
        # a pair.  Enumerating unordered links yields the same result without
        # host copies or a CUDA/Warp dependency.
        for name in collision_link_names:
            link_id = link_name_to_sphere_index[name]
            if isinstance(link_id, bool) or not isinstance(link_id, int):
                raise TypeError("link_name_to_sphere_index values must be integers")
            mask = link_ids == link_id
            if not bool(mask.any().item()):
                # V2 leaves links without spheres alone; retaining this makes
                # optional collision links harmless.
                continue
            value = self_collision_link_padding.get(name, 0.0)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("self_collision_link_padding values must be finite scalars")
            if value < 0:
                raise ValueError("self_collision_link_padding values must be non-negative")
            padding[mask] = float(value)

        for left_position, left_name in enumerate(collision_link_names):
            left_id = link_name_to_sphere_index[left_name]
            left_mask = link_ids == left_id
            if not bool(left_mask.any().item()):
                continue
            for right_name in collision_link_names[left_position + 1 :]:
                right_id = link_name_to_sphere_index[right_name]
                right_mask = link_ids == right_id
                if not bool(right_mask.any().item()):
                    continue
                if (
                    right_name in self_collision_link_pair_ignores.get(left_name, ())
                    or left_name in self_collision_link_pair_ignores.get(right_name, ())
                ):
                    continue
                active = left_mask[:, None] & right_mask[None, :]
                pair_distance = radii[:, None] + radii[None, :] + padding[:, None] + padding[None, :]
                distance = torch.where(active, pair_distance, distance)
                distance = torch.where(active.T, pair_distance, distance)
        return distance, padding

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
        """Create a portable self-collision configuration from link pairs."""
        distances, padding = (
            SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores(
                collision_link_names,
                link_name_to_sphere_index,
                self_collision_link_pair_ignores,
                self_collision_link_padding,
                all_link_spheres,
                link_index_to_sphere_index,
                device_cfg,
            )
        )
        return SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(distances, padding)


__all__ = ["SelfCollisionKinematicsCfg"]
