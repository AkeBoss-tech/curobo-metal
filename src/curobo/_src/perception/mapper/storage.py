"""Portable dense implementation of the cuRobo block-map storage surface.

The upstream object owns a CUDA/Warp hash table and a pool of fixed-size
blocks.  That ABI cannot be represented faithfully on Metal.  This module
instead exposes the same useful *lifecycle* over a bounded dense tensor map:
state can be reset, exported, imported, inspected, and queried on CPU or MPS.
Methods whose contract is the raw Warp hash/pool ABI reject rather than
pretending that a dense tensor is a sparse allocation pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from curobo_metal.ops.perception import PerceptionConfig, PerceptionMapper
from curobo_metal.ops.perception.core import DenseMap


def _device(value: str | torch.device) -> str | torch.device:
    """Map cuRobo's default CUDA spelling to the available portable device."""
    if str(value).startswith("cuda"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return value


@dataclass
class BlockSparseTSDFCfg:
    """Source-shaped configuration for a bounded dense TSDF replacement.

    ``max_blocks`` and ``hash_capacity`` are retained as planning metadata only.
    They never select a CUDA hash-table implementation.  RGB and learned
    feature grids are likewise unavailable because the current production
    depth mapper stores geometry only.
    """

    max_blocks: int = 100_000
    hash_capacity: int = 200_000
    voxel_size: float = 0.002
    origin: Optional[torch.Tensor] = None
    truncation_distance: float = 0.04
    device: str = "cuda:0"
    grid_shape: Optional[Tuple[int, int, int]] = None
    enable_dynamic: bool = True
    enable_static: bool = False
    static_obstacle_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    block_size: int = 8
    feature_dim: int = 0
    feature_block_grid_size: int = 1
    feature_grid_height: Optional[int] = None
    feature_grid_width: Optional[int] = None
    feature_channels_per_thread: int = 4
    max_feature_tile_channels: int = 4096
    max_support_pixels_per_block_camera: int = 32
    color_grid_size: int = 1
    accumulator_w_max: float = 1000.0

    def __post_init__(self) -> None:
        if self.grid_shape is None or len(self.grid_shape) != 3 or any(int(n) < 2 for n in self.grid_shape):
            raise ValueError("portable dense BlockSparseTSDFCfg.grid_shape must contain three integers >= 2")
        self.grid_shape = tuple(int(n) for n in self.grid_shape)
        if self.voxel_size <= 0 or self.truncation_distance <= 0 or self.accumulator_w_max <= 0:
            raise ValueError("voxel_size, truncation_distance, and accumulator_w_max must be positive")
        if self.block_size < 1 or self.block_size > 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be 1 or a power of two through 32")
        if self.max_blocks < 1 or self.hash_capacity < 1:
            raise ValueError("max_blocks and hash_capacity must be positive")
        if not self.enable_dynamic and not self.enable_static:
            raise ValueError("at least one of enable_dynamic or enable_static must be True")
        if self.feature_dim:
            raise NotImplementedError("feature-volume block storage requires CUDA/Warp")
        if self.color_grid_size != 1:
            raise NotImplementedError("RGB block control grids require CUDA/Warp")
        if self.feature_block_grid_size < 1 or self.feature_channels_per_thread < 1:
            raise ValueError("feature block/grid channel dimensions must be positive")
        if self.origin is None:
            self.origin = torch.zeros(3, dtype=torch.float32)
        else:
            self.origin = torch.as_tensor(self.origin, dtype=torch.float32)
        if self.origin.shape != (3,):
            raise ValueError("origin must be an xyz vector")


@dataclass
class BlockDataView:
    """Read-only dense per-voxel view with the pinned block-data query API."""

    rgb_grid: torch.Tensor
    coords: torch.Tensor
    num_allocated: int
    origin: torch.Tensor
    voxel_size: float
    block_size: int
    grid_shape: tuple
    color_grid_size: int = 1
    feature_block_grid_size: int = 1
    features: torch.Tensor | None = None
    feature_weight: torch.Tensor | None = None
    feature_dim: int = 0

    def _indices(self, values: torch.Tensor) -> torch.Tensor:
        indices = values.to(device=self.rgb_grid.device, dtype=torch.long).reshape(-1)
        if bool(((indices < 0) | (indices >= self.num_allocated)).any().item()):
            raise ValueError("block_idx_per_voxel is outside the portable dense storage")
        return indices

    def sample_rgbw_at_centers(self, centers: torch.Tensor, block_idx_per_voxel: torch.Tensor) -> torch.Tensor:
        del centers
        indices = self._indices(block_idx_per_voxel)
        if indices.numel() == 0:
            return self.rgb_grid.new_empty((0, 4), dtype=torch.float32)
        # A dense map has exactly one geometry cell per logical entry.  Color
        # fusion is unavailable, so this returns its deterministic zero RGBW
        # accumulator rather than making up RGB values.
        return self.rgb_grid[indices, 0].float()

    def _feature_node_indices_at_centers(self, centers: torch.Tensor, block_idx_per_voxel: torch.Tensor) -> torch.Tensor:
        del centers
        return torch.zeros_like(self._indices(block_idx_per_voxel))

    def sample_features_at_centers(self, centers: torch.Tensor, block_idx_per_voxel: torch.Tensor,
                                   eps: float = 1e-6) -> torch.Tensor:
        del eps
        if self.feature_dim == 0 or self.features is None or self.feature_weight is None:
            raise RuntimeError("sample_features_at_centers() requires an enabled CUDA/Warp feature volume")
        indices = self._indices(block_idx_per_voxel)
        values = self.features[indices, 0].float()
        return values / self.feature_weight[indices, 0].float().clamp_min(torch.finfo(values.dtype).eps).unsqueeze(-1)

    def features_normalized(self, eps: float = 1e-6) -> torch.Tensor:
        if self.feature_dim == 0 or self.features is None or self.feature_weight is None:
            raise RuntimeError("features_normalized() requires an enabled CUDA/Warp feature volume")
        values = self.features[:self.num_allocated].float().sum(1)
        weights = self.feature_weight[:self.num_allocated].float().sum(1).clamp_min(eps)
        return values / weights.unsqueeze(-1)


@dataclass
class OccupiedVoxels:
    centers: torch.Tensor
    block_idx_per_voxel: torch.Tensor
    block_data: BlockDataView
    texture_colors: Optional[torch.Tensor] = None
    texture_valid: Optional[torch.Tensor] = None
    subvoxel_factor: int = 1

    def __len__(self) -> int:
        return self.centers.shape[0]

    def colors_uint8(self, eps: float = 1e-6, prefer_texture: bool = True) -> torch.Tensor:
        rgbw = self.block_data.sample_rgbw_at_centers(self.centers, self.block_idx_per_voxel)
        observed = rgbw[:, 3] > eps
        colors = (rgbw[:, :3] / rgbw[:, 3:4].clamp_min(eps) * 255).clamp(0, 255).to(torch.uint8)
        if bool((~observed).any().item()):
            colors = colors.clone()
            colors[~observed] = 128
        if prefer_texture and self.texture_colors is not None and self.texture_valid is not None:
            valid = self.texture_valid.to(device=colors.device, dtype=torch.bool)
            if valid.shape != (len(colors),):
                raise ValueError("texture_valid must be parallel to voxel centers")
            if self.texture_colors.shape != colors.shape:
                raise ValueError("texture_colors must have shape [N, 3]")
            colors = torch.where(valid[:, None], self.texture_colors.to(device=colors.device, dtype=torch.uint8), colors)
        return colors

    def features(self, eps: float = 1e-6) -> torch.Tensor:
        return self.block_data.sample_features_at_centers(self.centers, self.block_idx_per_voxel, eps)


@dataclass
class MatchedVoxels:
    voxels: OccupiedVoxels
    block_pool_idx: torch.Tensor
    block_scores: torch.Tensor

    def __len__(self) -> int:
        return len(self.voxels)

    def scores_per_voxel(self, fill_value: float = float("nan")) -> torch.Tensor:
        idx = self.voxels.block_idx_per_voxel.reshape(-1, 1)
        matched = idx == self.block_pool_idx.to(device=idx.device, dtype=idx.dtype).reshape(1, -1)
        result = self.block_scores.new_full((len(self.voxels),), fill_value)
        if matched.numel():
            any_match = matched.any(-1)
            result[any_match] = self.block_scores.to(result.device)[matched[any_match].to(torch.bool).argmax(-1)]
        return result


@dataclass
class BlockSparseTSDFData:
    """Dense tensor record using pinned field names where they have meaning."""

    block_data: torch.Tensor
    block_grid_rgb: torch.Tensor
    block_coords: torch.Tensor
    block_size: int
    decay_factor: torch.Tensor
    free_count: torch.Tensor
    free_list: torch.Tensor
    frustum_flags: torch.Tensor
    grid_shape: Tuple[int, int, int]
    hash_capacity: int
    hash_table: torch.Tensor
    max_blocks: int
    new_block_count: torch.Tensor
    new_blocks: torch.Tensor
    num_allocated: torch.Tensor
    origin: torch.Tensor
    truncation_distance: float
    voxel_size: float
    allocation_failures: torch.Tensor
    block_sums: torch.Tensor
    block_to_hash_slot: torch.Tensor
    recycle_count: torch.Tensor
    static_block_data: torch.Tensor
    static_block_sums: torch.Tensor
    feature_dim: int = 0
    feature_block_grid_size: int = 1
    color_grid_size: int = 1
    has_dynamic: bool = True
    has_static: bool = False
    has_features: bool = False
    block_features: Optional[torch.Tensor] = None
    block_feature_weight: Optional[torch.Tensor] = None

    @property
    def grid_center(self) -> torch.Tensor:
        """Pinned metadata spelling for the bounded dense map center."""
        return self.origin

    def to_warp(self):
        raise NotImplementedError("raw Warp BlockSparseTSDFData conversion is unavailable on CPU/MPS")


class BlockSparseTSDF:
    """Bounded dense TSDF lifecycle with source-compatible entry points."""

    def __init__(self, config: BlockSparseTSDFCfg, kernels=None, *, _native: Optional[PerceptionMapper] = None):
        if kernels is not None:
            raise NotImplementedError("custom Warp block-sparse kernels are unavailable on CPU/MPS")
        self.config = config
        self._native = _native or PerceptionMapper(PerceptionConfig(
            shape=config.grid_shape, voxel_size=config.voxel_size,
            grid_center=tuple(config.origin.tolist()), truncation_distance=config.truncation_distance,
            max_weight=config.accumulator_w_max, block_size=config.block_size,
        ), device=_device(config.device))
        self._failure_count = 0
        # A dense map has no allocation queue, but callers use the V2 frame
        # counters to tell whether an integration pass created new coverage.
        # Keep a compact observed-mask snapshot at ``prepare_frame`` so those
        # diagnostics remain meaningful without claiming sparse-pool ABI.
        self._frame_observed: torch.Tensor | None = None

    @classmethod
    def from_native(cls, config: BlockSparseTSDFCfg, native: PerceptionMapper) -> "BlockSparseTSDF":
        return cls(config, _native=native)

    @property
    def state(self) -> DenseMap:
        return self._native.state

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def grid_center(self) -> torch.Tensor:
        return self.config.origin

    def _observed(self) -> torch.Tensor:
        """Return the dense dynamic-observation mask as a flattened bool tensor."""
        return (self.state.weight[0].reshape(-1) > 0)

    def _logical_block_count(self, observed: torch.Tensor | None = None) -> tuple[int, int]:
        """Return ``(active, capacity)`` in logical block units.

        ``BlockSparseTSDFData`` is intentionally indexed per dense voxel in
        this backend, while its monitoring API is naturally block-oriented.
        Computing active *logical* blocks here gives callers stable capacity
        and utilization measurements independent of the dense tensor layout.
        """
        shape = self.config.grid_shape
        capacity = math.prod((math.ceil(size / self.config.block_size) for size in shape))
        mask = self._observed() if observed is None else observed
        if not bool(mask.any().item()):
            return 0, capacity
        coordinates = torch.nonzero(mask.reshape(shape), as_tuple=False)
        keys = torch.div(coordinates, self.config.block_size, rounding_mode="floor")
        return int(torch.unique(keys, dim=0).shape[0]), capacity

    def _frame_new_indices(self, observed: torch.Tensor) -> torch.Tensor:
        if self._frame_observed is None:
            return torch.empty(0, device=observed.device, dtype=torch.int32)
        baseline = self._frame_observed.to(device=observed.device)
        if baseline.shape != observed.shape:
            # This should be impossible without replacing the native mapper,
            # but avoid reporting arbitrary counters if a caller does so.
            return torch.empty(0, device=observed.device, dtype=torch.int32)
        return torch.nonzero(observed & ~baseline, as_tuple=False).flatten().to(torch.int32)

    def _data(self) -> BlockSparseTSDFData:
        state = self.state
        n = int(state.tsdf[0].numel())
        observed = self._observed()
        new_blocks = self._frame_new_indices(observed)
        coords = torch.stack(torch.meshgrid(
            *[torch.arange(v, device=state.tsdf.device, dtype=torch.int32) for v in self.config.grid_shape],
            indexing="ij",
        ), -1).reshape(-1, 3)
        block_data = torch.stack((state.tsdf[0].reshape(-1), state.weight[0].reshape(-1)), -1).unsqueeze(0)
        empty_int = torch.empty(0, device=state.tsdf.device, dtype=torch.int32)
        # RGB and learned features are deliberately unsupported, but retain a
        # correctly-indexable zero accumulator so ``BlockDataView`` queries
        # over dense flattened voxel ids have source-shaped output.
        rgb = torch.zeros((n, 1, 4), device=state.tsdf.device, dtype=state.tsdf.dtype)
        return BlockSparseTSDFData(
            block_data=block_data, block_grid_rgb=rgb,
            block_coords=coords, block_size=self.config.block_size,
            decay_factor=torch.ones(n, device=state.tsdf.device, dtype=state.tsdf.dtype),
            free_count=torch.zeros(1, device=state.tsdf.device, dtype=torch.int32), free_list=empty_int,
            frustum_flags=observed.to(torch.int32), grid_shape=self.config.grid_shape,
            hash_capacity=self.config.hash_capacity, hash_table=empty_int, max_blocks=self.config.max_blocks,
            new_block_count=torch.tensor([len(new_blocks)], device=state.tsdf.device, dtype=torch.int32), new_blocks=new_blocks,
            num_allocated=torch.tensor([n], device=state.tsdf.device, dtype=torch.int32), origin=self.config.origin.to(state.tsdf.device),
            truncation_distance=self.config.truncation_distance, voxel_size=self.config.voxel_size,
            allocation_failures=torch.tensor([self._failure_count], device=state.tsdf.device, dtype=torch.int32),
            block_sums=state.weight[0].reshape(-1), block_to_hash_slot=empty_int, recycle_count=torch.zeros(1, device=state.tsdf.device, dtype=torch.int32),
            static_block_data=torch.empty(0, device=state.tsdf.device, dtype=state.tsdf.dtype),
            static_block_sums=torch.empty(0, device=state.tsdf.device, dtype=state.tsdf.dtype),
            has_dynamic=self.config.enable_dynamic,
            has_static=self.config.enable_static,
        )

    @property
    def data(self) -> BlockSparseTSDFData:
        return self._data()

    def get_warp_data(self):
        raise NotImplementedError("raw Warp BlockSparseTSDF storage is unavailable on CPU/MPS")

    def invalidate_cache(self) -> None:
        # The production dense mapper has no host hash/cache mirror.  A frame
        # snapshot however may no longer describe a caller-replaced state.
        self._frame_observed = None

    def reset(self) -> None:
        self._native.reset()
        self.reset_failure_counter()
        self._frame_observed = None

    def export_blocks(self) -> Dict[str, torch.Tensor]:
        state = self.state
        return {name: getattr(state, name).detach().clone() for name in
                ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")}

    def import_blocks(self, blocks: Dict[str, torch.Tensor]) -> None:
        expected = {"tsdf", "weight", "occupancy", "esdf", "gradient", "generation"}
        native_checkpoint = set(blocks) == expected | {"format", "version", "config"}
        if set(blocks) != expected and not native_checkpoint:
            raise NotImplementedError("importing CUDA/Warp block-pool payloads is unavailable; pass a portable dense state")
        if native_checkpoint:
            # PerceptionMapper owns a small self-describing checkpoint envelope.
            # Validate it through its public loader so configuration mismatches
            # are not silently accepted, then return.
            self._native.load_state_dict(blocks)
            return
        current = self.state
        for name in expected:
            value = blocks[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.shape != getattr(current, name).shape:
                raise ValueError(f"{name} shape does not match dense mapper storage")
        checkpoint = self._native.state_dict()
        checkpoint.update({name: blocks[name].to(device=current.tsdf.device) for name in expected})
        self._native.load_state_dict(checkpoint)

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False) -> Dict[str, float]:
        observed = self._observed()
        observed_voxels = int(observed.sum().item())
        active_blocks, capacity_blocks = self._logical_block_count(observed)
        # There is no free-list or hash table in a fixed dense tensor.  Keep
        # the V2 keys so monitoring integrations can run, but represent their
        # actual portable meaning rather than fabricated hash occupancy.
        stats: Dict[str, float] = {
            "num_allocated": active_blocks,
            "free_count": capacity_blocks - active_blocks,
            "active_blocks": active_blocks,
            "holes": 0,
            "recycled_last": 0,
            "tombstone_count": 0,
            "pool_usage_pct": active_blocks / max(capacity_blocks, 1) * 100.0,
            "fragmentation_pct": 0.0,
            "hash_load_pct": 0.0,
            "observed_voxels": observed_voxels,
            "dense_capacity_voxels": int(observed.numel()),
            "dense_logical_blocks": capacity_blocks,
            "dense_observed_fraction_pct": observed_voxels / max(int(observed.numel()), 1) * 100.0,
            "allocation_failures": self._failure_count,
            "storage": "dense_portable",
        }
        if not scan_pool:
            # The value is exact for dense storage; the flag only controls an
            # expensive sparse-pool invariant upstream.
            stats.pop("holes")
        if scan_hash:
            stats.update({"hash_empty": 0, "hash_tomb": 0, "hash_occ": 0})
        return stats

    def reset_failure_counter(self) -> None:
        self._failure_count = 0

    def compact_hash_table(self) -> None:
        # A dense tensor has neither probe chains nor tombstones.
        return None

    def memory_usage_bytes(self) -> int:
        state = self.state
        return sum(value.numel() * value.element_size() for value in (
            state.tsdf, state.weight, state.occupancy, state.esdf, state.gradient, state.generation,
        ))

    def memory_usage_mb(self) -> float:
        return self.memory_usage_bytes() / 2**20

    def prepare_frame(self) -> None:
        # Dense tensors are allocated at construction, but preserving the
        # observed mask lets ``data.new_blocks`` report first observations made
        # by the next integration pass.  Clone intentionally owns the
        # snapshot: callers commonly mutate the returned state in-place.
        self._frame_observed = self._observed().detach().clone()
