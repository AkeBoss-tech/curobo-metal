"""Portable source-shaped block-sparse kernel bundle.

The CUDA implementation builds Warp functions and kernels.  CPU/MPS callers
still need the bundle's configuration and host-side hash/coordinate helpers;
raw device launches remain explicit boundary errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, TypeAlias

import torch

from curobo._src.perception.mapper._portable import RAW_KERNEL_MESSAGE
from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    HASH_EMPTY,
    HashLayout,
    _validate_block_size,
    _validate_color_grid_size,
    _validate_feature_block_grid_size,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
)
from curobo._src.perception.mapper.kernel.builder.builder_camera_integrate import make_camera_integrate_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_coord import make_coord_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_decay import make_decay_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_esdf import make_esdf_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_hash import make_hash_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_lidar_integrate import make_lidar_integrate_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_mesh import make_mesh_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_raycast import make_raycast_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_rescale import make_rescale_kernels  # noqa: F401
from curobo._src.perception.mapper.kernel.builder.builder_stamp import make_stamp_kernels  # noqa: F401
from curobo._src.util.logging import log_and_raise  # noqa: F401

WarpKernel: TypeAlias = Any
WarpFunction: TypeAlias = Any


class _PortableKernel:
    """Named marker for a raw kernel unavailable on CPU/MPS."""

    def __init__(self, key: str):
        self.key = key

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NotImplementedError(RAW_KERNEL_MESSAGE)

    def __repr__(self) -> str:
        return f"_PortableKernel({self.key!r})"


def _pack_key(bx: int, by: int, bz: int, layout: HashLayout) -> int:
    x = (int(bx) + layout.coord_bias_xyz[0]) & layout.coord_masks_xyz[0]
    y = (int(by) + layout.coord_bias_xyz[1]) & layout.coord_masks_xyz[1]
    z = (int(bz) + layout.coord_bias_xyz[2]) & layout.coord_masks_xyz[2]
    return (x << layout.x_shift) | (y << layout.y_shift) | (z << layout.z_shift)


def _unpack_key(key: int, layout: HashLayout) -> tuple[int, int, int]:
    return tuple(
        int((int(key) >> shift) & mask) - bias
        for shift, mask, bias in zip(
            (layout.x_shift, layout.y_shift, layout.z_shift),
            layout.coord_masks_xyz,
            layout.coord_bias_xyz,
        )
    )


def _hash_exports(block_size: int, layout: HashLayout) -> dict[str, Any]:
    suffix = f"bs{block_size}_{layout.name}"

    def pack_entry(bx: int, by: int, bz: int, pool_idx: int) -> int:
        return _pack_key(bx, by, bz, layout) | (int(pool_idx) & layout.value_mask)

    def pack_key_only(bx: int, by: int, bz: int) -> int:
        return _pack_key(bx, by, bz, layout)

    def unpack_entry(entry: int) -> tuple[int, int, int, int]:
        return (*_unpack_key(int(entry), layout), int(entry) & layout.value_mask)

    def get_pool_idx(entry: int) -> int:
        return int(entry) & layout.value_mask

    def get_key_part(entry: int) -> int:
        return int(entry) & layout.key_mask

    def is_valid_block_key(bx: int, by: int, bz: int) -> bool:
        return all(lo <= int(v) <= hi for v, lo, hi in zip(
            (bx, by, bz), layout.coord_min_xyz, layout.coord_max_xyz
        ))

    def spatial_hash(bx: int, by: int, bz: int, capacity: int) -> int:
        if int(capacity) <= 0:
            raise ValueError("hash capacity must be positive")
        value = (int(bx) * 73856093) ^ (int(by) * 19349663) ^ (int(bz) * 83492791)
        return (value & 0x7FFFFFFFFFFFFFFF) % int(capacity)

    def pack_rgb(r: int, g: int, b: int) -> int:
        return (int(r) << 16) | (int(g) << 8) | int(b)

    def hash_lookup(hash_table: Any, bx: int, by: int, bz: int, capacity: int) -> int:
        if not is_valid_block_key(bx, by, bz):
            return -1
        target = pack_key_only(bx, by, bz)
        slot = spatial_hash(bx, by, bz, capacity)
        for _ in range(int(capacity)):
            entry = int(hash_table[slot])
            if entry == int(HASH_EMPTY):
                return -1
            if get_key_part(entry) == target:
                pool = get_pool_idx(entry)
                return -1 if pool == layout.pending_pool_idx else pool
            slot = (slot + 1) % int(capacity)
        return -1

    result = {
        "pack_entry": pack_entry, "pack_key_only": pack_key_only,
        "unpack_entry": unpack_entry, "get_pool_idx": get_pool_idx,
        "get_key_part": get_key_part, "is_valid_block_key": is_valid_block_key,
        "spatial_hash": spatial_hash, "pack_rgb": pack_rgb,
        "hash_lookup": hash_lookup, "pack_block_key": pack_key_only,
        "unpack_block_key": _unpack_key,
    }
    for name in (
        "free_list_pop", "free_list_push", "spin_until_ready", "find_or_insert_block",
        "hash_table_insert_with_pool_idx", "read_tsdf_voxel", "write_tsdf_voxel",
    ):
        result[name] = _PortableKernel(f"{name}_{suffix}")
    for name, fn in result.items():
        if not isinstance(fn, _PortableKernel):
            setattr(fn, "key", f"{name}_{suffix}")
    return result


_FUNCTION_FIELDS = {
    "pack_entry", "pack_key_only", "unpack_entry", "get_pool_idx", "get_key_part",
    "is_valid_block_key", "spatial_hash", "pack_rgb", "hash_lookup", "free_list_pop",
    "free_list_push", "spin_until_ready", "find_or_insert_block", "hash_table_insert_with_pool_idx",
    "read_tsdf_voxel", "write_tsdf_voxel", "pack_block_key", "unpack_block_key",
    "world_to_continuous_voxel", "voxel_to_world", "voxel_to_world_corner", "block_offsets",
    "block_grid_to_key_coords", "block_key_to_grid_coords", "block_key_to_voxel_base",
    "world_to_block_coords", "world_to_block_and_local", "block_local_to_world",
    "local_to_linear_index", "linear_to_local_coords", "sample_voxel", "sample_tsdf",
    "_sample_voxel_at_block_local", "sample_tsdf_trilinear", "sample_rgb", "compute_gradient",
    "compute_gradient_nearest", "refine_hit_bisection", "ray_block_exit_t",
    "lookup_combined_sdf_at_esdf_coords", "lookup_static_sdf_at_esdf_coords",
}

_KERNEL_NAMES = {
    "clear_new_blocks_kernel", "clear_new_block_features_kernel", "clear_new_block_grid_rgb_kernel",
    "mark_blocks_in_frustum_kernel", "mark_lidar_blocks_in_frustum_kernel", "recycle_empty_blocks_kernel",
    "preallocate_unique_blocks_kernel", "enumerate_blocks_from_aabb_kernel", "filter_blocks_by_sdf_kernel",
    "stamp_sdf_kernel", "update_block_grid_rgb_kernel", "raycast_block_sparse_kernel",
    "raycast_block_sparse_color_kernel", "count_surface_voxels_kernel", "count_occupied_voxels_kernel",
    "count_occupied_voxels_masked_kernel", "extract_occupied_voxels_kernel", "extract_occupied_voxels_masked_kernel",
    "extract_surface_voxels_kernel", "append_active_blocks_kernel", "count_surface_cubes_from_blocks_kernel",
    "append_surface_cubes_from_blocks_kernel", "count_total_triangles_kernel", "generate_mesh_kernel",
    "project_mesh_uvs_kernel", "sample_vertex_colors_kernel", "rescale_block_accumulators_kernel",
    "rescale_block_grid_rgb_kernel", "compute_block_keys_only_kernel", "allocate_visible_blocks_from_keys_kernel",
    "build_support_pixels_from_keys_kernel", "collect_blocks_in_aabb_kernel", "clear_blocks_by_pool_kernel",
    "clear_block_features_by_pool_kernel", "clear_block_grid_rgb_by_pool_kernel", "integrate_voxels_kernel",
    "integrate_block_grid_rgb_kernel", "integrate_features_from_support_grouped_kernel",
    "integrate_features_from_support_tiled_kernel", "integrate_features_grouped_kernel",
    "lidar_compute_block_keys_only_kernel", "lidar_build_support_pixels_from_keys_kernel",
    "lidar_integrate_voxels_kernel", "lidar_integrate_block_grid_rgb_kernel",
    "lidar_integrate_features_from_support_grouped_kernel", "lidar_integrate_features_from_support_tiled_kernel",
    "seed_esdf_sites_from_block_sparse_kernel", "seed_esdf_sites_gather_kernel",
    "compute_esdf_from_min_tsdf_kernel",
}


@dataclass(frozen=True, init=False)
class BlockSparseKernels:
    """Complete source-shaped bundle backed by portable Torch metadata."""

    block_size: int
    feature_dim: int = 0
    num_cameras: int = 1
    image_height: int = 1
    image_width: int = 1
    texture_num_cameras: int = 1
    texture_camera_image_height: int = 1
    texture_camera_image_width: int = 1
    lidar_num_sensors: int = 0
    lidar_image_height: int = 1
    lidar_image_width: int = 1
    num_samples: int = 1
    grid_shape: tuple[int, int, int] = (1, 1, 1)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    voxel_size: float = 1.0
    truncation_distance: float = 0.0
    feature_grid_shape: tuple[int, int] | None = None
    lidar_feature_grid_shape: tuple[int, int] | None = None
    esdf_grid_shape: tuple[int, int, int] = (1, 1, 1)
    feature_channels_per_thread: int = 4
    max_feature_tile_channels: int = 4096
    max_support_pixels_per_block_camera: int = 32
    max_support_pixels_per_block_lidar: int = 32
    color_grid_size: int = 1
    color_grid_voxels: int = 1
    feature_block_grid_size: int = 1
    feature_grid_voxels: int = 1
    hash_layout: HashLayout = DEFAULT_HASH_LAYOUT
    # Warp-shaped function/kernel slots.  These are named markers on the
    # portable backend, but remain dataclass fields for introspection parity.
    pack_entry: WarpFunction = None
    pack_key_only: WarpFunction = None
    unpack_entry: WarpFunction = None
    get_pool_idx: WarpFunction = None
    get_key_part: WarpFunction = None
    is_valid_block_key: WarpFunction = None
    spatial_hash: WarpFunction = None
    pack_rgb: WarpFunction = None
    hash_lookup: WarpFunction = None
    free_list_pop: WarpFunction = None
    free_list_push: WarpFunction = None
    spin_until_ready: WarpFunction = None
    find_or_insert_block: WarpFunction = None
    hash_table_insert_with_pool_idx: WarpFunction = None
    read_tsdf_voxel: WarpFunction = None
    write_tsdf_voxel: WarpFunction = None
    pack_block_key: WarpFunction = None
    unpack_block_key: WarpFunction = None
    world_to_continuous_voxel: WarpFunction = None
    voxel_to_world: WarpFunction = None
    voxel_to_world_corner: WarpFunction = None
    block_offsets: WarpFunction = None
    block_grid_to_key_coords: WarpFunction = None
    block_key_to_grid_coords: WarpFunction = None
    block_key_to_voxel_base: WarpFunction = None
    world_to_block_coords: WarpFunction = None
    world_to_block_and_local: WarpFunction = None
    block_local_to_world: WarpFunction = None
    local_to_linear_index: WarpFunction = None
    linear_to_local_coords: WarpFunction = None
    sample_voxel: WarpFunction = None
    sample_tsdf: WarpFunction = None
    _sample_voxel_at_block_local: WarpFunction = None
    sample_tsdf_trilinear: WarpFunction = None
    sample_rgb: WarpFunction = None
    compute_gradient: WarpFunction = None
    compute_gradient_nearest: WarpFunction = None
    refine_hit_bisection: WarpFunction = None
    ray_block_exit_t: WarpFunction = None
    lookup_combined_sdf_at_esdf_coords: WarpFunction = None
    lookup_static_sdf_at_esdf_coords: WarpFunction = None
    block_empty_threshold: Any = None
    clear_new_blocks_kernel: WarpKernel = None
    clear_new_block_features_kernel: WarpKernel = None
    clear_new_block_grid_rgb_kernel: WarpKernel = None
    mark_blocks_in_frustum_kernel: WarpKernel = None
    mark_lidar_blocks_in_frustum_kernel: WarpKernel = None
    recycle_empty_blocks_kernel: WarpKernel = None
    preallocate_unique_blocks_kernel: WarpKernel = None
    enumerate_blocks_from_aabb_kernel: WarpKernel = None
    filter_blocks_by_sdf_kernel: WarpKernel = None
    stamp_sdf_kernel: WarpKernel = None
    update_block_grid_rgb_kernel: WarpKernel = None
    raycast_block_sparse_kernel: WarpKernel = None
    raycast_block_sparse_color_kernel: WarpKernel = None
    count_surface_voxels_kernel: WarpKernel = None
    count_occupied_voxels_kernel: WarpKernel = None
    count_occupied_voxels_masked_kernel: WarpKernel = None
    extract_occupied_voxels_kernel: WarpKernel = None
    extract_occupied_voxels_masked_kernel: WarpKernel = None
    extract_surface_voxels_kernel: WarpKernel = None
    append_active_blocks_kernel: WarpKernel = None
    count_surface_cubes_from_blocks_kernel: WarpKernel = None
    append_surface_cubes_from_blocks_kernel: WarpKernel = None
    count_total_triangles_kernel: WarpKernel = None
    generate_mesh_kernel: WarpKernel = None
    project_mesh_uvs_kernel: WarpKernel = None
    sample_vertex_colors_kernel: WarpKernel = None
    rescale_block_accumulators_kernel: WarpKernel = None
    rescale_block_grid_rgb_kernel: WarpKernel = None
    compute_block_keys_only_kernel: WarpKernel = None
    allocate_visible_blocks_from_keys_kernel: WarpKernel = None
    build_support_pixels_from_keys_kernel: WarpKernel = None
    collect_blocks_in_aabb_kernel: WarpKernel = None
    clear_blocks_by_pool_kernel: WarpKernel = None
    clear_block_features_by_pool_kernel: WarpKernel = None
    clear_block_grid_rgb_by_pool_kernel: WarpKernel = None
    integrate_voxels_kernel: WarpKernel = None
    integrate_block_grid_rgb_kernel: WarpKernel = None
    integrate_features_from_support_grouped_kernel: WarpKernel = None
    integrate_features_from_support_tiled_kernel: WarpKernel = None
    integrate_features_grouped_kernel: WarpKernel = None
    lidar_compute_block_keys_only_kernel: WarpKernel = None
    lidar_build_support_pixels_from_keys_kernel: WarpKernel = None
    lidar_integrate_voxels_kernel: WarpKernel = None
    lidar_integrate_block_grid_rgb_kernel: WarpKernel = None
    lidar_integrate_features_from_support_grouped_kernel: WarpKernel = None
    lidar_integrate_features_from_support_tiled_kernel: WarpKernel = None
    seed_esdf_sites_from_block_sparse_kernel: WarpKernel = None
    seed_esdf_sites_gather_kernel: WarpKernel = None
    compute_esdf_from_min_tsdf_kernel: WarpKernel = None


def _bundle_init(self: BlockSparseKernels, **kwargs: Any) -> None:
    metadata = {field.name: field.default for field in fields(type(self))}
    if "block_size" not in kwargs:
        raise TypeError("block_size is required")
    for name, value in kwargs.items():
        if name in metadata:
            metadata[name] = value
    for name, value in metadata.items():
        object.__setattr__(self, name, value)
    for name, value in kwargs.items():
        if name not in metadata:
            object.__setattr__(self, name, value)


# Dynamic assignment keeps AST-visible dataclass constructor compatibility.
BlockSparseKernels.__init__ = _bundle_init

def _resolve(cfg: Any | None, name: str, default: Any) -> Any:
    if cfg is None or isinstance(cfg, int):
        return default
    return getattr(cfg, name, default)


def _geometry_exports(block_size: int, values: dict[str, Any]) -> dict[str, Any]:
    voxel, origin, grid = values["voxel_size"], values["origin_xyz"], values["grid_shape"]

    def world_to_continuous_voxel(pos: Any):
        p = torch.as_tensor(pos)
        return (p - p.new_tensor(origin)) / voxel + p.new_tensor(grid) / 2

    def world_to_block_coords(pos: Any):
        return torch.floor(world_to_continuous_voxel(pos) / block_size).to(torch.int32)

    def voxel_to_world(coords: Any):
        c = torch.as_tensor(coords)
        return (c + 0.5) * voxel + c.new_tensor(origin)

    def voxel_to_world_corner(coords: Any):
        c = torch.as_tensor(coords)
        return c * voxel + c.new_tensor(origin)

    def local_to_linear_index(lx: int, ly: int, lz: int) -> int:
        return int(lz) * block_size * block_size + int(ly) * block_size + int(lx)

    def linear_to_local_coords(index: int) -> tuple[int, int, int]:
        i = int(index)
        return i % block_size, (i // block_size) % block_size, i // (block_size * block_size)

    return {
        "world_to_continuous_voxel": world_to_continuous_voxel,
        "world_to_block_coords": world_to_block_coords,
        "voxel_to_world": voxel_to_world,
        "voxel_to_world_corner": voxel_to_world_corner,
        "local_to_linear_index": local_to_linear_index,
        "linear_to_local_coords": linear_to_local_coords,
    }


def make_block_sparse_kernels(
    cfg: Any | None = None,
    *,
    block_size: int | None = None,
    seeding_method: str | None = None,
    feature_channels_per_thread: int | None = None,
) -> BlockSparseKernels:
    """Create a fresh, source-shaped portable kernel bundle."""
    raw_size = block_size if block_size is not None else _resolve(cfg, "block_size", 8)
    _validate_block_size(raw_size)
    size = int(raw_size)
    feature_dim = int(_resolve(cfg, "feature_dim", 0))
    raw_fcpt = feature_channels_per_thread if feature_channels_per_thread is not None else _resolve(cfg, "feature_channels_per_thread", 4)
    _validate_feature_channels_per_thread(raw_fcpt)
    fcpt = int(raw_fcpt)
    grid = tuple(int(v) for v in _resolve(cfg, "grid_shape", (1, 1, 1)))
    raw_origin = _resolve(cfg, "origin", (0.0, 0.0, 0.0))
    if hasattr(raw_origin, "detach"):
        raw_origin = raw_origin.detach().cpu().reshape(-1).tolist()
    origin = tuple(float(v) for v in raw_origin)
    voxel, trunc = float(_resolve(cfg, "voxel_size", 1.0)), float(_resolve(cfg, "truncation_distance", 0.0))
    color, feature_grid = int(_resolve(cfg, "color_grid_size", 1)), int(_resolve(cfg, "feature_block_grid_size", 1))
    _validate_color_grid_size(color, size)
    _validate_feature_block_grid_size(feature_grid, size)
    fg_h, fg_w = _resolve(cfg, "feature_grid_height", None), _resolve(cfg, "feature_grid_width", None)
    _validate_feature_grid_shape(feature_dim, fg_h, fg_w)
    if seeding_method is not None and seeding_method not in {"gather", "scatter"}:
        raise ValueError("seeding_method must be 'gather' or 'scatter'")
    num_cameras = int(_resolve(cfg, "num_cameras", 1) or 1)
    image_height = int(_resolve(cfg, "image_height", 1) or 1)
    image_width = int(_resolve(cfg, "image_width", 1) or 1)
    texture_num_cameras = int(_resolve(cfg, "texture_num_cameras", num_cameras) or num_cameras)
    texture_height = int(_resolve(cfg, "texture_camera_image_height", image_height) or image_height)
    texture_width = int(_resolve(cfg, "texture_camera_image_width", image_width) or image_width)
    values = {
        "block_size": size, "feature_dim": feature_dim,
        "num_cameras": num_cameras, "image_height": image_height, "image_width": image_width,
        "texture_num_cameras": texture_num_cameras, "texture_camera_image_height": texture_height, "texture_camera_image_width": texture_width,
        "lidar_num_sensors": int(_resolve(cfg, "lidar_num_sensors", 0)), "lidar_image_height": int(_resolve(cfg, "lidar_image_height", 1) or 1), "lidar_image_width": int(_resolve(cfg, "lidar_image_width", 1) or 1),
        "num_samples": int(_resolve(cfg, "num_samples", max(1, math.ceil(2 * trunc / max(voxel * size, 1e-9)) + 1))),
        "grid_shape": grid, "origin_xyz": origin, "voxel_size": voxel, "truncation_distance": trunc,
        "feature_grid_shape": None if fg_h is None else (int(fg_h), int(fg_w)), "lidar_feature_grid_shape": None,
        "esdf_grid_shape": tuple(int(v) for v in _resolve(cfg, "esdf_grid_shape", (1, 1, 1))),
        "feature_channels_per_thread": fcpt, "max_feature_tile_channels": int(_resolve(cfg, "max_feature_tile_channels", 4096)), "max_support_pixels_per_block_camera": int(_resolve(cfg, "max_support_pixels_per_block_camera", 32)), "max_support_pixels_per_block_lidar": int(_resolve(cfg, "max_support_pixels_per_block_lidar", 32)),
        "color_grid_size": color, "color_grid_voxels": color ** 3, "feature_block_grid_size": feature_grid, "feature_grid_voxels": feature_grid ** 3, "hash_layout": DEFAULT_HASH_LAYOUT,
        "block_empty_threshold": float(_resolve(cfg, "block_empty_threshold", 0.1)),
    }
    exports = _hash_exports(size, DEFAULT_HASH_LAYOUT)
    exports.update(_geometry_exports(size, values))
    for name in _KERNEL_NAMES:
        if name not in exports:
            exports[name] = _PortableKernel(f"{name}_bs{size}_fcpt{fcpt}")
    values.update(exports)
    return BlockSparseKernels(**values)


__all__ = ["BlockSparseKernels", "make_block_sparse_kernels"]
