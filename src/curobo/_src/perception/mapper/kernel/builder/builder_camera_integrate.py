"""Portable declaration boundary for camera integration Warp kernels.

The high-level mapper provides CPU/MPS camera integration. The generated Warp
kernels themselves are CUDA ABI and deliberately unavailable here; this module
retains the pinned factory contract so callers fail explicitly rather than at
import or argument-binding time.
"""

from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    compute_tsdf_weight,
    floor_div,
)
from curobo._src.util.warp import warp_constant_suffix, warp_kernel, wp


def make_camera_integrate_kernels(
    block_size: int,
    *,
    feature_dim: int,
    num_cameras: int,
    image_height: int,
    image_width: int,
    num_samples: int,
    grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    truncation_distance: float,
    feature_grid_shape: tuple[int, int] | None,
    feature_channels_per_thread: int,
    max_feature_tile_channels: int,
    max_support_pixels_per_block_camera: int,
    color_grid_size: int,
    feature_block_grid_size: int,
    pack_key_only,
    unpack_block_key,
    find_or_insert_block,
    hash_lookup,
    voxel_to_world,
    voxel_to_world_corner,
    world_to_continuous_voxel,
    block_local_to_world,
    block_grid_to_key_coords,
    block_key_to_grid_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Reject raw CUDA/Warp camera-kernel construction on CPU/MPS."""
    return unsupported_kernel(
        block_size,
        feature_dim=feature_dim,
        num_cameras=num_cameras,
        image_height=image_height,
        image_width=image_width,
        num_samples=num_samples,
        grid_shape=grid_shape,
        origin_xyz=origin_xyz,
        voxel_size=voxel_size,
        truncation_distance=truncation_distance,
        feature_grid_shape=feature_grid_shape,
        feature_channels_per_thread=feature_channels_per_thread,
        max_feature_tile_channels=max_feature_tile_channels,
        max_support_pixels_per_block_camera=max_support_pixels_per_block_camera,
        color_grid_size=color_grid_size,
        feature_block_grid_size=feature_block_grid_size,
        pack_key_only=pack_key_only,
        unpack_block_key=unpack_block_key,
        find_or_insert_block=find_or_insert_block,
        hash_lookup=hash_lookup,
        voxel_to_world=voxel_to_world,
        voxel_to_world_corner=voxel_to_world_corner,
        world_to_continuous_voxel=world_to_continuous_voxel,
        block_local_to_world=block_local_to_world,
        block_grid_to_key_coords=block_grid_to_key_coords,
        block_key_to_grid_coords=block_key_to_grid_coords,
        block_key_to_voxel_base=block_key_to_voxel_base,
    )
