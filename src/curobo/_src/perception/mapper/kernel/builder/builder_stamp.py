from __future__ import annotations

from typing import Any

from curobo._src.perception.mapper._portable import unsupported_kernel

annotations = None
warp_func = warp_kernel = wp = None
compute_local_sdf = is_obs_enabled = load_obstacle_transform = unsupported_kernel
warp_constant_suffix = None


def make_stamp_kernels(
    block_size: int,
    *,
    grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    truncation_distance: float,
    pack_key_only,
    unpack_block_key,
    block_local_to_world,
    hash_lookup,
    hash_table_insert_with_pool_idx,
    free_list_pop,
    color_grid_size: int,
) -> dict[str, object]:
    return unsupported_kernel(
        block_size, grid_shape=grid_shape, origin_xyz=origin_xyz,
        voxel_size=voxel_size, truncation_distance=truncation_distance,
        pack_key_only=pack_key_only, unpack_block_key=unpack_block_key,
        block_local_to_world=block_local_to_world, hash_lookup=hash_lookup,
        hash_table_insert_with_pool_idx=hash_table_insert_with_pool_idx,
        free_list_pop=free_list_pop, color_grid_size=color_grid_size,
    )
