from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.util.warp import warp_func, warp_kernel, wp


def make_raycast_kernels(
    block_size: int,
    *,
    hash_lookup,
    voxel_to_world,
    voxel_to_world_corner,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
    world_to_block_coords,
    world_to_block_and_local,
    world_to_continuous_voxel,
    color_grid_size: int = 1,
) -> dict[str, object]:
    return unsupported_kernel(
        block_size,
        hash_lookup=hash_lookup,
        voxel_to_world=voxel_to_world,
        voxel_to_world_corner=voxel_to_world_corner,
        block_grid_to_key_coords=block_grid_to_key_coords,
        block_key_to_voxel_base=block_key_to_voxel_base,
        world_to_block_coords=world_to_block_coords,
        world_to_block_and_local=world_to_block_and_local,
        world_to_continuous_voxel=world_to_continuous_voxel,
        color_grid_size=color_grid_size,
    )
