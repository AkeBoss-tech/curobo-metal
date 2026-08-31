from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel

BlockSparseTSDFWarp = object
pack_site_coords = sample_combined_sdf = sample_static_sdf = unsupported_kernel
warp_constant_suffix = warp_func = warp_kernel = wp = None
annotations = None


def make_esdf_kernels(
    block_size: int,
    *,
    grid_shape: tuple[int, int, int],
    esdf_grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    truncation_distance: float,
    hash_lookup,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    return unsupported_kernel(
        block_size, grid_shape=grid_shape, esdf_grid_shape=esdf_grid_shape,
        origin_xyz=origin_xyz, voxel_size=voxel_size,
        truncation_distance=truncation_distance, hash_lookup=hash_lookup,
        block_grid_to_key_coords=block_grid_to_key_coords,
        block_key_to_voxel_base=block_key_to_voxel_base,
    )
