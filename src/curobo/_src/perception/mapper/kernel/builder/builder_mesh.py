"""Portable declaration boundary for block-sparse marching-cubes kernels."""

from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    get_edge_vertex,
)
from curobo._src.util.warp import warp_func, warp_kernel, wp


def make_mesh_kernels(
    block_size: int,
    *,
    num_cameras: int = 1,
    image_height: int = 1,
    image_width: int = 1,
    hash_lookup,
    sample_rgb,
    sample_voxel,
    sample_tsdf_trilinear,
    compute_gradient,
    compute_gradient_nearest,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Reject raw CUDA/Warp mesh-kernel construction on CPU/MPS."""
    return unsupported_kernel(
        block_size,
        num_cameras=num_cameras,
        image_height=image_height,
        image_width=image_width,
        hash_lookup=hash_lookup,
        sample_rgb=sample_rgb,
        sample_voxel=sample_voxel,
        sample_tsdf_trilinear=sample_tsdf_trilinear,
        compute_gradient=compute_gradient,
        compute_gradient_nearest=compute_gradient_nearest,
        block_grid_to_key_coords=block_grid_to_key_coords,
        block_key_to_voxel_base=block_key_to_voxel_base,
    )
