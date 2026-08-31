"""Portable declaration boundary for TSDF decay/recycling Warp kernels."""

from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.constants import HASH_TOMBSTONE, REFERENCE_BLOCK_SIZE
from curobo._src.util.warp import warp_constant_suffix, warp_kernel, wp


def make_decay_kernels(
    block_size: int,
    *,
    grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    num_cameras: int,
    image_height: int,
    image_width: int,
    free_list_push,
) -> dict[str, object]:
    """Reject raw CUDA/Warp decay-kernel construction on CPU/MPS."""
    return unsupported_kernel(
        block_size,
        grid_shape=grid_shape,
        origin_xyz=origin_xyz,
        voxel_size=voxel_size,
        num_cameras=num_cameras,
        image_height=image_height,
        image_width=image_width,
        free_list_push=free_list_push,
    )
