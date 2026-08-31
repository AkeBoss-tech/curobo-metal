from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.util.warp import warp_constant_suffix, warp_func, wp


def make_coord_kernels(
    block_size: int,
    *,
    grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
) -> dict[str, object]:
    return unsupported_kernel(
        block_size,
        grid_shape=grid_shape,
        origin_xyz=origin_xyz,
        voxel_size=voxel_size,
    )
