"""Pinned raw Warp marching-cubes sampling declarations.

The public functions in this module are Warp device functions upstream.  A
Metal/CPU process cannot execute that ABI, so they deliberately retain their
declaration shape and fail at the raw-kernel boundary.  Portable mesh
extraction is exposed through the Mapper API instead.
"""

from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.util.warp import wp

try:
    import warp as _raw_wp
except ImportError:
    _raw_wp = None

if _raw_wp is not None:
    wp = _raw_wp
if wp is None:
    class _PortableWarp:
        """Enough of Warp's declaration surface to import raw ABI modules."""

        @staticmethod
        def constant(value):
            return value

        @staticmethod
        def func(function):
            return function

    wp = _PortableWarp()


UNOBSERVED_SDF = wp.constant(1000.0)


@wp.func
def sample_sdf_with_weight(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
) -> wp.float32:
    return unsupported_kernel(
        sdf_weight_grid, weight_grid, i, j, k, minimum_tsdf_weight, level
    )


@wp.func
def trilinear_sample_sdf_weighted(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),
    pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
) -> wp.float32:
    return unsupported_kernel(
        sdf_weight_grid,
        weight_grid,
        pos,
        origin,
        voxel_size,
        minimum_tsdf_weight,
        level,
    )


@wp.func
def estimate_sdf_gradient_weighted(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),
    pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
) -> wp.vec3:
    return unsupported_kernel(
        sdf_weight_grid,
        weight_grid,
        pos,
        origin,
        voxel_size,
        minimum_tsdf_weight,
        level,
    )


@wp.func
def refine_vertex_weighted(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),
    vertex: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
    iterations: wp.int32,
) -> wp.vec3:
    return unsupported_kernel(
        sdf_weight_grid,
        weight_grid,
        vertex,
        origin,
        voxel_size,
        minimum_tsdf_weight,
        level,
        iterations,
    )
