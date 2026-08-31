"""Portable value-level TSDF sampling with cuRobo's Warp-facing API."""

from __future__ import annotations

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.util.warp import wp as _raw_wp


class _WarpCompat:
    """Declaration façade; it never creates or launches Warp objects."""

    int32 = int
    bool = bool

    @staticmethod
    def float32(value):
        return float(value)

    @staticmethod
    def constant(value):
        return value

    @staticmethod
    def func(function):
        return function


wp = _raw_wp if _raw_wp is not None else _WarpCompat()


SDF_INFINITY = wp.constant(wp.float32(1e10))


@wp.func
def sample_dynamic_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    min_weight: wp.float32,
) -> wp.float32:
    """Return the weighted dynamic SDF, or infinity when it is unobserved."""
    if not tsdf.has_dynamic:
        return SDF_INFINITY
    sdf_weight = tsdf.block_data[pool_idx, local_idx, 0]
    weight = tsdf.block_data[pool_idx, local_idx, 1]
    if weight > min_weight:
        return sdf_weight / weight
    return SDF_INFINITY


@wp.func
def sample_static_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.float32:
    """Return the primitive/static SDF, or infinity when that channel is off."""
    if not tsdf.has_static:
        return SDF_INFINITY
    return tsdf.static_block_data[pool_idx, local_idx]


@wp.func
def sample_combined_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    min_weight: wp.float32,
) -> wp.float32:
    """Combine dynamic and static obstacle channels conservatively."""
    dynamic_sdf = sample_dynamic_sdf(tsdf, pool_idx, local_idx, min_weight)
    static_sdf = sample_static_sdf(tsdf, pool_idx, local_idx)
    return dynamic_sdf if dynamic_sdf <= static_sdf else static_sdf


@wp.func
def has_valid_observation(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    min_weight: wp.float32,
) -> wp.bool:
    """Whether either enabled channel contains a valid voxel observation."""
    if tsdf.has_dynamic and tsdf.block_data[pool_idx, local_idx, 1] > min_weight:
        return True
    if tsdf.has_static and tsdf.static_block_data[pool_idx, local_idx] < SDF_INFINITY:
        return True
    return False
