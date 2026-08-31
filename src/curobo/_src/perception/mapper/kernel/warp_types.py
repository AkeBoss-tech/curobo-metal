"""Pinned Warp sparse-TSDF declaration with an explicit portable boundary."""

from __future__ import annotations


class _BlockSparseTSDFWarpUnavailable:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "raw Warp block-sparse struct ABI is unavailable on CPU/MPS"
        )


class _PortableWarpDeclarations:
    @staticmethod
    def struct(_declared_class):
        return _BlockSparseTSDFWarpUnavailable


wp = _PortableWarpDeclarations()


@wp.struct
class BlockSparseTSDFWarp:
    hash_table: wp.array(dtype=wp.int64)
    block_data: wp.array3d(dtype=wp.float16)
    block_grid_rgb: wp.array3d(dtype=wp.float16)
    block_features: wp.array3d(dtype=wp.float16)
    block_feature_weight: wp.array2d(dtype=wp.float16)
    static_block_data: wp.array2d(dtype=wp.float16)
    has_dynamic: wp.bool
    has_static: wp.bool
    has_features: wp.bool
    feature_dim: int
    color_grid_size: int
    feature_block_grid_size: int
    block_coords: wp.array(dtype=wp.int32)
    block_to_hash_slot: wp.array(dtype=wp.int32)
    free_list: wp.array(dtype=wp.int32)
    free_count: wp.array(dtype=wp.int32)
    num_allocated: wp.array(dtype=wp.int32)
    allocation_failures: wp.array(dtype=wp.int32)
    block_sums: wp.array(dtype=wp.float32)
    static_block_sums: wp.array(dtype=wp.int32)
    new_blocks: wp.array(dtype=wp.int32)
    new_block_count: wp.array(dtype=wp.int32)
    recycle_count: wp.array(dtype=wp.int32)
    origin: wp.vec3
    voxel_size: float
    hash_capacity: int
    max_blocks: int
    truncation_distance: float
    block_size: int
    grid_W: int
    grid_H: int
    grid_D: int


__all__ = ["BlockSparseTSDFWarp"]
