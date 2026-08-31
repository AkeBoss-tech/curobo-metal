import math
from typing import Tuple

from curobo._src.perception.mapper.constants import MAX_POOL_IDX, REFERENCE_BLOCK_SIZE
from curobo._src.util.logging import log_warn


_MIN_VOXEL_FLOOR = 10_000 * REFERENCE_BLOCK_SIZE**3
_DEFAULT_VOXEL_BUDGET = 100_000 * REFERENCE_BLOCK_SIZE**3


def default_max_blocks(block_size: int) -> int:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return min(max(1, _DEFAULT_VOXEL_BUDGET // block_size**3), MAX_POOL_IDX)


def calculate_tsdf_max_blocks(
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
    block_size: int,
    truncation_dist: float,
    roughness: float = 3.0,
) -> int:
    if voxel_size <= 0 or block_size < 1 or truncation_dist <= 0 or roughness <= 0:
        raise ValueError("voxel_size, block_size, truncation_dist, and roughness must be positive")
    nz, ny, nx = grid_shape
    bx, by, bz = nx/block_size, ny/block_size, nz/block_size
    area = 2*(bx*by+bx*bz+by*bz)
    thickness = math.ceil(2*truncation_dist/(voxel_size*block_size))+1
    estimate = max(
        max(1, _MIN_VOXEL_FLOOR // block_size**3),
        int(area * thickness * roughness),
    )
    return min(estimate, MAX_POOL_IDX)
