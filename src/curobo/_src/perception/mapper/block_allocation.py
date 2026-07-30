import math

REFERENCE_BLOCK_SIZE = 8


def default_max_blocks(block_size: int) -> int:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return max(1, 100_000 * REFERENCE_BLOCK_SIZE**3 // block_size**3)


def calculate_tsdf_max_blocks(grid_shape, voxel_size, block_size, truncation_dist, roughness=3.0):
    if voxel_size <= 0 or block_size < 1 or truncation_dist <= 0 or roughness <= 0:
        raise ValueError("voxel_size, block_size, truncation_dist, and roughness must be positive")
    nz, ny, nx = grid_shape
    bx, by, bz = nx/block_size, ny/block_size, nz/block_size
    area = 2*(bx*by+bx*bz+by*bz)
    thickness = math.ceil(2*truncation_dist/(voxel_size*block_size))+1
    return max(1, int(area*thickness*roughness))
