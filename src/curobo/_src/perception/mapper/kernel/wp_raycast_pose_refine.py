from __future__ import annotations

from typing import Callable

from . import raw_kernel

BlockSparseKernels = object
BlockSparseTSDFWarp = object
make_block_sparse_kernels = raw_kernel
warp_kernel = wp = None


def quat_from_wxyz_array(q: wp.array(dtype=wp.float32)) -> wp.quat:
    return raw_kernel(q)


def vec3_from_array(v: wp.array(dtype=wp.float32)) -> wp.vec3:
    return raw_kernel(v)


def create_ray_sdf_alignment_block_sparse_tiled_kernel(
    n_samples_per_ray: int = 5,
    kernels=None,
) -> Callable:
    return raw_kernel(n_samples_per_ray, kernels)
