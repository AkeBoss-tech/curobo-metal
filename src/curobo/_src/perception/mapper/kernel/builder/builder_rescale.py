from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.util.warp import warp_kernel, wp


def make_rescale_kernels(
    block_size: int,
    *,
    feature_dim: int,
    color_grid_size: int = 1,
    feature_block_grid_size: int = 1,
) -> dict[str, object]:
    return unsupported_kernel(
        block_size,
        feature_dim=feature_dim,
        color_grid_size=color_grid_size,
        feature_block_grid_size=feature_block_grid_size,
    )
