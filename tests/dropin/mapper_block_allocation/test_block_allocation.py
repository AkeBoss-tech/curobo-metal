"""Portable allocation estimates retain the pinned sparse-map bounds."""

import pytest

from curobo._src.perception.mapper.block_allocation import (
    calculate_tsdf_max_blocks,
    default_max_blocks,
)
from curobo._src.perception.mapper.constants import MAX_POOL_IDX, REFERENCE_BLOCK_SIZE


def test_default_budget_scales_in_voxels_then_respects_packed_pool_limit() -> None:
    assert default_max_blocks(REFERENCE_BLOCK_SIZE) == min(100_000, MAX_POOL_IDX)
    assert default_max_blocks(REFERENCE_BLOCK_SIZE // 2) == MAX_POOL_IDX


def test_surface_estimate_has_a_voxel_floor_and_never_returns_reserved_pool_id() -> None:
    assert calculate_tsdf_max_blocks((16, 16, 16), 0.1, 8, 0.01, roughness=1.0) == 10_000
    assert calculate_tsdf_max_blocks((16, 16, 16), 0.1, 4, 0.01, roughness=1.0) == MAX_POOL_IDX
    assert calculate_tsdf_max_blocks((100_000, 100_000, 100_000), 0.001, 1, 1.0) == MAX_POOL_IDX


@pytest.mark.parametrize(
    "args",
    [
        ((16, 16, 16), 0.0, 8, 0.01),
        ((16, 16, 16), 0.1, 0, 0.01),
        ((16, 16, 16), 0.1, 8, 0.0),
        ((16, 16, 16), 0.1, 8, 0.01, 0.0),
    ],
)
def test_estimates_reject_nonpositive_physical_parameters(args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_tsdf_max_blocks(*args)
