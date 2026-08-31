from __future__ import annotations

from typing import Optional, Tuple

import torch

from curobo._src.perception.mapper._portable import dense_state

BlockDataView = OccupiedVoxels = object
annotations = get_warp_device_stream = init_warp = wp = None


def extract_occupied_voxels_block_sparse(
    tsdf,
    surface_only: bool = False,
    sdf_threshold: float = None,
    minimum_tsdf_weight: float = 0.1,
    grid_shape: Tuple[int, int, int] = None,
) -> OccupiedVoxels:
    state = dense_state(tsdf)
    return state.occupancy.nonzero(as_tuple=False)


def extract_surface_voxels_block_sparse(
    tsdf,
    sdf_threshold: float = 0.04,
    minimum_tsdf_weight: float = 0.1,
    grid_shape: Tuple[int, int, int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return extract_occupied_voxels_block_sparse(
        tsdf, True, sdf_threshold, minimum_tsdf_weight, grid_shape
    )


def extract_matching_voxels_block_sparse(
    tsdf,
    block_mask: torch.Tensor,
    surface_only: bool = False,
    sdf_threshold: Optional[float] = None,
    minimum_tsdf_weight: float = 0.1,
) -> OccupiedVoxels:
    values = extract_occupied_voxels_block_sparse(tsdf, surface_only, sdf_threshold, minimum_tsdf_weight)
    return values[block_mask[: len(values)].bool()] if len(block_mask) >= len(values) else values
