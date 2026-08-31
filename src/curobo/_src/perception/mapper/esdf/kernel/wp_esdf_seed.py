"""Portable declaration boundary for CUDA ESDF seeding launchers."""

from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.storage import BlockSparseTSDF
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise

wp = None


def seed_esdf_sites_from_block_sparse_warp(
    tsdf: BlockSparseTSDF,
    esdf_site_index: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    minimum_tsdf_weight: float = 0.1,
) -> None:
    unsupported_kernel(
        tsdf, esdf_site_index, esdf_origin, esdf_voxel_size_tensor,
        esdf_grid_shape, minimum_tsdf_weight,
    )


def seed_esdf_sites_gather_warp(
    tsdf: BlockSparseTSDF,
    esdf_site_index: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    minimum_tsdf_weight: float = 0.1,
) -> None:
    unsupported_kernel(
        tsdf, esdf_site_index, esdf_origin, esdf_voxel_size_tensor,
        esdf_grid_shape, minimum_tsdf_weight,
    )
