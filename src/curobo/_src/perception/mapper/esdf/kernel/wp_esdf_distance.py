"""Portable declaration boundary for the CUDA ESDF distance launcher."""

from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.storage import BlockSparseTSDF
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise

wp = None


def compute_esdf_from_min_tsdf_warp(
    esdf_site_index: torch.Tensor,
    tsdf: BlockSparseTSDF,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    esdf_dist_field: torch.Tensor,
    esdf_origin: torch.Tensor,
    adjacent_skip_steps: float = 0.0,
    minimum_tsdf_weight: float = 0.1,
) -> None:
    """Reject the raw CUDA/Warp launcher while preserving its call contract."""
    unsupported_kernel(
        esdf_site_index, tsdf, esdf_voxel_size_tensor, esdf_grid_shape,
        esdf_dist_field, esdf_origin, adjacent_skip_steps, minimum_tsdf_weight,
    )
