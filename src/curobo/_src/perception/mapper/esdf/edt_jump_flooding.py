"""Portable declaration boundary for the CUDA/Warp jump-flooding EDT."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.esdf.kernel.wp_jfa import (
    compute_num_passes,
    jfa_propagate_kernel_18,
    jfa_propagate_kernel_26,
    validate_grid_size,
)
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import profile_class_methods
from curobo._src.util.warp import get_warp_device_stream

wp = None


@profile_class_methods
class JumpFloodingEDT:
    """JFA is a raw CUDA/Warp implementation and is not available on Metal."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        device: torch.device,
        single_buffer: bool = True,
        max_distance: Optional[float] = None,
        neighbors: int = 18,
    ):
        if neighbors not in (18, 26):
            raise ValueError(f"neighbors must be 18 or 26, got {neighbors}")
        self.grid_shape = grid_shape
        self.voxel_size = float(voxel_size)
        self.device = device
        self.single_buffer = single_buffer
        self.neighbors = neighbors
        self.n_voxels = validate_grid_size(grid_shape, "JumpFloodingEDT")
        self.num_passes = compute_num_passes(grid_shape)
        self._kernel = jfa_propagate_kernel_18 if neighbors == 18 else jfa_propagate_kernel_26

    def propagate(self, site_index: torch.Tensor) -> None:
        unsupported_kernel(site_index)
