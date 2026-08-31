"""Parallel Banding Algorithm compatibility surface.

PBA is an NVIDIA execution mechanism. This module keeps the pinned constructor
and launch contract exact and delegates to the backend, which provides the
repository's explicit unsupported-platform boundary away from CUDA.
"""

from typing import Tuple

import torch

from curobo._src.curobolib.backends import pba as pba_cu
from curobo._src.perception.mapper.esdf.kernel.wp_jfa import validate_grid_size
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import profile_class_methods


@profile_class_methods
class ParallelBandingEDT:
    """Exact 3D EDT using the pinned Parallel Banding Algorithm contract."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        device: torch.device,
        m3: int = 2,
    ):
        if device.type != "cuda":
            raise ValueError(f"ParallelBandingEDT requires CUDA device, got {device}")

        self.grid_shape = grid_shape
        self.voxel_size = float(voxel_size)
        self.device = device
        self.m3 = m3
        self.n_voxels = validate_grid_size(grid_shape, "ParallelBandingEDT")
        self._buffer = torch.empty(
            self.n_voxels,
            dtype=torch.int32,
            device=device,
        )

        mem_gb = self.n_voxels * 8 / (1024**3)
        log_info(
            f"ParallelBandingEDT: {grid_shape}, 5 kernel launches (exact), "
            f"{mem_gb:.2f} GB"
        )

    def propagate(self, site_index: torch.Tensor) -> None:
        """Run PBA propagation on ``site_index`` in place."""
        nx, ny, nz = self.grid_shape
        pba_cu.launch_pba3d(
            site_index.view(-1),
            self._buffer,
            nx,
            ny,
            nz,
            m3=self.m3,
        )
