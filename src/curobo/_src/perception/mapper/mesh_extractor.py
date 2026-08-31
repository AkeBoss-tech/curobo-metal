from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import MCLookupTables
from curobo._src.util.warp import get_warp_device_stream, wp


def extract_mesh_block_sparse(
    tsdf,
    level: float = 0.0,
    surface_only: bool = False,
    refine_iterations: int = 0,
    minimum_tsdf_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(tsdf, "extract_mesh"):
        mesh = tsdf.extract_mesh()
        normals = torch.zeros_like(mesh.vertices)
        return mesh.vertices, mesh.faces, normals
    raise NotImplementedError("use Mapper.extract_mesh for the portable dense backend")
