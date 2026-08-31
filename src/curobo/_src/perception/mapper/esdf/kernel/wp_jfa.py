from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.util.utils_quantization import (
    get_weight_from_float16,
    unpack_site_coords_torch,
    unpack_site_x,
    unpack_site_y,
    unpack_site_z,
)

wp = None


def validate_grid_size(grid_shape: Tuple[int, int, int], class_name: str) -> int:
    if len(grid_shape) != 3 or any(int(v) < 1 for v in grid_shape):
        raise ValueError(f"{class_name} requires a positive 3D grid")
    n_voxels = math.prod(int(v) for v in grid_shape)
    if n_voxels > 2**31 - 1:
        raise ValueError(f"{class_name} grid is too large for int32 site indices")
    return n_voxels


def compute_num_passes(grid_shape: Tuple[int, int, int]) -> int:
    return int(math.ceil(math.log2(max(grid_shape)))) if max(grid_shape) > 1 else 0


def get_nearest_surface_coords(
    voxel_idx: torch.Tensor,
    site_index: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    _, height, width = grid_shape
    flat_idx = voxel_idx[:, 0] * height * width + voxel_idx[:, 1] * width + voxel_idx[:, 2]
    packed = site_index.reshape(-1)[flat_idx]
    x, y, z = unpack_site_coords_torch(packed)
    return torch.stack((z, y, x), dim=1)


def smooth_esdf(
    dist_field: torch.Tensor,
    weight_grid: torch.Tensor,
    temp_buffer: torch.Tensor,
    minimum_tsdf_weight: float = 0.5,
    iterations: int = 2,
) -> None:
    result = dist_field
    for _ in range(iterations):
        result = torch.nn.functional.avg_pool3d(
            result[None, None].float(), 3, stride=1, padding=1
        )[0, 0].to(dist_field.dtype)
    dist_field.copy_(torch.where(weight_grid >= minimum_tsdf_weight, result, dist_field))


def dist_sq_to_site_packed(i: int, j: int, k: int, site_packed: wp.int32) -> int:
    return unsupported_kernel(i, j, k, site_packed)


def jfa_propagate_kernel_18(
    site_in: wp.array(dtype=wp.int32), site_out: wp.array(dtype=wp.int32),
    offset: int, D: int, H: int, W: int,
):
    return unsupported_kernel(site_in, site_out, offset, D, H, W)


def jfa_propagate_kernel_26(
    site_in: wp.array(dtype=wp.int32), site_out: wp.array(dtype=wp.int32),
    offset: int, D: int, H: int, W: int,
):
    return unsupported_kernel(site_in, site_out, offset, D, H, W)


def gaussian_smooth_1d_x_kernel(
    input_field: wp.array(dtype=wp.float16), output_field: wp.array(dtype=wp.float16),
    D: int, H: int, W: int,
):
    return unsupported_kernel(input_field, output_field, D, H, W)


def gaussian_smooth_1d_y_kernel(
    input_field: wp.array(dtype=wp.float16), output_field: wp.array(dtype=wp.float16),
    D: int, H: int, W: int,
):
    return unsupported_kernel(input_field, output_field, D, H, W)


def gaussian_smooth_1d_z_kernel(
    input_field: wp.array(dtype=wp.float16), output_field: wp.array(dtype=wp.float16),
    weight_grid: wp.array(dtype=wp.float16), minimum_tsdf_weight: float,
    D: int, H: int, W: int,
):
    return unsupported_kernel(input_field, output_field, weight_grid, minimum_tsdf_weight, D, H, W)
