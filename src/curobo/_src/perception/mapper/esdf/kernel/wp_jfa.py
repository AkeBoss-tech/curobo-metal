import math

import torch

from curobo._src.perception.mapper._portable import unsupported_kernel


def validate_grid_size(grid_shape, class_name):
    if len(grid_shape) != 3 or any(int(v) < 1 for v in grid_shape):
        raise ValueError(f"{class_name} requires a positive 3D grid")
    return tuple(int(v) for v in grid_shape)


def compute_num_passes(grid_shape):
    return int(math.ceil(math.log2(max(grid_shape)))) if max(grid_shape) > 1 else 0


def get_nearest_surface_coords(voxel_idx, site_index, grid_shape):
    packed = site_index.reshape(-1)[voxel_idx]
    from curobo._src.perception.mapper.util.utils_quantization import unpack_site_coords_torch
    return unpack_site_coords_torch(packed)


def smooth_esdf(dist_field, weight_grid, temp_buffer, minimum_tsdf_weight=0.5, iterations=2):
    result = dist_field
    for _ in range(iterations):
        result = torch.nn.functional.avg_pool3d(
            result[None, None].float(), 3, stride=1, padding=1
        )[0, 0].to(dist_field.dtype)
    dist_field.copy_(torch.where(weight_grid >= minimum_tsdf_weight, result, dist_field))
    return dist_field


dist_sq_to_site_packed = unsupported_kernel
jfa_propagate_kernel_18 = unsupported_kernel
jfa_propagate_kernel_26 = unsupported_kernel
gaussian_smooth_1d_x_kernel = unsupported_kernel
gaussian_smooth_1d_y_kernel = unsupported_kernel
gaussian_smooth_1d_z_kernel = unsupported_kernel
