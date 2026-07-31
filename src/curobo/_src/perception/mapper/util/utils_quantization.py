"""Torch equivalents of mapper coordinate packing helpers."""

import torch

SITE_COORD_BITS = 10
SITE_COORD_MASK = (1 << SITE_COORD_BITS) - 1


def pack_site_coords(x, y, z):
    return (x & SITE_COORD_MASK) | ((y & SITE_COORD_MASK) << 10) | ((z & SITE_COORD_MASK) << 20)


def unpack_site_x(packed):
    return packed & SITE_COORD_MASK


def unpack_site_y(packed):
    return (packed >> 10) & SITE_COORD_MASK


def unpack_site_z(packed):
    return (packed >> 20) & SITE_COORD_MASK


def unpack_site_coords_torch(packed: torch.Tensor):
    packed = packed.to(torch.int64)
    return torch.stack((unpack_site_x(packed), unpack_site_y(packed), unpack_site_z(packed)), -1)


def get_sdf_from_float16_grids(sdf_weight, weight):
    return sdf_weight / torch.as_tensor(weight).clamp_min(torch.finfo(torch.float16).tiny)


def get_weight_from_float16(weight):
    return weight.to(torch.float32) if isinstance(weight, torch.Tensor) else float(weight)
