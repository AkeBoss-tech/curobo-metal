"""Torch equivalents of mapper coordinate packing helpers."""

from __future__ import annotations

import torch

from curobo._src.util.warp import wp as _raw_wp


class _WarpCompat:
    """Declaration-only Warp façade; implementation remains tensor-native."""


wp = _raw_wp if _raw_wp is not None else _WarpCompat()

SITE_COORD_BITS = 10
SITE_COORD_MASK = (1 << SITE_COORD_BITS) - 1


def pack_site_coords(x: wp.int32, y: wp.int32, z: wp.int32) -> wp.int32:
    return (x & SITE_COORD_MASK) | ((y & SITE_COORD_MASK) << 10) | ((z & SITE_COORD_MASK) << 20)


def unpack_site_x(packed: wp.int32) -> wp.int32:
    return packed & SITE_COORD_MASK


def unpack_site_y(packed: wp.int32) -> wp.int32:
    return (packed >> 10) & SITE_COORD_MASK


def unpack_site_z(packed: wp.int32) -> wp.int32:
    return (packed >> 20) & SITE_COORD_MASK


def unpack_site_coords_torch(packed: torch.Tensor) -> tuple:
    packed = packed.to(torch.int64)
    return torch.stack((unpack_site_x(packed), unpack_site_y(packed), unpack_site_z(packed)), -1)


def get_sdf_from_float16_grids(
    sdf_weight: wp.float16, weight: wp.float16
) -> wp.float32:
    return sdf_weight / torch.as_tensor(weight).clamp_min(torch.finfo(torch.float16).tiny)


def get_weight_from_float16(weight: wp.float16) -> wp.float32:
    return weight.to(torch.float32) if isinstance(weight, torch.Tensor) else float(weight)
