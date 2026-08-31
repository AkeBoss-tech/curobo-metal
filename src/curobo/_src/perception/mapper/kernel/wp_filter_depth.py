"""Depth-filter declarations with portable scalar helpers.

The three image kernels below retain cuRobo's public Warp call signatures so
callers fail at the actual unsupported boundary rather than at import time.
Metal's mapper uses tensor-native operators instead; raw Warp array/kernel
launches are deliberately not emulated here.
"""

from __future__ import annotations

import math

from curobo._src.util.warp import wp as _raw_wp

from . import raw_kernel


class _WarpCompat:
    """Small declaration façade used only when NVIDIA Warp is absent."""

    @staticmethod
    def func(function):
        return function

    @staticmethod
    def kernel(function):
        return function


wp = _raw_wp if _raw_wp is not None else _WarpCompat()


@wp.func
def compute_flying_pixel_threshold(threshold: float) -> float:
    """Return cuRobo's exponentially interpolated relative-depth tolerance."""
    max_tol = 0.08
    min_tol = 0.005
    return max_tol * math.exp(float(threshold) * math.log(min_tol / max_tol))


@wp.func
def is_flying_pixel(
    depth_center: float,
    d_left: float,
    d_right: float,
    d_up: float,
    d_down: float,
    tolerance: float,
) -> bool:
    """Apply the V2 four-neighbour relative-gradient predicate."""
    max_difference = max(
        abs(depth_center - d_left),
        abs(depth_center - d_right),
        abs(depth_center - d_up),
        abs(depth_center - d_down),
    )
    return max_difference > tolerance * depth_center


@wp.kernel
def filter_depth_fused_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    valid_mask_out: wp.array3d(dtype=wp.uint8),
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    enable_flying_pixel: wp.int32,
    flying_tolerance: float,
    enable_bilateral: wp.int32,
    bilateral_radius: wp.int32,
    sigma_spatial_sq2: float,
    sigma_depth_sq2: float,
):
    """Raw Warp kernel ABI, explicitly unavailable on CPU/MPS."""
    return raw_kernel(
        depth_in,
        depth_out,
        valid_mask_out,
        depth_minimum_distance,
        depth_maximum_distance,
        enable_flying_pixel,
        flying_tolerance,
        enable_bilateral,
        bilateral_radius,
        sigma_spatial_sq2,
        sigma_depth_sq2,
    )


@wp.kernel
def bilateral_filter_separable_h_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    radius: wp.int32,
    sigma_spatial_sq2: float,
    sigma_depth_sq2: float,
    depth_minimum_distance: float,
    depth_maximum_distance: float,
):
    """Raw horizontal Warp kernel ABI, explicitly unavailable on CPU/MPS."""
    return raw_kernel(
        depth_in,
        depth_out,
        radius,
        sigma_spatial_sq2,
        sigma_depth_sq2,
        depth_minimum_distance,
        depth_maximum_distance,
    )


@wp.kernel
def bilateral_filter_separable_v_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    radius: wp.int32,
    sigma_spatial_sq2: float,
    sigma_depth_sq2: float,
    depth_minimum_distance: float,
    depth_maximum_distance: float,
):
    """Raw vertical Warp kernel ABI, explicitly unavailable on CPU/MPS."""
    return raw_kernel(
        depth_in,
        depth_out,
        radius,
        sigma_spatial_sq2,
        sigma_depth_sq2,
        depth_minimum_distance,
        depth_maximum_distance,
    )
