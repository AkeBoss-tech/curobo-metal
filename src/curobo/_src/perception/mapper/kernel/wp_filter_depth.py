"""Portable scalar filter predicates; raw kernels remain unavailable."""

import math

from . import raw_kernel


def compute_flying_pixel_threshold(threshold):
    return float(threshold)


def is_flying_pixel(depth_center, d_left, d_right, d_up, d_down, tolerance):
    if not math.isfinite(float(depth_center)):
        return True
    return any(
        math.isfinite(float(v)) and abs(float(v) - float(depth_center)) > tolerance
        for v in (d_left, d_right, d_up, d_down)
    )


filter_depth_fused_kernel = raw_kernel
bilateral_filter_separable_h_kernel = raw_kernel
bilateral_filter_separable_v_kernel = raw_kernel
