from __future__ import annotations

from . import raw_kernel
from curobo._src.util.warp import wp as _raw_wp


class _WarpCompat:
    """Declaration-only Warp façade for explicitly unsupported kernels."""

    @staticmethod
    def func(function):
        return function


wp = _raw_wp if _raw_wp is not None else _WarpCompat()


def quat_from_wxyz_array(cam_quaternion: wp.array(dtype=wp.float32)) -> wp.quat:
    return raw_kernel(cam_quaternion)


def vec3_from_array(arr: wp.array(dtype=wp.float32)) -> wp.vec3:
    return raw_kernel(arr)


def compute_tsdf_weight(depth: wp.float32, voxel_size: wp.float32) -> wp.float32:
    return max(0.001, min(2.0, 1.0 / max(float(depth) ** 2, 1e-12)))


def floor_div(a: wp.int32, b: wp.int32) -> wp.int32:
    return int(a) // int(b)


def floor_mod(a: wp.int32, b: wp.int32) -> wp.int32:
    return int(a) % int(b)
