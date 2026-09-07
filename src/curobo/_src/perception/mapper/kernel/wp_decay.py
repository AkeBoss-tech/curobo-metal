from __future__ import annotations

import torch

from curobo._src.perception.mapper._portable import dense_state

annotations = check_float32_tensors = get_warp_device_stream = log_and_raise = wp = None


def decay_and_recycle(
    tsdf,
    decay_factor: float = 0.95,
) -> int:
    if getattr(tsdf, "_portable_sparse", False):
        return tsdf.decay_and_recycle(decay_factor)
    state = dense_state(tsdf)
    state.weight.mul_(decay_factor)
    return 0


def launch_recycle(tsdf, num_blocks: int | None = None):
    if getattr(tsdf, "_portable_sparse", False):
        return tsdf.decay_and_recycle(1.0)
    return dense_state(tsdf)


def apply_decay_from_frustum_flags(
    tsdf,
    *,
    num_blocks: int,
    time_decay: float,
    frustum_decay: float,
):
    return decay_and_recycle(tsdf, time_decay * frustum_decay)


def decay_frustum_aware_multi_sensor(
    tsdf,
    *,
    camera_intrinsics: torch.Tensor | None = None,
    camera_positions: torch.Tensor | None = None,
    camera_quaternions: torch.Tensor | None = None,
    camera_img_shape: tuple[int, int] | None = None,
    camera_depth_minimum_distance: float = 0.1,
    camera_depth_maximum_distance: float = 10.0,
    lidar_positions: torch.Tensor | None = None,
    lidar_quaternions: torch.Tensor | None = None,
    lidar_valid_range_m: torch.Tensor | None = None,
    lidar_elevation_range_rad: torch.Tensor | None = None,
    lidar_img_shape: tuple[int, int] | None = None,
    time_decay: float = 1.0,
    frustum_decay: float = 0.5,
    num_blocks: int = None,
):
    if lidar_positions is not None:
        raise NotImplementedError("lidar frustum decay requires CUDA/Warp")
    return decay_and_recycle(tsdf, time_decay * frustum_decay)


def decay_frustum_aware_multi_camera(
    tsdf,
    intrinsics: torch.Tensor,
    cam_positions: torch.Tensor,
    cam_quaternions: torch.Tensor,
    img_shape: tuple,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 10.0,
    time_decay: float = 1.0,
    frustum_decay: float = 0.5,
    num_blocks: int = None,
):
    return decay_and_recycle(tsdf, time_decay * frustum_decay)
