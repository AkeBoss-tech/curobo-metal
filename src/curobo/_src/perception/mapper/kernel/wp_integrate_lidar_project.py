"""Portable declaration-compatible boundary for Warp LiDAR integration."""

from __future__ import annotations

import math
import torch

from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise

wp = None


class LidarProjectIntegrator:
    """CUDA/Warp raw LiDAR helper retained as an explicit unsupported API."""

    def __init__(self, lidar_num_sensors: int, lidar_image_height: int, lidar_image_width: int, voxel_size: float, block_size: int, truncation_distance: float, max_blocks: int, max_visible_blocks_per_lidar_integration: int | None = None, max_support_pixels_per_block_lidar: int = 32, device: str = "cuda:0", feature_channels_per_thread: int = 4, use_tiled_feature_kernel: bool = True, lidar_feature_grid_shape: tuple[int, int] | None = None, linear_interpolation_max_allowable_difference_vox: float = 2.0, nearest_interpolation_max_allowable_dist_to_ray_vox: float = 0.5, profile_kernel_timings: bool = False):
        raise NotImplementedError("LidarProjectIntegrator requires the upstream CUDA/Warp backend")

    def _timer_start(self) -> None:
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def _timer_stop(self, name: str) -> None:
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def _next_frame_epoch(self) -> int:
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def _prepare_frame_scratch(self) -> None:
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def _dedup_visible_blocks_hash(self, tsdf, kernels, data, n_keys, device, stream) -> int:
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def _build_support_pixels(self, tsdf, kernels, data, n_keys, frame_epoch, device, stream):
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def _clear_new_blocks(self, tsdf, kernels, data, num_visible_blocks, device, stream):
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")

    def integrate(self, tsdf, range_images: torch.Tensor, rgb_images: torch.Tensor, lidar_positions: torch.Tensor, lidar_quaternions: torch.Tensor, valid_range_m: torch.Tensor, elevation_range_rad: torch.Tensor, feature_grid: torch.Tensor | None = None) -> None:
        raise NotImplementedError("CUDA/Warp LiDAR integration is unavailable on CPU/MPS")
