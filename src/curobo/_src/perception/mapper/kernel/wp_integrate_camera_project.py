"""Portable declaration-compatible boundary for Warp camera integration."""

from __future__ import annotations

import math
import torch

from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise

# Warp kernels are intentionally not emulated through a misleading Python ABI.
wp = None


class CameraProjectIntegrator:
    """CUDA/Warp projective camera integration boundary."""

    def __init__(self, num_cameras: int, image_height: int, image_width: int, voxel_size: float, block_size: int, truncation_distance: float, max_blocks: int, max_visible_blocks_per_integration: int | None = None, max_support_pixels_per_block_camera: int = 32, device: str = "cuda:0", use_hash_dedup: bool = True, feature_channels_per_thread: int = 4, use_tiled_feature_kernel: bool = True, feature_grid_shape: tuple[int, int] | None = None, profile_kernel_timings: bool = False):
        raise NotImplementedError("CameraProjectIntegrator requires the upstream CUDA/Warp backend")

    def _timer_start(self) -> None:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _timer_stop(self, name: str) -> None:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _next_frame_epoch(self) -> int:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _prepare_frame_scratch(self):
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _dedup_visible_blocks_hash(self, tsdf, kernels, data, num_block_key_candidates, device, stream) -> int:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _build_support_pixels(self, tsdf, kernels, data, num_block_key_candidates, frame_epoch, device, stream) -> None:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _clear_new_blocks(self, tsdf, kernels, data, num_visible_blocks, device, stream):
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def _world_aabb_to_block_bounds(self, tsdf, bounds_min, bounds_max) -> tuple:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def clear_blocks(self, tsdf, pool_indices) -> int:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def clear_region(self, tsdf, bounds_min, bounds_max) -> int:
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")

    def integrate(self, tsdf, depth_images: torch.Tensor, rgb_images: torch.Tensor, cam_positions: torch.Tensor, cam_quaternions: torch.Tensor, intrinsics: torch.Tensor, depth_min: float = 0.1, depth_max: float = 5.0, grid_size: tuple = None, feature_grid: "torch.Tensor | None" = None):
        raise NotImplementedError("CUDA/Warp camera integration is unavailable on CPU/MPS")
