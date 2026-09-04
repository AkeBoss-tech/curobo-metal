"""Portable CPU/MPS implementation of cuRobo's batched depth filter.

The upstream implementation launches fused Warp kernels.  This module keeps
the source-level lifecycle (configuration, reusable output buffers, updates,
and caller-owned output buffers) while implementing the mathematical filter
with regular PyTorch operations.  It intentionally does not expose Warp
kernel handles or CUDA launch state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.util.logging import log_and_raise


@dataclass
class FilterDepthConfig:
    """Parameters for range, flying-pixel, and bilateral depth filtering."""

    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0
    flying_pixel_threshold: Optional[float] = 0.5
    bilateral_kernel_size: Optional[int] = 5
    bilateral_sigma_spatial: float = 2.0
    bilateral_sigma_depth: float = 0.05

    def __post_init__(self) -> None:
        if self.depth_minimum_distance < 0 or self.depth_maximum_distance <= self.depth_minimum_distance:
            raise ValueError("depth limits must satisfy 0 <= minimum < maximum")
        if self.flying_pixel_threshold is not None and self.flying_pixel_threshold < 0:
            raise ValueError("flying_pixel_threshold must be non-negative or None")
        if self.bilateral_kernel_size is not None and (
            not isinstance(self.bilateral_kernel_size, int)
            or self.bilateral_kernel_size < 1
            or self.bilateral_kernel_size % 2 == 0
        ):
            raise ValueError("bilateral_kernel_size must be an odd positive integer or None")
        if self.bilateral_sigma_spatial <= 0 or self.bilateral_sigma_depth <= 0:
            raise ValueError("bilateral sigmas must be positive")


def _portable_device(device: str | torch.device) -> torch.device:
    """Map the upstream CUDA default to the available portable accelerator."""

    requested = str(device)
    if requested.startswith("cuda"):
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


class FilterDepth:
    """Apply the V2 batched depth-filter contract on CPU or float32 MPS.

    Inputs may use either the source-shaped ``(B, H, W)`` layout or a single
    unbatched ``(H, W)`` image. Matching calls reuse public output buffers,
    while shape changes allocate temporary outputs just as the source
    implementation does.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        depth_minimum_distance: float = 0.1,
        depth_maximum_distance: float = 10.0,
        flying_pixel_threshold: Optional[float] = 0.5,
        bilateral_kernel_size: Optional[int] = 5,
        bilateral_sigma_spatial: float = 10.0,
        bilateral_sigma_depth: float = 0.1,
        device: str = "cuda",
        num_batch: int = 1,
    ):
        if len(image_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in image_shape):
            raise ValueError("image_shape must be a positive (height, width) pair")
        if int(num_batch) < 1:
            raise ValueError("num_batch must be positive")

        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.num_batch = int(num_batch)
        self.device = _portable_device(device)
        self.config = FilterDepthConfig(
            depth_minimum_distance=depth_minimum_distance,
            depth_maximum_distance=depth_maximum_distance,
            flying_pixel_threshold=flying_pixel_threshold,
            bilateral_kernel_size=bilateral_kernel_size,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_depth=bilateral_sigma_depth,
        )
        self._setup_kernel_params()
        self._allocate_buffers()

    def _setup_kernel_params(self) -> None:
        """Mirror the upstream derived filter parameters without Warp state."""

        cfg = self.config
        threshold = cfg.flying_pixel_threshold
        self._enable_flying = threshold is not None
        # V2 maps the documented 0--1 aggressiveness control logarithmically.
        # Older portable callers used values >1 as an absolute tolerance; keep
        # that harmless extension instead of silently changing their results.
        self._legacy_absolute_flying = threshold is not None and threshold > 1.0
        if threshold is None:
            self._flying_tolerance = 0.0
        elif self._legacy_absolute_flying:
            self._flying_tolerance = float(threshold)
        else:
            self._flying_tolerance = 0.08 * (0.005 / 0.08) ** float(threshold)

        kernel = cfg.bilateral_kernel_size
        self._enable_bilateral = kernel is not None and kernel > 1
        self._bilateral_radius = 0 if kernel is None else kernel // 2
        self._sigma_spatial_sq2 = 2.0 * cfg.bilateral_sigma_spatial**2
        self._sigma_depth_sq2 = 2.0 * cfg.bilateral_sigma_depth**2
        self._use_separable = bool(kernel is not None and kernel >= 7)

    def _allocate_buffers(self) -> None:
        shape = (self.num_batch, *self.image_shape)
        self._depth_out = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._valid_mask_out = torch.zeros(shape, dtype=torch.bool, device=self.device)
        # These attributes exist for source-compatible lifecycle inspection.
        # The portable path retains exact 2-D bilateral weights rather than
        # swapping to upstream's large-kernel separable Warp approximation.
        self._depth_temp = torch.zeros(shape, dtype=torch.float32, device=self.device) if self._use_separable else None
        self._depth_temp2 = torch.zeros(shape, dtype=torch.float32, device=self.device) if self._use_separable else None

    @staticmethod
    def _check_output_buffer(buffer: torch.Tensor, name: str, shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> None:
        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(buffer.shape) != shape:
            raise ValueError(f"{name} must have shape (B, H, W)={shape}; got {tuple(buffer.shape)}")
        if buffer.device != device:
            raise ValueError(f"{name} must be on {device}; got {buffer.device}")
        if buffer.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}; got {buffer.dtype}")

    def _acquire_buffers(
        self,
        batch: int,
        height: int,
        width: int,
        depth_out: Optional[torch.Tensor],
        valid_mask_out: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device if device is None else torch.device(device)
        shape = (batch, height, width)
        if depth_out is not None:
            self._check_output_buffer(depth_out, "depth_out", shape, device, torch.float32)
        if valid_mask_out is not None:
            self._check_output_buffer(valid_mask_out, "valid_mask_out", shape, device, torch.bool)
        matches_preallocation = device == self.device and shape == tuple(self._depth_out.shape)
        out_depth = depth_out if depth_out is not None else (
            self._depth_out if matches_preallocation else torch.zeros(shape, dtype=torch.float32, device=device)
        )
        out_mask = valid_mask_out if valid_mask_out is not None else (
            self._valid_mask_out if matches_preallocation else torch.zeros(shape, dtype=torch.bool, device=device)
        )
        return out_depth, out_mask

    def _flying_mask(self, value: torch.Tensor, range_valid: torch.Tensor) -> torch.Tensor:
        if not self._enable_flying:
            return torch.zeros_like(range_valid)
        # 4-neighbor, edge-clamped values, precisely matching the fused kernel.
        padded = F.pad(value[:, None], (1, 1, 1, 1), mode="replicate").squeeze(1)
        left, right = padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]
        up, down = padded[:, :-2, 1:-1], padded[:, 2:, 1:-1]
        # replication_pad2d does not support bool tensors on every backend.
        padded_valid = F.pad(
            range_valid[:, None].to(dtype=torch.float32), (1, 1, 1, 1), mode="replicate"
        ).squeeze(1).bool()
        neighbor_valid = [
            padded_valid[:, 1:-1, :-2],
            padded_valid[:, 1:-1, 2:],
            padded_valid[:, :-2, 1:-1],
            padded_valid[:, 2:, 1:-1],
        ]
        neighbors = [torch.where(is_valid, candidate, value) for candidate, is_valid in zip((left, right, up, down), neighbor_valid)]
        maximum_difference = torch.stack([(value - neighbor).abs() for neighbor in neighbors]).amax(dim=0)
        tolerance = self._flying_tolerance if self._legacy_absolute_flying else value * self._flying_tolerance
        return maximum_difference > tolerance

    def _bilateral(self, value: torch.Tensor, range_valid: torch.Tensor) -> torch.Tensor:
        if not self._enable_bilateral:
            return value
        radius = self._bilateral_radius
        kernel = radius * 2 + 1
        padded_value = F.pad(value[:, None], (radius, radius, radius, radius), mode="replicate").squeeze(1)
        windows = padded_value.unfold(1, kernel, 1).unfold(2, kernel, 1)
        # Upstream skips samples outside the image rather than using replicated
        # border samples, so an independently padded in-bounds mask is needed.
        in_bounds = F.pad(torch.ones_like(range_valid[:, None]), (radius, radius, radius, radius)).squeeze(1)
        in_bounds = in_bounds.unfold(1, kernel, 1).unfold(2, kernel, 1).bool()
        padded_valid = F.pad(range_valid[:, None], (radius, radius, radius, radius)).squeeze(1)
        valid_windows = padded_valid.unfold(1, kernel, 1).unfold(2, kernel, 1).bool() & in_bounds
        offsets = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        spatial = torch.exp(-(xx.square() + yy.square()) / self._sigma_spatial_sq2)
        range_weight = torch.exp(-(windows - value[..., None, None]).square() / self._sigma_depth_sq2)
        weights = spatial * range_weight * valid_windows
        denominator = weights.sum(dim=(-1, -2))
        filtered = (windows * weights).sum(dim=(-1, -2)) / denominator.clamp_min(torch.finfo(value.dtype).eps)
        return torch.where(denominator > 0, filtered, value)

    def __call__(
        self,
        depth_image: torch.Tensor,
        depth_out: Optional[torch.Tensor] = None,
        valid_mask_out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter a depth image/batch and return depth plus a validity mask.

        A 2-D input keeps its 2-D shape on return, including when caller-owned
        2-D output buffers are supplied.  Batched callers retain the original
        3-D behavior and buffer-reuse contract.
        """

        if not isinstance(depth_image, torch.Tensor) or depth_image.dtype != torch.float32:
            raise TypeError("depth_image must be a float32 torch.Tensor")
        if depth_image.ndim not in (2, 3):
            raise ValueError("FilterDepth expects a depth tensor of shape (H, W) or (B, H, W)")
        unbatched = depth_image.ndim == 2
        caller_depth_out = depth_out
        caller_valid_mask_out = valid_mask_out
        if unbatched:
            if depth_out is not None:
                if not isinstance(depth_out, torch.Tensor) or depth_out.ndim != 2:
                    raise ValueError("depth_out must have shape (H, W) for an unbatched input")
                depth_out = depth_out.unsqueeze(0)
            if valid_mask_out is not None:
                if not isinstance(valid_mask_out, torch.Tensor) or valid_mask_out.ndim != 2:
                    raise ValueError("valid_mask_out must have shape (H, W) for an unbatched input")
                valid_mask_out = valid_mask_out.unsqueeze(0)
            value = depth_image.unsqueeze(0)
        else:
            value = depth_image
        batch, height, width = value.shape
        # Caller-owned buffers define the destination device.  This permits a
        # CPU image/buffer lifecycle even when the filter was constructed with
        # its CUDA-labelled default, while preserving configured MPS reuse for
        # ordinary calls without external buffers.
        target_device = depth_out.device if depth_out is not None else self.device
        if (height, width) != self.image_shape and (depth_out is not None or valid_mask_out is not None):
            # Dynamic shapes are allowed, but caller buffers must still describe
            # the actual call layout; _acquire_buffers gives the precise error.
            pass
        out_depth, out_mask = self._acquire_buffers(
            batch, height, width, depth_out, valid_mask_out, target_device
        )
        value = value.to(target_device)
        range_valid = torch.isfinite(value) & (value >= self.config.depth_minimum_distance) & (value <= self.config.depth_maximum_distance)
        valid = range_valid & ~self._flying_mask(value, range_valid)
        filtered = self._bilateral(value, range_valid)
        filtered = torch.where(valid, filtered, torch.zeros_like(filtered))
        out_depth.copy_(filtered)
        out_mask.copy_(valid)
        if unbatched:
            return (
                caller_depth_out if caller_depth_out is not None else out_depth.squeeze(0),
                caller_valid_mask_out if caller_valid_mask_out is not None else out_mask.squeeze(0),
            )
        return out_depth, out_mask

    def update_config(
        self,
        depth_minimum_distance: Optional[float] = None,
        depth_maximum_distance: Optional[float] = None,
        flying_pixel_threshold: Optional[float] = None,
        bilateral_sigma_depth: Optional[float] = None,
    ):
        """Update source-supported runtime settings without reallocating buffers.

        Passing ``flying_pixel_threshold=0`` disables that filter, matching V2;
        use ``None`` for all other arguments to leave the existing value intact.
        """

        proposed = FilterDepthConfig(**self.config.__dict__)
        if depth_minimum_distance is not None:
            proposed.depth_minimum_distance = depth_minimum_distance
        if depth_maximum_distance is not None:
            proposed.depth_maximum_distance = depth_maximum_distance
        if flying_pixel_threshold is not None:
            proposed.flying_pixel_threshold = None if flying_pixel_threshold == 0 else flying_pixel_threshold
        if bilateral_sigma_depth is not None:
            proposed.bilateral_sigma_depth = bilateral_sigma_depth
        proposed.__post_init__()
        self.config = proposed
        self._setup_kernel_params()

    @classmethod
    def from_config(
        cls,
        config: FilterDepthConfig,
        image_shape: Tuple[int, int],
        device: str = "cuda",
        num_batch: int = 1,
    ) -> "FilterDepth":
        if not isinstance(config, FilterDepthConfig):
            raise TypeError("config must be a FilterDepthConfig")
        return cls(image_shape=image_shape, device=device, num_batch=num_batch, **config.__dict__)
