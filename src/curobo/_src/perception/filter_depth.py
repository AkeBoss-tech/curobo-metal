from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class FilterDepthConfig:
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0
    flying_pixel_threshold: Optional[float] = 0.5
    bilateral_kernel_size: Optional[int] = 5
    bilateral_sigma_spatial: float = 2.0
    bilateral_sigma_depth: float = 0.05


class FilterDepth:
    def __init__(
        self, image_shape: Tuple[int, int], depth_minimum_distance: float = 0.1,
        depth_maximum_distance: float = 10.0,
        flying_pixel_threshold: Optional[float] = 0.5,
        bilateral_kernel_size: Optional[int] = 5,
        bilateral_sigma_spatial: float = 10.0,
        bilateral_sigma_depth: float = 0.1, device: str = "cuda",
        num_batch: int = 1,
    ) -> None:
        if len(image_shape) != 2 or any(int(value) <= 0 for value in image_shape):
            raise ValueError("image_shape must be a positive (height, width) pair")
        if depth_minimum_distance < 0 or depth_maximum_distance <= depth_minimum_distance:
            raise ValueError("depth limits must satisfy 0 <= minimum < maximum")
        if bilateral_kernel_size is not None and (
            bilateral_kernel_size < 1 or bilateral_kernel_size % 2 == 0
        ):
            raise ValueError("bilateral_kernel_size must be an odd positive integer or None")
        if num_batch < 1:
            raise ValueError("num_batch must be positive")
        self.image_shape = tuple(image_shape)
        self.num_batch = num_batch
        self.device = "mps" if str(device).startswith("cuda") and torch.backends.mps.is_available() else ("cpu" if str(device).startswith("cuda") else device)
        self.config = FilterDepthConfig(
            depth_minimum_distance, depth_maximum_distance, flying_pixel_threshold,
            bilateral_kernel_size, bilateral_sigma_spatial, bilateral_sigma_depth,
        )

    def __call__(self, depth_image: torch.Tensor, depth_out=None, valid_mask_out=None):
        if not isinstance(depth_image, torch.Tensor) or not depth_image.is_floating_point():
            raise TypeError("depth_image must be a floating torch.Tensor")
        if depth_image.shape[-2:] != self.image_shape:
            raise ValueError("depth image shape does not match configured image_shape")
        value = depth_image.to(self.device)
        squeeze = value.ndim == 2
        if squeeze: value = value[None]
        valid = torch.isfinite(value) & (value >= self.config.depth_minimum_distance) & (value <= self.config.depth_maximum_distance)
        if self.config.flying_pixel_threshold is not None:
            padded = F.pad(value[:,None], (1,1,1,1), mode="replicate")
            local = F.avg_pool2d(padded, 3, 1).squeeze(1)
            valid &= (value-local).abs() <= self.config.flying_pixel_threshold
        filtered = torch.where(valid, value, torch.zeros_like(value))
        # The upstream path uses a separable bilateral filter.  The portable
        # equivalent deliberately remains tensor-only and uses the same range
        # weighting, which is available on CPU and MPS without a Warp kernel.
        kernel = self.config.bilateral_kernel_size
        if kernel is not None and kernel > 1:
            radius = kernel // 2
            padded = F.pad(filtered[:, None], (radius, radius, radius, radius), mode="replicate").squeeze(1)
            windows = padded.unfold(1, kernel, 1).unfold(2, kernel, 1)
            center = filtered[..., None, None]
            offsets = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
            yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
            spatial = torch.exp(-(xx.square() + yy.square()) / (2 * self.config.bilateral_sigma_spatial**2))
            range_weight = torch.exp(-(windows - center).square() / (2 * self.config.bilateral_sigma_depth**2))
            weights = spatial * range_weight * valid[..., None, None]
            filtered = (windows * weights).sum((-1, -2)) / weights.sum((-1, -2)).clamp_min(torch.finfo(value.dtype).eps)
            filtered = torch.where(valid, filtered, torch.zeros_like(filtered))
        if squeeze: filtered, valid = filtered[0], valid[0]
        if depth_out is not None: depth_out.copy_(filtered); filtered = depth_out
        if valid_mask_out is not None: valid_mask_out.copy_(valid); valid = valid_mask_out
        return filtered, valid

    def update_config(
        self, depth_minimum_distance: Optional[float] = None,
        depth_maximum_distance: Optional[float] = None,
        flying_pixel_threshold: Optional[float] = None,
        bilateral_sigma_depth: Optional[float] = None,
    ) -> None:
        updates = {
            "depth_minimum_distance": depth_minimum_distance,
            "depth_maximum_distance": depth_maximum_distance,
            "flying_pixel_threshold": flying_pixel_threshold,
            "bilateral_sigma_depth": bilateral_sigma_depth,
        }
        for name, value in updates.items():
            if value is not None:
                setattr(self.config, name, value)
        if self.config.depth_minimum_distance < 0 or self.config.depth_maximum_distance <= self.config.depth_minimum_distance:
            raise ValueError("depth limits must satisfy 0 <= minimum < maximum")

    @classmethod
    def from_config(cls, config: FilterDepthConfig, image_shape: Tuple[int, int],
                    device: str = "cuda", num_batch: int = 1) -> "FilterDepth":
        return cls(image_shape=image_shape, device=device, num_batch=num_batch, **config.__dict__)
