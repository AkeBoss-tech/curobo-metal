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
        self, image_shape: Tuple[int,int], depth_minimum_distance=0.1,
        depth_maximum_distance=10.0, flying_pixel_threshold=0.5,
        bilateral_kernel_size=5, bilateral_sigma_spatial=10.0,
        bilateral_sigma_depth=0.1, device="cuda", num_batch=1,
    ):
        self.image_shape = tuple(image_shape)
        self.num_batch = num_batch
        self.device = "mps" if str(device).startswith("cuda") and torch.backends.mps.is_available() else ("cpu" if str(device).startswith("cuda") else device)
        self.config = FilterDepthConfig(
            depth_minimum_distance, depth_maximum_distance, flying_pixel_threshold,
            bilateral_kernel_size, bilateral_sigma_spatial, bilateral_sigma_depth,
        )

    def __call__(self, depth_image, depth_out=None, valid_mask_out=None):
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
        if squeeze: filtered, valid = filtered[0], valid[0]
        if depth_out is not None: depth_out.copy_(filtered); filtered = depth_out
        if valid_mask_out is not None: valid_mask_out.copy_(valid); valid = valid_mask_out
        return filtered, valid

    def update_config(self, depth_minimum_distance=None, depth_maximum_distance=None, flying_pixel_threshold=None, bilateral_sigma_depth=None):
        for name, value in locals().copy().items():
            if name not in ("self",) and value is not None:
                setattr(self.config, name, value)

    @classmethod
    def from_config(cls, config, image_shape, device="cuda", num_batch=1):
        return cls(image_shape=image_shape, device=device, num_batch=num_batch, **config.__dict__)
