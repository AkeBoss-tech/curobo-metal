from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from curobo._src.types.pose import Pose
from curobo.logging import log_and_raise
wp = None


@dataclass
class BlockSparseTSDFRendererCfg:
    """Source-compatible rendering thresholds for a mapper integrator."""

    depth_minimum_distance: float = 0.2
    depth_maximum_distance: float = 15.0
    minimum_tsdf_weight: float = 0.2


def depth_to_colormap(
    depth: torch.Tensor,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 5.0,
    valid_mask: Optional[torch.Tensor] = None,
    invalid_color: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth >= depth_minimum_distance) & (depth <= depth_maximum_distance)
    if valid_mask is not None: valid &= valid_mask
    value = ((depth-depth_minimum_distance)/(depth_maximum_distance-depth_minimum_distance)).clamp(0,1)
    color = torch.stack((value, 1-(2*value-1).abs(), 1-value), -1)
    invalid = color.new_tensor(invalid_color)/255
    return torch.where(valid[...,None], color, invalid).mul(255).to(torch.uint8)


def normals_to_colormap(
    normals: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    color = ((normals+1)*127.5).clamp(0,255).to(torch.uint8)
    return torch.where(valid_mask[...,None], color, torch.zeros_like(color))


class BlockSparseTSDFRenderer:
    def __init__(
        self,
        integrator,
    ) -> None:
        self.integrator = integrator

    def render(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.integrator.render(intrinsics, pose, image_shape)

    def render_depth(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render(intrinsics, pose, image_shape)[0]

    def render_normals(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render(intrinsics, pose, image_shape)[1]

    def render_depth_colormap(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        depth, _, valid = self.render(intrinsics, pose, image_shape)
        return depth_to_colormap(depth, valid_mask=valid)

    def render_normal_colormap(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        return normals_to_colormap(normals, valid)

    def render_color(
        self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(self.integrator, "render_color"):
            return self.integrator.render_color(intrinsics, pose, image_shape)
        depth, normals, valid = self.render(intrinsics, pose, image_shape)
        return depth, normals, torch.zeros(depth.shape + (3,), device=depth.device, dtype=torch.uint8), valid

    def render_color_only(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render_color(intrinsics, pose, image_shape)[2]

    def render_shaded(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
        light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        ambient: float = 0.3,
        use_color: bool = True,
    ) -> torch.Tensor:
        if hasattr(self.integrator, "render_shaded"):
            return self.integrator.render_shaded(intrinsics, pose, image_shape, light_direction, ambient, use_color)
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        light = normals.new_tensor(light_direction)
        light = light / torch.linalg.vector_norm(light).clamp_min(torch.finfo(normals.dtype).eps)
        intensity = (normals * light).sum(-1).clamp_min(0) * (1 - ambient) + ambient
        return torch.where(valid[..., None], (255 * intensity[..., None]).to(torch.uint8), torch.zeros_like(normals, dtype=torch.uint8))
