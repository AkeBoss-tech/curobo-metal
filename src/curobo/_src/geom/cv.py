"""Portable pinhole-camera projection helpers."""

from __future__ import annotations

import torch


def get_projection_rays(
    height: int, width: int, intrinsics_matrix: torch.Tensor, depth_to_meter: float = 0.001
) -> torch.Tensor:
    if intrinsics_matrix.ndim == 2:
        intrinsics_matrix = intrinsics_matrix.unsqueeze(0)
    y, x = torch.meshgrid(
        torch.arange(height, device=intrinsics_matrix.device, dtype=intrinsics_matrix.dtype),
        torch.arange(width, device=intrinsics_matrix.device, dtype=intrinsics_matrix.dtype),
        indexing="ij",
    )
    rx = (x[None] - intrinsics_matrix[:, 0, 2, None, None]) / intrinsics_matrix[:, 0, 0, None, None]
    ry = (y[None] - intrinsics_matrix[:, 1, 2, None, None]) / intrinsics_matrix[:, 1, 1, None, None]
    return torch.stack((rx.expand_as(ry), ry, torch.ones_like(ry)), -1) * depth_to_meter


def project_depth_using_rays(
    depth_image: torch.Tensor,
    rays: torch.Tensor,
    filter_origin: bool = False,
    depth_threshold: float = 0.01,
) -> torch.Tensor:
    depth = depth_image.unsqueeze(0) if depth_image.ndim == 2 else depth_image
    result = depth[..., None] * rays
    if filter_origin:
        result = torch.where(
            (depth > depth_threshold).unsqueeze(-1), result, torch.zeros_like(result)
        )
    return result


def project_depth_to_pointcloud(depth_image: torch.Tensor, intrinsics_matrix: torch.Tensor) -> torch.Tensor:
    rays = get_projection_rays(*depth_image.shape[-2:], intrinsics_matrix)
    return project_depth_using_rays(depth_image, rays)


def extract_depth_from_structured_pointcloud(
    pointcloud: torch.Tensor, output_image: torch.Tensor
) -> torch.Tensor:
    depth = pointcloud[..., 2]
    output_image.copy_(depth)
    return output_image


__all__ = [
    "extract_depth_from_structured_pointcloud", "get_projection_rays",
    "project_depth_to_pointcloud", "project_depth_using_rays",
]
