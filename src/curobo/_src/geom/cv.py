"""Portable pinhole-camera projection helpers."""

import torch


def get_projection_rays(height, width, intrinsics, depth_to_meter=1.0):
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    y, x = torch.meshgrid(
        torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype),
        torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype),
        indexing="ij",
    )
    rx = (x[None] - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None]
    ry = (y[None] - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None]
    return torch.stack((rx.expand_as(ry), ry, torch.ones_like(ry)), -1) * depth_to_meter


def project_depth_using_rays(depth_image, projection_rays, out_points=None):
    depth = depth_image.unsqueeze(0) if depth_image.ndim == 2 else depth_image
    result = depth[..., None] * projection_rays
    if out_points is not None:
        out_points.copy_(result); return out_points
    return result


def project_depth_to_pointcloud(depth_image, intrinsics, depth_to_meter=1.0, out_points=None):
    rays = get_projection_rays(*depth_image.shape[-2:], intrinsics, depth_to_meter)
    return project_depth_using_rays(depth_image, rays, out_points)


def extract_depth_from_structured_pointcloud(pointcloud, output_image=None):
    depth = pointcloud[..., 2]
    if output_image is not None:
        output_image.copy_(depth); return output_image
    return depth


__all__ = [
    "extract_depth_from_structured_pointcloud", "get_projection_rays",
    "project_depth_to_pointcloud", "project_depth_using_rays",
]
