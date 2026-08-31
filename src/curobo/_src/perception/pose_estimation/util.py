"""Portable point-cloud and pose-estimation utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from curobo._src.geom.transform import matrix_to_quaternion
from curobo._src.types.camera import CameraObservation
from curobo._src.util.torch_util import get_profiler_decorator, get_torch_jit_decorator


def extract_observed_points(
    camera_obs: CameraObservation,
    min_depth: float = 0.1,
) -> torch.Tensor:
    depth = camera_obs.depth_image
    intr = camera_obs.intrinsics
    valid = torch.isfinite(depth) & (depth >= min_depth)
    y, x = torch.meshgrid(
        torch.arange(depth.shape[-2], device=depth.device, dtype=depth.dtype),
        torch.arange(depth.shape[-1], device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth
    points = torch.stack(
        ((x - intr[..., 0, 2]) * z / intr[..., 0, 0],
         (y - intr[..., 1, 2]) * z / intr[..., 1, 1], z), -1,
    )
    return points[valid]


def omega_to_quaternion(omega: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    half = angle / 2
    scale = torch.where(angle > 1e-8, torch.sin(half) / angle, 0.5 - angle.square() / 48)
    return torch.cat((torch.cos(half), omega * scale), -1)


def huber_loss(residuals: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    absolute = residuals.abs()
    return torch.where(absolute <= delta, 0.5 * residuals.square(), delta * (absolute - 0.5 * delta))


def find_nearest_neighbors(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    distance_threshold: float = float("inf"),
) -> torch.Tensor:
    distances = torch.cdist(source_points, target_points)
    values, indices = distances.min(-1)
    return indices, values, values <= distance_threshold


def resample_points(
    points: torch.Tensor,
    target_count: int,
    device: torch.device = None,
) -> torch.Tensor:
    if target_count < 0 or len(points) == 0 and target_count:
        raise ValueError("target_count requires a nonempty point set")
    if target_count == 0:
        return points[:0]
    indices = torch.arange(target_count, device=points.device) % len(points)
    return points[indices].to(device=device or points.device)


def _point_to_plane_system(source_points, target_points, target_normals, weights, use_huber, huber_delta):
    cross = torch.linalg.cross(source_points, target_normals)
    jac = torch.cat((target_normals, cross), -1)
    residual = ((source_points - target_points) * target_normals).sum(-1)
    if weights is None:
        weights = torch.ones_like(residual)
    if use_huber:
        weights = weights * torch.where(residual.abs() <= huber_delta, 1, huber_delta / residual.abs().clamp_min(1e-12))
    return jac, residual, weights


def compute_pose_point_to_plane_cholesky(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_huber: bool = False,
    huber_delta: float = 0.02,
    damping: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    jac, residual, weights = _point_to_plane_system(
        source_points, target_points, target_normals, weights, use_huber, huber_delta
    )
    jtj = jac.mT @ (weights[:, None] * jac)
    jtr = jac.mT @ (weights * residual)
    return torch.linalg.solve(jtj + damping * torch.eye(6, device=jtj.device, dtype=jtj.dtype), -jtr)


def compute_pose_point_to_plane_svd(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_huber: bool = False,
    huber_delta: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    jac, residual, weights = _point_to_plane_system(
        source_points, target_points, target_normals, weights, use_huber, huber_delta
    )
    return torch.linalg.lstsq(jac * weights.sqrt()[:, None], -residual * weights.sqrt()).solution
