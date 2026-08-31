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
    if camera_obs.depth_image is None:
        raise ValueError("camera observation requires a depth image")
    if camera_obs.projection_rays is not None:
        depth = camera_obs.depth_image.reshape(-1, 1)
        rays = camera_obs.projection_rays.reshape(-1, 3)
        if depth.shape[0] != rays.shape[0]:
            raise ValueError("depth image and projection rays must contain the same pixels")
        points = depth * rays
        if camera_obs.pose is not None:
            points = camera_obs.pose.batch_transform_points(points).reshape(-1, 3)
    else:
        points = camera_obs.get_pointcloud(project_to_pose=True).view(-1, 3)
    if camera_obs.image_segmentation is not None:
        points = points[camera_obs.image_segmentation.view(-1) > 0]
    valid = torch.isfinite(points).all(dim=1)
    valid &= points[:, 2].abs() > min_depth
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
    if distance_threshold < float("inf"):
        indices = indices.clone()
        indices[values > distance_threshold] = -1
    return indices


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
    distances = ((target_points - source_points) * target_normals).sum(dim=1)
    if use_huber:
        absolute = distances.abs()
        huber_weights = torch.where(
            absolute < huber_delta,
            torch.ones_like(absolute),
            huber_delta / (absolute + 1e-10),
        )
        weights = huber_weights if weights is None else huber_weights * weights
    cross = torch.cross(source_points, target_normals, dim=1)
    normals = target_normals
    if weights is not None:
        root = weights.sqrt().unsqueeze(1)
        cross = cross * root
        normals = normals * root
        distances = distances * root.squeeze(1)
    jacobian = torch.cat((cross, normals), dim=1)
    normal_matrix = jacobian.T @ jacobian
    right_hand_side = jacobian.T @ distances
    normal_matrix = normal_matrix + damping * torch.eye(
        6, device=normal_matrix.device, dtype=normal_matrix.dtype
    )
    factor, info = torch.linalg.cholesky_ex(normal_matrix)
    if int(info.item()) == 0:
        solution = torch.cholesky_solve(
            right_hand_side.unsqueeze(1), factor
        ).squeeze(1)
    else:
        # CPU is an explicit numerical fallback for an unsupported/degenerate
        # least-squares solve; the result returns to the caller's device.
        solution = torch.linalg.lstsq(
            jacobian.cpu(), distances.cpu().unsqueeze(1)
        ).solution.squeeze(1).to(jacobian.device)
    return solution[3:], omega_to_quaternion(solution[:3])


def compute_pose_point_to_plane_svd(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_huber: bool = False,
    huber_delta: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    difference = target_points - source_points
    distances = (difference * target_normals).sum(dim=1, keepdim=True)
    if use_huber:
        absolute = distances.abs().squeeze(-1)
        huber_weights = torch.where(
            absolute < huber_delta,
            torch.ones_like(absolute),
            huber_delta / (absolute + 1e-10),
        )
        weights = huber_weights if weights is None else huber_weights * weights
    projected = source_points + distances * target_normals
    if weights is None:
        source_mean = source_points.mean(dim=0, keepdim=True)
        target_mean = projected.mean(dim=0, keepdim=True)
        source_centered = source_points - source_mean
        target_centered = projected - target_mean
        covariance = target_centered.T @ source_centered
    else:
        normalized = weights / (weights.sum() + 1e-10)
        source_mean = (source_points * normalized.unsqueeze(1)).sum(dim=0, keepdim=True)
        target_mean = (projected * normalized.unsqueeze(1)).sum(dim=0, keepdim=True)
        source_centered = source_points - source_mean
        target_centered = projected - target_mean
        covariance = target_centered.T @ (source_centered * weights.unsqueeze(1))
    # MPS has no native SVD kernel.  Make the tiny 3x3 numerical solve an
    # explicit CPU operation so execution never depends on PyTorch's implicit
    # MPS fallback setting, then return the rigid transform to the input device.
    solve_matrix = covariance.cpu() if covariance.device.type == "mps" else covariance
    left, _, right_h = torch.linalg.svd(solve_matrix)
    right = right_h.T
    rotation = left @ right.T
    if bool((torch.det(rotation) < 0).item()):
        right = right.clone()
        right[:, -1] *= -1
        rotation = left @ right.T
    rotation = rotation.to(covariance.device)
    position = (target_mean.T - rotation @ source_mean.T).squeeze()
    quaternion = matrix_to_quaternion(rotation.unsqueeze(0)).squeeze(0)
    return position, quaternion
