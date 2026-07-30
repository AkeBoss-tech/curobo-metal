"""Portable PyTorch implementation of cuRobo's trajectory speed metric."""

from __future__ import annotations

import torch


def apply_speed_metric(
    distance: torch.Tensor,
    gradient: torch.Tensor,
    query_spheres: torch.Tensor,
    trajectory_dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale positive interior costs by central-difference sphere speed.

    The return is differentiable and deterministic. CUDA/Warp's in-place launch
    details are intentionally not reproduced.
    """
    if query_spheres.ndim != 4 or query_spheres.shape[-1] != 4:
        raise ValueError("query_spheres must have shape [batch,horizon,spheres,4]")
    if distance.shape != query_spheres.shape[:-1]:
        raise ValueError("distance shape must match query_spheres[...,0]")
    if gradient.shape != query_spheres.shape:
        raise ValueError("gradient shape must match query_spheres")
    if query_spheres.shape[1] <= 2:
        return distance, gradient
    dt = torch.as_tensor(trajectory_dt, device=distance.device, dtype=distance.dtype)
    dt = dt.reshape(-1)[0].clamp_min(torch.finfo(distance.dtype).eps)
    velocity = (query_spheres[:, 2:, :, :3] - query_spheres[:, :-2, :, :3]) / (2 * dt)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    factor = torch.where(distance[:, 1:-1] > 0, speed, torch.ones_like(speed))
    out_distance = distance.clone()
    out_gradient = gradient.clone()
    out_distance[:, 1:-1] = out_distance[:, 1:-1] * factor
    out_gradient[:, 1:-1] = out_gradient[:, 1:-1] * factor[..., None]
    return out_distance, out_gradient


__all__ = ["apply_speed_metric"]
