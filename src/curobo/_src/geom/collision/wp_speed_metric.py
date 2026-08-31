"""Portable form of V2's speed-metric post-processing kernel."""

from __future__ import annotations

import torch

class _PortableWarpAnnotations:
    @staticmethod
    def kernel(function):
        del function
        return _apply_speed_metric_portable


wp = _PortableWarpAnnotations()


def _apply_tensor_speed_metric(spheres, distance, gradient, speed_dt):
    if spheres.ndim != 4 or spheres.shape[-1] != 4:
        raise ValueError("spheres must have shape [batch, horizon, spheres, 4]")
    if distance.shape != spheres.shape[:-1]:
        raise ValueError("distance must have shape [batch, horizon, spheres]")
    if gradient.shape != spheres.shape:
        raise ValueError("gradient must have shape [batch, horizon, spheres, 4]")
    if spheres.shape[1] <= 2:
        return distance, gradient
    dt = torch.as_tensor(speed_dt, device=distance.device, dtype=distance.dtype)
    dt = dt.reshape(-1)[0].clamp_min(1.0e-6)
    velocity = (spheres[:, 2:, :, :3] - spheres[:, :-2, :, :3]) / (2 * dt)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    out_distance, out_gradient = distance.clone(), gradient.clone()
    current_distance = distance[:, 1:-1]
    active = (current_distance > 0.0) & (speed >= 1.0e-3)
    safe_speed = speed.clamp_min(1.0e-3)
    normalized_velocity = velocity / safe_speed.unsqueeze(-1)
    acceleration = (
        spheres[:, :-2, :, :3]
        + spheres[:, 2:, :, :3]
        - 2.0 * spheres[:, 1:-1, :, :3]
    ) / (dt * dt)
    curvature = acceleration / safe_speed.square().unsqueeze(-1)
    current_gradient = gradient[:, 1:-1, :, :3]
    orthogonal_gradient = current_gradient - (
        normalized_velocity * current_gradient
    ).sum(dim=-1, keepdim=True) * normalized_velocity
    orthogonal_curvature = curvature - (
        normalized_velocity * curvature
    ).sum(dim=-1, keepdim=True) * normalized_velocity
    new_gradient = safe_speed.unsqueeze(-1) * (
        orthogonal_gradient - current_distance.unsqueeze(-1) * orthogonal_curvature
    )
    out_distance[:, 1:-1] = torch.where(active, speed * current_distance, current_distance)
    out_gradient[:, 1:-1, :, :3] = torch.where(
        active.unsqueeze(-1), new_gradient, current_gradient
    )
    return out_distance, out_gradient


def _apply_speed_metric_portable(
    spheres: torch.Tensor,
    distance: torch.Tensor,
    gradient: torch.Tensor,
    speed_dt: torch.Tensor,
    batch_size=None,
    horizon=None,
    num_spheres=None,
):
    """Apply the speed metric with the pinned Warp-kernel argument layout.

    Seven arguments model an in-place raw kernel launch and return ``None``.
    The historical portable four-argument order ``(distance, gradient,
    spheres, speed_dt)`` remains accepted and returns new tensors so existing
    Metal callers continue to work.
    """
    # Earlier portable releases accidentally exposed the four first arguments
    # in a result-first order.  Detect it structurally rather than by device.
    legacy_order = isinstance(spheres, torch.Tensor) and spheres.ndim == 3 and (
        isinstance(gradient, torch.Tensor) and gradient.ndim == 4 and gradient.shape[-1] == 4
    )
    if legacy_order:
        return _apply_tensor_speed_metric(gradient, spheres, distance, speed_dt)
    result_distance, result_gradient = _apply_tensor_speed_metric(
        spheres, distance, gradient, speed_dt
    )
    if batch_size is None and horizon is None and num_spheres is None:
        return result_distance, result_gradient
    expected = (int(batch_size), int(horizon), int(num_spheres))
    if tuple(spheres.shape[:3]) != expected:
        raise ValueError("batch_size, horizon, and num_spheres do not match spheres")
    distance.copy_(result_distance)
    gradient.copy_(result_gradient)
    return None


@wp.kernel
def apply_speed_metric(
    spheres: wp.array(dtype=wp.vec4),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    speed_dt: wp.array(dtype=wp.float32),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
):
    """Pinned declaration; the portable runtime implementation is rebound below."""
    return _apply_speed_metric_portable(
        spheres, distance, gradient, speed_dt, batch_size, horizon, num_spheres
    )


__all__ = ["apply_speed_metric"]
