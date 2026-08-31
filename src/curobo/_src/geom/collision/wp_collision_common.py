"""Portable equivalents of the value-level parts of V2's Warp helpers.

The functions preserve the raw-kernel argument layout where it is meaningful,
but operate only on regular PyTorch tensors.  Atomic Warp accumulation is
replaced by deterministic, caller-serial accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from curobo._src.util.warp import wp as _raw_wp


class _WarpCompat:
    """Declaration-only Warp façade; portable calls use tensor operations."""

    @staticmethod
    def func(function):
        return function


wp = _raw_wp if _raw_wp is not None else _WarpCompat()


@dataclass
class SphereQueryData:
    """A sphere decoded from ``(x, y, z, radius)`` query storage."""

    center: torch.Tensor
    radius: torch.Tensor
    radius_adjusted: torch.Tensor


def apply_collision_activation(dist: wp.float32, eta: wp.float32) -> wp.vec2:
    """Return V2's C1 ``(cost, gradient_scale)`` activation pair.

    ``dist`` is penetration (positive in collision).  The result is a tensor
    with final dimension two for tensor input, or a two-item tuple for scalar
    input, matching the indexable ``wp.vec2`` value layout.
    """
    tensor_input = isinstance(dist, torch.Tensor) or isinstance(eta, torch.Tensor)
    value = torch.as_tensor(dist)
    if not value.is_floating_point():
        value = value.to(torch.get_default_dtype())
    threshold = torch.as_tensor(eta, device=value.device, dtype=value.dtype)
    positive = value > 0
    # eta==0 naturally takes the linear branch for positive penetration.
    quadratic = positive & (value <= threshold) & (threshold > 0)
    cost = torch.where(
        quadratic, 0.5 * value.square() / threshold.clamp_min(torch.finfo(value.dtype).eps),
        torch.where(positive, value - 0.5 * threshold, torch.zeros_like(value)),
    )
    scale = torch.where(
        quadratic, value / threshold.clamp_min(torch.finfo(value.dtype).eps),
        positive.to(value.dtype),
    )
    if not tensor_input and value.ndim == 0:
        return float(cost), float(scale)
    return torch.stack((cost, scale), dim=-1)


def load_sphere_query(
    spheres: wp.array(dtype=wp.vec4), idx: wp.int32, eta: wp.float32
) -> SphereQueryData:
    """Decode one V2-layout query sphere from a tensor or numeric array."""
    values = torch.as_tensor(spheres)
    sphere = values.reshape(-1, values.shape[-1])[int(idx)]
    if sphere.numel() != 4:
        raise ValueError("spheres must have a final (x, y, z, radius) dimension of 4")
    eta_value = torch.as_tensor(eta, device=sphere.device, dtype=sphere.dtype)
    return SphereQueryData(sphere[:3], sphere[3], sphere[3] + eta_value)


def accumulate_collision(
    sph_flat_idx: wp.int32,
    cost: wp.float32,
    grad: wp.vec3,
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
):
    """Deterministically accumulate one raw-layout result into tensor buffers."""
    if not isinstance(distance, torch.Tensor) or not isinstance(gradient, torch.Tensor):
        raise NotImplementedError(
            "accumulate_collision requires PyTorch tensor buffers; Warp atomic arrays "
            "are unavailable on the Metal backend"
        )
    index = int(sph_flat_idx)
    flat_distance = distance.reshape(-1)
    flat_gradient = gradient.reshape(-1, 4)
    if index < 0 or index >= flat_distance.numel() or index >= flat_gradient.shape[0]:
        raise IndexError("sph_flat_idx is outside the collision output buffers")
    cost_value = torch.as_tensor(cost, device=flat_distance.device, dtype=flat_distance.dtype)
    grad_value = torch.as_tensor(grad, device=flat_gradient.device, dtype=flat_gradient.dtype)
    if grad_value.numel() != 3:
        raise ValueError("grad must contain exactly three spatial components")
    flat_distance[index].add_(cost_value)
    flat_gradient[index, :3].add_(grad_value.reshape(3))


def process_collision_result(
    sdf_result: wp.vec4,
    radius_adjusted: wp.float32,
    weight: wp.float32,
    eta: wp.float32,
    sph_flat_idx: wp.int32,
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
):
    """Apply the V2 activation rule and accumulate a tensor SDF result."""
    result = torch.as_tensor(sdf_result)
    if result.numel() != 4:
        raise ValueError("sdf_result must contain signed distance followed by xyz gradient")
    adjusted = torch.as_tensor(radius_adjusted, device=result.device, dtype=result.dtype)
    penetration = -result[0] + adjusted
    activation = apply_collision_activation(penetration, eta)
    if isinstance(activation, tuple):
        cost, scale = activation
    else:
        cost, scale = activation.unbind(-1)
    if torch.as_tensor(penetration).item() > 0:
        weighted = torch.as_tensor(weight, device=result.device, dtype=result.dtype)
        accumulate_collision(sph_flat_idx, weighted * cost, weighted * scale * result[1:], distance, gradient)


__all__ = [
    "SphereQueryData", "accumulate_collision", "apply_collision_activation",
    "load_sphere_query", "process_collision_result",
]
