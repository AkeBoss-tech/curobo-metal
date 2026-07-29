"""Batched differentiable IK costs for CPU and Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from curobo_metal.backend import validate_tensor_device
from curobo_metal.ops.collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
    transform_spheres,
)


@dataclass(frozen=True)
class CollisionModel:
    """Device-resident primitive collision metadata."""

    local_spheres: torch.Tensor
    link_indices: torch.Tensor
    self_pairs: torch.Tensor | None = None
    cuboid_centers: torch.Tensor | None = None
    cuboid_rotations: torch.Tensor | None = None
    cuboid_half_extents: torch.Tensor | None = None
    padding: float = 0.0
    activation_distance: float = 0.0
    weight: float = 1.0


@dataclass(frozen=True)
class CostTerm:
    """Named composable cost with scalar or per-run weights."""

    function: Callable[[torch.Tensor], torch.Tensor]
    weight: float | torch.Tensor = 1.0
    enabled: bool = True

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        result = self.function(value)
        if not self.enabled:
            return result * 0
        weight = torch.as_tensor(self.weight, device=result.device, dtype=result.dtype)
        return result * weight


@dataclass(frozen=True)
class CostManager:
    terms: Mapping[str, CostTerm]

    def evaluate(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = {name: term(value) for name, term in self.terms.items()}
        if not components:
            return value.sum(-1) * 0, components
        iterator = iter(components.values())
        total = next(iterator)
        for component in iterator:
            total = total + component
        return total, components


def waypoint_cost(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    offset: int = 0,
    run_weight: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Squared waypoint error with deterministic offset and broadcast run weights."""
    trajectory = _floating(value, "value")
    goal = _floating(target, "target")
    if trajectory.ndim < 2 or goal.shape[-1] != trajectory.shape[-1]:
        raise ValueError("value must end in [T,J] and target in [J] or [W,J]")
    index = offset if offset >= 0 else trajectory.shape[-2] + offset
    if index < 0 or index >= trajectory.shape[-2]:
        raise ValueError("waypoint offset is outside the trajectory")
    residual = trajectory[..., index, :] - goal
    weight = torch.as_tensor(run_weight, device=value.device, dtype=value.dtype)
    return 0.5 * residual.square().sum(-1) * weight


def _floating(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(value)
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if value.device.type == "mps" and value.dtype != torch.float32:
        raise TypeError("MPS costs support only float32")
    return value


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions ``[...,4]`` to rotation matrices."""
    q = _floating(quaternion, "quaternion")
    if q.ndim < 1 or q.shape[-1] != 4:
        raise ValueError("quaternion must have shape [...,4]")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if bool((~torch.isfinite(q)).any().item()) or bool((norm <= 0).any().item()):
        raise ValueError("quaternion must be finite and nonzero")
    w, x, y, z = (q / norm).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _rotation_vector(rotation: torch.Tensor) -> torch.Tensor:
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    cosine = 0.5 * (
        rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2] - 1
    )
    angle = torch.atan2(sine, cosine)
    scale = torch.where(
        sine > 1e-7,
        angle / (2 * sine).clamp_min(torch.finfo(rotation.dtype).eps),
        0.5 + angle * angle / 12,
    )
    regular = scale[..., None] * vee

    # At exactly pi the skew part vanishes. Select the largest diagonal
    # quaternion component, then canonicalize the first nonzero axis sign.
    eps = torch.finfo(rotation.dtype).eps
    x = (0.5 * (rotation[..., 0, 0] + 1)).clamp_min(0).sqrt()
    y = (0.5 * (rotation[..., 1, 1] + 1)).clamp_min(0).sqrt()
    z = (0.5 * (rotation[..., 2, 2] + 1)).clamp_min(0).sqrt()
    candidates = torch.stack(
        (
            torch.stack((x, (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4 * x.clamp_min(eps)),
                         (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4 * x.clamp_min(eps))), -1),
            torch.stack(((rotation[..., 0, 1] + rotation[..., 1, 0]) / (4 * y.clamp_min(eps)), y,
                         (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4 * y.clamp_min(eps))), -1),
            torch.stack(((rotation[..., 0, 2] + rotation[..., 2, 0]) / (4 * z.clamp_min(eps)),
                         (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4 * z.clamp_min(eps)), z), -1),
        ),
        -2,
    )
    largest = torch.stack((x, y, z), -1).argmax(-1)
    axis = candidates.gather(
        -2, largest[..., None, None].expand(largest.shape + (1, 3))
    ).squeeze(-2)
    first = (axis.abs() > 1e-6).to(torch.int64).argmax(-1)
    first_value = axis.gather(-1, first[..., None]).squeeze(-1)
    axis = torch.where((first_value < 0)[..., None], -axis, axis)
    at_pi = (sine <= 1e-6) & (cosine < 0)
    return torch.where(at_pi[..., None], torch.pi * axis, regular)


def pose_error(
    transform: torch.Tensor,
    target_position: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> torch.Tensor:
    """Return ``[...,6]`` world-translation and target-frame rotation error."""
    matrix = _floating(transform, "transform")
    position = _floating(target_position, "target_position")
    quaternion = _floating(target_quaternion, "target_quaternion")
    if matrix.shape[-2:] != (4, 4) or position.shape[-1:] != (3,):
        raise ValueError("transform and target_position must end in [4,4] and [3]")
    if (
        matrix.device.type != position.device.type
        or matrix.device.type != quaternion.device.type
    ):
        raise ValueError("pose inputs must be on the same device")
    if matrix.dtype != position.dtype or matrix.dtype != quaternion.dtype:
        raise TypeError("pose inputs must have the same dtype")
    target_rotation = quaternion_to_matrix(quaternion)
    relative = target_rotation.transpose(-1, -2) @ matrix[..., :3, :3]
    return torch.cat((matrix[..., :3, 3] - position, _rotation_vector(relative)), -1)


def pose_cost(error: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    residual = _floating(error, "error")
    if residual.shape[-1:] != (6,):
        raise ValueError("error must end in shape [6]")
    weight = torch.ones(6, device=residual.device, dtype=residual.dtype) if weights is None else weights
    weight = _floating(weight, "weights")
    if (
        weight.shape[-1:] != (6,)
        or weight.device.type != residual.device.type
        or weight.dtype != residual.dtype
    ):
        raise ValueError("weights must end in [6] and match error dtype/device")
    if bool((weight < 0).any().item()):
        raise ValueError("weights must be nonnegative")
    return 0.5 * (weight * residual.square()).sum(dim=-1)


def joint_limit_cost(
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    margin: float = 0.0,
    weight: float = 1.0,
) -> torch.Tensor:
    value = _floating(q, "q")
    if lower.shape != value.shape[-1:] or upper.shape != value.shape[-1:]:
        raise ValueError("lower and upper must match the final q dimension")
    if lower.device.type != value.device.type or upper.device.type != value.device.type:
        raise ValueError("joint limit tensors must be on the q device")
    if lower.dtype != value.dtype or upper.dtype != value.dtype:
        raise TypeError("joint limit tensors must match q dtype")
    if margin < 0 or weight < 0 or bool((lower > upper).any().item()):
        raise ValueError("limits and nonnegative cost parameters are invalid")
    below = (lower + margin - value).clamp_min(0)
    above = (value - (upper - margin)).clamp_min(0)
    return 0.5 * weight * (below.square() + above.square()).sum(dim=-1)


def smoothness_cost(
    q: torch.Tensor, *, velocity_weight: float = 1.0, acceleration_weight: float = 0.0
) -> torch.Tensor:
    trajectory = _floating(q, "q")
    if trajectory.ndim < 2:
        raise ValueError("q must end in shape [T,J]")
    if velocity_weight < 0 or acceleration_weight < 0:
        raise ValueError("smoothness weights must be nonnegative")
    velocity = torch.diff(trajectory, dim=-2)
    acceleration = torch.diff(trajectory, n=2, dim=-2)
    return 0.5 * (
        velocity_weight * velocity.square().sum(dim=(-2, -1))
        + acceleration_weight * acceleration.square().sum(dim=(-2, -1))
    )


def collision_cost(
    clearances: torch.Tensor, *, activation_distance: float = 0.0, weight: float = 1.0
) -> torch.Tensor:
    distance = _floating(clearances, "clearances")
    if activation_distance < 0 or weight < 0:
        raise ValueError("collision cost parameters must be nonnegative")
    penetration = (activation_distance - distance).clamp_min(0)
    return 0.5 * weight * penetration.square().sum(dim=-1)


def robot_collision_cost(
    transforms: torch.Tensor, model: CollisionModel
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose production FK outputs with production primitive collision paths."""
    spheres = transform_spheres(
        transforms, model.local_spheres, model.link_indices
    ).spheres
    values: list[torch.Tensor] = []
    if model.self_pairs is not None:
        values.append(
            sphere_sphere_signed_distance(
                spheres, model.self_pairs, padding=model.padding
            ).distances
        )
    if model.cuboid_centers is not None:
        if model.cuboid_rotations is None or model.cuboid_half_extents is None:
            raise ValueError("complete cuboid metadata is required")
        world = sphere_cuboid_signed_distance(
            spheres,
            model.cuboid_centers,
            model.cuboid_rotations,
            model.cuboid_half_extents,
            padding=model.padding,
        ).distances
        values.append(world.flatten(start_dim=1))
    values = [value for value in values if value.shape[1] > 0]
    if not values:
        zero = spheres.sum(dim=(1, 2)) * 0
        return zero, spheres.new_full((spheres.shape[0],), torch.inf)
    clearances = torch.cat([value.flatten(start_dim=1) for value in values], dim=1)
    finite = torch.where(torch.isfinite(clearances), clearances, torch.inf)
    cost = collision_cost(
        finite, activation_distance=model.activation_distance, weight=model.weight
    )
    return cost, finite.min(dim=1).values
