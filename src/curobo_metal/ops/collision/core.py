"""Differentiable sphere collision operations for CPU and Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from curobo_metal.backend import validate_tensor_device


@dataclass(frozen=True)
class SphereTransformResult:
    spheres: torch.Tensor
    center_transform_jacobian: torch.Tensor
    input_was_batched: bool


@dataclass(frozen=True)
class PairDistanceResult:
    distances: torch.Tensor
    gradients: torch.Tensor
    reduced_distance: torch.Tensor
    reduced_gradient: torch.Tensor
    winning_pair: torch.Tensor
    input_was_unbatched: bool


@dataclass(frozen=True)
class CuboidDistanceResult:
    distances: torch.Tensor
    sphere_gradients: torch.Tensor
    reduced_distance: torch.Tensor
    reduced_sphere_gradient: torch.Tensor
    winning_cuboid: torch.Tensor
    input_was_unbatched: bool


def _floating(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(tensor)
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if tensor.device.type == "mps" and tensor.dtype != torch.float32:
        raise TypeError("MPS collision operations support only float32")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _same_floating(
    tensor: torch.Tensor, name: str, reference: torch.Tensor
) -> torch.Tensor:
    tensor = _floating(tensor, name)
    if tensor.device != reference.device:
        raise ValueError(
            f"{name} must be on {reference.device}, got {tensor.device}"
        )
    if tensor.dtype != reference.dtype:
        raise TypeError(
            f"{name} dtype {tensor.dtype} does not match {reference.dtype}"
        )
    return tensor


def _index(
    tensor: torch.Tensor, name: str, reference: torch.Tensor
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(tensor, expected=reference.device)
    if tensor.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype int64")
    return tensor


def _mask(
    value: torch.Tensor | None,
    length: int,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return torch.ones(length, dtype=torch.bool, device=reference.device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(value, expected=reference.device)
    if value.dtype != torch.bool or value.shape != (length,):
        raise ValueError(f"{name} must be boolean with shape [{length}]")
    return value


def _padding(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("padding must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("padding must be a finite nonnegative scalar")
    return result


def _batched_spheres(spheres: torch.Tensor) -> tuple[torch.Tensor, bool]:
    values = _floating(spheres, "spheres")
    unbatched = values.ndim == 2
    if unbatched:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[-1] != 4:
        raise ValueError("spheres must have shape [S, 4] or [B, S, 4]")
    if bool((values[..., 3] < 0).any().item()):
        raise ValueError("sphere radii must be nonnegative")
    return values, unbatched


def transform_spheres(
    transforms: torch.Tensor,
    local_spheres: torch.Tensor,
    link_indices: torch.Tensor,
    active: torch.Tensor | None = None,
) -> SphereTransformResult:
    """Transform link-local spheres, retaining a batch dimension."""
    matrices = _floating(transforms, "transforms")
    input_was_batched = matrices.ndim == 4
    if matrices.ndim == 3:
        matrices = matrices.unsqueeze(0)
    if matrices.ndim != 4 or matrices.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape [L,4,4] or [B,L,4,4]")
    local = _same_floating(local_spheres, "local_spheres", matrices)
    if local.ndim != 2 or local.shape[1] != 4:
        raise ValueError("local_spheres must have shape [S,4]")
    if bool((local[:, 3] < 0).any().item()):
        raise ValueError("sphere radii must be nonnegative")
    links = _index(link_indices, "link_indices", matrices)
    if links.shape != (local.shape[0],):
        raise ValueError("link_indices must have shape [S]")
    if bool(((links < 0) | (links >= matrices.shape[1])).any().item()):
        raise ValueError("link_indices contains an out-of-range link")
    enabled = _mask(active, local.shape[0], "active", matrices)

    count = local.shape[0]
    homogeneous = torch.cat((local[:, :3], local.new_ones((count, 1))), dim=1)
    selected = matrices[:, links]
    centers = torch.einsum("bsij,sj->bsi", selected, homogeneous)[..., :3]
    output = torch.cat(
        (centers, local[:, 3].expand(matrices.shape[0], count).unsqueeze(-1)),
        dim=-1,
    )
    output = torch.where(enabled[None, :, None], output, torch.zeros_like(output))

    eye3 = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    jacobian = torch.einsum("ki,sj->skij", eye3, homogeneous)
    jacobian = jacobian.unsqueeze(0).expand(matrices.shape[0], -1, -1, -1, -1)
    jacobian = torch.where(
        enabled[None, :, None, None, None], jacobian, torch.zeros_like(jacobian)
    )
    return SphereTransformResult(output, jacobian, input_was_batched)


def sphere_sphere_signed_distance(
    spheres: torch.Tensor,
    pairs: torch.Tensor,
    *,
    sphere_active: torch.Tensor | None = None,
    pair_active: torch.Tensor | None = None,
    padding: float = 0.0,
) -> PairDistanceResult:
    """Compute configured sphere-pair clearances and first-tie reduction."""
    values, unbatched = _batched_spheres(spheres)
    pad = _padding(padding)
    pair_table = _index(pairs, "pairs", values)
    if pair_table.ndim != 2 or pair_table.shape[1] != 2:
        raise ValueError("pairs must have shape [P,2]")
    if bool(((pair_table < 0) | (pair_table >= values.shape[1])).any().item()):
        raise ValueError("pairs contains an out-of-range sphere")
    if bool((pair_table[:, 0] == pair_table[:, 1]).any().item()):
        raise ValueError("a sphere cannot be paired with itself")
    sphere_enabled = _mask(
        sphere_active, values.shape[1], "sphere_active", values
    )
    pair_enabled = _mask(pair_active, pair_table.shape[0], "pair_active", values)
    if pair_table.shape[0]:
        pair_enabled = (
            pair_enabled
            & sphere_enabled[pair_table[:, 0]]
            & sphere_enabled[pair_table[:, 1]]
        )

    first, second = pair_table[:, 0], pair_table[:, 1]
    delta = values[:, first, :3] - values[:, second, :3]
    length = torch.linalg.vector_norm(delta, dim=-1)
    # PyTorch chooses zero at the origin; the contract selects +x.
    selected_length = length + torch.where(
        length == 0,
        delta[..., 0] - delta[..., 0].detach(),
        torch.zeros_like(length),
    )
    raw = (
        selected_length
        - values[:, first, 3]
        - values[:, second, 3]
        - pad
    )
    distances = torch.where(
        pair_enabled[None, :], raw, torch.full_like(raw, torch.inf)
    )

    direction = torch.where(
        (length > 0)[..., None],
        delta / torch.where(length > 0, length, torch.ones_like(length))[..., None],
        delta.new_tensor((1.0, 0.0, 0.0)),
    )
    gradients = values.new_zeros(
        (values.shape[0], pair_table.shape[0], values.shape[1], 4)
    )
    if pair_table.shape[0]:
        batch_index = torch.arange(values.shape[0], device=values.device)[:, None]
        pair_index = torch.arange(pair_table.shape[0], device=values.device)[None, :]
        gradients[batch_index, pair_index, first[None, :], :3] = direction
        gradients[batch_index, pair_index, second[None, :], :3] = -direction
        gradients[batch_index, pair_index, first[None, :], 3] = -1
        gradients[batch_index, pair_index, second[None, :], 3] = -1
        gradients = torch.where(
            pair_enabled[None, :, None, None], gradients, torch.zeros_like(gradients)
        )

    if pair_table.shape[0] and bool(pair_enabled.any().item()):
        reduced, winners = torch.min(distances, dim=1)
        batch_index = torch.arange(values.shape[0], device=values.device)
        reduced_gradient = gradients[batch_index, winners]
    else:
        reduced = values.sum(dim=(1, 2)) * 0 + torch.inf
        winners = torch.full(
            (values.shape[0],), -1, dtype=torch.int64, device=values.device
        )
        reduced_gradient = values.new_zeros(values.shape)
    return PairDistanceResult(
        distances, gradients, reduced, reduced_gradient, winners, unbatched
    )


class _BoxSDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
        q = local.abs() - half
        outside = q.clamp_min(0)
        outside_length = torch.linalg.vector_norm(outside, dim=-1)
        inside = q.max(dim=-1).values.clamp_max(0)
        ctx.save_for_backward(local, q, outside, outside_length)
        return outside_length + inside

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local, q, outside, outside_length = ctx.saved_tensors
        outside_case = outside_length > 0
        sign = torch.where(local < 0, -torch.ones_like(local), torch.ones_like(local))
        safe_length = torch.where(
            outside_case, outside_length, torch.ones_like(outside_length)
        )
        outside_gradient = outside / safe_length[..., None] * sign
        axis = q.argmax(dim=-1)
        inside_gradient = torch.zeros_like(local).scatter_(
            -1, axis[..., None], sign.gather(-1, axis[..., None])
        )
        local_gradient = torch.where(
            outside_case[..., None], outside_gradient, inside_gradient
        )
        q_gradient = local_gradient * sign
        return (
            grad_output[..., None] * local_gradient,
            -grad_output[..., None] * q_gradient,
        )


def sphere_cuboid_signed_distance(
    spheres: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_rotations: torch.Tensor,
    cuboid_half_extents: torch.Tensor,
    *,
    sphere_active: torch.Tensor | None = None,
    cuboid_active: torch.Tensor | None = None,
    padding: float = 0.0,
) -> CuboidDistanceResult:
    """Compute sphere-to-oriented-cuboid clearance and first-tie reduction."""
    values, unbatched = _batched_spheres(spheres)
    centers = _same_floating(cuboid_centers, "cuboid_centers", values)
    rotations = _same_floating(cuboid_rotations, "cuboid_rotations", values)
    half = _same_floating(cuboid_half_extents, "cuboid_half_extents", values)
    cuboids = centers.shape[0] if centers.ndim == 2 else -1
    if (
        centers.shape != (cuboids, 3)
        or rotations.shape != (cuboids, 3, 3)
        or half.shape != (cuboids, 3)
    ):
        raise ValueError(
            "cuboids require centers [C,3], rotations [C,3,3], half_extents [C,3]"
        )
    if bool((half < 0).any().item()):
        raise ValueError("cuboid half extents must be nonnegative")
    if cuboids:
        identity = torch.eye(3, dtype=values.dtype, device=values.device)
        tolerance = 1e-5 if values.dtype == torch.float32 else 1e-12
        if not bool(
            torch.allclose(
                rotations @ rotations.transpose(1, 2),
                identity.expand(cuboids, 3, 3),
                rtol=0,
                atol=tolerance,
            )
        ) or not bool(
            torch.allclose(
                torch.linalg.det(rotations),
                torch.ones(cuboids, dtype=values.dtype, device=values.device),
                rtol=0,
                atol=tolerance,
            )
        ):
            raise ValueError("cuboid rotations must be proper orthonormal matrices")
    pad = _padding(padding)
    sphere_enabled = _mask(
        sphere_active, values.shape[1], "sphere_active", values
    )
    cuboid_enabled = _mask(cuboid_active, cuboids, "cuboid_active", values)

    offset = values[:, :, None, :3] - centers[None, None, :, :]
    local = torch.einsum("cij,bscj->bsci", rotations.transpose(1, 2), offset)
    expanded_half = half[None, None, :, :].expand_as(local)
    box_distance = _BoxSDF.apply(local, expanded_half)
    raw = box_distance - values[:, :, None, 3] - pad
    enabled = sphere_enabled[None, :, None] & cuboid_enabled[None, None, :]
    distances = torch.where(enabled, raw, torch.full_like(raw, torch.inf))

    q = local.abs() - half[None, None, :, :]
    outside = q.clamp_min(0)
    outside_length = torch.linalg.vector_norm(outside, dim=-1)
    sign = torch.where(local < 0, -torch.ones_like(local), torch.ones_like(local))
    safe_length = torch.where(
        outside_length > 0, outside_length, torch.ones_like(outside_length)
    )
    outside_gradient = outside / safe_length[..., None] * sign
    axis = q.argmax(dim=-1)
    inside_gradient = torch.zeros_like(local).scatter_(
        -1, axis[..., None], sign.gather(-1, axis[..., None])
    )
    local_gradient = torch.where(
        (outside_length > 0)[..., None], outside_gradient, inside_gradient
    )
    world_gradient = torch.einsum("cij,bscj->bsci", rotations, local_gradient)
    gradients = torch.cat(
        (world_gradient, -torch.ones_like(world_gradient[..., :1])), dim=-1
    )
    gradients = torch.where(enabled[..., None], gradients, torch.zeros_like(gradients))

    if cuboids and bool(cuboid_enabled.any().item()):
        reduced, winners = torch.min(distances, dim=2)
        gather = winners[..., None, None].expand(-1, -1, 1, 4)
        reduced_gradient = gradients.gather(2, gather).squeeze(2)
        winners = torch.where(
            sphere_enabled[None, :],
            winners,
            torch.full_like(winners, -1),
        )
    else:
        reduced = values.sum(dim=-1) * 0 + torch.inf
        winners = torch.full(
            values.shape[:2], -1, dtype=torch.int64, device=values.device
        )
        reduced_gradient = torch.zeros_like(values)
    return CuboidDistanceResult(
        distances, gradients, reduced, reduced_gradient, winners, unbatched
    )
