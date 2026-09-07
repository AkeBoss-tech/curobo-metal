"""Differentiable mesh and voxel world-collision queries for CPU and Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from curobo_metal.backend import validate_tensor_device


def _floating(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(value)
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if value.device.type == "mps" and value.dtype != torch.float32:
        raise TypeError("MPS world-collision operations support only float32")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _same(value: torch.Tensor, name: str, reference: torch.Tensor) -> torch.Tensor:
    value = _floating(value, name)
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")
    if value.dtype != reference.dtype:
        raise TypeError(f"{name} must have dtype {reference.dtype}")
    return value


def _points(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    points = _floating(points, "points")
    unbatched = points.ndim == 2
    if unbatched:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [Q,3] or [B,Q,3]")
    return points, unbatched


def _env_indices(
    value: torch.Tensor | None, batch: int, environments: int, reference: torch.Tensor
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch, dtype=torch.int64, device=reference.device)
    if not isinstance(value, torch.Tensor):
        raise TypeError("env_indices must be a torch.Tensor")
    validate_tensor_device(value, expected=reference.device)
    if value.dtype != torch.int64 or value.shape != (batch,):
        raise ValueError(f"env_indices must be int64 with shape [{batch}]")
    if bool(((value < 0) | (value >= environments)).any().item()):
        raise ValueError("env_indices contains an out-of-range environment")
    return value


@dataclass(frozen=True)
class Mesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    watertight: bool = False

    def __post_init__(self) -> None:
        vertices = _floating(self.vertices, "vertices")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("vertices must have shape [V,3], V > 0")
        if not isinstance(self.faces, torch.Tensor):
            raise TypeError("faces must be a torch.Tensor")
        validate_tensor_device(self.faces, expected=vertices.device)
        if self.faces.dtype != torch.int64 or self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces):
            raise ValueError("faces must be int64 with shape [F,3], F > 0")
        if bool(((self.faces < 0) | (self.faces >= len(vertices))).any().item()):
            raise ValueError("faces contains an out-of-range vertex")
        triangles = vertices[self.faces]
        area = torch.linalg.vector_norm(
            torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1),
            dim=-1,
        )
        if bool((area <= 1e-15).any().item()):
            raise ValueError("mesh faces must be nondegenerate triangles")
        if not isinstance(self.watertight, bool):
            raise TypeError("watertight must be boolean")


@dataclass(frozen=True)
class MeshDistanceResult:
    distances: torch.Tensor
    gradients: torch.Tensor
    closest_points: torch.Tensor
    winning_face: torch.Tensor
    reduced_distance: torch.Tensor
    reduced_gradient: torch.Tensor
    winning_mesh: torch.Tensor
    input_was_unbatched: bool


def _closest_triangles(p: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Ericson region tests for p[N,3] against triangles[F,3,3]."""
    p = p[:, None, :]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cpv = p - c
    d5, d6 = (ab * cpv).sum(-1), (ac * cpv).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    denom_ab = torch.where(d1 - d3 != 0, d1 - d3, torch.ones_like(d1))
    denom_ac = torch.where(d2 - d6 != 0, d2 - d6, torch.ones_like(d2))
    denom_bc = torch.where((d4 - d3) + (d5 - d6) != 0, (d4 - d3) + (d5 - d6), torch.ones_like(d1))
    denom_face = torch.where(va + vb + vc != 0, va + vb + vc, torch.ones_like(va))
    result = a + (vb / denom_face)[..., None] * ab + (vc / denom_face)[..., None] * ac
    result = torch.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None],
                         b + ((d4 - d3) / denom_bc)[..., None] * (c - b), result)
    result = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
                         a + (d2 / denom_ac)[..., None] * ac, result)
    result = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
                         a + (d1 / denom_ab)[..., None] * ab, result)
    result = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c, result)
    result = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b, result)
    result = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a, result)
    return result


def _inside_mesh(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    direction = points.new_tensor([1.0, 0.3713906763541037, 0.6947465906068658])
    direction = direction / torch.linalg.vector_norm(direction)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    h = torch.cross(direction.expand_as(edge2), edge2, dim=-1)
    det = (edge1 * h).sum(-1)
    safe = torch.where(det.abs() > 1e-12, det, torch.ones_like(det))
    inv = safe.reciprocal()
    s = points[:, None, :] - a
    u = inv * (s * h).sum(-1)
    q = torch.cross(s, edge1.unsqueeze(0).expand_as(s), dim=-1)
    v = inv * (direction * q).sum(-1)
    t = inv * (edge2 * q).sum(-1)
    hits = (det.abs() > 1e-12) & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 1e-12)
    # Fixture-quality watertight meshes serialize shared-edge intersections
    # consistently. Unique-hit de-duplication is only relevant on ray degeneracy.
    return hits.sum(-1).remainder(2).bool()


def mesh_distance(
    points: torch.Tensor,
    meshes: Sequence[Mesh],
    translations: torch.Tensor,
    rotations: torch.Tensor,
    *,
    env_mesh_active: torch.Tensor | None = None,
    env_indices: torch.Tensor | None = None,
    signed: bool = True,
) -> MeshDistanceResult:
    query, unbatched = _points(points)
    if not meshes:
        raise ValueError("at least one mesh is required")
    m = len(meshes)
    translations = _same(translations, "translations", query)
    rotations = _same(rotations, "rotations", query)
    if translations.ndim != 3 or translations.shape[1:] != (m, 3):
        raise ValueError("translations must have shape [E,M,3]")
    e = translations.shape[0]
    if rotations.shape != (e, m, 3, 3):
        raise ValueError("rotations must have shape [E,M,3,3]")
    if signed and any(not mesh.watertight for mesh in meshes):
        raise ValueError("signed mesh distance requires every mesh to be declared watertight")
    for mesh in meshes:
        _same(mesh.vertices, "mesh vertices", query)
    env = _env_indices(env_indices, len(query), e, query)
    if env_mesh_active is None:
        active = torch.ones((e, m), dtype=torch.bool, device=query.device)
    else:
        if not isinstance(env_mesh_active, torch.Tensor):
            raise TypeError("env_mesh_active must be a torch.Tensor")
        validate_tensor_device(env_mesh_active, expected=query.device)
        if env_mesh_active.dtype != torch.bool or env_mesh_active.shape != (e, m):
            raise ValueError("env_mesh_active must be boolean with shape [E,M]")
        active = env_mesh_active

    all_d, all_g, all_cp, all_f = [], [], [], []
    for j, mesh in enumerate(meshes):
        r, t = rotations[env, j], translations[env, j]
        local = torch.einsum("bij,bqj->bqi", r.transpose(-1, -2), query - t[:, None])
        flat = local.reshape(-1, 3)
        triangles = mesh.vertices[mesh.faces]
        cps = _closest_triangles(flat, triangles)
        delta = flat[:, None] - cps
        unsigned = torch.linalg.vector_norm(delta, dim=-1)
        face = unsigned.argmin(-1)
        row = torch.arange(len(flat), device=query.device)
        best_cp = cps[row, face]
        best = unsigned[row, face]
        inside = _inside_mesh(flat.detach(), triangles.detach()) if signed else torch.zeros_like(best, dtype=torch.bool)
        sign = torch.where(inside, -torch.ones_like(best), torch.ones_like(best))
        normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
        normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        normal = normals[face]
        # At the surface choose the selected face normal as the autograd subgradient.
        surrogate = ((flat - best_cp) * normal).sum(-1)
        distance = sign * torch.where(best > 0, best, surrogate)
        local_grad = torch.where(
            (best > 0)[:, None], sign[:, None] * (flat - best_cp) / best.clamp_min(torch.finfo(best.dtype).tiny)[:, None], normal
        )
        # Closest-point branch derivatives are undefined on region boundaries.
        # Install the contract's deterministic selected subgradient without
        # changing the forward value.
        distance = distance.detach() + ((flat - flat.detach()) * local_grad.detach()).sum(-1)
        world_grad = torch.einsum("bij,bqj->bqi", r, local_grad.reshape(local.shape))
        world_cp = torch.einsum("bij,bqj->bqi", r, best_cp.reshape(local.shape)) + t[:, None]
        enabled = active[env, j][:, None]
        all_d.append(torch.where(enabled, distance.reshape(local.shape[:-1]), torch.full_like(distance.reshape(local.shape[:-1]), torch.inf)))
        all_g.append(torch.where(enabled[..., None], world_grad, torch.zeros_like(world_grad)))
        all_cp.append(torch.where(enabled[..., None], world_cp, torch.zeros_like(world_cp)))
        all_f.append(torch.where(enabled, face.reshape(local.shape[:-1]), torch.full_like(face.reshape(local.shape[:-1]), -1)))
    distances = torch.stack(all_d, -1)
    gradients = torch.stack(all_g, -2)
    closest = torch.stack(all_cp, -2)
    faces = torch.stack(all_f, -1)
    reduced, winner = distances.min(-1)
    valid = torch.isfinite(reduced)
    gather = winner[..., None, None].expand(*winner.shape, 1, 3)
    reduced_gradient = gradients.gather(-2, gather).squeeze(-2)
    reduced_gradient = torch.where(valid[..., None], reduced_gradient, torch.zeros_like(reduced_gradient))
    winner = torch.where(valid, winner, torch.full_like(winner, -1))
    return MeshDistanceResult(distances, gradients, closest, faces, reduced, reduced_gradient, winner, unbatched)


@dataclass(frozen=True)
class VoxelGrid:
    values: torch.Tensor
    voxel_size: float
    translation: torch.Tensor
    rotation: torch.Tensor
    out_of_bounds: float

    def __post_init__(self) -> None:
        values = _floating(self.values, "values")
        if values.ndim != 3 or min(values.shape) < 2:
            raise ValueError("values must have shape [nx,ny,nz], dimensions >= 2")
        _same(self.translation, "translation", values)
        _same(self.rotation, "rotation", values)
        if self.translation.shape != (3,) or self.rotation.shape != (3, 3):
            raise ValueError("translation/rotation must have shapes [3] and [3,3]")
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        if not math.isfinite(self.out_of_bounds):
            raise ValueError("out_of_bounds must be finite")


@dataclass(frozen=True)
class VoxelSampleResult:
    values: torch.Tensor
    gradients: torch.Tensor
    valid: torch.Tensor
    input_was_unbatched: bool


def sample_voxel_sdf(
    points: torch.Tensor,
    environments: Sequence[Sequence[VoxelGrid]],
    *,
    env_indices: torch.Tensor | None = None,
) -> VoxelSampleResult:
    query, unbatched = _points(points)
    if not environments:
        raise ValueError("at least one environment is required")
    widths = {len(x) for x in environments}
    if len(widths) != 1 or not next(iter(widths)):
        raise ValueError("environments require the same positive grid slot count")
    g = next(iter(widths))
    env = _env_indices(env_indices, len(query), len(environments), query)
    output, gradients, valids = [], [], []
    for j in range(g):
        slot_values, slot_gradients, slot_valids = [], [], []
        for b in range(len(query)):
            grid = environments[int(env[b].item())][j]
            _same(grid.values, "grid values", query)
            local = (query[b] - grid.translation) @ grid.rotation
            shape = query.new_tensor(grid.values.shape)
            coord = local / grid.voxel_size + shape / 2 - 0.5
            base = torch.floor(coord).to(torch.int64)
            frac = coord - base
            base_columns, frac_columns = [], []
            for axis, size in enumerate(grid.values.shape):
                last = coord[:, axis] == size - 1
                base_columns.append(torch.where(last, base.new_full((), size - 2), base[:, axis]))
                # Preserve the positive-cell one-sided derivative at the last center.
                endpoint = frac.new_ones(()) + coord[:, axis] - coord[:, axis].detach()
                frac_columns.append(torch.where(last, endpoint, frac[:, axis]))
            base = torch.stack(base_columns, -1)
            frac = torch.stack(frac_columns, -1)
            valid = ((base >= 0) & (base + 1 < shape.to(torch.int64))).all(-1)
            safe = torch.maximum(torch.minimum(base, shape.to(torch.int64) - 2), torch.zeros_like(base))
            value = query.new_zeros(len(query[b]))
            grad = query.new_zeros((len(query[b]), 3))
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        sample = grid.values[safe[:, 0] + x, safe[:, 1] + y, safe[:, 2] + z]
                        wx = frac[:, 0] if x else 1 - frac[:, 0]
                        wy = frac[:, 1] if y else 1 - frac[:, 1]
                        wz = frac[:, 2] if z else 1 - frac[:, 2]
                        value = value + sample * wx * wy * wz
                        grad[:, 0] += sample * (1 if x else -1) * wy * wz
                        grad[:, 1] += sample * wx * (1 if y else -1) * wz
                        grad[:, 2] += sample * wx * wy * (1 if z else -1)
            world_grad = (grad / grid.voxel_size) @ grid.rotation.transpose(-1, -2)
            slot_values.append(torch.where(valid, value, value.new_full((), grid.out_of_bounds)))
            slot_gradients.append(torch.where(valid[:, None], world_grad, torch.zeros_like(world_grad)))
            slot_valids.append(valid)
        output.append(torch.cat(slot_values))
        gradients.append(torch.cat(slot_gradients))
        valids.append(torch.cat(slot_valids))
    shape = (len(query), query.shape[1], g)
    return VoxelSampleResult(
        torch.stack(output, -1).reshape(shape),
        torch.stack(gradients, -2).reshape(shape + (3,)),
        torch.stack(valids, -1).reshape(shape),
        unbatched,
    )


@dataclass(frozen=True)
class ESDFQueryResult:
    distance: torch.Tensor
    gradient: torch.Tensor
    winning_grid: torch.Tensor
    valid: torch.Tensor


def query_esdf(
    points: torch.Tensor,
    environments: Sequence[Sequence[VoxelGrid]],
    *,
    env_indices: torch.Tensor | None = None,
    grid_active: torch.Tensor | None = None,
    padding: float = 0.0,
) -> ESDFQueryResult:
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    sampled = sample_voxel_sdf(points, environments, env_indices=env_indices)
    batch, _, g = sampled.values.shape
    env = _env_indices(env_indices, batch, len(environments), sampled.values)
    if grid_active is None:
        active = torch.ones((len(environments), g), dtype=torch.bool, device=sampled.values.device)
    else:
        if grid_active.dtype != torch.bool or grid_active.shape != (len(environments), g):
            raise ValueError("grid_active must be boolean with shape [E,G]")
        validate_tensor_device(grid_active, expected=sampled.values.device)
        active = grid_active
    candidates = sampled.valid & active[env, None, :]
    # Reducing an all-``inf`` row with ``min`` on MPS can produce a ``-1``
    # winner, and autograd then uses that invalid index in the reduction's
    # scatter backward.  Use the largest finite value for the internal
    # reduction and restore the public infinity sentinel after selection.
    finite_sentinel = torch.finfo(sampled.values.dtype).max
    masked = torch.where(
        candidates,
        sampled.values,
        torch.full_like(sampled.values, finite_sentinel),
    )
    distance, winner = masked.min(-1)
    valid = candidates.any(-1)
    # Metal may use ``-1`` as the reduction index when every candidate is
    # infinite.  Preserve that sentinel in the public result, but gather from
    # a safe row before masking the invalid gradient to zero.
    safe_winner = winner.clamp_min(0)
    # Avoid differentiating through ``gather`` here.  MPS may retain the
    # reduction's ``-1`` sentinel in gather's backward metadata even though
    # the forward index was clamped, which later raises an asynchronous
    # out-of-bounds scatter.  A one-hot weighted selection has the identical
    # value and gradient while keeping the invalid case maskable below.
    selector = (
        torch.arange(g, device=sampled.values.device)
        .view(*(1 for _ in safe_winner.shape), g)
        .eq(safe_winner[..., None])
        .to(sampled.gradients.dtype)
    )
    gradient = (sampled.gradients * selector[..., None]).sum(dim=-2)
    return ESDFQueryResult(
        torch.where(valid, distance - padding, torch.full_like(distance, torch.inf)),
        torch.where(valid[..., None], gradient, torch.zeros_like(gradient)),
        torch.where(valid, winner, torch.full_like(winner, -1)),
        valid,
    )


def sphere_world_collision(
    spheres: torch.Tensor,
    sdf_distance: torch.Tensor,
    sdf_gradient: torch.Tensor,
    *,
    activation_distance: float = 0.0,
    padding: float = 0.0,
    weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    spheres = _floating(spheres, "spheres")
    distance = _same(sdf_distance, "sdf_distance", spheres)
    gradient = _same(sdf_gradient, "sdf_gradient", spheres)
    if spheres.shape[-1] != 4 or distance.shape != spheres.shape[:-1] or gradient.shape != distance.shape + (3,):
        raise ValueError("sphere/SDF shapes are incompatible")
    if bool((spheres[..., 3] < 0).any().item()):
        raise ValueError("sphere radii must be nonnegative")
    if any(not math.isfinite(x) or x < 0 for x in (activation_distance, padding, weight)):
        raise ValueError("activation_distance, padding, and weight must be finite and nonnegative")
    x = spheres[..., 3] + padding + activation_distance - distance
    positive = x > 0
    if activation_distance == 0:
        cost = torch.where(positive, x, torch.zeros_like(x))
        scale = positive.to(x.dtype)
    else:
        linear = x > activation_distance
        quadratic = positive & ~linear
        cost = torch.where(linear, x - 0.5 * activation_distance,
                           torch.where(quadratic, 0.5 * x.square() / activation_distance, torch.zeros_like(x)))
        scale = torch.where(linear, torch.ones_like(x),
                            torch.where(quadratic, x / activation_distance, torch.zeros_like(x)))
    return weight * cost, -weight * scale[..., None] * gradient
