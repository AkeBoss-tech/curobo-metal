"""Deterministic NumPy references for the first sphere-collision primitives.

These routines are executable specifications, not performance implementations.
They use float64 internally and return explicit gradients/subgradients so future
device implementations can be tested without cuRobo, Warp, CUDA, or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

COLLISION_FORMAT = "curobo-metal-collision-case"
COLLISION_VERSION = 1


def _floating(value: Any, name: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating-point dtype")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _batched_spheres(value: Any) -> tuple[FloatArray, bool]:
    spheres = _floating(value, "spheres")
    unbatched = spheres.ndim == 2
    if unbatched:
        spheres = spheres[None, ...]
    if spheres.ndim != 3 or spheres.shape[-1] != 4:
        raise ValueError("spheres must have shape [S, 4] or [B, S, 4]")
    if np.any(spheres[..., 3] < 0.0):
        raise ValueError("sphere radii must be nonnegative")
    return spheres, unbatched


def _mask(value: Any | None, length: int, name: str) -> NDArray[np.bool_]:
    if value is None:
        return np.ones(length, dtype=np.bool_)
    result = np.asarray(value)
    if result.dtype.kind != "b" or result.shape != (length,):
        raise ValueError(f"{name} must be boolean with shape [{length}]")
    return np.asarray(result, dtype=np.bool_)


@dataclass(frozen=True)
class SphereTransformResult:
    spheres: FloatArray  # [B, S, 4]
    center_transform_jacobian: FloatArray  # [B, S, 3, 4, 4]
    input_was_batched: bool


def transform_spheres(
    transforms: Any,
    local_spheres: Any,
    link_indices: Sequence[int],
    active: Any | None = None,
) -> SphereTransformResult:
    """Transform link-local ``(x, y, z, radius)`` spheres into world space."""
    matrices = _floating(transforms, "transforms")
    input_was_batched = matrices.ndim == 4
    if matrices.ndim == 3:
        matrices = matrices[None, ...]
    if matrices.ndim != 4 or matrices.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape [L, 4, 4] or [B, L, 4, 4]")
    local = _floating(local_spheres, "local_spheres")
    if local.ndim != 2 or local.shape[1] != 4:
        raise ValueError("local_spheres must have shape [S, 4]")
    if np.any(local[:, 3] < 0.0):
        raise ValueError("sphere radii must be nonnegative")
    links = np.asarray(link_indices)
    if links.dtype.kind not in "iu" or links.shape != (local.shape[0],):
        raise ValueError("link_indices must be integers with shape [S]")
    links = np.asarray(links, dtype=np.int64)
    if np.any(links < 0) or np.any(links >= matrices.shape[1]):
        raise ValueError("link_indices contains an out-of-range link")
    enabled = _mask(active, local.shape[0], "active")

    batch, count = matrices.shape[0], local.shape[0]
    output = np.zeros((batch, count, 4), dtype=np.float64)
    jacobian = np.zeros((batch, count, 3, 4, 4), dtype=np.float64)
    homogeneous = np.concatenate((local[:, :3], np.ones((count, 1))), axis=1)
    for b in range(batch):
        for sphere_index, link_index in enumerate(links):
            if not enabled[sphere_index]:
                continue
            output[b, sphere_index, :3] = (
                matrices[b, link_index] @ homogeneous[sphere_index]
            )[:3]
            output[b, sphere_index, 3] = local[sphere_index, 3]
            for coordinate in range(3):
                jacobian[b, sphere_index, coordinate, coordinate, :] = homogeneous[sphere_index]
    return SphereTransformResult(output, jacobian, input_was_batched)


@dataclass(frozen=True)
class PairDistanceResult:
    distances: FloatArray  # [B, P]
    gradients: FloatArray  # [B, P, S, 4]
    reduced_distance: FloatArray  # [B]
    reduced_gradient: FloatArray  # [B, S, 4]
    winning_pair: IntArray  # [B], -1 if no active pair
    input_was_unbatched: bool


def sphere_sphere_signed_distance(
    spheres: Any,
    pairs: Any,
    *,
    sphere_active: Any | None = None,
    pair_active: Any | None = None,
    padding: float = 0.0,
) -> PairDistanceResult:
    """Compute configured sphere-pair clearances and their minimum."""
    values, unbatched = _batched_spheres(spheres)
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("padding must be a finite nonnegative scalar")
    pair_table = np.asarray(pairs)
    if pair_table.dtype.kind not in "iu" or pair_table.ndim != 2 or pair_table.shape[1] != 2:
        raise ValueError("pairs must be integers with shape [P, 2]")
    pair_table = np.asarray(pair_table, dtype=np.int64)
    if np.any(pair_table < 0) or np.any(pair_table >= values.shape[1]):
        raise ValueError("pairs contains an out-of-range sphere")
    if np.any(pair_table[:, 0] == pair_table[:, 1]):
        raise ValueError("a sphere cannot be paired with itself")
    sphere_enabled = _mask(sphere_active, values.shape[1], "sphere_active")
    pair_enabled = _mask(pair_active, pair_table.shape[0], "pair_active")
    pair_enabled &= sphere_enabled[pair_table[:, 0]] & sphere_enabled[pair_table[:, 1]]

    batch, sphere_count, pair_count = values.shape[0], values.shape[1], pair_table.shape[0]
    distances = np.full((batch, pair_count), np.inf, dtype=np.float64)
    gradients = np.zeros((batch, pair_count, sphere_count, 4), dtype=np.float64)
    for b in range(batch):
        for p, (first, second) in enumerate(pair_table):
            if not pair_enabled[p]:
                continue
            delta = values[b, first, :3] - values[b, second, :3]
            length = float(np.linalg.norm(delta))
            direction = delta / length if length > 0.0 else np.array([1.0, 0.0, 0.0])
            distances[b, p] = length - values[b, first, 3] - values[b, second, 3] - padding
            gradients[b, p, first, :3] = direction
            gradients[b, p, second, :3] = -direction
            gradients[b, p, first, 3] = -1.0
            gradients[b, p, second, 3] = -1.0

    winners = np.full(batch, -1, dtype=np.int64)
    reduced = np.full(batch, np.inf, dtype=np.float64)
    reduced_gradient = np.zeros((batch, sphere_count, 4), dtype=np.float64)
    for b in range(batch):
        if np.any(pair_enabled):
            winners[b] = int(np.argmin(distances[b]))
            reduced[b] = distances[b, winners[b]]
            reduced_gradient[b] = gradients[b, winners[b]]
    return PairDistanceResult(
        distances, gradients, reduced, reduced_gradient, winners, unbatched
    )


@dataclass(frozen=True)
class CuboidDistanceResult:
    distances: FloatArray  # [B, S, C]
    sphere_gradients: FloatArray  # [B, S, C, 4]
    reduced_distance: FloatArray  # [B, S]
    reduced_sphere_gradient: FloatArray  # [B, S, 4]
    winning_cuboid: IntArray  # [B, S], -1 if inactive/no cuboid
    input_was_unbatched: bool


def _box_sdf_gradient(point: FloatArray, half: FloatArray) -> tuple[float, FloatArray]:
    q = np.abs(point) - half
    outside = np.maximum(q, 0.0)
    outside_length = float(np.linalg.norm(outside))
    if outside_length > 0.0:
        local_gradient = outside / outside_length * np.where(point < 0.0, -1.0, 1.0)
        return outside_length, local_gradient
    # Interior/boundary: max(q), with first-axis tie and sign(0) := +1.
    axis = int(np.argmax(q))
    local_gradient = np.zeros(3, dtype=np.float64)
    local_gradient[axis] = -1.0 if point[axis] < 0.0 else 1.0
    return float(q[axis]), local_gradient


def sphere_cuboid_signed_distance(
    spheres: Any,
    cuboid_centers: Any,
    cuboid_rotations: Any,
    cuboid_half_extents: Any,
    *,
    sphere_active: Any | None = None,
    cuboid_active: Any | None = None,
    padding: float = 0.0,
) -> CuboidDistanceResult:
    """Sphere-to-oriented-cuboid clearances and gradients w.r.t. spheres."""
    values, unbatched = _batched_spheres(spheres)
    centers = _floating(cuboid_centers, "cuboid_centers")
    rotations = _floating(cuboid_rotations, "cuboid_rotations")
    half = _floating(cuboid_half_extents, "cuboid_half_extents")
    cuboids = centers.shape[0] if centers.ndim == 2 else -1
    if centers.shape != (cuboids, 3) or rotations.shape != (cuboids, 3, 3) or half.shape != (cuboids, 3):
        raise ValueError("cuboids require centers [C,3], rotations [C,3,3], half_extents [C,3]")
    if np.any(half < 0.0):
        raise ValueError("cuboid half extents must be nonnegative")
    identities = rotations @ rotations.transpose(0, 2, 1)
    if not np.allclose(identities, np.eye(3), rtol=0.0, atol=1e-12) or not np.allclose(
        np.linalg.det(rotations), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("cuboid rotations must be proper orthonormal matrices")
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("padding must be a finite nonnegative scalar")
    sphere_enabled = _mask(sphere_active, values.shape[1], "sphere_active")
    cuboid_enabled = _mask(cuboid_active, cuboids, "cuboid_active")

    batch, sphere_count = values.shape[:2]
    distances = np.full((batch, sphere_count, cuboids), np.inf, dtype=np.float64)
    gradients = np.zeros((batch, sphere_count, cuboids, 4), dtype=np.float64)
    for b in range(batch):
        for s in range(sphere_count):
            if not sphere_enabled[s]:
                continue
            for c in range(cuboids):
                if not cuboid_enabled[c]:
                    continue
                local = rotations[c].T @ (values[b, s, :3] - centers[c])
                box_distance, local_gradient = _box_sdf_gradient(local, half[c])
                distances[b, s, c] = box_distance - values[b, s, 3] - padding
                gradients[b, s, c, :3] = rotations[c] @ local_gradient
                gradients[b, s, c, 3] = -1.0

    winners = np.full((batch, sphere_count), -1, dtype=np.int64)
    reduced = np.full((batch, sphere_count), np.inf, dtype=np.float64)
    reduced_gradient = np.zeros((batch, sphere_count, 4), dtype=np.float64)
    for b in range(batch):
        for s in range(sphere_count):
            if sphere_enabled[s] and np.any(cuboid_enabled):
                winners[b, s] = int(np.argmin(distances[b, s]))
                reduced[b, s] = distances[b, s, winners[b, s]]
                reduced_gradient[b, s] = gradients[b, s, winners[b, s]]
    return CuboidDistanceResult(
        distances, gradients, reduced, reduced_gradient, winners, unbatched
    )


def load_collision_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        case = json.load(stream)
    if case.get("format") != COLLISION_FORMAT or case.get("version") != COLLISION_VERSION:
        raise ValueError("unsupported collision replay format")
    if not isinstance(case.get("cases"), list):
        raise ValueError("collision replay cases must be a list")
    return case


def save_collision_case(path: str | Path, case: Mapping[str, Any]) -> None:
    destination = Path(path)
    text = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    destination.write_text(text, encoding="utf-8")
    load_collision_case(destination)
