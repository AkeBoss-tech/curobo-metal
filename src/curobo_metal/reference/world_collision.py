"""Independent NumPy oracles for mesh and voxel/ESDF world collision."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

WORLD_COLLISION_FORMAT = "curobo-metal-world-collision-case"
WORLD_COLLISION_VERSION = 1


def _float(value: Any, name: str) -> FloatArray:
    a = np.asarray(value)
    if a.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating-point dtype")
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a


def _rotation(value: Any, count: int, name: str) -> FloatArray:
    a = _float(value, name)
    if a.shape != (count, 3, 3):
        raise ValueError(f"{name} must have shape [{count},3,3]")
    if not np.allclose(a @ a.transpose(0, 2, 1), np.eye(3), atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.allclose(np.linalg.det(a), 1.0, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be proper rotations")
    return a


def _mask(value: Any | None, count: int, name: str) -> NDArray[np.bool_]:
    if value is None:
        return np.ones(count, dtype=bool)
    a = np.asarray(value)
    if a.dtype.kind != "b" or a.shape != (count,):
        raise ValueError(f"{name} must be boolean with shape [{count}]")
    return np.asarray(a, dtype=bool)


def _points(value: Any) -> tuple[FloatArray, bool]:
    a = _float(value, "points")
    unbatched = a.ndim == 2
    if unbatched:
        a = a[None]
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("points must have shape [Q,3] or [B,Q,3]")
    return a, unbatched


def _env_indices(value: Sequence[int] | None, batch: int, environments: int) -> IntArray:
    if value is None:
        result = np.zeros(batch, dtype=np.int64)
    else:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iu" or raw.shape != (batch,):
            raise ValueError(f"env_indices must be integer with shape [{batch}]")
        result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= environments):
        raise ValueError("env_indices contains an out-of-range environment")
    return result


@dataclass(frozen=True)
class Mesh:
    vertices: FloatArray
    faces: IntArray
    watertight: bool = False

    def __post_init__(self) -> None:
        vertices = _float(self.vertices, "vertices")
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("vertices must have shape [V,3], V > 0")
        if faces.dtype.kind not in "iu" or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("faces must be integer with shape [F,3], F > 0")
        faces = np.asarray(faces, dtype=np.int64)
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces contains an out-of-range vertex")
        if not isinstance(self.watertight, (bool, np.bool_)):
            raise TypeError("watertight must be boolean")
        for face in faces:
            if len(set(face.tolist())) != 3 or np.linalg.norm(
                np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            ) <= 1e-15:
                raise ValueError("mesh faces must be nondegenerate triangles")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True)
class MeshDistanceResult:
    distances: FloatArray  # [B,Q,M]
    gradients: FloatArray  # d distance / d world point, [B,Q,M,3]
    closest_points: FloatArray  # [B,Q,M,3]
    winning_face: IntArray  # [B,Q,M]
    reduced_distance: FloatArray  # [B,Q]
    reduced_gradient: FloatArray  # [B,Q,3]
    winning_mesh: IntArray  # [B,Q]
    input_was_unbatched: bool


def _closest_triangle(p: FloatArray, a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    # Ericson's region tests; serialized face order resolves equal-distance ties.
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + vb * denom * ab + vc * denom * ac


def _inside_mesh(p: FloatArray, vertices: FloatArray, faces: IntArray) -> bool:
    # Odd/even ray test in a deliberately non-axis-aligned direction.
    direction = np.array([1.0, 0.3713906763541037, 0.6947465906068658])
    direction /= np.linalg.norm(direction)
    hits: list[float] = []
    eps = 1e-12
    for face in faces:
        a, b, c = vertices[face]
        edge1, edge2 = b - a, c - a
        h = np.cross(direction, edge2)
        det = float(edge1 @ h)
        if abs(det) <= eps:
            continue
        inv = 1.0 / det
        s = p - a
        u = inv * float(s @ h)
        q = np.cross(s, edge1)
        v = inv * float(direction @ q)
        t = inv * float(edge2 @ q)
        if u >= -eps and v >= -eps and u + v <= 1 + eps and t > eps:
            if not any(abs(t - old) <= eps for old in hits):
                hits.append(t)
    return len(hits) % 2 == 1


def mesh_distance(
    points: Any,
    meshes: Sequence[Mesh],
    translations: Any,
    rotations: Any,
    *,
    env_mesh_active: Any | None = None,
    env_indices: Sequence[int] | None = None,
    signed: bool = True,
) -> MeshDistanceResult:
    """Query brute-force point-to-triangle distance in transformed environments."""
    query, unbatched = _points(points)
    m = len(meshes)
    translations = _float(translations, "translations")
    if translations.ndim != 3 or translations.shape[1:] != (m, 3):
        raise ValueError("translations must have shape [E,M,3]")
    e = translations.shape[0]
    rotations = _float(rotations, "rotations")
    if rotations.shape != (e, m, 3, 3):
        raise ValueError("rotations must have shape [E,M,3,3]")
    _rotation(rotations.reshape(e * m, 3, 3), e * m, "rotations")
    active = np.ones((e, m), dtype=bool) if env_mesh_active is None else np.asarray(env_mesh_active)
    if active.dtype.kind != "b" or active.shape != (e, m):
        raise ValueError("env_mesh_active must be boolean with shape [E,M]")
    env = _env_indices(env_indices, len(query), e)
    if signed and any(not mesh.watertight for mesh in meshes):
        raise ValueError("signed mesh distance requires every mesh to be declared watertight")

    shape = (len(query), query.shape[1], m)
    distances = np.full(shape, np.inf)
    gradients = np.zeros(shape + (3,))
    closest = np.zeros(shape + (3,))
    faces = np.full(shape, -1, dtype=np.int64)
    for b in range(len(query)):
        for q, world_p in enumerate(query[b]):
            for j, mesh in enumerate(meshes):
                if not active[env[b], j]:
                    continue
                r, t = rotations[env[b], j], translations[env[b], j]
                local_p = r.T @ (world_p - t)
                best = np.inf
                best_cp = np.zeros(3)
                best_face = -1
                for f, face in enumerate(mesh.faces):
                    cp = _closest_triangle(local_p, *mesh.vertices[face])
                    d = float(np.linalg.norm(local_p - cp))
                    if d < best:  # strict: first serialized face wins exact ties
                        best, best_cp, best_face = d, cp, f
                sign = -1.0 if signed and _inside_mesh(local_p, mesh.vertices, mesh.faces) else 1.0
                delta = local_p - best_cp
                if best > 0:
                    local_grad = sign * delta / best
                else:
                    face = mesh.faces[best_face]
                    normal = np.cross(mesh.vertices[face[1]] - mesh.vertices[face[0]],
                                      mesh.vertices[face[2]] - mesh.vertices[face[0]])
                    local_grad = normal / np.linalg.norm(normal)
                distances[b, q, j] = sign * best
                gradients[b, q, j] = r @ local_grad
                closest[b, q, j] = r @ best_cp + t
                faces[b, q, j] = best_face
    reduced = np.full(shape[:2], np.inf)
    reduced_grad = np.zeros(shape[:2] + (3,))
    winner = np.full(shape[:2], -1, dtype=np.int64)
    for b in range(shape[0]):
        for q in range(shape[1]):
            finite = np.isfinite(distances[b, q])
            if np.any(finite):
                winner[b, q] = int(np.argmin(distances[b, q]))
                reduced[b, q] = distances[b, q, winner[b, q]]
                reduced_grad[b, q] = gradients[b, q, winner[b, q]]
    return MeshDistanceResult(distances, gradients, closest, faces, reduced,
                              reduced_grad, winner, unbatched)


@dataclass(frozen=True)
class VoxelGrid:
    values: FloatArray  # [nx,ny,nz], samples at voxel centers
    voxel_size: float
    translation: FloatArray
    rotation: FloatArray
    out_of_bounds: float

    def __post_init__(self) -> None:
        values = _float(self.values, "values")
        if values.ndim != 3 or np.any(np.asarray(values.shape) < 2):
            raise ValueError("values must have shape [nx,ny,nz] with every dimension >= 2")
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        translation = _float(self.translation, "translation")
        if translation.shape != (3,):
            raise ValueError("translation must have shape [3]")
        rotation = _rotation(np.asarray(self.rotation)[None], 1, "rotation")[0]
        if not np.isfinite(self.out_of_bounds):
            raise ValueError("out_of_bounds must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True)
class VoxelSampleResult:
    values: FloatArray  # [B,Q,G]
    gradients: FloatArray  # d ESDF / d world point, [B,Q,G,3]
    valid: NDArray[np.bool_]  # all eight corners in bounds
    input_was_unbatched: bool


def sample_voxel_sdf(
    points: Any,
    environments: Sequence[Sequence[VoxelGrid]],
    *,
    env_indices: Sequence[int] | None = None,
) -> VoxelSampleResult:
    """Trilinearly sample center-aligned grids; any OOB stencil returns default."""
    query, unbatched = _points(points)
    if len(environments) == 0:
        raise ValueError("at least one environment is required")
    widths = {len(x) for x in environments}
    if len(widths) != 1:
        raise ValueError("all environments must have the same grid slot count")
    g = widths.pop()
    env = _env_indices(env_indices, len(query), len(environments))
    values = np.empty((len(query), query.shape[1], g))
    gradients = np.zeros(values.shape + (3,))
    valid = np.zeros(values.shape, dtype=bool)
    for b in range(len(query)):
        for q, p in enumerate(query[b]):
            for j, grid in enumerate(environments[env[b]]):
                local = grid.rotation.T @ (p - grid.translation)
                coord = local / grid.voxel_size + np.asarray(grid.values.shape) / 2.0 - 0.5
                base = np.floor(coord).astype(np.int64)
                frac = coord - base
                # The last sample center belongs to the final cell at fraction 1.
                for axis, size in enumerate(grid.values.shape):
                    if coord[axis] == size - 1:
                        base[axis], frac[axis] = size - 2, 1.0
                if np.any(base < 0) or np.any(base + 1 >= grid.values.shape):
                    values[b, q, j] = grid.out_of_bounds
                    continue
                corners = grid.values[
                    base[0]:base[0] + 2, base[1]:base[1] + 2, base[2]:base[2] + 2
                ]
                wx, wy, wz = ([1.0 - frac[k], frac[k]] for k in range(3))
                value = 0.0
                grad = np.zeros(3)
                for x in range(2):
                    for y in range(2):
                        for z in range(2):
                            s = corners[x, y, z]
                            value += s * wx[x] * wy[y] * wz[z]
                            grad[0] += s * (-1 if x == 0 else 1) * wy[y] * wz[z]
                            grad[1] += s * wx[x] * (-1 if y == 0 else 1) * wz[z]
                            grad[2] += s * wx[x] * wy[y] * (-1 if z == 0 else 1)
                values[b, q, j] = value
                gradients[b, q, j] = grid.rotation @ (grad / grid.voxel_size)
                valid[b, q, j] = True
    return VoxelSampleResult(values, gradients, valid, unbatched)


@dataclass(frozen=True)
class ESDFQueryResult:
    distance: FloatArray  # [B,Q], minimum point ESDF minus padding
    gradient: FloatArray  # [B,Q,3]
    winning_grid: IntArray  # [B,Q]
    valid: NDArray[np.bool_]


def query_esdf(
    points: Any,
    environments: Sequence[Sequence[VoxelGrid]],
    *,
    env_indices: Sequence[int] | None = None,
    grid_active: Any | None = None,
    padding: float = 0.0,
) -> ESDFQueryResult:
    """Return the minimum valid ESDF across grid slots in each selected environment."""
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    sampled = sample_voxel_sdf(points, environments, env_indices=env_indices)
    e, g = len(environments), sampled.values.shape[2]
    active = np.ones((e, g), dtype=bool) if grid_active is None else np.asarray(grid_active)
    if active.dtype.kind != "b" or active.shape != (e, g):
        raise ValueError("grid_active must be boolean with shape [E,G]")
    batch = sampled.values.shape[0]
    env = _env_indices(env_indices, batch, e)
    distance = np.full(sampled.values.shape[:2], np.inf)
    gradient = np.zeros(sampled.values.shape[:2] + (3,))
    winner = np.full(sampled.values.shape[:2], -1, dtype=np.int64)
    valid = np.zeros(sampled.values.shape[:2], dtype=bool)
    for b in range(batch):
        for q in range(sampled.values.shape[1]):
            candidates = sampled.valid[b, q] & active[env[b]]
            if np.any(candidates):
                masked = np.where(candidates, sampled.values[b, q], np.inf)
                winner[b, q] = int(np.argmin(masked))
                distance[b, q] = masked[winner[b, q]] - padding
                gradient[b, q] = sampled.gradients[b, q, winner[b, q]]
                valid[b, q] = True
    return ESDFQueryResult(distance, gradient, winner, valid)


def sphere_world_collision(
    spheres: Any,
    sdf_distance: Any,
    sdf_gradient: Any,
    *,
    activation_distance: float = 0.0,
    padding: float = 0.0,
    weight: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Apply cuRoboV2-style sphere radius, padding, and C1 activation to an SDF."""
    sphere = _float(spheres, "spheres")
    distance = _float(sdf_distance, "sdf_distance")
    gradient = _float(sdf_gradient, "sdf_gradient")
    if sphere.shape[-1] != 4 or distance.shape != sphere.shape[:-1] or gradient.shape != distance.shape + (3,):
        raise ValueError("sphere/SDF shapes are incompatible")
    if np.any(sphere[..., 3] < 0):
        raise ValueError("sphere radii must be nonnegative")
    if any(not np.isfinite(x) or x < 0 for x in (activation_distance, padding, weight)):
        raise ValueError("activation_distance, padding, and weight must be finite and nonnegative")
    penetration = sphere[..., 3] + padding + activation_distance - distance
    cost = np.zeros(distance.shape)
    scale = np.zeros(distance.shape)
    positive = penetration > 0
    if activation_distance == 0:
        cost[positive] = penetration[positive]
        scale[positive] = 1.0
    else:
        linear = penetration > activation_distance
        quadratic = positive & ~linear
        cost[linear] = penetration[linear] - 0.5 * activation_distance
        scale[linear] = 1.0
        cost[quadratic] = 0.5 * penetration[quadratic] ** 2 / activation_distance
        scale[quadratic] = penetration[quadratic] / activation_distance
    # This is the gradient of cost. Upstream stores the opposite SDF gradient
    # and relies on its custom backward convention; this oracle is calculus-facing.
    return weight * cost, -weight * scale[..., None] * gradient


def load_world_collision_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        case = json.load(stream)
    if case.get("format") != WORLD_COLLISION_FORMAT or case.get("version") != WORLD_COLLISION_VERSION:
        raise ValueError("unsupported world collision replay format")
    return case


def save_world_collision_case(path: str | Path, case: Mapping[str, Any]) -> None:
    text = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    load_world_collision_case(path)
