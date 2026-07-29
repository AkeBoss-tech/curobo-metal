"""Backend-neutral NumPy cost references used by the IK oracle."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
    transform_spheres,
)
from .forward_kinematics import SerialRobot, forward_kinematics

FloatArray = NDArray[np.float64]


def _float(value: Any, name: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind != "f" or np.iscomplexobj(array):
        raise TypeError(f"{name} must be real floating point")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def normalize_quaternion(quaternion: Any) -> FloatArray:
    """Normalize a scalar-first ``[w, x, y, z]`` quaternion."""
    value = _float(quaternion, "quaternion")
    if value.shape != (4,):
        raise ValueError("quaternion must have shape [4]")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("quaternion must be nonzero")
    value = value / norm
    # q and -q are identical. Canonicalization makes serialization deterministic.
    if value[0] < 0.0 or (value[0] == 0.0 and value[np.flatnonzero(value)[0]] < 0.0):
        value = -value
    return value


def quaternion_to_matrix(quaternion: Any) -> FloatArray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation: Any) -> FloatArray:
    matrix = _float(rotation, "rotation")
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("rotation must be proper")
    # Stable largest-component construction, scalar-first.
    candidates = np.array(
        [
            1 + np.trace(matrix),
            1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
            1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
            1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
        ]
    )
    index = int(np.argmax(candidates))
    root = 0.5 * np.sqrt(max(candidates[index], 0.0))
    if root <= 1e-15:
        raise ValueError("rotation could not be converted to a quaternion")
    if index == 0:
        value = [root, (matrix[2, 1] - matrix[1, 2]) / (4 * root),
                 (matrix[0, 2] - matrix[2, 0]) / (4 * root),
                 (matrix[1, 0] - matrix[0, 1]) / (4 * root)]
    elif index == 1:
        value = [(matrix[2, 1] - matrix[1, 2]) / (4 * root), root,
                 (matrix[0, 1] + matrix[1, 0]) / (4 * root),
                 (matrix[0, 2] + matrix[2, 0]) / (4 * root)]
    elif index == 2:
        value = [(matrix[0, 2] - matrix[2, 0]) / (4 * root),
                 (matrix[0, 1] + matrix[1, 0]) / (4 * root), root,
                 (matrix[1, 2] + matrix[2, 1]) / (4 * root)]
    else:
        value = [(matrix[1, 0] - matrix[0, 1]) / (4 * root),
                 (matrix[0, 2] + matrix[2, 0]) / (4 * root),
                 (matrix[1, 2] + matrix[2, 1]) / (4 * root), root]
    return normalize_quaternion(np.asarray(value, dtype=np.float64))


def _rotation_vector(rotation: FloatArray) -> FloatArray:
    quaternion = matrix_to_quaternion(rotation)
    vector = quaternion[1:]
    length = float(np.linalg.norm(vector))
    if length <= 1e-15:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(length, quaternion[0])
    return angle * vector / length


def pose_error(transform: Any, target_position: Any, target_quaternion: Any) -> FloatArray:
    """Return ``[world translation, target-frame rotation-vector]`` error."""
    matrix = _float(transform, "transform")
    position = _float(target_position, "target_position")
    if matrix.shape != (4, 4) or position.shape != (3,):
        raise ValueError("transform and target_position must have shapes [4,4] and [3]")
    target_rotation = quaternion_to_matrix(target_quaternion)
    return np.concatenate(
        (matrix[:3, 3] - position, _rotation_vector(target_rotation.T @ matrix[:3, :3]))
    )


@dataclass(frozen=True)
class ScalarCost:
    value: float
    gradient: FloatArray


def pose_cost(error: Any, weights: Any = np.ones(6)) -> ScalarCost:
    residual = _float(error, "error")
    weight = _float(weights, "weights")
    if residual.shape != (6,) or weight.shape != (6,) or np.any(weight < 0.0):
        raise ValueError("error and nonnegative weights must have shape [6]")
    return ScalarCost(float(0.5 * np.sum(weight * residual**2)), weight * residual)


def joint_limit_cost(q: Any, lower: Any, upper: Any, *, margin: float = 0.0,
                     weight: float = 1.0) -> ScalarCost:
    value, low, high = _float(q, "q"), _float(lower, "lower"), _float(upper, "upper")
    if value.shape != low.shape or value.shape != high.shape or np.any(low > high):
        raise ValueError("q, lower, and upper must have equal shapes and lower <= upper")
    if not np.isfinite(margin) or margin < 0 or weight < 0:
        raise ValueError("margin and weight must be finite and nonnegative")
    below = np.maximum(low + margin - value, 0.0)
    above = np.maximum(value - (high - margin), 0.0)
    return ScalarCost(
        float(0.5 * weight * np.sum(below**2 + above**2)),
        weight * (-below + above),
    )


def smoothness_cost(q: Any, *, velocity_weight: float = 1.0,
                    acceleration_weight: float = 0.0,
                    jerk_weight: float = 0.0, dt: float = 1.0) -> ScalarCost:
    """Integral of squared finite-difference derivatives.

    Derivatives use a uniform knot spacing ``dt`` and each squared sample is
    multiplied by ``dt``.  The historical defaults retain the Wave 3 result.
    """
    trajectory = _float(q, "q")
    if trajectory.ndim != 2:
        raise ValueError("q must have shape [T, J]")
    weights = (velocity_weight, acceleration_weight, jerk_weight)
    if (not np.isfinite(dt) or dt <= 0.0
            or not all(np.isfinite(weight) and weight >= 0.0 for weight in weights)):
        raise ValueError("dt must be positive and smoothness weights nonnegative")
    gradient = np.zeros_like(trajectory)
    value = 0.0
    for order, weight in enumerate(weights, start=1):
        if weight == 0.0 or trajectory.shape[0] <= order:
            continue
        difference = np.diff(trajectory, n=order, axis=0)
        scale = weight * dt ** (1 - 2 * order)
        value += 0.5 * scale * float(np.sum(difference**2))
        # D_n^T D_n with coefficients (-1)^(n-k) C(n,k).
        for offset in range(order + 1):
            coefficient = (-1) ** (order - offset) * comb(order, offset)
            gradient[offset:offset + difference.shape[0]] += (
                scale * coefficient * difference
            )
    return ScalarCost(value, gradient)


def collision_cost(clearances: Any, *, activation_distance: float = 0.0,
                   weight: float = 1.0) -> ScalarCost:
    distance = _float(clearances, "clearances")
    if activation_distance < 0 or weight < 0:
        raise ValueError("activation_distance and weight must be nonnegative")
    penetration = np.maximum(activation_distance - distance, 0.0)
    return ScalarCost(
        float(0.5 * weight * np.sum(penetration**2)),
        -weight * penetration,
    )


@dataclass(frozen=True)
class CollisionModel:
    local_spheres: FloatArray
    link_indices: NDArray[np.int64]
    self_pairs: NDArray[np.int64] | None = None
    cuboid_centers: FloatArray | None = None
    cuboid_rotations: FloatArray | None = None
    cuboid_half_extents: FloatArray | None = None
    padding: float = 0.0
    activation_distance: float = 0.0
    weight: float = 1.0


def robot_collision_cost(robot: SerialRobot, q: Any, model: CollisionModel) -> float:
    """Compose FK, sphere transforms, collision references, and hinge cost."""
    transforms = forward_kinematics(robot, _float(q, "q")).transforms
    spheres = transform_spheres(
        transforms, model.local_spheres, model.link_indices
    ).spheres
    values: list[FloatArray] = []
    if model.self_pairs is not None:
        pair = sphere_sphere_signed_distance(
            spheres, model.self_pairs, padding=model.padding
        )
        values.append(pair.distances[np.isfinite(pair.distances)])
    if model.cuboid_centers is not None:
        world = sphere_cuboid_signed_distance(
            spheres, model.cuboid_centers, model.cuboid_rotations,
            model.cuboid_half_extents, padding=model.padding,
        )
        values.append(world.distances[np.isfinite(world.distances)])
    if not values:
        return 0.0
    return collision_cost(
        np.concatenate(values), activation_distance=model.activation_distance,
        weight=model.weight,
    ).value
