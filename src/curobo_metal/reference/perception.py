"""Small independent NumPy oracle for depth-fused TSDF and dense ESDF maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraObservation:
    """Metric pinhole depth and a camera-to-world transform."""

    depth: FloatArray  # [C,H,W]
    intrinsics: FloatArray  # [C,3,3]
    camera_to_world: FloatArray  # [C,4,4]


@dataclass(frozen=True)
class PerceptionConfig:
    shape: tuple[int, int, int]
    voxel_size: float
    grid_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    truncation_distance: float = 0.04
    depth_min: float = 0.1
    depth_max: float = 10.0
    max_weight: float = 100.0
    occupancy_threshold: float = 0.0
    unobserved_esdf: float = 1.0

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(not isinstance(v, int) or v < 2 for v in self.shape):
            raise ValueError("shape must contain three integers >= 2")
        scalars = (self.voxel_size, self.truncation_distance, self.depth_min,
                   self.depth_max, self.max_weight, self.unobserved_esdf)
        if not all(np.isfinite(v) for v in scalars):
            raise ValueError("configuration scalars must be finite")
        if self.voxel_size <= 0 or self.truncation_distance <= 0 or self.max_weight <= 0:
            raise ValueError("voxel_size, truncation_distance, and max_weight must be positive")
        if self.depth_min < 0 or self.depth_min >= self.depth_max:
            raise ValueError("depth_min must be nonnegative and less than depth_max")
        if self.unobserved_esdf <= 0:
            raise ValueError("unobserved_esdf must be positive")


@dataclass(frozen=True)
class MapState:
    tsdf: FloatArray
    weight: FloatArray
    occupancy: NDArray[np.bool_]
    esdf: FloatArray
    gradient: FloatArray


def voxel_centers(config: PerceptionConfig) -> FloatArray:
    axes = [
        (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * config.voxel_size + c
        for n, c in zip(config.shape, config.grid_center)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def empty_state(config: PerceptionConfig) -> MapState:
    shape = config.shape
    return MapState(
        np.ones(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=bool),
        np.full(shape, config.unobserved_esdf, dtype=np.float64),
        np.zeros(shape + (3,), dtype=np.float64),
    )


def _observation(value: CameraObservation) -> CameraObservation:
    depth = np.asarray(value.depth, dtype=np.float64)
    if depth.ndim == 2:
        depth = depth[None]
    intrinsics = np.asarray(value.intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None]
    poses = np.asarray(value.camera_to_world, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None]
    c = depth.shape[0]
    if depth.ndim != 3 or intrinsics.shape != (c, 3, 3) or poses.shape != (c, 4, 4):
        raise ValueError("depth/intrinsics/camera_to_world must be [C,H,W]/[C,3,3]/[C,4,4]")
    if c == 0 or depth.shape[-2] == 0 or depth.shape[-1] == 0:
        raise ValueError("cameras, height, and width must be positive")
    if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(poses)):
        raise ValueError("intrinsics and camera_to_world must be finite")
    if np.any(intrinsics[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("fx and fy must be positive")
    if not np.allclose(intrinsics[:, 2], (0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("intrinsics bottom row must be [0,0,1]")
    if not np.allclose(poses[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("camera_to_world bottom row must be [0,0,0,1]")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-8):
        raise ValueError("camera_to_world rotations must be orthonormal")
    return CameraObservation(depth, intrinsics, poses)


def integrate(
    config: PerceptionConfig,
    state: MapState,
    observation: CameraObservation,
) -> MapState:
    """Project voxel centers into every camera and fuse unit-weight TSDF samples."""
    obs = _observation(observation)
    centers = voxel_centers(config).reshape(-1, 3)
    old_t = np.asarray(state.tsdf).reshape(-1).copy()
    old_w = np.asarray(state.weight).reshape(-1).copy()
    accum = old_t * old_w
    weight = old_w.copy()
    h, w = obs.depth.shape[-2:]
    for camera in range(len(obs.depth)):
        rotation = obs.camera_to_world[camera, :3, :3]
        translation = obs.camera_to_world[camera, :3, 3]
        local = (centers - translation) @ rotation
        z = local[:, 2]
        u = np.rint(obs.intrinsics[camera, 0, 0] * local[:, 0] / np.where(z != 0, z, 1)
                    + obs.intrinsics[camera, 0, 2]).astype(np.int64)
        v = np.rint(obs.intrinsics[camera, 1, 1] * local[:, 1] / np.where(z != 0, z, 1)
                    + obs.intrinsics[camera, 1, 2]).astype(np.int64)
        in_image = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        sampled = np.zeros(len(z))
        sampled[in_image] = obs.depth[camera, v[in_image], u[in_image]]
        valid_depth = (np.isfinite(sampled) & (sampled >= config.depth_min)
                       & (sampled <= config.depth_max))
        sdf = sampled - z
        valid = in_image & valid_depth & (sdf >= -config.truncation_distance)
        normalized = np.clip(sdf / config.truncation_distance, -1.0, 1.0)
        accum[valid] += normalized[valid]
        weight[valid] += 1.0
    raw_weight = weight
    scale = np.ones_like(raw_weight)
    saturated = raw_weight > config.max_weight
    scale[saturated] = config.max_weight / raw_weight[saturated]
    weight = raw_weight * scale
    # A saturated cell scales history and the new frame equally.
    tsdf = np.where(weight > 0, accum * scale / np.maximum(weight, 1e-30), 1.0)
    tsdf = np.clip(tsdf, -1.0, 1.0).reshape(config.shape)
    weight = weight.reshape(config.shape)
    occupancy = (weight > 0) & (tsdf <= config.occupancy_threshold)
    esdf, gradient = dense_esdf(occupancy, config.voxel_size, config.unobserved_esdf)
    return MapState(tsdf, weight, occupancy, esdf, gradient)


def dense_esdf(
    occupancy: NDArray[np.bool_], voxel_size: float, empty_value: float
) -> tuple[FloatArray, FloatArray]:
    """Exact brute-force center ESDF with a half-voxel surface convention."""
    occupied = np.asarray(occupancy, dtype=bool)
    shape = occupied.shape
    coords = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1)
    flat = coords.reshape(-1, 3).astype(np.float64)
    occ_flat = occupied.reshape(-1)
    values = np.full(len(flat), empty_value, dtype=np.float64)
    gradients = np.zeros((len(flat), 3), dtype=np.float64)
    if not np.any(occ_flat):
        return values.reshape(shape), gradients.reshape(shape + (3,))
    for i, point in enumerate(flat):
        targets = flat[~occ_flat] if occ_flat[i] else flat[occ_flat]
        if len(targets) == 0:
            values[i] = -empty_value
            continue
        delta = point - targets
        distances = np.linalg.norm(delta, axis=1)
        winner = int(np.argmin(distances))
        center_distance = distances[winner] * voxel_size
        magnitude = max(center_distance - 0.5 * voxel_size, 0.5 * voxel_size)
        sign = -1.0 if occ_flat[i] else 1.0
        values[i] = sign * magnitude
        if distances[winner] > 0:
            gradients[i] = sign * delta[winner] / distances[winner]
    return values.reshape(shape), gradients.reshape(shape + (3,))


def fuse_sequence(
    config: PerceptionConfig, observations: Sequence[CameraObservation]
) -> MapState:
    state = empty_state(config)
    for observation in observations:
        state = integrate(config, state, observation)
    return state
