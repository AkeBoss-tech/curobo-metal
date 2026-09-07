"""Differentiable dense depth fusion implemented with portable PyTorch operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from curobo_metal.backend import validate_tensor_device
from curobo_metal.ops.world_collision import VoxelGrid, query_esdf


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
    environments: int = 1
    block_size: int = 8

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(not isinstance(v, int) or v < 2 for v in self.shape):
            raise ValueError("shape must contain three integers >= 2")
        values = (self.voxel_size, self.truncation_distance, self.depth_min,
                  self.depth_max, self.max_weight, self.unobserved_esdf)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("configuration scalars must be finite")
        if self.voxel_size <= 0 or self.truncation_distance <= 0 or self.max_weight <= 0:
            raise ValueError("voxel_size, truncation_distance, and max_weight must be positive")
        if self.depth_min < 0 or self.depth_min >= self.depth_max:
            raise ValueError("depth_min must be nonnegative and less than depth_max")
        if self.unobserved_esdf <= 0 or not isinstance(self.environments, int) or self.environments < 1:
            raise ValueError("unobserved_esdf and environments must be positive")
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")

    @classmethod
    def from_mapper_config(cls, value: Mapping[str, Any]) -> "PerceptionConfig":
        """Adapt portable fields from upstream-style mapper dictionaries."""
        if not isinstance(value, Mapping):
            raise TypeError("mapper config must be a mapping")
        aliases = {
            "voxel_size_m": "voxel_size",
            "map_size": "shape",
            "map_center": "grid_center",
            "truncation_distance_m": "truncation_distance",
            "num_environments": "environments",
        }
        fields = cls.__dataclass_fields__
        normalized = {aliases.get(k, k): v for k, v in value.items()}
        unknown = set(normalized) - set(fields)
        if unknown:
            raise ValueError(f"unsupported mapper config fields: {sorted(unknown)}")
        for key in ("shape", "grid_center"):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


@dataclass(frozen=True)
class CameraObservation:
    depth: torch.Tensor  # [B,C,H,W] or [C,H,W] or [H,W], metres
    intrinsics: torch.Tensor  # [B,C,3,3], broadcast forms accepted
    camera_to_world: torch.Tensor  # [B,C,4,4], broadcast forms accepted

    @classmethod
    def from_camera_frame(
        cls, value: Mapping[str, torch.Tensor] | Any
    ) -> "CameraObservation":
        """Adapt dict/object camera frames without copying tensor storage."""
        def field(*names: str) -> torch.Tensor:
            for name in names:
                if isinstance(value, Mapping) and name in value:
                    return value[name]
                if hasattr(value, name):
                    return getattr(value, name)
            raise ValueError(f"camera frame is missing {names[0]}")
        return cls(
            field("depth", "depth_image"),
            field("intrinsics", "projection_matrix"),
            field("camera_to_world", "pose", "camera_pose"),
        )


@dataclass(frozen=True)
class DenseMap:
    tsdf: torch.Tensor  # [E,nx,ny,nz], normalized to [-1,1]
    weight: torch.Tensor
    occupancy: torch.Tensor
    esdf: torch.Tensor  # metres; positive free, negative occupied
    gradient: torch.Tensor  # [E,nx,ny,nz,3]
    generation: torch.Tensor  # int64 [E]


@dataclass(frozen=True)
class SparseTSDF:
    block_indices: torch.Tensor  # int64 [K,4]: environment,x,y,z
    tsdf: torch.Tensor  # [K,B,B,B]
    weight: torch.Tensor
    block_size: int
    generation: torch.Tensor


@dataclass(frozen=True)
class TriangleMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    environment: int


@dataclass(frozen=True)
class RenderResult:
    depth: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class PoseRefinementResult:
    camera_to_world: torch.Tensor
    loss: torch.Tensor
    iterations: int


def _float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(value)
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if value.device.type == "mps" and value.dtype != torch.float32:
        raise TypeError("MPS perception supports only float32")
    return value


def voxel_centers(
    config: PerceptionConfig, *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    axes = [
        (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * config.voxel_size + c
        for n, c in zip(config.shape, config.grid_center)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def _canonical_observation(
    observation: CameraObservation, environments: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = _float_tensor(observation.depth, "depth")
    if depth.ndim == 2:
        depth = depth[None, None]
    elif depth.ndim == 3:
        depth = depth[None]
    if depth.ndim != 4:
        raise ValueError("depth must have shape [H,W], [C,H,W], or [B,C,H,W]")
    b, c = depth.shape[:2]
    if b == 0 or c == 0 or depth.shape[-2] == 0 or depth.shape[-1] == 0:
        raise ValueError("depth batch, cameras, height, and width must be positive")
    if b not in (1, environments):
        raise ValueError("depth batch must be 1 or equal environments")
    if b == 1 and environments > 1:
        depth = depth.expand(environments, -1, -1, -1)
        b = environments

    intrinsics = _float_tensor(observation.intrinsics, "intrinsics")
    poses = _float_tensor(observation.camera_to_world, "camera_to_world")
    if intrinsics.device != depth.device or poses.device != depth.device:
        raise ValueError("observation tensors must share a device")
    if intrinsics.dtype != depth.dtype or poses.dtype != depth.dtype:
        raise TypeError("observation tensors must share a dtype")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None, None]
    elif intrinsics.ndim == 3:
        intrinsics = intrinsics[None]
    if poses.ndim == 2:
        poses = poses[None, None]
    elif poses.ndim == 3:
        poses = poses[None]
    try:
        intrinsics = intrinsics.expand(b, c, 3, 3)
        poses = poses.expand(b, c, 4, 4)
    except RuntimeError as error:
        raise ValueError("intrinsics/poses must broadcast to [B,C,3,3]/[B,C,4,4]") from error
    if not bool(torch.isfinite(intrinsics).all().item()) or not bool(torch.isfinite(poses).all().item()):
        raise ValueError("intrinsics and camera_to_world must be finite")
    if bool((intrinsics[..., (0, 1), (0, 1)] <= 0).any().item()):
        raise ValueError("fx and fy must be positive")
    expected_i = depth.new_tensor([0.0, 0.0, 1.0])
    expected_p = depth.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(intrinsics[..., 2, :], expected_i.expand_as(intrinsics[..., 2, :]), atol=1e-6, rtol=0):
        raise ValueError("intrinsics bottom row must be [0,0,1]")
    if not torch.allclose(poses[..., 3, :], expected_p.expand_as(poses[..., 3, :]), atol=1e-6, rtol=0):
        raise ValueError("camera_to_world bottom row must be [0,0,0,1]")
    rotations = poses[..., :3, :3]
    eye = torch.eye(3, dtype=depth.dtype, device=depth.device)
    if not torch.allclose(rotations @ rotations.transpose(-1, -2), eye, atol=1e-5, rtol=0):
        raise ValueError("camera_to_world rotations must be orthonormal")
    return depth, intrinsics, poses


def dense_esdf(
    occupancy: torch.Tensor, voxel_size: float, empty_value: float,
    *, dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact dense EDT using cdist; intended for bounded planning volumes."""
    if occupancy.dtype != torch.bool or occupancy.ndim != 4:
        raise ValueError("occupancy must be boolean [E,nx,ny,nz]")
    device = occupancy.device
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be float32 or float64")
    if device.type == "mps" and dtype != torch.float32:
        raise TypeError("MPS dense ESDF supports only float32")
    axes = [torch.arange(n, device=device, dtype=dtype) for n in occupancy.shape[1:]]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    values, gradients = [], []
    for env in range(len(occupancy)):
        mask = occupancy[env].reshape(-1)
        if not bool(mask.any().item()):
            values.append(torch.full((len(coords),), empty_value, device=device, dtype=dtype))
            gradients.append(torch.zeros((len(coords), 3), device=device, dtype=dtype))
            continue
        # Computing the full N x N matrix made otherwise modest mapper
        # volumes (for example 50 x 40 x 40) consume tens of gigabytes.  The
        # nearest opposite-class point is identical when evaluated in bounded
        # query/target tiles; retain only the current winner for each query.
        best = torch.full((len(coords),), torch.inf, device=device, dtype=dtype)
        winner = torch.zeros((len(coords),), device=device, dtype=torch.long)
        for source_value, target_value in ((False, True), (True, False)):
            source_indices = torch.nonzero(mask == source_value, as_tuple=False).flatten()
            target_indices = torch.nonzero(mask == target_value, as_tuple=False).flatten()
            if not len(source_indices) or not len(target_indices):
                continue
            for query_start in range(0, len(source_indices), 1024):
                query_indices = source_indices[query_start : query_start + 1024]
                query_best = torch.full(
                    (len(query_indices),), torch.inf, device=device, dtype=dtype
                )
                query_winner = torch.zeros(
                    (len(query_indices),), device=device, dtype=torch.long
                )
                for target_start in range(0, len(target_indices), 4096):
                    candidate_indices = target_indices[target_start : target_start + 4096]
                    distances = torch.cdist(coords[query_indices], coords[candidate_indices])
                    candidate_best, local_winner = distances.min(-1)
                    improve = candidate_best < query_best
                    query_best = torch.where(improve, candidate_best, query_best)
                    query_winner = torch.where(
                        improve, candidate_indices[local_winner], query_winner
                    )
                best[query_indices] = query_best
                winner[query_indices] = query_winner
        all_occupied = ~torch.isfinite(best)
        delta = coords - coords[winner]
        unit = delta / best.clamp_min(torch.finfo(dtype).tiny)[:, None]
        sign = torch.where(mask, -torch.ones_like(best), torch.ones_like(best))
        magnitude = (best * voxel_size - 0.5 * voxel_size).clamp_min(0.5 * voxel_size)
        value = sign * magnitude
        value = torch.where(all_occupied, torch.full_like(value, -empty_value), value)
        gradient = torch.where(all_occupied[:, None], torch.zeros_like(unit), sign[:, None] * unit)
        values.append(value)
        gradients.append(gradient)
    shape = occupancy.shape
    return (torch.stack(values).reshape(shape),
            torch.stack(gradients).reshape(shape + (3,)))


def integrate_depth(
    config: PerceptionConfig, state: DenseMap, observation: CameraObservation
) -> DenseMap:
    depth, intrinsics, poses = _canonical_observation(observation, config.environments)
    if state.tsdf.device != depth.device or state.tsdf.dtype != depth.dtype:
        raise ValueError("state and observation must share device and dtype")
    centers = voxel_centers(config, device=depth.device, dtype=depth.dtype).reshape(-1, 3)
    n = len(centers)
    accum = state.tsdf.reshape(config.environments, n) * state.weight.reshape(config.environments, n)
    added_weight = torch.zeros_like(accum)
    h, w = depth.shape[-2:]
    for camera in range(depth.shape[1]):
        rotation = poses[:, camera, :3, :3]
        translation = poses[:, camera, :3, 3]
        local = torch.einsum("bij,bnj->bni", rotation.transpose(-1, -2),
                             centers[None] - translation[:, None])
        z = local[..., 2]
        safe_z = torch.where(z != 0, z, torch.ones_like(z))
        u = torch.round(intrinsics[:, camera, 0, 0, None] * local[..., 0] / safe_z
                        + intrinsics[:, camera, 0, 2, None]).to(torch.int64)
        v = torch.round(intrinsics[:, camera, 1, 1, None] * local[..., 1] / safe_z
                        + intrinsics[:, camera, 1, 2, None]).to(torch.int64)
        in_image = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        linear = v.clamp(0, h - 1) * w + u.clamp(0, w - 1)
        sampled = depth[:, camera].reshape(len(depth), -1).gather(1, linear)
        valid_depth = (torch.isfinite(sampled) & (sampled >= config.depth_min)
                       & (sampled <= config.depth_max))
        sdf = sampled - z
        valid = in_image & valid_depth & (sdf >= -config.truncation_distance)
        normalized = (sdf / config.truncation_distance).clamp(-1, 1)
        accum = accum + torch.where(valid, normalized, torch.zeros_like(normalized))
        added_weight = added_weight + valid.to(depth.dtype)
    old_weight = state.weight.reshape(config.environments, n)
    raw_weight = old_weight + added_weight
    # Once saturated, scale both history and this frame equally.
    scale = torch.where(raw_weight > config.max_weight, config.max_weight / raw_weight, torch.ones_like(raw_weight))
    weight = raw_weight * scale
    tsdf = torch.where(raw_weight > 0, accum * scale / weight.clamp_min(1e-30), torch.ones_like(accum))
    tsdf = tsdf.clamp(-1, 1).reshape(state.tsdf.shape)
    weight = weight.reshape(state.weight.shape)
    occupancy = (weight > 0) & (tsdf <= config.occupancy_threshold)
    esdf, gradient = dense_esdf(
        occupancy, config.voxel_size, config.unobserved_esdf, dtype=depth.dtype
    )
    generation = state.generation + (added_weight.sum(-1) > 0).to(torch.int64)
    return DenseMap(tsdf, weight, occupancy, esdf, gradient, generation)


def integrate_lidar(
    config: PerceptionConfig,
    state: DenseMap,
    range_image: torch.Tensor,
    lidar_to_world: torch.Tensor,
    valid_range_m: torch.Tensor,
    elevation_range_rad: torch.Tensor,
) -> DenseMap:
    """Fuse spherical LiDAR range images into a dense TSDF on CPU or MPS.

    The projection follows cuRobo's range-image convention: columns span
    ``[-pi, pi)`` and row zero is the maximum elevation.  Nearest-pixel
    sampling is deterministic and keeps the observable one-sided TSDF update
    used by the CUDA implementation.
    """
    ranges = _float_tensor(range_image, "range_image")
    poses = _float_tensor(lidar_to_world, "lidar_to_world")
    valid_ranges = _float_tensor(valid_range_m, "valid_range_m")
    elevations = _float_tensor(elevation_range_rad, "elevation_range_rad")
    if ranges.ndim != 3:
        raise ValueError("range_image must have shape [num_lidars,H,W]")
    sensors, height, width = ranges.shape
    if sensors < 1 or height < 1 or width < 1:
        raise ValueError("range_image dimensions must be positive")
    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if tuple(poses.shape) != (sensors, 4, 4):
        raise ValueError("lidar_to_world must have shape [num_lidars,4,4]")
    if tuple(valid_ranges.shape) != (sensors, 2):
        raise ValueError("valid_range_m must have shape [num_lidars,2]")
    if tuple(elevations.shape) != (sensors, 2):
        raise ValueError("elevation_range_rad must have shape [num_lidars,2]")
    tensors = (poses, valid_ranges, elevations)
    if any(value.device != ranges.device for value in tensors):
        raise ValueError("LiDAR tensors must share a device")
    if any(value.dtype != ranges.dtype for value in tensors):
        raise TypeError("LiDAR tensors must share a dtype")
    if state.tsdf.device != ranges.device or state.tsdf.dtype != ranges.dtype:
        raise ValueError("state and LiDAR observation must share device and dtype")
    if bool((valid_ranges[:, 0] > valid_ranges[:, 1]).any().item()):
        raise ValueError("valid_range_m must contain [minimum, maximum]")
    if bool((elevations[:, 0] > elevations[:, 1]).any().item()):
        raise ValueError("elevation_range_rad must contain [minimum, maximum]")
    if height == 1 and not bool(torch.allclose(elevations[:, 0], elevations[:, 1])):
        raise ValueError("planar LiDAR requires equal elevation_range_rad bounds")

    centers = voxel_centers(config, device=ranges.device, dtype=ranges.dtype).reshape(-1, 3)
    voxel_count = len(centers)
    old_weight = state.weight.reshape(config.environments, voxel_count)
    accum = state.tsdf.reshape(config.environments, voxel_count) * old_weight
    added_weight = torch.zeros_like(accum)
    tiny = torch.finfo(ranges.dtype).tiny
    for sensor in range(sensors):
        transform = poses[sensor]
        local = (centers - transform[:3, 3]) @ transform[:3, :3]
        radius = torch.linalg.vector_norm(local, dim=-1)
        xy_radius = torch.linalg.vector_norm(local[:, :2], dim=-1)
        azimuth = torch.atan2(local[:, 1], local[:, 0])
        u_float = (azimuth + torch.pi) * (width / (2.0 * torch.pi))
        u = torch.round(u_float).to(torch.int64).remainder(width)
        elevation = torch.atan2(local[:, 2], xy_radius)
        minimum, maximum = elevations[sensor]
        if height == 1:
            v = torch.zeros_like(u)
            in_elevation = (
                (elevation - minimum).abs()
                <= config.voxel_size / radius.clamp_min(tiny)
            )
        else:
            v_float = (maximum - elevation) * ((height - 1) / (maximum - minimum))
            v = torch.round(v_float).to(torch.int64)
            in_elevation = (elevation >= minimum) & (elevation <= maximum)
        linear = v.clamp(0, height - 1) * width + u
        sampled = ranges[sensor].reshape(-1)[linear]
        valid = (
            torch.isfinite(sampled)
            & (sampled >= valid_ranges[sensor, 0])
            & (sampled <= valid_ranges[sensor, 1])
            & in_elevation
        )
        sdf = sampled - radius
        valid &= sdf >= -config.truncation_distance
        normalized = (sdf / config.truncation_distance).clamp(-1.0, 1.0)
        contribution = torch.where(valid, normalized, torch.zeros_like(normalized))
        weight = valid.to(ranges.dtype)
        accum = accum + contribution.unsqueeze(0)
        added_weight = added_weight + weight.unsqueeze(0)

    raw_weight = old_weight + added_weight
    scale = torch.where(
        raw_weight > config.max_weight,
        config.max_weight / raw_weight.clamp_min(tiny),
        torch.ones_like(raw_weight),
    )
    weight = (raw_weight * scale).reshape(state.weight.shape)
    tsdf = torch.where(
        raw_weight > 0,
        accum * scale / (raw_weight * scale).clamp_min(tiny),
        torch.ones_like(accum),
    ).clamp(-1.0, 1.0).reshape(state.tsdf.shape)
    occupancy = (weight > 0) & (tsdf <= config.occupancy_threshold)
    esdf, gradient = dense_esdf(
        occupancy, config.voxel_size, config.unobserved_esdf, dtype=ranges.dtype
    )
    generation = state.generation + (added_weight.sum(-1) > 0).to(torch.int64)
    return DenseMap(tsdf, weight, occupancy, esdf, gradient, generation)


def sparse_blocks(state: DenseMap, block_size: int = 8) -> SparseTSDF:
    """Pack observed blocks in lexicographic order with deterministic padding."""
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    e, nx, ny, nz = state.tsdf.shape
    indices, values, weights = [], [], []
    for env in range(e):
        for x in range(0, nx, block_size):
            for y in range(0, ny, block_size):
                for z in range(0, nz, block_size):
                    w = state.weight[env, x:x + block_size, y:y + block_size, z:z + block_size]
                    if not bool((w > 0).any().item()):
                        continue
                    tpad = state.tsdf.new_ones((block_size,) * 3)
                    wpad = state.weight.new_zeros((block_size,) * 3)
                    extent = w.shape
                    tpad[:extent[0], :extent[1], :extent[2]] = state.tsdf[
                        env, x:x + block_size, y:y + block_size, z:z + block_size
                    ]
                    wpad[:extent[0], :extent[1], :extent[2]] = w
                    indices.append((env, x // block_size, y // block_size, z // block_size))
                    values.append(tpad)
                    weights.append(wpad)
    index = torch.tensor(indices, device=state.tsdf.device, dtype=torch.int64).reshape(-1, 4)
    empty_shape = (0, block_size, block_size, block_size)
    return SparseTSDF(
        index,
        torch.stack(values) if values else state.tsdf.new_empty(empty_shape),
        torch.stack(weights) if weights else state.weight.new_empty(empty_shape),
        block_size,
        state.generation.clone(),
    )


def extract_mesh(
    config: PerceptionConfig, state: DenseMap, environment: int = 0
) -> TriangleMesh:
    """Extract a deterministic watertight voxel-surface mesh."""
    if environment < 0 or environment >= config.environments:
        raise ValueError("environment is out of range")
    occupied = state.occupancy[environment]
    centers = voxel_centers(config, device=occupied.device, dtype=state.tsdf.dtype)
    directions = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    # Face corners in a stable outward winding; vertex duplication is intentional.
    corners = (
        ((-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1)),
        ((1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)),
        ((-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)),
        ((-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)),
        ((-1,-1,-1),(-1,1,-1),(1,1,-1),(1,-1,-1)),
        ((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)),
    )
    vertices, faces = [], []
    for cell in torch.nonzero(occupied, as_tuple=False).tolist():
        for face_id, direction in enumerate(directions):
            neighbor = tuple(cell[i] + direction[i] for i in range(3))
            inside = all(0 <= neighbor[i] < config.shape[i] for i in range(3))
            if inside and bool(occupied[neighbor].item()):
                continue
            base = len(vertices)
            center = centers[tuple(cell)]
            vertices.extend(center + center.new_tensor(c) * (config.voxel_size / 2) for c in corners[face_id])
            faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    return TriangleMesh(
        torch.stack(vertices) if vertices else state.tsdf.new_empty((0, 3)),
        torch.tensor(faces, device=state.tsdf.device, dtype=torch.int64).reshape(-1, 3),
        environment,
    )


def render_depth(
    config: PerceptionConfig, state: DenseMap, intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor, image_size: tuple[int, int], *, environment: int = 0,
) -> RenderResult:
    """Render occupied voxel centres with a deterministic nearest-depth z-buffer."""
    obs = CameraObservation(
        state.tsdf.new_zeros(image_size), intrinsics, camera_to_world
    )
    _, intr, poses = _canonical_observation(obs, 1)
    h, w = image_size
    points = voxel_centers(config, device=state.tsdf.device, dtype=state.tsdf.dtype)[
        state.occupancy[environment]
    ]
    depth = state.tsdf.new_full((h * w,), torch.inf)
    if len(points):
        local = (points - poses[0, 0, :3, 3]) @ poses[0, 0, :3, :3]
        z = local[:, 2]
        u = torch.round(intr[0, 0, 0, 0] * local[:, 0] / z.clamp_min(1e-30) + intr[0, 0, 0, 2]).long()
        v = torch.round(intr[0, 0, 1, 1] * local[:, 1] / z.clamp_min(1e-30) + intr[0, 0, 1, 2]).long()
        valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        linear = v[valid] * w + u[valid]
        # scatter_reduce is supported natively by the fallback-disabled MPS path.
        depth = depth.scatter_reduce(0, linear, z[valid], reduce="amin", include_self=True)
    valid_pixels = torch.isfinite(depth)
    return RenderResult(torch.where(valid_pixels, depth, torch.zeros_like(depth)).reshape(h, w),
                        valid_pixels.reshape(h, w))


class PerceptionMapper:
    """Mutable multi-environment facade over functional differentiable fusion."""

    def __init__(
        self, config: PerceptionConfig, *, device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        device = torch.device(device)
        if device.type not in ("cpu", "mps"):
            raise ValueError("perception supports only CPU and MPS devices")
        if dtype not in (torch.float32, torch.float64) or (device.type == "mps" and dtype != torch.float32):
            raise TypeError("dtype must be float32/float64 (float32 only on MPS)")
        self.config = config
        shape = (config.environments,) + config.shape
        self.state = DenseMap(
            torch.ones(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=torch.bool),
            torch.full(shape, config.unobserved_esdf, device=device, dtype=dtype),
            torch.zeros(shape + (3,), device=device, dtype=dtype),
            torch.zeros(config.environments, device=device, dtype=torch.int64),
        )

    def update(
        self, observation: CameraObservation, *, env_indices: torch.Tensor | None = None
    ) -> DenseMap:
        if env_indices is None:
            self.state = integrate_depth(self.config, self.state, observation)
            return self.state
        if not isinstance(env_indices, torch.Tensor) or env_indices.dtype != torch.int64 or env_indices.ndim != 1:
            raise ValueError("env_indices must be an int64 vector")
        if len(env_indices) == 0 or bool(((env_indices < 0) | (env_indices >= self.config.environments)).any().item()):
            raise ValueError("env_indices must be nonempty and in range")
        if len(torch.unique(env_indices)) != len(env_indices):
            raise ValueError("env_indices must not contain duplicates")
        depth, intrinsics, poses = _canonical_observation(observation, len(env_indices))
        subconfig = PerceptionConfig(**{**self.config.__dict__, "environments": len(env_indices)})
        index = env_indices.to(self.state.tsdf.device)
        substate = DenseMap(*(getattr(self.state, field)[index] for field in
                              ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")))
        updated = integrate_depth(subconfig, substate, CameraObservation(depth, intrinsics, poses))
        fields = {}
        for field in ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation"):
            fields[field] = getattr(self.state, field).index_copy(0, index, getattr(updated, field))
        self.state = DenseMap(**fields)
        return self.state

    def allocate_blocks(self) -> SparseTSDF:
        return sparse_blocks(self.state, self.config.block_size)

    def extract_mesh(self, environment: int = 0) -> TriangleMesh:
        return extract_mesh(self.config, self.state, environment)

    def render(
        self, intrinsics: torch.Tensor, camera_to_world: torch.Tensor,
        image_size: tuple[int, int], *, environment: int = 0,
    ) -> RenderResult:
        return render_depth(self.config, self.state, intrinsics, camera_to_world,
                            image_size, environment=environment)

    def refine_pose(
        self, observation: CameraObservation, *, environment: int = 0,
        iterations: int = 5, learning_rate: float = 0.1,
    ) -> PoseRefinementResult:
        """Refine camera z translation against rendered valid depth pixels."""
        depth, intrinsics, poses = _canonical_observation(observation, 1)
        if iterations < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("iterations must be nonnegative and learning_rate positive")
        pose = poses[0, 0].clone()
        losses = []
        for _ in range(iterations):
            rendered = self.render(intrinsics[0, 0], pose, depth.shape[-2:],
                                   environment=environment)
            valid = rendered.valid & torch.isfinite(depth[0, 0]) & (depth[0, 0] > 0)
            residual = depth[0, 0][valid] - rendered.depth[valid]
            loss = residual.square().mean() if bool(valid.any().item()) else depth.new_zeros(())
            losses.append(loss)
            # Translation along camera z changes rendered depth with derivative -1.
            if bool(valid.any().item()):
                pose[:3, 3] = pose[:3, 3] - learning_rate * residual.mean() * pose[:3, 2]
        final = losses[-1] if losses else depth.new_zeros(())
        return PoseRefinementResult(pose, final, iterations)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "curobo-metal-perception-checkpoint",
            "version": 1,
            "config": dict(self.config.__dict__),
            **{field: getattr(self.state, field).clone() for field in
               ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")},
        }

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != "curobo-metal-perception-checkpoint" or checkpoint.get("version") != 1:
            raise ValueError("unsupported perception checkpoint")
        if PerceptionConfig.from_mapper_config(checkpoint["config"]) != self.config:
            raise ValueError("checkpoint configuration does not match mapper")
        fields = {}
        for field in ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation"):
            value = checkpoint[field]
            expected = getattr(self.state, field)
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"checkpoint field {field} has incompatible shape or dtype")
            fields[field] = value.to(expected.device).clone()
        self.state = DenseMap(**fields)

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        fresh = PerceptionMapper(self.config, device=self.state.tsdf.device,
                                 dtype=self.state.tsdf.dtype).state
        if env_indices is None:
            self.state = fresh
            return
        if not isinstance(env_indices, torch.Tensor) or env_indices.dtype != torch.int64 or env_indices.ndim != 1:
            raise ValueError("env_indices must be an int64 vector")
        index = env_indices.to(self.state.tsdf.device)
        if len(index) == 0 or bool(((index < 0) | (index >= self.config.environments)).any().item()):
            raise ValueError("env_indices must be nonempty and in range")
        if len(torch.unique(index)) != len(index):
            raise ValueError("env_indices must not contain duplicates")
        self.state = DenseMap(**{
            field: getattr(self.state, field).index_copy(0, index, getattr(fresh, field)[index])
            for field in ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")
        })

    def voxel_grids(self) -> list[list[VoxelGrid]]:
        identity = torch.eye(3, device=self.state.tsdf.device, dtype=self.state.tsdf.dtype)
        center = self.state.tsdf.new_tensor(self.config.grid_center)
        return [[VoxelGrid(self.state.esdf[e], self.config.voxel_size, center, identity,
                           self.config.unobserved_esdf)]
                for e in range(self.config.environments)]

    def query(
        self, points: torch.Tensor, *, env_indices: torch.Tensor | None = None,
        padding: float = 0.0,
    ):
        return query_esdf(points, self.voxel_grids(), env_indices=env_indices, padding=padding)
