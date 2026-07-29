"""Differentiable dense depth fusion implemented with portable PyTorch operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

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


@dataclass(frozen=True)
class CameraObservation:
    depth: torch.Tensor  # [B,C,H,W] or [C,H,W] or [H,W], metres
    intrinsics: torch.Tensor  # [B,C,3,3], broadcast forms accepted
    camera_to_world: torch.Tensor  # [B,C,4,4], broadcast forms accepted


@dataclass(frozen=True)
class DenseMap:
    tsdf: torch.Tensor  # [E,nx,ny,nz], normalized to [-1,1]
    weight: torch.Tensor
    occupancy: torch.Tensor
    esdf: torch.Tensor  # metres; positive free, negative occupied
    gradient: torch.Tensor  # [E,nx,ny,nz,3]
    generation: torch.Tensor  # int64 [E]


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
        distance = torch.cdist(coords, coords)
        target_mask = torch.where(mask[:, None], ~mask[None], mask[None])
        candidates = torch.where(target_mask, distance, torch.full_like(distance, torch.inf))
        best, winner = candidates.min(-1)
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
