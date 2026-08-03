"""Portable ESDF voxel-cache data and vectorised sampling helpers.

The cache mirrors V2's named per-environment lifecycle while retaining plain
PyTorch tensors on CPU/MPS.  Warp structs and kernel-only overloads remain an
explicit boundary; callers can use these helpers or :class:`SceneCollision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from curobo._src.geom.types import SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, raw_warp


def _as_index(value, *, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(value, device=device)
    if value.shape[-1:] != (3,):
        raise ValueError("voxel indices and dimensions must end in dimension 3")
    return value.to(dtype=torch.long)


def _feature_scalar(value, features: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=features.device, dtype=features.dtype)


def _shape_for_grid(grid: VoxelGrid) -> tuple[int, int, int]:
    shape = tuple(int(item) for item in grid.get_grid_shape()[0])
    if any(item < 1 for item in shape):
        raise ValueError("VoxelGrid dimensions must resolve to positive voxel counts")
    return shape


def _features_for_grid(grid: VoxelGrid, cfg: DeviceCfg) -> torch.Tensor:
    if grid.feature_tensor is None:
        raise ValueError(f"VoxelGrid {grid.name!r} requires feature_tensor for portable collision data")
    values = grid.feature_tensor.reshape(-1).to(**cfg.as_torch_dict())
    shape = _shape_for_grid(grid)
    expected = shape[0] * shape[1] * shape[2]
    if values.numel() != expected:
        raise ValueError(f"feature tensor size {values.numel()} does not match voxel shape {shape} ({expected})")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("voxel feature tensor must be finite floating-point values")
    return values


def voxel_idx_to_flat(idx, grid_dims) -> torch.Tensor:
    """Convert ``[..., x, y, z]`` indices to C-order flattened indices."""
    device = idx.device if isinstance(idx, torch.Tensor) else (grid_dims.device if isinstance(grid_dims, torch.Tensor) else torch.device("cpu"))
    index, dims = _as_index(idx, device=device), _as_index(grid_dims, device=device)
    return index[..., 0] * dims[..., 1] * dims[..., 2] + index[..., 1] * dims[..., 2] + index[..., 2]


def is_voxel_valid(idx, grid_dims) -> torch.Tensor:
    """Return whether all index coordinates lie in bounds."""
    device = idx.device if isinstance(idx, torch.Tensor) else (grid_dims.device if isinstance(grid_dims, torch.Tensor) else torch.device("cpu"))
    index, dims = _as_index(idx, device=device), _as_index(grid_dims, device=device)
    return ((index >= 0) & (index < dims)).all(dim=-1)


def world_to_voxel_idx(local_pt, grid_dims, voxel_size) -> torch.Tensor:
    """Map local coordinates to the V2 truncation-toward-zero voxel index."""
    point = local_pt if isinstance(local_pt, torch.Tensor) else torch.as_tensor(local_pt, dtype=torch.get_default_dtype())
    if point.shape[-1:] != (3,):
        raise ValueError("local_pt must end in dimension 3")
    dims = _as_index(grid_dims, device=point.device).to(dtype=point.dtype)
    size = torch.as_tensor(voxel_size, device=point.device, dtype=point.dtype)
    if size.ndim > 0 and size.numel() != 1:
        raise ValueError("voxel_size must be scalar")
    if bool((size <= 0).any()) or not bool(torch.isfinite(size).all()):
        raise ValueError("voxel_size must be finite and positive")
    return (point / size + dims * 0.5).to(dtype=torch.long)


def sample_voxel_sdf(features, layer_start_idx, idx, grid_dims, default_val) -> torch.Tensor:
    """Nearest ESDF query returning ``[..., sdf, valid]``."""
    values = features if isinstance(features, torch.Tensor) else torch.as_tensor(features)
    if values.ndim != 1 or not values.is_floating_point():
        raise ValueError("features must be a flat floating-point tensor")
    values = values.float() if values.dtype == torch.float16 else values
    index, dims = _as_index(idx, device=values.device), _as_index(grid_dims, device=values.device)
    if bool((dims <= 0).any()):
        raise ValueError("grid_dims must be positive")
    start = torch.as_tensor(layer_start_idx, device=values.device, dtype=torch.long)
    valid = is_voxel_valid(index, dims)
    safe = torch.minimum(torch.maximum(index, torch.zeros_like(index)), dims - 1)
    flat = voxel_idx_to_flat(safe, dims)
    if bool(((start + flat) < 0).any()) or bool(((start + flat) >= values.numel()).any()):
        raise ValueError("feature buffer is too small for grid_dims and layer_start_idx")
    sampled = values[start + flat]
    sampled = torch.where(valid, sampled, _feature_scalar(default_val, values))
    return torch.stack((sampled, valid.to(dtype=sampled.dtype)), dim=-1)


def sample_voxel_sdf_with_grad(features, layer_start_idx, local_pt, grid_dims, voxel_size, default_val) -> torch.Tensor:
    """Trilinearly query ESDF values, returning ``[..., sdf, dx, dy, dz]``.

    At incomplete border stencils V2's portable contract renormalizes valid
    corner weights.  Gradients use a pair only when both values exist.
    """
    values = features if isinstance(features, torch.Tensor) else torch.as_tensor(features)
    if values.ndim != 1 or not values.is_floating_point():
        raise ValueError("features must be a flat floating-point tensor")
    values = values.float() if values.dtype == torch.float16 else values
    point = local_pt if isinstance(local_pt, torch.Tensor) else torch.as_tensor(local_pt, device=values.device, dtype=values.dtype)
    point = point.to(device=values.device, dtype=values.dtype)
    if point.shape[-1:] != (3,):
        raise ValueError("local_pt must end in dimension 3")
    dims_i = _as_index(grid_dims, device=values.device)
    if bool((dims_i <= 0).any()):
        raise ValueError("grid_dims must be positive")
    voxel = torch.as_tensor(voxel_size, device=values.device, dtype=values.dtype)
    if voxel.numel() != 1 or bool((voxel <= 0).any()) or not bool(torch.isfinite(voxel).all()):
        raise ValueError("voxel_size must be one finite positive scalar")
    default = _feature_scalar(default_val, values)
    if bool((dims_i < 2).any()):
        nearest = sample_voxel_sdf(values, layer_start_idx, world_to_voxel_idx(point, dims_i, voxel), dims_i, default)
        return torch.cat((nearest[..., :1], torch.zeros_like(point)), dim=-1)

    coordinate = point / voxel + dims_i.to(dtype=values.dtype) * 0.5 - 0.5
    base = torch.floor(coordinate).to(dtype=torch.long)
    fraction = coordinate - base.to(dtype=values.dtype)
    offsets = torch.tensor([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], device=values.device, dtype=torch.long)
    corner = base.unsqueeze(-2) + offsets
    valid = is_voxel_valid(corner, dims_i)
    safe = torch.minimum(torch.maximum(corner, torch.zeros_like(corner)), dims_i - 1)
    flat = voxel_idx_to_flat(safe, dims_i)
    start = torch.as_tensor(layer_start_idx, device=values.device, dtype=torch.long)
    requested = start + flat
    if bool((requested < 0).any()) or bool((requested >= values.numel()).any()):
        raise ValueError("feature buffer is too small for grid_dims and layer_start_idx")
    samples = torch.where(valid, values[requested], default)
    bit = offsets.to(dtype=values.dtype)
    weights = torch.where(bit == 1, fraction.unsqueeze(-2), 1 - fraction.unsqueeze(-2)).prod(dim=-1)
    valid_weight = weights * valid.to(dtype=values.dtype)
    denominator = valid_weight.sum(dim=-1)
    sdf = torch.where(denominator > 0, (samples * valid_weight).sum(dim=-1) / denominator.clamp_min(torch.finfo(values.dtype).eps), default)

    pairs = (([0, 1, 2, 3], [4, 5, 6, 7]), ([0, 1, 4, 5], [2, 3, 6, 7]), ([0, 2, 4, 6], [1, 3, 5, 7]))
    gradients = []
    for axis, (lo, hi) in enumerate(pairs):
        low, high = torch.tensor(lo, device=values.device), torch.tensor(hi, device=values.device)
        other = [index for index in range(3) if index != axis]
        pair_weight = torch.where(
            offsets[low][:, other].to(dtype=values.dtype) == 1,
            fraction[..., other].unsqueeze(-2),
            1 - fraction[..., other].unsqueeze(-2),
        ).prod(dim=-1)
        pair_valid = valid[..., low] & valid[..., high]
        pair_sum = (pair_weight * pair_valid.to(dtype=values.dtype)).sum(dim=-1)
        gradients.append(torch.where(
            pair_sum > 0,
            ((samples[..., high] - samples[..., low]) * pair_weight * pair_valid.to(dtype=values.dtype)).sum(dim=-1) / pair_sum.clamp_min(torch.finfo(values.dtype).eps) / voxel,
            torch.zeros_like(pair_sum),
        ))
    return torch.cat((sdf.unsqueeze(-1), torch.stack(gradients, dim=-1)), dim=-1)


class VoxelDataWarp(PortableWarpStruct):
    """Pinned Warp value name; unavailable without Warp."""


@dataclass(init=False)
class VoxelData(PortableObstacleData):
    """Mutable flat ESDF cache with strict capacity and reconstruction rules."""

    @classmethod
    def create_cache(
        cls, max_n: int, num_envs: int, device_cfg: DeviceCfg,
        grid_dims: Sequence[float] | None = None, voxel_size: float | None = None,
        max_esdf_distance: float = 100.0, *, max_voxels: int | None = None,
    ) -> "VoxelData":
        max_n, num_envs = int(max_n), int(num_envs)
        if max_n < 1 or num_envs < 1:
            raise ValueError("max_n and num_envs must be positive")
        if max_voxels is None:
            if grid_dims is None or voxel_size is None:
                raise ValueError("grid_dims and voxel_size are required when max_voxels is omitted")
            probe = VoxelGrid("cache", dims=list(grid_dims), voxel_size=float(voxel_size))
            shape = _shape_for_grid(probe)
            max_voxels = shape[0] * shape[1] * shape[2]
        if int(max_voxels) < 1:
            raise ValueError("max_voxels must be positive")
        if not torch.isfinite(torch.tensor(float(max_esdf_distance))) or float(max_esdf_distance) <= 0:
            raise ValueError("max_esdf_distance must be finite and positive")
        result = cls._base(max_n, num_envs, device_cfg)
        result.max_voxels = int(max_voxels)
        result.max_esdf_distance = float(max_esdf_distance)
        result.features = torch.full((num_envs, max_n, result.max_voxels), result.max_esdf_distance, **device_cfg.as_torch_dict())
        result.xyzr = torch.zeros((num_envs, max_n, result.max_voxels, 4), **device_cfg.as_torch_dict())
        result.params = torch.zeros((num_envs, max_n, 4), **device_cfg.as_torch_dict())
        result.dims = torch.zeros((num_envs, max_n, 4), **device_cfg.as_torch_dict())
        result._grids: dict[tuple[int, str], VoxelGrid] = {}
        return result

    @classmethod
    def create_from_voxel_grids(
        cls, voxel_grids: list[VoxelGrid], device_cfg: DeviceCfg, env_idx: int = 0,
        num_envs: int = 1, max_n: int | None = None, max_esdf_distance: float = 100.0,
    ) -> "VoxelData":
        """Create data from grids; portable storage copies rather than aliases.

        Alias-based CUDA buffer lifetime is inherently unsafe on MPS.  The
        content, metadata and update semantics are preserved, while the copy
        boundary is explicit.
        """
        if not voxel_grids:
            raise ValueError("voxel_grids must not be empty")
        capacity = max_n if max_n is not None else len(voxel_grids)
        max_values = max(_features_for_grid(item, device_cfg).numel() for item in voxel_grids)
        result = cls.create_cache(capacity, num_envs, device_cfg, max_esdf_distance=max_esdf_distance, max_voxels=max_values)
        result.load_batch(voxel_grids, env_idx)
        return result

    @classmethod
    def from_voxel_grid(cls, voxel_grid: VoxelGrid, device_cfg: DeviceCfg, env_idx: int = 0, num_envs: int = 1, max_n: int = 1) -> "VoxelData":
        return cls.create_from_voxel_grids([voxel_grid], device_cfg, env_idx, num_envs, max_n)

    @classmethod
    def from_scene_cfg(cls, scene_cfg: SceneCfg, device_cfg: DeviceCfg, env_idx: int = 0, num_envs: int = 1, max_n: int | None = None, max_esdf_distance: float = 100.0) -> "VoxelData":
        grids = list(scene_cfg.voxel or [])
        if not grids:
            return cls.create_cache(max_n or 1, num_envs, device_cfg, grid_dims=[1, 1, 1], voxel_size=1.0, max_esdf_distance=max_esdf_distance)
        return cls.create_from_voxel_grids(grids, device_cfg, env_idx, num_envs, max_n or len(grids), max_esdf_distance)

    @classmethod
    def from_batch_scene_cfg(cls, scene_cfg_list: list[SceneCfg], device_cfg: DeviceCfg, max_n: int | None = None, max_esdf_distance: float = 100.0) -> "VoxelData":
        if not scene_cfg_list:
            raise ValueError("scene_cfg_list must not be empty")
        grids = [grid for scene in scene_cfg_list for grid in (scene.voxel or [])]
        capacity = max_n if max_n is not None else max(max(len(scene.voxel or []), 1) for scene in scene_cfg_list)
        if not grids:
            return cls.create_cache(capacity, len(scene_cfg_list), device_cfg, grid_dims=[1, 1, 1], voxel_size=1.0, max_esdf_distance=max_esdf_distance)
        result = cls.create_cache(capacity, len(scene_cfg_list), device_cfg, max_esdf_distance=max_esdf_distance, max_voxels=max(_features_for_grid(grid, device_cfg).numel() for grid in grids))
        for env_idx, scene in enumerate(scene_cfg_list):
            result.load_batch(list(scene.voxel or []), env_idx)
        return result

    def _validate_batch(self, grids: list[VoxelGrid], env_idx: int) -> list[VoxelGrid]:
        self._check_env(env_idx)
        if len(grids) > self.max_n:
            raise ValueError(f"voxel cache capacity exceeded: {len(grids)} > {self.max_n}")
        names = [grid.name for grid in grids]
        if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
            raise ValueError("voxel names must be unique non-empty strings within an environment")
        for grid in grids:
            if _features_for_grid(grid, self.device_cfg).numel() > self.max_voxels:
                raise ValueError("feature tensor exceeds voxel cache capacity")
        return grids

    def load_batch(self, grids: list[VoxelGrid], env_idx: int) -> None:
        values = self._validate_batch(list(grids), env_idx)
        self.clear(env_idx)
        for grid in values:
            self.add(grid, env_idx)

    def add(self, grid: VoxelGrid, env_idx: int = 0, name: str | None = None) -> int:
        self._check_env(env_idx)
        key = name or grid.name
        if not isinstance(key, str) or not key:
            raise ValueError("voxel name must be a non-empty string")
        if self.has_name(key, env_idx):
            raise ValueError(f"voxel grid already exists with name: {key!r}")
        index = self.get_active_count(env_idx)
        if index >= self.max_n:
            raise ValueError(f"voxel cache is full ({self.max_n} layers)")
        self.names[env_idx][index] = key
        self.count[env_idx] += 1
        self._write_grid(grid, env_idx, index, key)
        return index

    def _write_grid(self, grid: VoxelGrid, env_idx: int, index: int, key: str) -> None:
        features = _features_for_grid(grid, self.device_cfg)
        shape = _shape_for_grid(grid)
        self.features[env_idx, index].fill_(self.max_esdf_distance)
        self.features[env_idx, index, : features.numel()].copy_(features)
        self.xyzr[env_idx, index].zero_()
        if grid.xyzr_tensor is not None:
            xyzr = grid.xyzr_tensor.to(**self.device_cfg.as_torch_dict()).reshape(-1, 4)
            if xyzr.shape[0] != features.numel():
                raise ValueError("xyzr_tensor must contain one row per feature")
            self.xyzr[env_idx, index, : xyzr.shape[0]].copy_(xyzr)
        dims = torch.as_tensor(grid.dims, **self.device_cfg.as_torch_dict())
        self.params[env_idx, index].copy_(torch.tensor([*shape, grid.voxel_size], **self.device_cfg.as_torch_dict()))
        self.dims[env_idx, index].zero_()
        self.dims[env_idx, index, :3].copy_(dims)
        self.dims[env_idx, index, 3] = grid.voxel_size
        self.inv_pose[env_idx, index, :7].copy_(inverse_pose(grid.pose or [0, 0, 0, 1, 0, 0, 0], self.device_cfg))
        self.enable[env_idx, index] = 1
        self._grids[(env_idx, key)] = grid.clone()

    def update_data(self, grid: VoxelGrid, env_idx: int = 0, name: str | None = None) -> None:
        self._check_env(env_idx)
        key = name or grid.name
        index = self.get_idx(key, env_idx)
        self._write_grid(grid, env_idx, index, key)

    def update_features(self, features, name: str, env_idx: int = 0) -> None:
        self._check_env(env_idx)
        index = self.get_idx(name, env_idx)
        shape = tuple(int(value) for value in self.params[env_idx, index, :3].detach().cpu().tolist())
        expected = shape[0] * shape[1] * shape[2]
        values = torch.as_tensor(features, **self.device_cfg.as_torch_dict()).reshape(-1)
        if values.numel() != expected or not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError(f"features for {name!r} must be {expected} finite floating-point values")
        self.features[env_idx, index].fill_(self.max_esdf_distance)
        self.features[env_idx, index, :expected].copy_(values)
        stored = self._grids[(env_idx, name)].clone()
        stored.feature_tensor = values.clone()
        self._grids[(env_idx, name)] = stored

    def get_voxel_grid(self, name: str, env_idx: int = 0) -> VoxelGrid:
        self._check_env(env_idx)
        index = self.get_idx(name, env_idx)
        shape = tuple(int(value) for value in self.params[env_idx, index, :3].detach().cpu().tolist())
        count = shape[0] * shape[1] * shape[2]
        # Stored pose is world-to-object; VoxelGrid presents object-in-world.
        world_pose = inverse_pose(self.inv_pose[env_idx, index, :7], self.device_cfg).detach().cpu().tolist()
        dims = self.dims[env_idx, index, :3].detach().cpu().tolist()
        size = float(self.params[env_idx, index, 3].item())
        return VoxelGrid(name, pose=world_pose, dims=dims, voxel_size=size, feature_tensor=self.features[env_idx, index, :count].clone(), xyzr_tensor=self.xyzr[env_idx, index, :count].clone())

    def get_grid_shape(self, env_idx: int = 0, name: str | None = None, idx: int = 0) -> torch.Size:
        self._check_env(env_idx)
        if name is not None:
            idx = self.get_idx(name, env_idx)
        if not 0 <= int(idx) < self.max_n:
            raise IndexError("voxel layer index is outside cache capacity")
        return torch.Size([int(value) for value in self.params[env_idx, idx, :3].detach().cpu().tolist()])

    def clear(self, env_idx: int | None = None) -> None:
        ids = range(self.num_envs) if env_idx is None else [env_idx]
        for index in ids:
            self._check_env(index)
            self.features[index].fill_(self.max_esdf_distance)
            self.xyzr[index].zero_()
            self.params[index].zero_()
            self.dims[index].zero_()
            for key in [key for key in self._grids if key[0] == index]:
                del self._grids[key]
        super().clear(env_idx)


is_obs_enabled = load_obstacle_transform = compute_local_sdf = compute_local_sdf_with_grad = raw_warp

__all__ = [
    "VoxelData", "VoxelDataWarp", "is_voxel_valid", "sample_voxel_sdf",
    "sample_voxel_sdf_with_grad", "voxel_idx_to_flat", "world_to_voxel_idx",
    "is_obs_enabled", "load_obstacle_transform", "compute_local_sdf", "compute_local_sdf_with_grad",
]
