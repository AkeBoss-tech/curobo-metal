"""Portable tensor storage for oriented cuboid scene obstacles.

The pinned implementation exposes mutable Warp-backed buffers.  This module
keeps the same useful cache lifecycle on CPU and MPS, but deliberately does
not manufacture a Warp struct or kernel entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg

from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, raw_warp


class CuboidDataWarp(PortableWarpStruct):
    """Pinned Warp value name; unavailable without the Warp runtime."""


def _validate_capacity(max_n: int, num_envs: int) -> tuple[int, int]:
    max_n, num_envs = int(max_n), int(num_envs)
    if max_n < 1:
        raise ValueError("max_n must be positive")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return max_n, num_envs


def _dims(value, cfg: DeviceCfg) -> torch.Tensor:
    result = torch.as_tensor(value, **cfg.as_torch_dict()).reshape(-1)
    if result.shape != (3,):
        raise ValueError("cuboid dims must contain exactly three values")
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise ValueError("cuboid dims must be finite and positive")
    return result


@dataclass(init=False)
class CuboidData(PortableObstacleData):
    """Per-environment OBB cache with deterministic mutable-name semantics."""

    @classmethod
    def create_cache(cls, max_n: int, num_envs: int, device_cfg: DeviceCfg) -> "CuboidData":
        max_n, num_envs = _validate_capacity(max_n, num_envs)
        result = cls._base(max_n, num_envs, device_cfg)
        # V2 initializes inactive dimensions to a small valid cuboid instead
        # of zeros, avoiding accidental invalid extents when a raw consumer
        # observes an unused cache slot.
        result.dims = torch.full((num_envs, max_n, 4), 0.01, **device_cfg.as_torch_dict())
        return result

    @classmethod
    def from_scene_cfg(
        cls, scene_cfg: SceneCfg, device_cfg: DeviceCfg, env_idx: int = 0,
        num_envs: int = 1, max_n: int | None = None,
    ) -> "CuboidData":
        cuboids = list(scene_cfg.cuboid or [])
        result = cls.create_cache(max_n if max_n is not None else max(len(cuboids), 1), num_envs, device_cfg)
        result.load_batch(cuboids, env_idx)
        return result

    @classmethod
    def from_batch_scene_cfg(
        cls, scene_cfg_list: list[SceneCfg], device_cfg: DeviceCfg, max_n: int | None = None,
    ) -> "CuboidData":
        if not scene_cfg_list:
            raise ValueError("scene_cfg_list must not be empty")
        capacity = max_n if max_n is not None else max(max(len(scene.cuboid or []), 1) for scene in scene_cfg_list)
        result = cls.create_cache(capacity, len(scene_cfg_list), device_cfg)
        for env_idx, scene in enumerate(scene_cfg_list):
            result.load_batch(list(scene.cuboid or []), env_idx)
        return result

    def _validate_batch(self, cuboids: Iterable[Cuboid], env_idx: int) -> list[Cuboid]:
        self._check_env(env_idx)
        values = list(cuboids)
        if len(values) > self.max_n:
            raise ValueError(f"cuboid cache capacity exceeded: {len(values)} > {self.max_n}")
        names = [item.name for item in values]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every cuboid requires a non-empty string name")
        if len(set(names)) != len(names):
            raise ValueError("cuboid names must be unique within an environment")
        for item in values:
            _dims(item.dims, self.device_cfg)
        return values

    def load_batch(self, cuboids: list[Cuboid], env_idx: int) -> None:
        """Replace one environment atomically after validating all records."""
        values = self._validate_batch(cuboids, env_idx)
        self.clear(env_idx)
        for item in values:
            self.add(item, env_idx)

    def add(self, cuboid: Cuboid, env_idx: int = 0) -> int:
        return self.add_from_raw(
            cuboid.name,
            cuboid.dims,
            env_idx,
            w_obj_pose=cuboid.pose or [0, 0, 0, 1, 0, 0, 0],
        )

    def add_from_raw(
        self, name: str, dims, env_idx: int = 0, w_obj_pose=None, obj_w_pose=None,
    ) -> int:
        self._check_env(env_idx)
        if not isinstance(name, str) or not name:
            raise ValueError("cuboid name must be a non-empty string")
        if self.has_name(name, env_idx):
            raise ValueError(f"cuboid already exists with name: {name!r}")
        if w_obj_pose is None and obj_w_pose is None:
            raise ValueError("w_obj_pose or obj_w_pose is required")
        index = self.get_active_count(env_idx)
        if index >= self.max_n:
            raise ValueError(f"cuboid cache is full ({self.max_n} cuboids)")
        extent = _dims(dims, self.device_cfg)
        pose = inverse_pose(w_obj_pose, self.device_cfg) if w_obj_pose is not None else torch.as_tensor(
            obj_w_pose.get_pose_vector() if hasattr(obj_w_pose, "get_pose_vector") else obj_w_pose,
            **self.device_cfg.as_torch_dict(),
        ).reshape(-1, 7)[0]
        if pose.shape != (7,) or not bool(torch.isfinite(pose).all()):
            raise ValueError("cuboid pose must contain seven finite values")
        self.dims[env_idx, index, :3].copy_(extent)
        self.inv_pose[env_idx, index, :7].copy_(pose)
        self.enable[env_idx, index] = 1
        self.names[env_idx][index] = name
        self.count[env_idx] += 1
        return index

    def update_dims(self, name: str, dims, env_idx: int = 0) -> None:
        self._check_env(env_idx)
        self.dims[env_idx, self.get_idx(name, env_idx), :3].copy_(_dims(dims, self.device_cfg))


is_obs_enabled = load_obstacle_transform = compute_local_sdf = compute_local_sdf_with_grad = raw_warp

__all__ = [
    "CuboidData", "CuboidDataWarp", "is_obs_enabled", "load_obstacle_transform",
    "compute_local_sdf", "compute_local_sdf_with_grad",
]
