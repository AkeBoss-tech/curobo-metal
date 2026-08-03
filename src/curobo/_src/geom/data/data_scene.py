"""Portable aggregate storage for cuboid, mesh and ESDF world data.

This mirrors V2's useful Python lifecycle API while deliberately stopping at
the CUDA/Warp conversion boundary.  The contained tensors are ordinary PyTorch
CPU/MPS tensors and are consumed by the portable scene-collision backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from curobo._src.geom.data.data_cuboid import CuboidData
from curobo._src.geom.data.data_mesh import MeshData
from curobo._src.geom.data.data_voxel import VoxelData
from curobo._src.geom.types import Cuboid, Mesh, Obstacle, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


@dataclass
class SceneData:
    """Mutable, per-environment portable geometry storage."""

    cuboids: CuboidData | None = None
    meshes: MeshData | None = None
    voxels: VoxelData | None = None
    num_envs: int = 1
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    scene_model: Optional[Union[SceneCfg, List[SceneCfg]]] = None

    def get_valid_data(self) -> list[object]:
        return [item for item in (self.cuboids, self.meshes, self.voxels) if item is not None]

    def _get_obstacle_data(self, name: str, env_idx: int = 0):
        for data in self.get_valid_data():
            if data.has_name(name, env_idx):
                return data
        return None

    @classmethod
    def create_cache(
        cls, num_envs: int, device_cfg: DeviceCfg,
        cuboid_cache: int | None = None, mesh_cache: int | None = None,
        mesh_max_dist: float = 0.1, voxel_cache: dict | None = None,
    ) -> "SceneData":
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if cuboid_cache is not None and cuboid_cache < 0:
            raise ValueError("cuboid_cache must be non-negative")
        if mesh_cache is not None and mesh_cache < 0:
            raise ValueError("mesh_cache must be non-negative")
        cuboids = CuboidData.create_cache(cuboid_cache, num_envs, device_cfg) if cuboid_cache else None
        meshes = MeshData.create_cache(mesh_cache, num_envs, device_cfg, mesh_max_dist) if mesh_cache else None
        voxels = None
        if voxel_cache is not None:
            if not isinstance(voxel_cache, dict):
                raise TypeError("voxel_cache must be a dictionary")
            dims = voxel_cache.get("dims", [1.0, 1.0, 1.0])
            size = voxel_cache.get("voxel_size", 0.02)
            layers = int(voxel_cache.get("layers", 1))
            voxels = VoxelData.create_cache(
                layers, num_envs, device_cfg, grid_dims=dims, voxel_size=size,
                max_esdf_distance=float(voxel_cache.get("max_esdf_distance", 100.0)),
                max_voxels=voxel_cache.get("max_voxels"),
            )
        return cls(cuboids, meshes, voxels, num_envs, device_cfg)

    @classmethod
    def from_scene_cfg(
        cls, scene_cfg: SceneCfg, device_cfg: DeviceCfg, num_envs: int = 1,
        env_idx: int = 0, cuboid_cache: int | None = None,
        mesh_cache: int | None = None, mesh_max_dist: float = 0.1,
        voxel_cache: dict | None = None,
    ) -> "SceneData":
        if not 0 <= env_idx < num_envs:
            raise IndexError("env_idx is outside num_envs")
        cuboid_capacity = cuboid_cache if cuboid_cache is not None else len(scene_cfg.cuboid)
        mesh_capacity = mesh_cache if mesh_cache is not None else len(scene_cfg.mesh)
        voxel_spec = voxel_cache
        if voxel_spec is None and scene_cfg.voxel:
            first = scene_cfg.voxel[0]
            voxel_spec = {
                "layers": len(scene_cfg.voxel), "dims": first.dims, "voxel_size": first.voxel_size,
                "max_voxels": max(
                    int(grid.feature_tensor.numel()) if grid.feature_tensor is not None
                    else int(torch.tensor(grid.get_grid_shape()[0]).prod().item())
                    for grid in scene_cfg.voxel
                ),
            }
        result = cls.create_cache(num_envs, device_cfg, cuboid_capacity, mesh_capacity, mesh_max_dist, voxel_spec)
        result.load_from_scene_cfg(scene_cfg, env_idx, store_reference=True)
        return result

    @classmethod
    def from_batch_scene_cfg(
        cls, scene_cfg_list: list[SceneCfg], device_cfg: DeviceCfg,
        cuboid_cache: int | None = None, mesh_cache: int | None = None,
        mesh_max_dist: float = 0.1, voxel_cache: dict | None = None,
    ) -> "SceneData":
        if not scene_cfg_list:
            raise ValueError("scene_cfg_list must not be empty")
        cuboid_capacity = cuboid_cache if cuboid_cache is not None else max(len(scene.cuboid) for scene in scene_cfg_list)
        mesh_capacity = mesh_cache if mesh_cache is not None else max(len(scene.mesh) for scene in scene_cfg_list)
        voxel_spec = voxel_cache
        if voxel_spec is None:
            grids = [grid for scene in scene_cfg_list for grid in scene.voxel]
            if grids:
                first = grids[0]
                voxel_spec = {
                    "layers": max(len(scene.voxel) for scene in scene_cfg_list),
                    "dims": first.dims,
                    "voxel_size": first.voxel_size,
                    "max_voxels": max(
                        int(grid.feature_tensor.numel()) if grid.feature_tensor is not None
                        else int(torch.tensor(grid.get_grid_shape()[0]).prod().item())
                        for grid in grids
                    ),
                }
        result = cls.create_cache(len(scene_cfg_list), device_cfg, cuboid_capacity, mesh_capacity, mesh_max_dist, voxel_spec)
        for env_idx, scene in enumerate(scene_cfg_list):
            result.load_from_scene_cfg(scene, env_idx, store_reference=False)
        result.scene_model = scene_cfg_list
        return result

    @classmethod
    def from_scene_model(cls, scene_model, device_cfg: DeviceCfg = DeviceCfg()):
        if isinstance(scene_model, list):
            return cls.from_batch_scene_cfg(scene_model, device_cfg)
        return cls.from_scene_cfg(scene_model, device_cfg)

    def add_obstacle(self, obstacle: Obstacle, env_idx: int = 0) -> int:
        if not 0 <= int(env_idx) < self.num_envs:
            raise IndexError("env_idx is outside num_envs")
        if self.check_obstacle_exists(obstacle.name, env_idx):
            raise ValueError(f"obstacle already exists with name: {obstacle.name!r}")
        if isinstance(obstacle, Cuboid):
            if self.cuboids is None:
                raise ValueError("Cuboid cache is not initialized")
            return self.cuboids.add(obstacle, env_idx)
        if isinstance(obstacle, Mesh):
            if self.meshes is None:
                raise ValueError("Mesh cache is not initialized")
            return self.meshes.add(obstacle, env_idx)
        if isinstance(obstacle, VoxelGrid):
            if self.voxels is None:
                raise ValueError("Voxel cache is not initialized")
            return self.voxels.add(obstacle, env_idx)
        raise NotImplementedError(
            f"{type(obstacle).__name__} is not a native SceneData collision layer; "
            "convert it with SceneCfg.get_collision_check_world()"
        )

    def update_obstacle_pose(self, name: str, pose: Pose, env_idx: int = 0) -> None:
        data = self._get_obstacle_data(name, env_idx)
        if data is None:
            raise ValueError(f"Obstacle {name!r} not found in environment {env_idx}")
        data.update_pose(name, w_obj_pose=pose, env_idx=env_idx)

    def enable_obstacle(self, name: str, enabled: bool = True, env_idx: int = 0) -> None:
        data = self._get_obstacle_data(name, env_idx)
        if data is None:
            raise ValueError(f"Obstacle {name!r} not found in environment {env_idx}")
        data.set_enabled(name, enabled, env_idx)

    def get_obstacle_names(self, env_idx: int = 0) -> list[str]:
        return [name for data in self.get_valid_data() for name in data.get_names(env_idx)]

    def check_obstacle_exists(self, name: str, env_idx: int = 0) -> bool:
        return self._get_obstacle_data(name, env_idx) is not None

    def clear(self, env_idx: int | None = None) -> None:
        for data in self.get_valid_data():
            data.clear(env_idx)

    def load_from_scene_cfg(self, scene_cfg: SceneCfg, env_idx: int = 0, store_reference: bool = True) -> None:
        if not 0 <= int(env_idx) < self.num_envs:
            raise IndexError("env_idx is outside num_envs")
        names = [obstacle.name for obstacle in (*scene_cfg.cuboid, *scene_cfg.mesh, *scene_cfg.voxel)]
        if len(names) != len(set(names)):
            raise ValueError("scene obstacle names must be unique across collision layers")
        if store_reference:
            self.scene_model = scene_cfg
        self.clear(env_idx)
        for obstacle in (*scene_cfg.cuboid, *scene_cfg.mesh, *scene_cfg.voxel):
            self.add_obstacle(obstacle, env_idx)

    def has_cuboids(self) -> bool:
        return self.cuboids is not None

    def has_meshes(self) -> bool:
        return self.meshes is not None

    def has_voxels(self) -> bool:
        return self.voxels is not None

    def get_active_types(self) -> dict[str, bool]:
        return {"cuboid": self.has_cuboids(), "mesh": self.has_meshes(), "voxel": self.has_voxels()}

    def to_warp(self, mesh_max_dist: float | None = None):
        raise NotImplementedError(
            "Warp scene structs are unavailable on the portable backend; use SceneData tensors or SceneCollision"
        )


class SceneDataWarp:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Warp scene data is unavailable on the portable backend")


__all__ = ["SceneData", "SceneDataWarp"]
