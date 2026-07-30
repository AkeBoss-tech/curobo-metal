"""Pinned scene collision interface over :mod:`curobo_metal` production operators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Union

import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo_metal.collision_checker import (
    MeshCacheConfig, PrimitiveCacheConfig, VoxelCacheConfig, WorldCollision, WorldCollisionConfig,
)
from curobo_metal.ops.world_collision import Mesh as BackendMesh
from curobo_metal.ops.world_collision import VoxelGrid as BackendVoxelGrid


def _rotation(pose: list[float], device_cfg: DeviceCfg) -> torch.Tensor:
    q = device_cfg.to_device(pose[3:])
    q = q / torch.linalg.vector_norm(q)
    w, x, y, z = q.unbind()
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )).reshape(3, 3)


@dataclass
class SceneCollisionCfg:
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    scene_model: Optional[Union[SceneCfg, List[SceneCfg]]] = None
    num_envs: int = 1
    max_distance: float = 0.1
    cache: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.scene_model, list):
            self.num_envs = len(self.scene_model)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")


class SceneCollision:
    def __init__(self, config: SceneCollisionCfg) -> None:
        self.device_cfg = config.device_cfg
        self.scene_model = config.scene_model
        cache = config.cache or {}
        scenes = config.scene_model if isinstance(config.scene_model, list) else (
            [] if config.scene_model is None else [config.scene_model]
        )
        required = {
            "cuboid": max([len(x.cuboid) for x in scenes] or [0]),
            "mesh": max([len(x.mesh) for x in scenes] or [0]),
            "voxel": max([len(x.voxel) for x in scenes] or [0]),
        }
        capacities = {
            "cuboid": cache.get("cuboid", cache.get("primitive", cache.get("obb", required["cuboid"]))),
            "mesh": cache.get("mesh", required["mesh"]),
            "voxel": cache.get("voxel", required["voxel"]),
        }
        if any(capacities[k] < required[k] for k in required):
            raise ValueError("collision cache is smaller than the supplied scene")
        self._world = WorldCollision(WorldCollisionConfig(
            environments=config.num_envs,
            primitive_cache=PrimitiveCacheConfig(capacities["cuboid"]),
            mesh_cache=MeshCacheConfig(capacities["mesh"]),
            voxel_cache=VoxelCacheConfig(capacities["voxel"]),
            allow_cpu_fallback=False,
        ))
        self._names: list[dict[str, tuple[str, int]]] = [dict() for _ in range(config.num_envs)]
        for index, scene in enumerate(scenes):
            self.load_collision_model(scene, index)

    @classmethod
    def from_config(cls, config: SceneCollisionCfg) -> "SceneCollision":
        if not isinstance(config, SceneCollisionCfg):
            raise TypeError("config must be a SceneCollisionCfg")
        return cls(config)

    @property
    def collision_types(self) -> Dict[str, bool]:
        return {
            "cuboid": bool(self._world.primitive_cache.active.any()),
            "mesh": bool(self._world.mesh_cache.active.any()),
            "voxel": bool(self._world.voxel_cache.active.any()),
        }

    @property
    def num_envs(self) -> int:
        return self._world.config.environments

    def _load_cuboid(self, value: Cuboid, env_idx: int, slot: int) -> None:
        pose = value.pose
        self._world.update_cuboid(
            env_idx, slot, self.device_cfg.to_device(pose[:3]), _rotation(pose, self.device_cfg),
            self.device_cfg.to_device(value.dims) * 0.5,
        )

    def _load_mesh(self, value: Mesh, env_idx: int, slot: int) -> None:
        vertices = self.device_cfg.to_device(value.vertices)
        pose = value.pose or [0, 0, 0, 1, 0, 0, 0]
        vertices = vertices @ _rotation(pose, self.device_cfg).T + self.device_cfg.to_device(pose[:3])
        faces = torch.as_tensor(value.faces, device=self.device_cfg.device, dtype=torch.int64)
        if faces.ndim == 1:
            if faces.numel() % 3:
                raise ValueError("mesh faces must contain triangles")
            faces = faces.reshape(-1, 3)
        self._world.update_mesh(env_idx, slot, BackendMesh(vertices, faces, watertight=False))

    def _load_voxel(self, value: VoxelGrid, env_idx: int, slot: int) -> None:
        shape = value.get_grid_shape()[0]
        if value.feature_tensor is None:
            raise ValueError("VoxelGrid requires feature_tensor ESDF values")
        values = value.feature_tensor.to(
            device=self.device_cfg.device, dtype=self.device_cfg.collision_geometry_dtype
        ).reshape(shape)
        pose = value.pose or [0, 0, 0, 1, 0, 0, 0]
        self._world.update_voxel(env_idx, slot, BackendVoxelGrid(
            values, value.voxel_size, self.device_cfg.to_device(pose[:3]),
            _rotation(pose, self.device_cfg), -self._world.config.activation_distance,
        ))

    def load_collision_model(self, scene_model: SceneCfg, env_idx: int = 0):
        if not isinstance(scene_model, SceneCfg):
            raise TypeError("scene_model must be a SceneCfg")
        if scene_model.sphere or scene_model.capsule or scene_model.cylinder:
            raise NotImplementedError("sphere, capsule, and cylinder scene obstacles are unsupported")
        self.clear_cache(env_idx)
        for kind, values, loader in (
            ("primitive", scene_model.cuboid, self._load_cuboid),
            ("mesh", scene_model.mesh, self._load_mesh),
            ("voxel", scene_model.voxel, self._load_voxel),
        ):
            cache = getattr(self._world, f"{'primitive' if kind == 'primitive' else kind}_cache")
            if len(values) > cache.capacity:
                raise RuntimeError(f"{kind} cache capacity {cache.capacity} exceeded")
            for slot, value in enumerate(values):
                if value.name in self._names[env_idx]:
                    raise ValueError(f"duplicate obstacle name: {value.name}")
                loader(value, env_idx, slot)
                self._names[env_idx][value.name] = (kind, slot)
        self.scene_model = scene_model

    def _query(
        self, query_spheres: torch.Tensor, collision_buffer: CollisionBuffer,
        weight: torch.Tensor, activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor], swept: bool,
    ) -> torch.Tensor:
        if query_spheres.ndim != 4 or query_spheres.shape[-1] != 4:
            raise ValueError("query_spheres must have shape [batch, horizon, num_spheres, 4]")
        if weight.numel() != 1 or activation_distance.numel() != 1:
            raise ValueError("weight and activation_distance must be scalar tensors")
        activation = float(activation_distance.item())
        if activation < 0:
            raise ValueError("activation_distance must be nonnegative")
        old = self._world.config
        self._world.config = replace(
            old, activation_distance=activation,
            interpolation_steps=16 if swept else old.interpolation_steps,
        )
        try:
            if swept:
                if query_spheres.shape[1] < 2:
                    raise ValueError("swept query requires at least two trajectory knots")
                segments, gradients = [], []
                for knot in range(query_spheres.shape[1] - 1):
                    result = self._world.get_swept_sphere_distance(
                        query_spheres[:, knot], query_spheres[:, knot + 1],
                        env_indices=env_query_idx,
                    )
                    segments.append(result.distance)
                    gradients.append(result.gradient)
                segment_distance = torch.stack(segments, dim=1)
                segment_gradient = torch.stack(gradients, dim=1)
                distance = torch.cat((segment_distance, segment_distance[:, -1:]), dim=1)
                gradient = torch.cat((segment_gradient, segment_gradient[:, -1:]), dim=1)
            else:
                batch, horizon, count, _ = query_spheres.shape
                env = None if env_query_idx is None else env_query_idx[:, None].expand(batch, horizon).reshape(-1)
                result = self._world.get_sphere_distance(query_spheres.reshape(-1, count, 4), env_indices=env)
                distance = result.distance.reshape(batch, horizon, count)
                gradient = result.gradient.reshape(batch, horizon, count, 3)
        finally:
            self._world.config = old
        scale = weight.to(distance)
        output = distance * scale
        collision_buffer.resize(query_spheres.shape, self.device_cfg)
        collision_buffer.distance.copy_(output.detach())
        collision_buffer.gradient.zero_()
        collision_buffer.gradient[..., :3].copy_((gradient * scale).detach())
        return output

    def get_sphere_distance_raw(self, query_spheres, collision_buffer, weight, activation_distance,
                                env_query_idx=None, return_loss=False):
        return self._query(query_spheres, collision_buffer, weight, activation_distance, env_query_idx, False)

    def get_sphere_distance(self, state, collision_buffer, weight, activation_distance,
                            env_query_idx=None, return_loss=False):
        return self.get_sphere_distance_raw(state.robot_spheres, collision_buffer, weight,
                                            activation_distance, env_query_idx, return_loss)

    get_sphere_collision = get_sphere_distance

    def get_swept_sphere_distance_raw(self, query_spheres, collision_buffer, weight,
                                      activation_distance, trajectory_dt,
                                      enable_speed_metric=False, env_query_idx=None, return_loss=False):
        if enable_speed_metric:
            raise NotImplementedError("swept speed metric is not implemented")
        if trajectory_dt.numel() not in (1, query_spheres.shape[1]):
            raise ValueError("trajectory_dt must be scalar or horizon-sized")
        return self._query(query_spheres, collision_buffer, weight, activation_distance, env_query_idx, True)

    def get_swept_sphere_distance(self, state, collision_buffer, weight, activation_distance,
                                  trajectory_dt, enable_speed_metric=False,
                                  env_query_idx=None, return_loss=False):
        return self.get_swept_sphere_distance_raw(
            state.robot_spheres, collision_buffer, weight, activation_distance, trajectory_dt,
            enable_speed_metric, env_query_idx, return_loss,
        )

    get_swept_sphere_collision = get_swept_sphere_distance

    def enable_obstacle(self, name: str, enable: bool = True, env_idx: int = 0):
        kind, slot = self._names[env_idx][name]
        self._world.enable_obstacle(kind, env_idx, slot, enable)

    def get_obstacle_names(self, env_idx: int = 0) -> List[str]:
        return list(self._names[env_idx])

    def check_obstacle_exists(self, name: str, env_idx: int = 0) -> bool:
        return name in self._names[env_idx]

    def clear_cache(self, env_idx: Optional[int] = None):
        self._world.clear_cache(env_idx)
        if env_idx is None:
            for names in self._names:
                names.clear()
        else:
            self._names[env_idx].clear()

    def get_num_scene_collision_checkers(self) -> int:
        return sum(self.collision_types.values())

    def update_obstacle_pose(self, name: str, w_obj_pose: Pose, env_idx: int = 0):
        raise NotImplementedError("pose mutation requires reloading the owning obstacle")

    def update_voxel_data(self, voxel_coords: torch.Tensor, features: torch.Tensor, env_idx: int = 0):
        raise NotImplementedError("sparse voxel updates are not supported; reload a dense VoxelGrid")

    def get_voxel_grid(self, env_idx: int = 0) -> Optional[Cuboid]:
        value = next((x for x in self._world.voxel_cache._slots[env_idx] if x is not None), None)
        if value is None:
            return None
        dims = [float(x * value.voxel_size) for x in value.values.shape]
        return Cuboid(name=f"voxel_grid_{env_idx}", pose=[*value.translation.tolist(), 1, 0, 0, 0], dims=dims)


def create_scene_collision(config: SceneCollisionCfg) -> SceneCollision:
    return SceneCollision.from_config(config)
