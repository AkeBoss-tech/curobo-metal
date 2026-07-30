"""Pinned scene collision interface over :mod:`curobo_metal` production operators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Union

import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.types import Capsule, Cuboid, Cylinder, Mesh, SceneCfg, Sphere, VoxelGrid
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
        self._analytic: list[list[Union[Sphere, Capsule, Cylinder]]] = [
            [] for _ in range(config.num_envs)
        ]
        self._analytic_active: list[list[bool]] = [[] for _ in range(config.num_envs)]
        self._scene_models: list[Optional[SceneCfg]] = [None] * config.num_envs
        for index, scene in enumerate(scenes):
            self.load_collision_model(scene, index)
        self.scene_model = config.scene_model

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
            "sphere": any(
                active and isinstance(value, Sphere)
                for values, enabled in zip(self._analytic, self._analytic_active)
                for value, active in zip(values, enabled)
            ),
            "capsule": any(
                active and isinstance(value, Capsule)
                for values, enabled in zip(self._analytic, self._analytic_active)
                for value, active in zip(values, enabled)
            ),
            "cylinder": any(
                active and isinstance(value, Cylinder)
                for values, enabled in zip(self._analytic, self._analytic_active)
                for value, active in zip(values, enabled)
            ),
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
        for value in (*scene_model.sphere, *scene_model.capsule, *scene_model.cylinder):
            if value.name in self._names[env_idx]:
                raise ValueError(f"duplicate obstacle name: {value.name}")
            slot = len(self._analytic[env_idx])
            self._analytic[env_idx].append(value)
            self._analytic_active[env_idx].append(True)
            self._names[env_idx][value.name] = ("analytic", slot)
        self._scene_models[env_idx] = scene_model
        if self.num_envs == 1:
            self.scene_model = scene_model
        else:
            self.scene_model = list(self._scene_models)

    def _analytic_distance(
        self,
        query_spheres: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = query_spheres.reshape(-1, query_spheres.shape[-2], 4)
        if env_query_idx is None:
            env = torch.zeros(len(flat), dtype=torch.int64, device=flat.device)
        else:
            batch, horizon = query_spheres.shape[:2]
            if env_query_idx.shape != (batch,):
                raise ValueError("env_query_idx must have shape [batch]")
            env = env_query_idx.to(device=flat.device, dtype=torch.int64)
            env = env[:, None].expand(batch, horizon).reshape(-1)
        if bool(((env < 0) | (env >= self.num_envs)).any().item()):
            raise ValueError("env_query_idx contains an out-of-range environment")
        candidates: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        xyz, query_radius = flat[..., :3], flat[..., 3]
        eps = torch.finfo(flat.dtype).eps
        for environment in range(self.num_envs):
            selected = env == environment
            for value, active in zip(
                self._analytic[environment], self._analytic_active[environment]
            ):
                if not active:
                    continue
                if isinstance(value, Sphere):
                    pose = value.pose or [0, 0, 0, 1, 0, 0, 0]
                    center = flat.new_tensor(pose[:3])
                    delta = xyz - center
                    norm = torch.linalg.vector_norm(delta, dim=-1)
                    distance = norm - query_radius - value.radius
                    gradient = delta / norm.clamp_min(eps)[..., None]
                    gradient = torch.where(
                        (norm > eps)[..., None], gradient,
                        torch.tensor([1.0, 0.0, 0.0], device=flat.device, dtype=flat.dtype),
                    )
                elif isinstance(value, Capsule):
                    pose = value.pose or [0, 0, 0, 1, 0, 0, 0]
                    rotation = _rotation(pose, self.device_cfg)
                    translation = flat.new_tensor(pose[:3])
                    base = flat.new_tensor(value.base) @ rotation.T + translation
                    tip = flat.new_tensor(value.tip) @ rotation.T + translation
                    segment = tip - base
                    denom = segment.square().sum().clamp_min(eps)
                    t = ((xyz - base) * segment).sum(-1).div(denom).clamp(0, 1)
                    closest = base + t[..., None] * segment
                    delta = xyz - closest
                    norm = torch.linalg.vector_norm(delta, dim=-1)
                    distance = norm - query_radius - value.radius
                    gradient = delta / norm.clamp_min(eps)[..., None]
                    gradient = torch.where(
                        (norm > eps)[..., None], gradient,
                        torch.tensor([1.0, 0.0, 0.0], device=flat.device, dtype=flat.dtype),
                    )
                else:
                    pose = value.pose or [0, 0, 0, 1, 0, 0, 0]
                    rotation = _rotation(pose, self.device_cfg)
                    translation = flat.new_tensor(pose[:3])
                    local = (xyz - translation) @ rotation
                    radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
                    d = torch.stack(
                        (radial - value.radius, local[..., 2].abs() - value.height * 0.5),
                        dim=-1,
                    )
                    outside = torch.linalg.vector_norm(d.clamp_min(0), dim=-1)
                    inside = torch.minimum(d.max(-1).values, torch.zeros_like(outside))
                    sdf = outside + inside
                    distance = sdf - query_radius
                    radial_grad = local.new_zeros(local.shape)
                    radial_grad[..., :2] = local[..., :2] / radial.clamp_min(eps)[..., None]
                    cap_grad = local.new_zeros(local.shape)
                    cap_grad[..., 2] = local[..., 2].sign()
                    choose_cap = d[..., 1] > d[..., 0]
                    gradient = torch.where(
                        choose_cap[..., None], cap_grad, radial_grad
                    ) @ rotation.T
                distance = torch.where(
                    selected[:, None], distance, torch.full_like(distance, torch.inf)
                )
                gradient = torch.where(
                    selected[:, None, None], gradient, torch.zeros_like(gradient)
                )
                candidates.append(distance)
                gradients.append(gradient)
        if not candidates:
            shape = flat.shape[:-1]
            return flat.new_full(shape, torch.inf), flat.new_zeros(shape + (3,))
        values = torch.stack(candidates, dim=-1)
        result, winner = values.min(-1)
        all_grad = torch.stack(gradients, dim=-2)
        gather = winner[..., None, None].expand(*winner.shape, 1, 3)
        gradient = all_grad.gather(-2, gather).squeeze(-2)
        return result, gradient

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
            has_backend = any(
                bool(cache.active.any())
                for cache in (
                    self._world.primitive_cache,
                    self._world.mesh_cache,
                    self._world.voxel_cache,
                )
            )
            if not has_backend:
                distance = torch.full_like(query_spheres[..., 0], torch.inf)
                gradient = torch.zeros_like(query_spheres[..., :3])
            elif swept:
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
            analytic_distance, analytic_gradient = self._analytic_distance(
                query_spheres, env_query_idx
            )
            analytic_distance = analytic_distance.reshape_as(distance)
            analytic_gradient = analytic_gradient.reshape_as(gradient)
            choose_analytic = analytic_distance < distance
            distance = torch.minimum(distance, analytic_distance)
            gradient = torch.where(
                choose_analytic[..., None], analytic_gradient, gradient
            )
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

    def get_sphere_collision(self, state, collision_buffer, weight, activation_distance,
                             env_query_idx=None, return_loss=False):
        return self.get_sphere_distance(
            state, collision_buffer, weight, activation_distance,
            env_query_idx, return_loss,
        )

    def get_swept_sphere_distance_raw(self, query_spheres, collision_buffer, weight,
                                      activation_distance, trajectory_dt,
                                      enable_speed_metric=False, env_query_idx=None, return_loss=False):
        if trajectory_dt.numel() not in (1, query_spheres.shape[1]):
            raise ValueError("trajectory_dt must be scalar or horizon-sized")
        result = self._query(
            query_spheres, collision_buffer, weight, activation_distance,
            env_query_idx, True,
        )
        if enable_speed_metric and query_spheres.shape[1] > 2:
            dt = trajectory_dt.reshape(-1)[0].to(result).clamp_min(1e-6)
            velocity = 0.5 * (
                query_spheres[:, 2:, :, :3] - query_spheres[:, :-2, :, :3]
            ) / dt
            speed = torch.linalg.vector_norm(velocity, dim=-1)
            scaled = result.clone()
            scaled[:, 1:-1] = torch.where(
                result[:, 1:-1] > 0, result[:, 1:-1] * speed, result[:, 1:-1]
            )
            result = scaled
            collision_buffer.distance.copy_(result.detach())
        return result

    def get_swept_sphere_distance(self, state, collision_buffer, weight, activation_distance,
                                  trajectory_dt, enable_speed_metric=False,
                                  env_query_idx=None, return_loss=False):
        return self.get_swept_sphere_distance_raw(
            state.robot_spheres, collision_buffer, weight, activation_distance, trajectory_dt,
            enable_speed_metric, env_query_idx, return_loss,
        )

    def get_swept_sphere_collision(self, state, collision_buffer, weight,
                                   activation_distance, trajectory_dt,
                                   enable_speed_metric=False, env_query_idx=None,
                                   return_loss=False):
        return self.get_swept_sphere_distance(
            state, collision_buffer, weight, activation_distance, trajectory_dt,
            enable_speed_metric, env_query_idx, return_loss,
        )

    def enable_obstacle(self, name: str, enable: bool = True, env_idx: int = 0):
        kind, slot = self._names[env_idx][name]
        if kind == "analytic":
            self._analytic_active[env_idx][slot] = bool(enable)
        else:
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
            for index in range(self.num_envs):
                self._analytic[index].clear()
                self._analytic_active[index].clear()
                self._scene_models[index] = None
        else:
            self._names[env_idx].clear()
            self._analytic[env_idx].clear()
            self._analytic_active[env_idx].clear()
            self._scene_models[env_idx] = None

    def get_num_scene_collision_checkers(self) -> int:
        return sum(self.collision_types.values())

    def update_obstacle_pose(self, name: str, w_obj_pose: Pose, env_idx: int = 0):
        if w_obj_pose.position.numel() != 3 or w_obj_pose.quaternion.numel() != 4:
            raise ValueError("w_obj_pose must contain one position and quaternion")
        pose = [
            *w_obj_pose.position.reshape(-1).detach().cpu().tolist(),
            *w_obj_pose.quaternion.reshape(-1).detach().cpu().tolist(),
        ]
        kind, slot = self._names[env_idx][name]
        if kind == "analytic":
            self._analytic[env_idx][slot].pose = pose
        elif kind == "primitive":
            value = next(x for x in self._scene_models[env_idx].cuboid if x.name == name)
            value.pose = pose
            self._load_cuboid(value, env_idx, slot)
        elif kind == "mesh":
            raise NotImplementedError(
                "mesh pose mutation requires reloading vertices from their serialized frame"
            )
        else:
            value = next(x for x in self._scene_models[env_idx].voxel if x.name == name)
            value.pose = pose
            self._load_voxel(value, env_idx, slot)

    def update_voxel_data(self, voxel_coords: torch.Tensor, features: torch.Tensor, env_idx: int = 0):
        if voxel_coords.ndim != 2 or voxel_coords.shape[-1] != 3:
            raise ValueError("voxel_coords must have shape [N,3]")
        if features.reshape(-1).shape[0] != voxel_coords.shape[0]:
            raise ValueError("features must contain one value per voxel coordinate")
        grid = next(
            (x for x in self._world.voxel_cache._slots[env_idx] if x is not None),
            None,
        )
        if grid is None:
            raise RuntimeError("no voxel grid is configured")
        indices = voxel_coords.to(device=grid.values.device, dtype=torch.int64)
        shape = torch.tensor(grid.values.shape, device=indices.device)
        if bool(((indices < 0) | (indices >= shape)).any().item()):
            raise ValueError("voxel_coords contains an out-of-range index")
        values = grid.values.clone()
        values[indices[:, 0], indices[:, 1], indices[:, 2]] = features.to(values).reshape(-1)
        slot = next(i for i, x in enumerate(self._world.voxel_cache._slots[env_idx]) if x is grid)
        self._world.update_voxel(
            env_idx, slot,
            BackendVoxelGrid(
                values, grid.voxel_size, grid.translation, grid.rotation, grid.out_of_bounds
            ),
        )

    def get_voxel_grid(self, env_idx: int = 0) -> Optional[Cuboid]:
        value = next((x for x in self._world.voxel_cache._slots[env_idx] if x is not None), None)
        if value is None:
            return None
        dims = [float(x * value.voxel_size) for x in value.values.shape]
        return Cuboid(name=f"voxel_grid_{env_idx}", pose=[*value.translation.tolist(), 1, 0, 0, 0], dims=dims)


def create_scene_collision(config: SceneCollisionCfg) -> SceneCollision:
    return SceneCollision.from_config(config)
