"""cuRobo-style collision checker interfaces backed by production operators."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from curobo_metal.ops.collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
)
from curobo_metal.ops.world_collision import (
    ESDFQueryResult,
    mesh_distance,
    query_esdf,
    sphere_world_collision,
)
from .cache import Cuboid, MeshCache, PrimitiveCache, VoxelCache
from .config import (
    RobotCollisionCheckerConfig,
    RobotSceneCollisionConfig,
    WorldCollisionConfig,
)


@dataclass(frozen=True)
class CollisionQueryResult:
    """Per-sphere collision cost and world-center gradient."""

    distance: torch.Tensor
    gradient: torch.Tensor
    primitive_distance: torch.Tensor
    mesh_distance: torch.Tensor
    voxel_distance: torch.Tensor
    input_was_unbatched: bool

    @property
    def max_distance(self) -> torch.Tensor:
        return self.distance.max(dim=-1).values


@dataclass(frozen=True)
class SweptCollisionResult:
    distance: torch.Tensor
    gradient: torch.Tensor
    samples: torch.Tensor
    sample_distance: torch.Tensor


def _spheres(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("spheres must be a torch.Tensor")
    unbatched = value.ndim == 2
    if unbatched:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] != 4:
        raise ValueError("spheres must have shape [S,4] or [B,S,4]")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("spheres must have dtype float32 or float64")
    if value.device.type == "mps" and value.dtype != torch.float32:
        raise TypeError("MPS collision checking supports only float32")
    if bool((value[..., 3] < 0).any().item()):
        raise ValueError("sphere radii must be nonnegative")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("spheres must contain only finite values")
    return value, unbatched


def _env(
    env_indices: torch.Tensor | None, batch: int, count: int, reference: torch.Tensor
) -> torch.Tensor:
    if env_indices is None:
        return torch.zeros(batch, dtype=torch.int64, device=reference.device)
    if (
        not isinstance(env_indices, torch.Tensor)
        or env_indices.dtype != torch.int64
        or env_indices.shape != (batch,)
        or env_indices.device != reference.device
    ):
        raise ValueError(f"env_indices must be int64 on {reference.device} with shape [{batch}]")
    if bool(((env_indices < 0) | (env_indices >= count)).any().item()):
        raise ValueError("env_indices contains an out-of-range environment")
    return env_indices


class WorldCollision:
    """Mutable multi-environment world checker.

    Cache mutations increment ``generation``. Queries never retain derived
    tensors across generations, so updates and activation changes are visible
    immediately and cannot return stale collision results.
    """

    def __init__(self, config: WorldCollisionConfig = WorldCollisionConfig()) -> None:
        if not isinstance(config, WorldCollisionConfig):
            raise TypeError("config must be a WorldCollisionConfig")
        self.config = config
        self.primitive_cache = PrimitiveCache(
            config.environments, config.primitive_cache.max_cuboids
        )
        self.mesh_cache = MeshCache(config.environments, config.mesh_cache.max_meshes)
        self.voxel_cache = VoxelCache(config.environments, config.voxel_cache.max_voxels)

    @property
    def generation(self) -> tuple[int, int, int]:
        return (
            self.primitive_cache.generation,
            self.mesh_cache.generation,
            self.voxel_cache.generation,
        )

    def update_cuboid(
        self, environment: int, slot: int, center: torch.Tensor,
        rotation: torch.Tensor, half_extents: torch.Tensor, *, active: bool = True
    ) -> None:
        self.primitive_cache.update(
            environment, slot, Cuboid(center, rotation, half_extents), active=active
        )

    def enable_obstacle(
        self, kind: str, environment: int, slot: int, enabled: bool = True
    ) -> None:
        caches = {
            "primitive": self.primitive_cache,
            "mesh": self.mesh_cache,
            "voxel": self.voxel_cache,
        }
        if kind not in caches:
            raise ValueError("kind must be 'primitive', 'mesh', or 'voxel'")
        caches[kind].set_active(environment, slot, enabled)

    def _primitive_clearance(
        self, spheres: torch.Tensor, environments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, gradients = [], []
        for b, e_tensor in enumerate(environments):
            e = int(e_tensor.item())
            slots = [
                item for i, item in enumerate(self.primitive_cache._slots[e])
                if item is not None and bool(self.primitive_cache._active[e, i])
            ]
            if not slots:
                rows.append(torch.full_like(spheres[b, :, 0], torch.inf))
                gradients.append(torch.zeros_like(spheres[b, :, :3]))
                continue
            centers = torch.stack([x.center.to(spheres) for x in slots])
            rotations = torch.stack([x.rotation.to(spheres) for x in slots])
            half = torch.stack([x.half_extents.to(spheres) for x in slots])
            result = sphere_cuboid_signed_distance(
                spheres[b], centers, rotations, half, padding=self.config.padding
            )
            rows.append(result.reduced_distance[0])
            gradients.append(result.reduced_sphere_gradient[0, :, :3])
        return torch.stack(rows), torch.stack(gradients)

    def _mesh_clearance(
        self, spheres: torch.Tensor, environments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distances, gradients = [], []
        for b, e_tensor in enumerate(environments):
            e = int(e_tensor.item())
            slots = [
                item for i, item in enumerate(self.mesh_cache._slots[e])
                if item is not None and bool(self.mesh_cache._active[e, i])
            ]
            if not slots:
                distances.append(torch.full_like(spheres[b, :, 0], torch.inf))
                gradients.append(torch.zeros_like(spheres[b, :, :3]))
                continue
            count = len(slots)
            translations = spheres.new_zeros((1, count, 3))
            rotations = torch.eye(3, dtype=spheres.dtype, device=spheres.device).reshape(1, 1, 3, 3).expand(1, count, 3, 3)
            result = mesh_distance(
                spheres[b, :, :3], slots, translations, rotations, signed=False
            )
            distances.append(result.reduced_distance[0] - spheres[b, :, 3] - self.config.padding)
            gradients.append(result.reduced_gradient[0])
        return torch.stack(distances), torch.stack(gradients)

    def _voxel_clearance(
        self, spheres: torch.Tensor, environments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distances, gradients = [], []
        for b, e_tensor in enumerate(environments):
            e = int(e_tensor.item())
            slots = [
                item for i, item in enumerate(self.voxel_cache._slots[e])
                if item is not None and bool(self.voxel_cache._active[e, i])
            ]
            if not slots:
                distances.append(torch.full_like(spheres[b, :, 0], torch.inf))
                gradients.append(torch.zeros_like(spheres[b, :, :3]))
                continue
            result = query_esdf(spheres[b, :, :3], [slots], padding=self.config.padding)
            distances.append(result.distance[0] - spheres[b, :, 3])
            gradients.append(result.gradient[0])
        return torch.stack(distances), torch.stack(gradients)

    def get_sphere_distance(
        self, spheres: torch.Tensor, *, env_indices: torch.Tensor | None = None
    ) -> CollisionQueryResult:
        values, unbatched = _spheres(spheres)
        environments = _env(env_indices, len(values), self.config.environments, values)
        primitive, primitive_gradient = self._primitive_clearance(values, environments)
        mesh, mesh_gradient = self._mesh_clearance(values, environments)
        voxel, voxel_gradient = self._voxel_clearance(values, environments)
        clearance = torch.stack((primitive, mesh, voxel), dim=-1)
        winning_clearance, winner = clearance.min(dim=-1)
        candidate_gradient = torch.stack(
            (primitive_gradient, mesh_gradient, voxel_gradient), dim=-2
        )
        world_gradient = candidate_gradient.gather(
            -2, winner[..., None, None].expand(*winner.shape, 1, 3)
        ).squeeze(-2)
        cost, gradient = sphere_world_collision(
            values,
            torch.where(
                torch.isfinite(winning_clearance),
                winning_clearance + values[..., 3],
                torch.full_like(winning_clearance, torch.finfo(values.dtype).max / 4),
            ),
            world_gradient,
            activation_distance=self.config.activation_distance,
        )
        valid = torch.isfinite(winning_clearance)
        return CollisionQueryResult(
            torch.where(valid, cost, torch.zeros_like(cost)),
            torch.where(valid[..., None], gradient, torch.zeros_like(gradient)),
            primitive, mesh, voxel, unbatched,
        )

    get_collision_distance = get_sphere_distance

    def get_swept_sphere_distance(
        self,
        spheres: torch.Tensor,
        end_spheres: torch.Tensor | None = None,
        *,
        env_indices: torch.Tensor | None = None,
        interpolation_steps: int | None = None,
    ) -> SweptCollisionResult:
        """Query linearly interpolated sphere trajectories.

        Accepts start/end ``[B,S,4]`` tensors or a trajectory ``[B,H,S,4]``.
        Radius is interpolated as part of the packed sphere representation.
        """
        steps = self.config.interpolation_steps if interpolation_steps is None else interpolation_steps
        if steps < 1:
            raise ValueError("interpolation_steps must be positive")
        if end_spheres is None:
            trajectory = spheres
            if trajectory.ndim == 3:
                trajectory = trajectory.unsqueeze(0)
            if trajectory.ndim != 4 or trajectory.shape[-1] != 4:
                raise ValueError("trajectory must have shape [H,S,4] or [B,H,S,4]")
        else:
            start, _ = _spheres(spheres)
            end, _ = _spheres(end_spheres)
            if start.shape != end.shape:
                raise ValueError("start and end spheres must have matching shapes")
            trajectory = torch.stack((start, end), dim=1)
        alpha = torch.linspace(
            0, 1, steps + 1, dtype=trajectory.dtype, device=trajectory.device
        )
        pieces = []
        for i in range(trajectory.shape[1] - 1):
            segment = (
                trajectory[:, i, None] * (1 - alpha[None, :, None, None])
                + trajectory[:, i + 1, None] * alpha[None, :, None, None]
            )
            pieces.append(segment if i == 0 else segment[:, 1:])
        samples = torch.cat(pieces, dim=1) if pieces else trajectory
        batch, horizon, count, _ = samples.shape
        expanded_env = None if env_indices is None else env_indices[:, None].expand(batch, horizon).reshape(-1)
        query = self.get_sphere_distance(
            samples.reshape(batch * horizon, count, 4), env_indices=expanded_env
        )
        sample_distance = query.distance.reshape(batch, horizon, count)
        distance, winning_time = sample_distance.max(dim=1)
        sample_gradient = query.gradient.reshape(batch, horizon, count, 3)
        gradient = sample_gradient.gather(
            1, winning_time[:, None, :, None].expand(batch, 1, count, 3)
        ).squeeze(1)
        return SweptCollisionResult(distance, gradient, samples, sample_distance)

    get_swept_collision_distance = get_swept_sphere_distance

    def get_esdf(
        self, points: torch.Tensor, *, env_indices: torch.Tensor | None = None
    ) -> ESDFQueryResult:
        environments = []
        for e in range(self.config.environments):
            slots = [
                item for i, item in enumerate(self.voxel_cache._slots[e])
                if item is not None and bool(self.voxel_cache._active[e, i])
            ]
            if not slots:
                raise RuntimeError("ESDF query requires an active voxel in every environment")
            environments.append(slots)
        widths = {len(x) for x in environments}
        if len(widths) != 1:
            raise RuntimeError("ESDF routing requires equal active voxel counts per environment")
        return query_esdf(points, environments, env_indices=env_indices)

    get_esdf_in_bounding_box = get_esdf


class RobotCollisionChecker:
    def __init__(self, config: RobotCollisionCheckerConfig = RobotCollisionCheckerConfig()) -> None:
        self.config = config

    def get_self_collision_distance(
        self, spheres: torch.Tensor, *, sphere_active: torch.Tensor | None = None
    ) -> torch.Tensor:
        values, _ = _spheres(spheres)
        pairs = torch.as_tensor(
            self.config.self_collision_pairs, dtype=torch.int64, device=values.device
        )
        if pairs.numel() == 0:
            pairs = pairs.reshape(0, 2)
        result = sphere_sphere_signed_distance(
            values, pairs, sphere_active=sphere_active,
            padding=self.config.self_collision_padding,
        )
        # cuRobo collision distances are nonnegative violation magnitudes and
        # reduce with max; primitive ops expose signed clearance and min.
        return torch.where(
            torch.isfinite(result.reduced_distance),
            (-result.reduced_distance).clamp_min(0),
            torch.zeros_like(result.reduced_distance),
        )

    get_scene_self_collision_distance = get_self_collision_distance


class RobotSceneCollision(RobotCollisionChecker):
    def __init__(self, config: RobotSceneCollisionConfig = RobotSceneCollisionConfig()) -> None:
        if not isinstance(config, RobotSceneCollisionConfig):
            raise TypeError("config must be a RobotSceneCollisionConfig")
        super().__init__(config.robot)
        self.world_collision = WorldCollision(config.world)

    def get_collision_distance(
        self, spheres: torch.Tensor, *, env_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.world_collision.get_sphere_distance(
            spheres, env_indices=env_indices
        ).max_distance

    def get_swept_collision_distance(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.world_collision.get_swept_sphere_distance(
            *args, **kwargs
        ).distance.max(dim=-1).values
