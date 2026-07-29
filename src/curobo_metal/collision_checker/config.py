"""Configuration records for cuRobo-style collision-checker wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(frozen=True)
class PrimitiveCacheConfig:
    max_cuboids: int = 0

    def __post_init__(self) -> None:
        if self.max_cuboids < 0:
            raise ValueError("max_cuboids must be nonnegative")


@dataclass(frozen=True)
class MeshCacheConfig:
    max_meshes: int = 0

    def __post_init__(self) -> None:
        if self.max_meshes < 0:
            raise ValueError("max_meshes must be nonnegative")


@dataclass(frozen=True)
class VoxelCacheConfig:
    max_voxels: int = 0

    def __post_init__(self) -> None:
        if self.max_voxels < 0:
            raise ValueError("max_voxels must be nonnegative")


@dataclass(frozen=True)
class WorldCollisionConfig:
    environments: int = 1
    primitive_cache: PrimitiveCacheConfig = field(default_factory=PrimitiveCacheConfig)
    mesh_cache: MeshCacheConfig = field(default_factory=MeshCacheConfig)
    voxel_cache: VoxelCacheConfig = field(default_factory=VoxelCacheConfig)
    activation_distance: float = 0.0
    padding: float = 0.0
    interpolation_steps: int = 1
    allow_cpu_fallback: bool = True

    def __post_init__(self) -> None:
        if self.environments <= 0:
            raise ValueError("environments must be positive")
        if self.activation_distance < 0 or self.padding < 0:
            raise ValueError("activation_distance and padding must be nonnegative")
        if self.interpolation_steps < 1:
            raise ValueError("interpolation_steps must be positive")


@dataclass(frozen=True)
class RobotCollisionCheckerConfig:
    self_collision_pairs: torch.Tensor | Sequence[Sequence[int]] = ()
    self_collision_padding: float = 0.0

    def __post_init__(self) -> None:
        if self.self_collision_padding < 0:
            raise ValueError("self_collision_padding must be nonnegative")


@dataclass(frozen=True)
class RobotSceneCollisionConfig:
    world: WorldCollisionConfig = field(default_factory=WorldCollisionConfig)
    robot: RobotCollisionCheckerConfig = field(default_factory=RobotCollisionCheckerConfig)


RobotSceneCollisionCfg = RobotSceneCollisionConfig
RobotCollisionCheckerCfg = RobotCollisionCheckerConfig
