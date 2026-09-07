"""Established geometry record import path."""

from curobo._src.geom.types import (
    Capsule, Cuboid, Cylinder, Mesh, Obstacle, SceneCfg, Sphere, VoxelGrid,
)

WorldConfig = SceneCfg
WorldCfg = SceneCfg
__all__ = [
    "Capsule", "Cuboid", "Cylinder", "Mesh", "Obstacle", "SceneCfg", "Sphere",
    "VoxelGrid", "WorldCfg", "WorldConfig",
]
