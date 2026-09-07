"""Established world-collision import path."""

from enum import Enum

from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg


class CollisionCheckerType(Enum):
    PRIMITIVE = "primitive"
    MESH = "mesh"
    BLOX = "blox"
    VOXEL = "voxel"


WorldConfig = SceneCfg
WorldCollisionConfig = SceneCollisionCfg
WorldPrimitiveCollision = SceneCollision
__all__ = [
    "CollisionCheckerType", "WorldCollisionConfig", "WorldConfig",
    "WorldPrimitiveCollision",
]
