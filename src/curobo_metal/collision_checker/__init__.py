"""cuRobo-style mutable collision-checker compatibility layer."""

from .cache import (
    CacheCapacityError, Cuboid, MeshCache, PrimitiveCache, VoxelCache,
)
from .config import (
    MeshCacheConfig, PrimitiveCacheConfig, RobotCollisionCheckerCfg,
    RobotCollisionCheckerConfig, RobotSceneCollisionCfg,
    RobotSceneCollisionConfig, VoxelCacheConfig, WorldCollisionConfig,
)
from .core import (
    CollisionQueryResult, RobotCollisionChecker, RobotSceneCollision,
    SweptCollisionResult, WorldCollision,
)

__all__ = [
    "CacheCapacityError", "CollisionQueryResult", "Cuboid", "MeshCache",
    "MeshCacheConfig", "PrimitiveCache", "PrimitiveCacheConfig",
    "RobotCollisionChecker", "RobotCollisionCheckerCfg",
    "RobotCollisionCheckerConfig", "RobotSceneCollision",
    "RobotSceneCollisionCfg", "RobotSceneCollisionConfig",
    "SweptCollisionResult", "VoxelCache", "VoxelCacheConfig",
    "WorldCollision", "WorldCollisionConfig",
]
