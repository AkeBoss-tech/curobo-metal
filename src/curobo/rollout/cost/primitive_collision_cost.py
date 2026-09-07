"""Established primitive-collision cost names."""

from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg

PrimitiveCollisionCost = SceneCollisionCost
PrimitiveCollisionCostConfig = SceneCollisionCostCfg
__all__ = ["PrimitiveCollisionCost", "PrimitiveCollisionCostConfig"]
