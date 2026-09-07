"""Established self-collision cost names."""

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg

SelfCollisionCostConfig = SelfCollisionCostCfg
__all__ = ["SelfCollisionCost", "SelfCollisionCostConfig"]
