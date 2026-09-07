"""Established joint-bound cost names."""

from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_type import CSpaceCostType

BoundCost = PositionCSpaceCost
BoundCostConfig = CSpaceCostCfg
BoundCostType = CSpaceCostType
__all__ = ["BoundCost", "BoundCostConfig", "BoundCostType"]
