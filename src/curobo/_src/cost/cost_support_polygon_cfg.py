from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from .portable import BaseCostCfg, CostSupportPolygon


@dataclass
class CostSupportPolygonCfg(BaseCostCfg):
    """Portable configuration for :class:`CostSupportPolygon`."""

    class_type: Type[CostSupportPolygon] = CostSupportPolygon
    foot_sphere_indices: Optional[torch.Tensor] = None
    foot_link_names: Optional[List[str]] = None
    inside_cost_weight: float = 0.001


__all__ = ["BaseCostCfg", "CostSupportPolygon", "CostSupportPolygonCfg"]
