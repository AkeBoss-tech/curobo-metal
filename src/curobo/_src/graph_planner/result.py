"""Pinned graph planner result model."""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch


@dataclass
class GraphPlannerResult:
    success: torch.Tensor
    plan_waypoints: Optional[List[Union[torch.Tensor, None]]] = None
    interpolated_waypoints: Optional[torch.Tensor] = None
    joint_names: Optional[List[str]] = None
    path_length: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    valid_query: bool = True
    debug_info: Optional[Any] = None


__all__ = ["GraphPlannerResult"]
