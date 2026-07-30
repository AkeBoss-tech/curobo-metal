"""Pinned trajectory optimization result model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional

import torch

from curobo._src.state.state_joint import JointState


@dataclass
class TrajOptSolverResult:
    success: torch.Tensor
    solution: Optional[torch.Tensor] = None
    js_solution: Optional[JointState] = None
    position_error: Optional[torch.Tensor] = None
    rotation_error: Optional[torch.Tensor] = None
    cspace_error: Optional[torch.Tensor] = None
    goalset_index: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    total_time: float = 0.0
    debug_info: Dict = field(default_factory=dict)
    optimized_seeds: Optional[torch.Tensor] = None
    metrics: Optional[object] = None
    position_tolerance: float = 0.0
    orientation_tolerance: float = 0.0
    seed_rank: Optional[torch.Tensor] = None
    seed_cost: Optional[torch.Tensor] = None
    batch_size: int = 0
    num_seeds: int = 0
    total_cost_reshaped: Optional[torch.Tensor] = None
    solution_state: Optional[object] = None
    feasible: Optional[torch.Tensor] = None
    interpolated_trajectory: Optional[JointState] = None
    interpolated_last_tstep: Optional[torch.Tensor] = None
    interpolated_metrics: Optional[object] = None
    maximum_trajectory_dt: Optional[torch.Tensor] = None
    minimum_trajectory_dt: Optional[torch.Tensor] = None

    @property
    def motion_time(self) -> torch.Tensor:
        if self.maximum_trajectory_dt is None:
            if self.js_solution is None or self.js_solution.dt is None:
                return self.success.new_zeros(self.success.shape, dtype=torch.float32)
            dt = self.js_solution.dt
        else:
            dt = self.maximum_trajectory_dt
        horizon = 0 if self.js_solution is None else self.js_solution.position.shape[-2] - 1
        return dt * horizon

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.clone() if hasattr(value, "clone") else (
                dict(value) if isinstance(value, dict) else value
            )
        return type(self)(**values)

    def get_interpolated_plan(self):
        return self.interpolated_trajectory

    def copy_at_batch_indices(self, other, mask):
        for name in (
            "success", "solution", "position_error", "rotation_error", "cspace_error",
            "goalset_index", "seed_rank", "seed_cost", "total_cost_reshaped",
        ):
            left, right = getattr(self, name), getattr(other, name)
            if left is not None and right is not None:
                left[mask] = right[mask]

    def get_topk_seeds(self, topk: int):
        if self.seed_cost is None:
            raise ValueError("seed_cost is unavailable")
        return torch.topk(self.seed_cost, topk, dim=-1, largest=False).indices

    def copy_successful_solutions(self, other):
        self.copy_at_batch_indices(other, other.success)


__all__ = ["TrajOptSolverResult"]
