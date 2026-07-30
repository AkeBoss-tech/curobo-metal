from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import torch

from curobo._src.state.state_joint import JointState


@dataclass
class BaseSolverResult:
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
    metrics: object = None
    position_tolerance: float = 0.0
    orientation_tolerance: float = 0.0
    seed_rank: Optional[torch.Tensor] = None
    seed_cost: Optional[torch.Tensor] = None
    batch_size: int = 0
    num_seeds: int = 0
    total_cost_reshaped: Optional[torch.Tensor] = None
    solution_state: object = None
    feasible: Optional[torch.Tensor] = None

    def clone(self):
        values = {}
        for key, value in self.__dict__.items():
            values[key] = value.clone() if hasattr(value, "clone") else (
                dict(value) if isinstance(value, dict) else value
            )
        return type(self)(**values)

    def copy_successful_solutions(self, other):
        mask = other.success
        self.success[mask] = True
        for name in ("solution", "position_error", "rotation_error", "goalset_index", "seed_cost"):
            left, right = getattr(self, name), getattr(other, name)
            if left is not None and right is not None:
                left[mask] = right[mask]

    def copy_at_batch_indices(self, other, mask):
        for name in ("success", "solution", "position_error", "rotation_error", "goalset_index"):
            left, right = getattr(self, name), getattr(other, name)
            if left is not None and right is not None:
                left[mask] = right[mask]


__all__ = ["BaseSolverResult"]
