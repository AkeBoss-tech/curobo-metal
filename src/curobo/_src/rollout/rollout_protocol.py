"""Typed portable rollout contract shared by solvers and optimizers.

The protocol deliberately models the *Python* rollout boundary rather than
CUDA Graph capture or Warp buffers.  A conforming implementation may use
ordinary eager PyTorch on CPU or MPS, while callers retain the V2 action,
metrics, reset, and sampling lifecycle without device-specific duck typing.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import torch

from curobo._src.rollout.metrics import (
    CostCollection,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.state.state_joint import JointState


__all__ = [
    "Rollout",
    "RolloutResult",
    "RolloutMetrics",
    "CostsAndConstraints",
    "CostCollection",
]


@runtime_checkable
class Rollout(Protocol):
    """Structural contract for an optimizer-driven action rollout.

    This matches the pinned V2 public contract.  ``use_cuda_graph`` is
    intentionally absent: on Metal the corresponding executor is a
    shape-stable eager lifecycle owned by individual rollout implementations,
    not a portable raw-CUDA requirement.
    """

    @property
    def action_dim(self) -> int:
        """Number of action dimensions at each timestep."""
        ...

    @property
    def action_horizon(self) -> int:
        """Number of timesteps in an action sequence."""
        ...

    @property
    def action_bound_lows(self) -> torch.Tensor:
        """Lower action bounds on the rollout device."""
        ...

    @property
    def action_bound_highs(self) -> torch.Tensor:
        """Upper action bounds on the rollout device."""
        ...

    @property
    def dt(self) -> float:
        """Integration timestep in seconds."""
        ...

    @property
    def sum_horizon(self) -> bool:
        """Whether cost terms are reduced across the time horizon."""
        ...

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutResult:
        """Simulate an action sequence and return state plus cost terms."""
        ...

    def compute_metrics_from_state(
        self, state: JointState, **kwargs
    ) -> RolloutMetrics:
        """Compute feasibility, convergence, costs, and constraints for a state."""
        ...

    def compute_metrics_from_action(
        self, act_seq: torch.Tensor, **kwargs
    ) -> RolloutMetrics:
        """Simulate actions and compute metrics for their resulting state."""
        ...

    def update_params(self, **kwargs) -> bool:
        """Update targets or rollout parameters for the next solve."""
        ...

    def update_batch_size(self, batch_size: int) -> None:
        """Resize persistent state for ``batch_size`` parallel problems."""
        ...

    def update_dt(
        self, dt: Union[float, torch.Tensor], **kwargs
    ) -> bool:
        """Update the scalar or per-problem integration timestep."""
        ...

    def reset(
        self, reset_problem_ids: Optional[torch.Tensor] = None, **kwargs
    ) -> bool:
        """Reset all, or selected, problem-local state."""
        ...

    def reset_shape(self) -> bool:
        """Discard shape-dependent cached rollout state."""
        ...

    def reset_seed(self) -> None:
        """Reset the deterministic action sampler."""
        ...
