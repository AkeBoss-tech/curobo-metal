"""Small, device-preserving optimizer debug trace recorder."""

from __future__ import annotations

from typing import Any

import torch

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState


class _DebugTrace(list[OptimizationIterationState]):
    """List-compatible legacy trace with V2 dictionary-style field access.

    Earlier portable solvers consumed a list of iteration states; current V2
    exposes action/cost arrays in a mapping.  Supporting both access forms
    prevents a recorder upgrade from changing an optimizer's public trace.
    """

    def __init__(self, states, actions, costs) -> None:
        super().__init__(states)
        self._actions = actions
        self._costs = costs

    def __getitem__(self, item):
        if item == "debug":
            return self._actions
        if item == "debug_cost":
            return self._costs
        return super().__getitem__(item)

    def keys(self):
        return ("debug", "debug_cost")

    def items(self):
        return (("debug", self._actions), ("debug_cost", self._costs))

    def values(self):
        return (self._actions, self._costs)

    def get(self, key, default=None):
        return self[key] if key in {"debug", "debug_cost"} else default


class DebugRecorder:
    """Record immutable action/cost snapshots for each optimization iteration.

    This is deliberately eager-PyTorch state, not CUDA graph debug storage.
    Snapshots are detached to avoid retaining entire autograd graphs while an
    optimizer is used over many MPC iterations.
    """

    def __init__(self) -> None:
        self.clear()

    def record(
        self,
        iteration_state: OptimizationIterationState,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        if not isinstance(iteration_state, OptimizationIterationState):
            raise TypeError("iteration_state must be an OptimizationIterationState")
        action_horizon, action_dim = int(action_horizon), int(action_dim)
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        action = iteration_state.action
        if not isinstance(action, torch.Tensor):
            raise TypeError("iteration_state.action must be a tensor")
        elements = action_horizon * action_dim
        if action.numel() % elements:
            raise ValueError("action cannot be reshaped into [batch, action_horizon, action_dim]")
        action_snapshot = action.reshape(-1, action_horizon, action_dim).detach().clone()
        self.actions.append(action_snapshot)
        cost_snapshot = None
        if iteration_state.cost is not None:
            if not isinstance(iteration_state.cost, torch.Tensor):
                raise TypeError("iteration_state.cost must be a tensor or None")
            cost_snapshot = iteration_state.cost.detach().clone()
            self.costs.append(cost_snapshot)
        self._states.append(OptimizationIterationState(action=action_snapshot, cost=cost_snapshot))

    def clear(self) -> None:
        self.actions: list[torch.Tensor] = []
        self.costs: list[torch.Tensor] = []
        self._states: list[OptimizationIterationState] = []

    @property
    def trace(self) -> list[OptimizationIterationState]:
        """Legacy state-shaped view of the independently stored snapshots."""
        return list(self._states)

    def get_trace(self) -> Any:
        """Return V2 fields and retain legacy list iteration compatibility."""
        return _DebugTrace(self._states, self.actions, self.costs)


__all__ = ["DebugRecorder"]
