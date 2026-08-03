"""Per-problem best-solution tracking for portable gradient optimizers."""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg


class BestTracker:
    """Track the best cost/action and convergence age for each problem.

    The buffers deliberately mirror V2's public fields while relying only on
    ordinary tensors, so they remain valid on CPU and fallback-disabled MPS.
    """

    _initial_cost = 5_000_000.0
    _reset_cost = 5_000_000_000_000.0

    def __init__(self, device_cfg: Optional[DeviceCfg] = None, *_, **__) -> None:
        self.device_cfg = device_cfg if device_cfg is not None else DeviceCfg()
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        self.cost: Optional[torch.Tensor] = None
        self.action: Optional[torch.Tensor] = None
        self.iteration: Optional[torch.Tensor] = None
        self.current_iteration: Optional[torch.Tensor] = None
        self.previous_step_direction: Optional[torch.Tensor] = None
        self.converged: Optional[torch.Tensor] = None

    # Backward-compatible aliases from the first portable surface.
    @property
    def best_cost(self) -> Optional[torch.Tensor]:
        return self.cost

    @property
    def best_action(self) -> Optional[torch.Tensor]:
        return self.action

    def resize(self, num_problems: int, action_horizon: int, action_dim: int) -> None:
        num_problems, action_horizon, action_dim = map(int, (num_problems, action_horizon, action_dim))
        if min(num_problems, action_horizon, action_dim) <= 0:
            raise ValueError("num_problems, action_horizon, and action_dim must be positive")
        kwargs = {"device": self.device_cfg.device, "dtype": self.device_cfg.dtype}
        self.cost = torch.full((num_problems,), self._initial_cost, **kwargs)
        self.action = torch.zeros((num_problems, action_horizon, action_dim), **kwargs)
        self.iteration = torch.zeros((num_problems,), device=self.device_cfg.device, dtype=torch.int32)
        self.current_iteration = torch.zeros_like(self.iteration)
        self.previous_step_direction = torch.zeros_like(self.action)
        self.converged = torch.zeros((num_problems,), device=self.device_cfg.device, dtype=torch.uint8)

    def _require_initialized(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(value is None for value in (self.cost, self.action, self.iteration, self.current_iteration, self.converged)):
            raise RuntimeError("BestTracker.resize must be called before using tracking buffers")
        assert self.cost is not None and self.action is not None and self.iteration is not None
        assert self.current_iteration is not None and self.converged is not None
        return self.cost, self.action, self.iteration, self.current_iteration, self.converged

    def clear(self, mask: Optional[torch.Tensor] = None) -> None:
        """Clear all or selected tracker entries in place."""
        cost, action, iteration, current_iteration, converged = self._require_initialized()
        if mask is None:
            mask = torch.ones_like(cost, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=cost.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(cost.shape):
                raise ValueError(f"mask must have shape {tuple(cost.shape)}")
        cost[mask] = self._reset_cost
        iteration[mask] = 0
        current_iteration[mask] = 0
        action[mask] = 0.0
        assert self.previous_step_direction is not None
        self.previous_step_direction[mask] = 0.0
        converged[mask] = 0

    def reset(self) -> None:
        self.clear()

    @staticmethod
    def _cost_vector(cost: torch.Tensor, batch: int) -> torch.Tensor:
        if not isinstance(cost, torch.Tensor):
            raise TypeError("iteration_state.cost must be a tensor")
        if cost.numel() != batch:
            raise ValueError(f"iteration_state.cost must contain {batch} values")
        return cost.reshape(batch)

    def update(
        self,
        iteration_state: OptimizationIterationState,
        action_horizon: int,
        action_dim: int,
        cost_delta_threshold: float,
        cost_relative_threshold: float,
        convergence_iteration: int,
    ) -> OptimizationIterationState:
        """Update state and component buffers with V2's strict improvement rule."""
        if not isinstance(iteration_state, OptimizationIterationState):
            raise TypeError("iteration_state must be an OptimizationIterationState")
        cost, stored_action, stored_iteration, stored_current, stored_converged = self._require_initialized()
        action_horizon, action_dim = int(action_horizon), int(action_dim)
        if tuple(stored_action.shape[1:]) != (action_horizon, action_dim):
            raise ValueError("action_horizon/action_dim do not match allocated tracker buffers")
        action = iteration_state.action
        if not isinstance(action, torch.Tensor) or action.numel() != stored_action.numel():
            raise ValueError("iteration_state.action does not match allocated tracker shape")
        action = action.reshape_as(stored_action)
        current_cost = self._cost_vector(iteration_state.cost, cost.shape[0])
        if current_cost.device != cost.device:
            raise ValueError("iteration_state.cost must be on the tracker device")

        previous_cost = iteration_state.best_cost
        previous_action = iteration_state.best_action
        previous_iteration = iteration_state.best_iteration
        previous_current = iteration_state.current_iteration
        previous_converged = iteration_state.converged
        base_cost = cost if previous_cost is None else self._cost_vector(previous_cost, cost.shape[0])
        base_action = stored_action if previous_action is None else previous_action.reshape_as(stored_action)
        base_iteration = stored_iteration if previous_iteration is None else torch.as_tensor(previous_iteration, device=cost.device, dtype=torch.int32).reshape_as(stored_iteration)
        base_current = stored_current if previous_current is None else torch.as_tensor(previous_current, device=cost.device, dtype=torch.int32).reshape_as(stored_current)
        del previous_converged

        delta = base_cost - current_cost.detach()
        relative = delta / (base_cost.abs() + 1e-8)
        improved = torch.isfinite(current_cost) & (delta > float(cost_delta_threshold)) & (relative > float(cost_relative_threshold))
        next_current = base_current + 1
        next_cost = torch.where(improved, current_cost.detach(), base_cost)
        next_action = torch.where(improved[:, None, None], action.detach(), base_action.detach())
        next_iteration = torch.where(improved, next_current, base_iteration)
        next_converged = (next_iteration + int(convergence_iteration) <= next_current)

        # Preserve V2's persistent component buffers and expose independent
        # snapshots in the returned state so callers can safely retain it.
        cost.copy_(next_cost)
        stored_action.copy_(next_action)
        stored_iteration.copy_(next_iteration)
        stored_current.copy_(next_current)
        stored_converged.copy_(next_converged.to(torch.uint8))

        iteration_state.best_cost = next_cost.clone()
        iteration_state.best_action = next_action.clone()
        iteration_state.best_iteration = next_iteration.clone()
        iteration_state.current_iteration = next_current.clone()
        iteration_state.converged = next_converged.to(torch.uint8).clone()
        return iteration_state

    @staticmethod
    def check_convergence(converged: torch.Tensor, converged_ratio: float) -> bool:
        """Return whether strictly more than the requested ratio converged."""
        if not isinstance(converged, torch.Tensor) or converged.ndim != 1:
            raise ValueError("converged must be a rank-one tensor")
        if converged.numel() == 0:
            return False
        converged_ratio = float(converged_ratio)
        if not 0.0 <= converged_ratio <= 1.0:
            raise ValueError("converged_ratio must be in [0, 1]")
        return bool(torch.count_nonzero(converged) > converged.numel() * converged_ratio)


__all__ = ["BestTracker"]
