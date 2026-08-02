"""Portable scaled gradient descent compatible with cuRobo V2's public API.

The original implementation has CUDA graph and rollout-buffer special cases.
This version deliberately keeps the ordinary tensor/autograd behaviour on CPU
and MPS and rejects graph capture through :class:`PortableOptimizer`.
"""

from dataclasses import dataclass
import math

import torch

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer, _objective


@dataclass
class GradientDescentOptCfg(PortableOptCfg):
    solver_type: str = "gradient_descent"
    solver_name: str = "gradient_descent"
    inner_iters: int = 25
    step_scale: float = 1.0
    fixed_iters: bool = True
    cost_convergence: float = 1.0e-11
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    convergence_iteration: int = 0
    minimum_iters: int | None = None
    return_best_action: bool = True
    gradient_descent_step_scale: float = 0.001

    def __post_init__(self) -> None:
        if self.num_particles is None:
            self.num_particles = 1
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if self.inner_iters <= 0:
            raise ValueError("inner_iters must be positive")
        if self.cost_relative_threshold >= 1.0:
            raise ValueError("cost_relative_threshold must be less than 1.0")
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0

    @property
    def outer_iters(self) -> int:
        return math.ceil(self.num_iters / self.inner_iters)


class GradientDescentOpt(PortableOptimizer):
    """Autograd gradient descent with V2's best-action return policy."""

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._best_action: torch.Tensor | None = None
        self._best_cost: torch.Tensor | None = None

    def _objective_value(self, action: torch.Tensor) -> torch.Tensor:
        value = _objective(self.rollout_fn)(action)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=action.device, dtype=action.dtype)
        # A rollout commonly retains horizon costs.  Every leading item is an
        # independent problem; only trailing cost dimensions are reduced.
        while value.ndim > max(0, action.ndim - 2):
            value = value.sum(dim=-1)
        return value

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        if not isinstance(seed_action, torch.Tensor):
            raise TypeError("seed_action must be a torch.Tensor")
        objective = _objective(self.rollout_fn)
        x = seed_action
        best = x.detach().clone()
        best_cost = self._objective_value(best)
        previous_cost = best_cost
        # ``step_scale`` was accepted by the earlier portable facade and is
        # still used by applications that configure all optimizers uniformly.
        # Preserve that behaviour when it was explicitly changed; otherwise
        # use V2's dedicated GD multiplier.
        step = (
            float(self.config.step_scale)
            if float(self.config.step_scale) != 1.0
            else float(self.config.gradient_descent_step_scale)
        )
        active = torch.isfinite(best_cost)
        iterations: list[torch.Tensor] = []
        for _ in range(self.config.num_iters):
            leaf = x.detach().requires_grad_(True)
            value = objective(leaf)
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=leaf.device, dtype=leaf.dtype)
            scalar = torch.where(torch.isfinite(value), value, torch.zeros_like(value)).sum()
            gradient = torch.autograd.grad(scalar, leaf, create_graph=torch.is_grad_enabled(), allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(leaf)
            candidate = (leaf - step * gradient).detach()
            candidate_cost = self._objective_value(candidate)
            finite = torch.isfinite(candidate_cost)
            while finite.ndim < candidate.ndim:
                finite = finite.unsqueeze(-1)
            x = torch.where(finite, candidate, x.detach())
            reduced = self._objective_value(x)
            improved = torch.isfinite(reduced) & (reduced < best_cost)
            item_mask = improved
            while item_mask.ndim < best.ndim:
                item_mask = item_mask.unsqueeze(-1)
            best = torch.where(item_mask, x, best)
            best_cost = torch.where(improved, reduced, best_cost)
            active = active & torch.isfinite(reduced)
            if self.config.store_debug:
                iterations.append(best_cost.detach().clone())
            if not self.config.fixed_iters:
                delta = (previous_cost - best_cost).abs()
                threshold = max(float(self.config.cost_convergence), float(self.config.cost_delta_threshold))
                if bool((delta <= threshold).all().item()):
                    break
                previous_cost = best_cost
        self._best_action = best.detach().clone()
        self._best_cost = best_cost.detach().clone()
        self.debug = {"objective": tuple(iterations)} if self.config.store_debug else None
        return best if self.config.return_best_action else x

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        super().reinitialize(action, mask, clear_optimizer_state, reset_num_iters)
        if reset_num_iters:
            self.config.update_niters(self.config.num_iters)
        self._best_action = None
        self._best_cost = None

    def get_recorded_trace(self):
        return self.debug if self.debug is not None else {"debug": [], "debug_cost": []}

    def update_solver_params(self, solver_params):
        if self.config.solver_name not in solver_params:
            raise ValueError(f"Optimizer {self.config.solver_name} not found in {solver_params}")
        for name, value in solver_params[self.config.solver_name].items():
            setattr(self.config, name, value)
        return True

    def reset_seed(self):
        return True

    def reset_shape(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_shape", None)
            if callable(callback):
                callback()

    def reset_cuda_graph(self):
        # No graph is captured on portable devices.  Keep the upstream
        # lifecycle method so consumers can safely reset either backend.
        return None

    def debug_dump(self, file_path=""):
        del file_path
        return None


class LineSearchGradientDescentOpt(GradientDescentOpt):
    """Legacy V2 name; portable GD intentionally does not capture a graph."""
