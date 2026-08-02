"""Portable nonlinear conjugate-gradient optimizer for CPU and Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from .line_search_strategy import LineSearchType
from curobo._src.optim._portable import _objective


def _cg_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    aliases = {
        "fr": "FR", "fletcher_reeves": "FR", "fletcherreeves": "FR",
        "pr": "PR", "polak_ribiere": "PR", "polakribiere": "PR",
        "dy": "DY", "dai_yuan": "DY", "daiyuan": "DY",
    }
    if normalized not in aliases:
        raise ValueError("cg_method must be one of FR, PR, or DY")
    return aliases[normalized]


def jit_cg_compute_step_direction(
    grad: torch.Tensor,
    prev_grad: torch.Tensor,
    prev_step: torch.Tensor,
    max_beta: float,
    method: str,
):
    """Return a Polak-Ribiere/Fletcher-Reeves/Dai-Yuan CG direction.

    The returned history tensors are updated out-of-place, avoiding in-place
    writes that would invalidate an autograd caller's saved values.
    """

    if grad.shape != prev_grad.shape or grad.shape != prev_step.shape:
        raise ValueError("grad, prev_grad, and prev_step must have matching shape")
    if max_beta < 0:
        raise ValueError("max_beta must be nonnegative")
    kind = _cg_method(method)
    eps = torch.finfo(grad.dtype).eps
    axes = tuple(range(1, grad.ndim))
    denominator = prev_grad.square().sum(dim=axes).clamp_min(eps)
    if kind == "FR":
        beta = grad.square().sum(dim=axes) / denominator
    elif kind == "PR":
        beta = (grad * (grad - prev_grad)).sum(dim=axes) / denominator
    else:
        dy_denom = -(prev_step * (grad - prev_grad)).sum(dim=axes)
        beta = grad.square().sum(dim=axes) / torch.where(
            dy_denom.abs() > eps, dy_denom, torch.full_like(dy_denom, eps)
        )
    beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, max_beta)
    direction = -grad + beta.reshape(beta.shape + (1,) * (grad.ndim - beta.ndim)) * prev_step
    return direction, grad.clone(), direction.clone()


def jit_cg_shift_buffers(
    prev_grad: torch.Tensor, prev_step: torch.Tensor, shift_steps: int, action_dim: int
):
    """Shift MPC history and zero-fill the newly exposed terminal actions."""

    if shift_steps < 0 or action_dim <= 0:
        raise ValueError("shift_steps must be nonnegative and action_dim positive")
    if prev_grad.shape != prev_step.shape:
        raise ValueError("prev_grad and prev_step must have matching shape")
    amount = shift_steps * action_dim
    output = []
    for value in (prev_grad, prev_step):
        shifted = torch.zeros_like(value)
        if amount < value.shape[-1]:
            shifted[..., :-amount] = value[..., amount:]
        output.append(shifted)
    return tuple(output)


@dataclass
class ConjugateGradientOptCfg(GradientDescentOptCfg):
    solver_type: str = "conjugate_gradient"
    solver_name: str = "conjugate_gradient"
    line_search_scale: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])
    line_search_type: LineSearchType | str = LineSearchType.APPROX_WOLFE
    use_cuda_kernel_line_search: bool = False
    fix_terminal_action: bool = False
    line_search_wolfe_c_1: float = 1e-5
    line_search_wolfe_c_2: float = 0.9
    initial_step_scale: float = 0.1
    cg_method: str = "FR"
    max_beta: float = 10.0
    # Earlier portable releases exposed this spelling.  Keep it as an input
    # alias while normalizing execution to pinned V2's ``cg_method``.
    beta_type: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.line_search_scale or any(value < 0 for value in self.line_search_scale):
            raise ValueError("line_search_scale must contain nonnegative values")
        if self.beta_type is not None:
            self.cg_method = self.beta_type
        self.cg_method = _cg_method(self.cg_method)
        if self.max_beta < 0:
            raise ValueError("max_beta must be nonnegative")
        self.line_search_type = LineSearchType(self.line_search_type)
        self.use_cuda_kernel_line_search = False


class ConjugateGradientOpt(GradientDescentOpt):
    """Batched nonlinear CG with deterministic fixed-candidate line search."""

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._prev_grad_q: torch.Tensor | None = None
        self._prev_step: torch.Tensor | None = None

    def _cost(self, action: torch.Tensor) -> torch.Tensor:
        value = _objective(self.rollout_fn)(action)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=action.device, dtype=action.dtype)
        if action.ndim < 3:
            return value.reshape(-1).sum().reshape(1)
        return value.reshape(action.shape[0], -1).sum(-1)

    def _candidate_step(self, action: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Include a zero step, so an invalid/non-improving candidate cannot
        # make a problem worse.  ``argmin`` supplies deterministic first ties.
        scales = torch.as_tensor(
            [0.0, *self.config.line_search_scale], device=action.device, dtype=action.dtype
        )
        candidates = action[:, None] + scales.reshape(1, -1, *([1] * (action.ndim - 1))) * direction[:, None]
        flattened = candidates.reshape((-1,) + tuple(action.shape[1:]))
        costs = self._cost(flattened).reshape(action.shape[0], scales.numel())
        selection_cost = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, torch.inf))
        index = selection_cost.argmin(dim=1)
        gather = index.reshape(-1, 1, *([1] * (action.ndim - 1))).expand(-1, 1, *action.shape[1:])
        return candidates.gather(1, gather).squeeze(1).detach(), costs.gather(1, index[:, None]).squeeze(1)

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        if not isinstance(seed_action, torch.Tensor) or seed_action.ndim < 3:
            raise ValueError("ConjugateGradientOpt expects [batch, horizon, action_dim] seed_action")
        x = seed_action.detach()
        best = x.clone()
        best_cost = self._cost(best)
        previous_gradient = None
        previous_direction = None
        trace: list[torch.Tensor] = []
        for _ in range(self.config.num_iters):
            leaf = x.detach().requires_grad_(True)
            cost = self._cost(leaf)
            gradient = torch.autograd.grad(
                torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost)).sum(),
                leaf,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(leaf)
            if previous_gradient is None:
                direction = -gradient
            else:
                direction, _previous_gradient, _previous_direction = jit_cg_compute_step_direction(
                    gradient, previous_gradient, previous_direction, self.config.max_beta, self.config.cg_method
                )
            x, current_cost = self._candidate_step(leaf.detach(), direction.detach())
            improved = torch.isfinite(current_cost) & (current_cost < best_cost)
            best = torch.where(improved.reshape(-1, 1, 1), x, best)
            best_cost = torch.where(improved, current_cost, best_cost)
            previous_gradient = gradient.detach()
            previous_direction = direction.detach()
            if self.config.store_debug:
                trace.append(best_cost.detach().clone())
        self._prev_grad_q = previous_gradient
        self._prev_step = previous_direction
        self._best_action, self._best_cost = best.clone(), best_cost.clone()
        self.debug = {"objective": tuple(trace)} if self.config.store_debug else None
        return best if self.config.return_best_action else x

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        super().reinitialize(action, mask, clear_optimizer_state, reset_num_iters)
        if clear_optimizer_state:
            self._prev_grad_q = None
            self._prev_step = None

    def shift(self, shift_steps: int = 0):
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        if self._prev_grad_q is not None and self._prev_step is not None:
            self._prev_grad_q, self._prev_step = jit_cg_shift_buffers(
                self._prev_grad_q, self._prev_step, shift_steps, self.action_dim
            )
        return True

    _shift = shift


__all__ = [
    "ConjugateGradientOptCfg", "ConjugateGradientOpt", "jit_cg_compute_step_direction",
    "jit_cg_shift_buffers",
]
