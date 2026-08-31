"""Portable nonlinear conjugate-gradient optimizer for CPU and Apple MPS."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import math
from numbers import Real
import time
from typing import Any, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from .gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from .line_search_strategy import LineSearchType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.optim._portable import _objective

# CUDA-oriented upstream reexports are declaration aliases on the portable
# backend; eagerly importing their package graph is circular here.
GradientOptCore = OptimizationIterationState = None
Rollout = None
get_torch_jit_decorator = log_and_raise = shift_buffer = None


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

    This public helper returns only the search direction, matching the
    historical cuRobo call surface.  The optimizer owns its history buffers
    and updates them separately, so callers do not receive a surprising
    tuple in place of a tensor.
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
    prev_step.copy_(direction)
    prev_grad.copy_(grad)
    return direction, prev_grad, prev_step


def jit_cg_shift_buffers(prev_grad, prev_step, shift_steps: int, action_dim: int):
    """Shift MPC history and zero-fill the newly exposed terminal actions."""

    if shift_steps < 0 or action_dim <= 0:
        raise ValueError("shift_steps must be nonnegative and action_dim positive")
    if prev_grad.shape != prev_step.shape:
        raise ValueError("prev_grad and prev_step must have matching shape")
    amount = shift_steps * action_dim
    output = []
    for value in (prev_grad, prev_step):
        shifted = torch.zeros_like(value)
        if amount == 0:
            shifted.copy_(value)
        elif amount < value.shape[-1]:
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
        if isinstance(self.line_search_scale, (str, bytes)):
            raise TypeError("line_search_scale must be a finite sequence of scales")
        try:
            self.line_search_scale = [float(value) for value in self.line_search_scale]
        except TypeError as error:
            raise TypeError("line_search_scale must be a finite sequence of scales") from error
        if (
            not self.line_search_scale
            or any(not math.isfinite(value) or value < 0.0 for value in self.line_search_scale)
        ):
            raise ValueError("line_search_scale must contain finite nonnegative values")
        if self.beta_type is not None:
            self.cg_method = self.beta_type
        self.cg_method = _cg_method(self.cg_method)
        for name in ("max_beta", "line_search_wolfe_c_1", "line_search_wolfe_c_2", "initial_step_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.max_beta < 0.0 or self.initial_step_scale < 0.0:
            raise ValueError("max_beta and initial_step_scale must be nonnegative")
        if not 0.0 <= self.line_search_wolfe_c_1 <= 1.0:
            raise ValueError("line_search_wolfe_c_1 must be in [0, 1]")
        if not 0.0 <= self.line_search_wolfe_c_2 <= 1.0:
            raise ValueError("line_search_wolfe_c_2 must be in [0, 1]")
        if not isinstance(self.fix_terminal_action, bool):
            raise TypeError("fix_terminal_action must be bool")
        if not isinstance(self.use_cuda_kernel_line_search, bool):
            raise TypeError("use_cuda_kernel_line_search must be bool")
        self.line_search_type = LineSearchType(self.line_search_type)
        # A fixed CPU/MPS tensor candidate loop substitutes for the pinned
        # CUDA kernel.  Keep the public field inspectable but never claim a
        # CUDA kernel was selected.
        self.use_cuda_kernel_line_search = False

    @property
    def num_rollout_instances(self):
        return self._num_rollout_instances

    @property
    def outer_iters(self):
        return math.ceil(self.num_iters / self.inner_iters)

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        return super().create_data_dict(data_dict, device_cfg, child_dict)

    def update_niters(self, niters: int):
        self.num_iters = niters


class _ConjugateGradientOptPortable(GradientDescentOpt):
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

    def _scale_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Apply portable V2 step and terminal-action constraints."""
        output = direction.detach().clone()
        if self.config.step_scale not in (0.0, 1.0) and self.action_horizon_step_max is not None:
            limit = torch.as_tensor(
                self.action_horizon_step_max, device=output.device, dtype=output.dtype
            )
            if bool((limit <= 0).any().item()):
                raise ValueError("action_horizon_step_max entries must be positive")
            ratio = (output.abs() / limit).reshape(output.shape[0], -1).amax(dim=-1)
            output = output / ratio.clamp_min(1.0).reshape((-1,) + (1,) * (output.ndim - 1))
        if self.config.fix_terminal_action and output.shape[-2] > 1:
            output[:, -1:] = 0.0
        return output

    def _candidate_step(
        self,
        action: torch.Tensor,
        direction: torch.Tensor,
        base_cost: torch.Tensor,
        base_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a deterministic batched Greedy/Armijo/Wolfe line search."""
        scales = torch.as_tensor(
            [0.0, *self.config.line_search_scale], device=action.device, dtype=action.dtype
        )
        candidates = action[:, None] + scales.reshape(1, -1, *([1] * (action.ndim - 1))) * direction[:, None]
        flattened = self._apply_action_bounds(
            candidates.reshape((-1,) + tuple(action.shape[1:]))
        ).detach().requires_grad_(True)
        costs = self._cost(flattened).reshape(action.shape[0], scales.numel())
        gradient = (
            torch.autograd.grad(costs.sum(), flattened, allow_unused=True)[0]
            if costs.requires_grad
            else None
        )
        if gradient is None:
            gradient = torch.zeros_like(flattened)
        candidate_gradient = gradient.reshape_as(candidates)
        selection_cost = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, torch.inf))
        if self.config.line_search_type == LineSearchType.GREEDY:
            index = selection_cost.argmin(dim=1)
        else:
            directional = (base_gradient.reshape(base_gradient.shape[0], -1) * direction.reshape(direction.shape[0], -1)).sum(-1)
            armijo = torch.isfinite(costs) & (
                costs <= base_cost[:, None] + float(self.config.line_search_wolfe_c_1) * scales[None] * directional[:, None]
            )
            candidate_directional = (
                candidate_gradient.reshape(action.shape[0], scales.numel(), -1)
                * direction[:, None].reshape(action.shape[0], 1, -1)
            ).sum(-1)
            kind = self.config.line_search_type
            if kind in (LineSearchType.WOLFE, LineSearchType.APPROX_WOLFE):
                accepted = armijo & (candidate_directional >= float(self.config.line_search_wolfe_c_2) * directional[:, None])
            elif kind in (LineSearchType.STRONG_WOLFE, LineSearchType.APPROX_STRONG_WOLFE):
                accepted = armijo & (candidate_directional.abs() <= float(self.config.line_search_wolfe_c_2) * directional.abs()[:, None])
            else:  # ARMlJO
                accepted = armijo
            # Match V2's deterministic largest-accepted-scale rule.  Zero is
            # always present as a finite fallback when the base action is valid.
            ranked = torch.where(accepted, scales[None], torch.full_like(scales[None], -torch.inf))
            index = ranked.argmax(dim=1)
        gather = index.reshape(-1, 1, *([1] * (action.ndim - 1))).expand(-1, 1, *action.shape[1:])
        return (
            flattened.reshape_as(candidates).gather(1, gather).squeeze(1).detach(),
            costs.gather(1, index[:, None]).squeeze(1).detach(),
        )

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        # Upstream's gradient core adopts the batch dimension supplied by a
        # rollout seed.  Keep that ergonomic path for lightweight callers
        # that leave ``num_problems`` at its default of one.
        if isinstance(seed_action, torch.Tensor) and self.config.num_problems != seed_action.shape[0]:
            if (
                seed_action.ndim == 3
                and tuple(seed_action.shape[-2:]) == (self.action_horizon, self.action_dim)
            ) or (
                seed_action.ndim == 2
                and seed_action.shape[-1] == self.action_horizon * self.action_dim
            ):
                self.update_num_problems(int(seed_action.shape[0]))
        original_shape = tuple(seed_action.shape) if isinstance(seed_action, torch.Tensor) else None
        x = self._apply_action_bounds(self._canonical_action(seed_action)).detach()
        if self._reinitialized_action is not None:
            if self._reinitialized_action.shape != x.shape:
                raise ValueError("reinitialized action shape no longer matches optimizer seed shape")
            if self._reinitialized_action.device != x.device or self._reinitialized_action.dtype != x.dtype:
                raise ValueError("reinitialized action must share seed device and dtype")
            x = self._reinitialized_action
            self._reinitialized_action = None
        best = x.clone()
        best_cost = self._cost(best)
        previous_cost = best_cost
        previous_gradient = self._prev_grad_q if self._prev_grad_q is not None and self._prev_grad_q.shape == x.shape and self._prev_grad_q.device == x.device and self._prev_grad_q.dtype == x.dtype else None
        previous_direction = (
            self._prev_step
            if previous_gradient is not None
            and self._prev_step is not None
            and self._prev_step.shape == x.shape
            and self._prev_step.device == x.device
            and self._prev_step.dtype == x.dtype
            else None
        )
        if previous_gradient is not None and (
            not bool(torch.isfinite(previous_gradient).all())
            or previous_direction is None
            or not bool(torch.isfinite(previous_direction).all())
        ):
            previous_gradient = previous_direction = None
        self._converged = None
        self._convergence_count = None
        self._iteration = 0
        trace: list[torch.Tensor] = []
        start = time.perf_counter()
        with torch.enable_grad():
            for iteration in range(self.config.num_iters):
                leaf = x.detach().requires_grad_(True)
                cost = self._cost(leaf)
                scalar = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost)).sum()
                gradient = (
                    torch.autograd.grad(scalar, leaf, create_graph=False, allow_unused=True)[0]
                    if scalar.requires_grad
                    else None
                )
                if gradient is None:
                    gradient = torch.zeros_like(leaf)
                if previous_gradient is None:
                    direction = -gradient
                else:
                    direction, previous_gradient, previous_direction = jit_cg_compute_step_direction(
                        gradient, previous_gradient, previous_direction, self.config.max_beta, self.config.cg_method
                    )
                direction = self._scale_direction(direction)
                x, current_cost = self._candidate_step(leaf.detach(), direction, cost.detach(), gradient.detach())
                improved = torch.isfinite(current_cost) & (current_cost < best_cost)
                best = torch.where(improved.reshape(-1, 1, 1), x, best)
                best_cost = torch.where(improved, current_cost, best_cost)
                previous_gradient = gradient.detach()
                previous_direction = direction.detach()
                if self.config.store_debug:
                    trace.append(best_cost.detach().clone())
                self._iteration = iteration + 1
                if self._should_stop(previous_cost, best_cost, self._iteration):
                    break
                previous_cost = best_cost
        if x.device.type == "mps":
            torch.mps.synchronize()
        self.opt_dt = time.perf_counter() - start
        self._prev_grad_q = previous_gradient
        self._prev_step = previous_direction
        self._best_action, self._best_cost = best.clone(), best_cost.clone()
        self.debug = (
            {"objective": tuple(trace), "converged": self._converged.detach().clone() if self._converged is not None else None}
            if self.config.store_debug else None
        )
        current_finite = torch.isfinite(self._cost(x))
        current = torch.where(current_finite.reshape((-1,) + (1,) * (x.ndim - 1)), x, best)
        output = best if self.config.return_best_action else current
        assert original_shape is not None
        return output.reshape(original_shape)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        super().reinitialize(action, mask, clear_optimizer_state, reset_num_iters)
        if clear_optimizer_state:
            self._prev_grad_q = None
            self._prev_step = None

    def reset(self):
        super().reset()
        self._prev_grad_q = None
        self._prev_step = None

    def reset_shape(self):
        super().reset_shape()
        self._prev_grad_q = None
        self._prev_step = None

    def update_num_problems(self, num_problems: int):
        super().update_num_problems(num_problems)
        self._prev_grad_q = None
        self._prev_step = None

    def shift(self, shift_steps: int = 0):
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        super().shift(shift_steps)
        if self._prev_grad_q is not None and self._prev_step is not None:
            shape = self._prev_grad_q.shape
            gradient, step = jit_cg_shift_buffers(
                self._prev_grad_q.reshape(shape[0], 1, -1),
                self._prev_step.reshape(shape[0], 1, -1),
                shift_steps,
                self.action_dim,
            )
            self._prev_grad_q, self._prev_step = gradient.reshape(shape), step.reshape(shape)
        return True

    _shift = shift


class ConjugateGradientOpt(_ConjugateGradientOptPortable):
    """Pinned declaration façade rebound to the portable CG implementation."""

    def __init__(self, config: ConjugateGradientOptCfg, rollout_list: List[Rollout], use_cuda_graph: bool = False): pass
    def action_bound_highs(self): pass
    def action_bound_lows(self): pass
    def action_dim(self): pass
    def action_horizon(self): pass
    def action_step_max(self): pass
    def compute_metrics(self, action): pass
    def config(self): pass
    def debug_dump(self, file_path=""): pass
    def device_cfg(self): pass
    def disable(self): pass
    def enable(self): pass
    def enabled(self): pass
    def get_all_rollout_instances(self): pass
    def get_recorded_trace(self): pass
    def horizon(self): pass
    def opt_dim(self): pass
    def opt_dt(self, value): pass
    def optimize(self, seed_action): pass
    def outer_iters(self): pass
    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False): pass
    def reset_cuda_graph(self): pass
    def reset_seed(self): pass
    def reset_shape(self): pass
    def rollout_fn(self): pass
    def shift(self, shift_steps=0): pass
    def solve_time(self): pass
    def solver_names(self): pass
    def update_goal_dt(self, goal): pass
    def update_niters(self, niters): pass
    def update_num_problems(self, num_problems): pass
    def update_rollout_params(self, goal): pass
    def update_solver_params(self, solver_params): pass
    def use_cuda_graph(self): pass


def _install_portable_cg_runtime():
    for base in reversed(_ConjugateGradientOptPortable.__mro__):
        for name, value in base.__dict__.items():
            if not (name.startswith("__") and name != "__init__"):
                setattr(ConjugateGradientOpt, name, value)


_install_portable_cg_runtime()


__all__ = [
    "ConjugateGradientOptCfg", "ConjugateGradientOpt", "jit_cg_compute_step_direction",
    "jit_cg_shift_buffers",
]
