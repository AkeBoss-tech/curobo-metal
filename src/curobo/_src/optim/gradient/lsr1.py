"""Portable limited-memory SR1 optimizer compatibility surface.

The pinned implementation combines a CUDA-oriented gradient core with raw
quasi-Newton buffers.  This module instead shares the eager, batched lifecycle
from :class:`LBFGSOpt` and substitutes its inverse-Hessian direction with the
SR1 rank-one update.  No CUDA graph or Warp ABI is emulated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from .lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.util.logging import log_info

# Declaration aliases avoid importing the CUDA-oriented components package
# during portable optimizer package initialization.
GradientOptCore = OptimizationIterationState = None
Rollout = None
get_torch_jit_decorator = None


def _history_rows(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    """Normalize supported portable/upstream-like history layouts to [B, M, N]."""
    if value.ndim == 3 and value.shape[0] == batch:
        return value.reshape(batch, value.shape[1], value.shape[2])
    if value.ndim == 4 and value.shape[1] == batch:
        # Upstream kernel layout: [M, B, N, 1].
        return value.permute(1, 0, 2, 3).reshape(batch, value.shape[0], value.shape[2] * value.shape[3])
    raise ValueError(f"{name} must have shape [B, M, N] or [M, B, N, 1]")


def jit_lsr1_compute_step_direction(
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    grad: torch.Tensor,
    m: int,
    epsilon: float,
    stable_mode: bool,
    hessian_0: torch.Tensor,
):
    """Return a finite batched L-SR1 inverse-Hessian search direction.

    Histories may be portable ``[B, M, N]`` tensors or the pinned kernel's
    ``[M, B, N, 1]`` layout.  The result preserves the shape of ``grad``.
    Pairs with an ill-conditioned SR1 denominator are skipped independently
    for each problem, which is essential for deterministic CPU/MPS batches.
    """
    if not isinstance(m, int) or isinstance(m, bool) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not isinstance(stable_mode, bool) or not stable_mode:
        raise ValueError("stable_mode must be true")
    if grad.ndim < 2:
        raise ValueError("grad must have a batch dimension and event dimensions")
    batch = grad.shape[0]
    flat_grad = grad.reshape(batch, -1)
    y = _history_rows(y_buffer, batch, "y_buffer")
    s = _history_rows(s_buffer, batch, "s_buffer")
    if y.shape != s.shape or y.shape[-1] != flat_grad.shape[-1]:
        raise ValueError("history and gradient dimensions must agree")
    if hessian_0.numel() not in (1, batch):
        raise ValueError("hessian_0 must be scalar or have one value per problem")
    scale = hessian_0.to(device=grad.device, dtype=grad.dtype).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(batch)
    if not torch.isfinite(scale).all():
        raise ValueError("hessian_0 must be finite")

    history = min(m, y.shape[1])
    # Pick the newest useful pair for the customary scalar initial Hessian.
    gamma = scale.clone()
    selected = torch.zeros(batch, dtype=torch.bool, device=grad.device)
    for index in range(history):
        sy = (s[:, index] * y[:, index]).sum(-1)
        yy = y[:, index].square().sum(-1)
        valid = (~selected) & torch.isfinite(sy) & torch.isfinite(yy) & (sy > epsilon) & (yy > epsilon)
        gamma = torch.where(valid, scale * (sy / yy.clamp_min(epsilon)), gamma)
        selected |= valid
    result = gamma[:, None] * flat_grad
    for index in range(history):
        current_y, current_s = y[:, index], s[:, index]
        u = current_s - gamma[:, None] * current_y
        denominator = (u * current_y).sum(-1)
        numerator = (u * flat_grad).sum(-1)
        valid = torch.isfinite(denominator) & torch.isfinite(numerator) & (denominator.abs() > epsilon)
        update = torch.where(
            valid[:, None],
            u * (numerator / torch.where(valid, denominator, torch.ones_like(denominator)))[:, None],
            torch.zeros_like(u),
        )
        result = result + update
    return (-result).reshape_as(grad)


class _LSR1OptPortable(LBFGSOpt):
    """Batched eager L-SR1 with independent per-problem rank-one history."""

    strategy = "lsr1"

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        # A limited-memory method cannot profitably retain more independent
        # pairs than action dimensions.  Match the pinned lifecycle by
        # reducing an oversized configured history before buffers are used.
        opt_dim = int(getattr(rollout_list[0], "action_horizon", 1)) * int(
            getattr(rollout_list[0], "action_dim", 1)
        )
        if config.history >= opt_dim:
            config.history = max(1, opt_dim - 1)
        config.use_cuda_kernel_step_direction = False
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._hessian_0: torch.Tensor | None = None
        self._qn = QuasiNewtonBuffers(self.device_cfg, self.config.history)
        self._qn.resize(self.config.num_problems, self.opt_dim)

    def _record_pair(self, action: torch.Tensor, gradient: torch.Tensor) -> None:
        current_action = action.detach().reshape(action.shape[0], -1)
        current_gradient = gradient.detach().reshape(gradient.shape[0], -1)
        if (
            self._reference_action is None
            or self._reference_gradient is None
            or self._reference_action.shape != current_action.shape
        ):
            self._reference_action = current_action.clone()
            self._reference_gradient = current_gradient.clone()
            return
        s = current_action - self._reference_action
        y = current_gradient - self._reference_gradient
        moved = torch.linalg.vector_norm(s, dim=-1) > self.config.epsilon
        finite = torch.isfinite(s).all(-1) & torch.isfinite(y).all(-1)
        # SR1 allows indefinite curvature.  Screen only its own denominator
        # under the initial diagonal model rather than applying BFGS's
        # positive-curvature rule.
        u = s - y
        denominator = (u * y).sum(-1)
        valid = finite & moved & (denominator.abs() > self.config.epsilon)
        safe_s = torch.where(valid[:, None], s, torch.zeros_like(s))
        safe_y = torch.where(valid[:, None], y, torch.zeros_like(y))
        marker = valid.to(dtype=current_action.dtype)
        if self._s_history is None or self._s_history.shape[0] != action.shape[0]:
            self._s_history, self._y_history, self._rho_history = safe_s[:, None], safe_y[:, None], marker[:, None]
        else:
            self._s_history = torch.cat((safe_s[:, None], self._s_history), dim=1)[:, : self.config.history]
            self._y_history = torch.cat((safe_y[:, None], self._y_history), dim=1)[:, : self.config.history]
            self._rho_history = torch.cat((marker[:, None], self._rho_history), dim=1)[:, : self.config.history]
        self._reference_action = current_action.clone()
        self._reference_gradient = current_gradient.clone()

    def _two_loop(self, gradient: torch.Tensor) -> torch.Tensor:
        if self._s_history is None or self._y_history is None:
            return -gradient.detach()
        hessian = self._hessian_0
        if hessian is None or hessian.shape[0] != gradient.shape[0] or hessian.device != gradient.device:
            hessian = torch.ones((gradient.shape[0], 1, 1), dtype=gradient.dtype, device=gradient.device)
            self._hessian_0 = hessian
        return jit_lsr1_compute_step_direction(
            self._y_history,
            self._s_history,
            gradient,
            self.config.history,
            self.config.epsilon,
            self.config.stable_mode,
            hessian,
        )

    def _clear_history(self, mask: torch.Tensor | None = None) -> None:
        super()._clear_history(mask)
        if mask is None:
            self._hessian_0 = None

    def update_num_problems(self, num_problems):
        super().update_num_problems(num_problems)
        self._hessian_0 = None
        if hasattr(self, "_qn"):
            self._qn.resize(num_problems, self.opt_dim)

    def reset_shape(self):
        super().reset_shape()
        self._hessian_0 = None


class LSR1Opt(_LSR1OptPortable):
    """Pinned declaration façade rebound to the portable L-SR1 lifecycle."""

    def __init__(self, config: LBFGSOptCfg, rollout_list: List[Rollout], use_cuda_graph: bool = False): pass
    def action_bound_highs(self): pass
    def action_bound_lows(self): pass
    def action_dim(self): pass
    def action_horizon(self): pass
    def action_horizon_bounds_highs(self): pass
    def action_horizon_bounds_lows(self): pass
    def action_horizon_step_max(self): pass
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


def _install_portable_lsr1_runtime():
    for base in reversed(_LSR1OptPortable.__mro__):
        for name, value in base.__dict__.items():
            if not (name.startswith("__") and name != "__init__"):
                setattr(LSR1Opt, name, value)


_install_portable_lsr1_runtime()


__all__ = ["LSR1Opt", "jit_lsr1_compute_step_direction"]
