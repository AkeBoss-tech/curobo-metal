from __future__ import annotations
import torch
from typing import List
from .optimization_config import LBFGSLaunchCfg, LineSearchLaunchCfg, OptimizationKernelCfg

LaunchConfig = get_runtime = launch_kernel = log_and_raise = None


def launch_lbfgs_step(step_vec: torch.Tensor, rho_buffer: torch.Tensor, y_buffer: torch.Tensor, s_buffer: torch.Tensor, q: torch.Tensor, grad_q: torch.Tensor, x_0: torch.Tensor, grad_0: torch.Tensor, epsilon: float, batch_size: int, history_m: int, v_dim: int, stable_mode: bool, use_shared_buffers: bool) -> List[torch.Tensor]:
    if history_m < 0 or history_m > 31:
        raise ValueError("history_m must be in [0, 31]")
    # Portable first/update step. Higher-level LBFGS owns the full history
    # lifecycle; this low-level seam preserves in-place buffer semantics.
    s = q - x_0
    y = grad_q - grad_0
    slot = 0 if history_m == 0 else int(getattr(launch_lbfgs_step, "_slot", 0) % history_m)
    if history_m:
        s_buffer[slot].copy_(s.reshape_as(s_buffer[slot]))
        y_buffer[slot].copy_(y.reshape_as(y_buffer[slot]))
        denom = (s * y).sum(dim=-1).clamp_min(torch.finfo(q.dtype).eps)
        rho_buffer[slot].copy_(denom.reciprocal().reshape_as(rho_buffer[slot]))
        launch_lbfgs_step._slot = slot + 1
    direction = -grad_q
    if stable_mode:
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(float(epsilon))
    step_vec.copy_(direction)
    x_0.copy_(q)
    grad_0.copy_(grad_q)
    return [step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]


def launch_line_search(best_cost: torch.Tensor, best_action: torch.Tensor, best_iteration: torch.Tensor, current_iteration: torch.Tensor, converged_global: torch.Tensor, convergence_iteration: int, cost_delta_threshold: float, cost_relative_threshold: float, exploration_cost: torch.Tensor, exploration_action: torch.Tensor, exploration_gradient: torch.Tensor, exploration_idx: torch.Tensor, selected_cost: torch.Tensor, selected_action: torch.Tensor, selected_gradient: torch.Tensor, selected_idx: torch.Tensor, search_cost: torch.Tensor, search_action: torch.Tensor, search_gradient: torch.Tensor, step_direction: torch.Tensor, search_magnitudes: torch.Tensor, armijo_threshold_c_1: float, curvature_threshold_c_2: float, strong_wolfe: bool, approx_wolfe: bool, n_linesearch: int, opt_dim: int, batchsize: int):
    index = search_cost.argmin(dim=-1)
    gather_action = index[..., None, None].expand(*index.shape, 1, search_action.shape[-1])
    gather_gradient = index[..., None, None].expand(*index.shape, 1, search_gradient.shape[-1])
    selected_cost.copy_(search_cost.gather(-1, index[..., None]).squeeze(-1))
    selected_action.copy_(search_action.gather(-2, gather_action).squeeze(-2))
    selected_gradient.copy_(search_gradient.gather(-2, gather_gradient).squeeze(-2))
    selected_idx.copy_(index.to(selected_idx.dtype))
    improved = selected_cost < best_cost
    best_cost.copy_(torch.where(improved, selected_cost, best_cost))
    best_action.copy_(torch.where(improved[..., None], selected_action, best_action))
    return None
