from __future__ import annotations
import torch
from .optimization_config import LBFGSLaunchCfg, LineSearchLaunchCfg, OptimizationKernelCfg


def launch_lbfgs_step(step_vec, rho_buffer, y_buffer, s_buffer, q, grad_q, x_0, grad_0, epsilon, batch_size, history_m, v_dim, stable_mode, use_shared_buffers):
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


def launch_line_search(best_cost, best_action, best_iteration, current_iteration, converged_global, convergence_iteration, cost_delta_threshold, cost_relative_threshold, exploration_cost, exploration_action, exploration_gradient, exploration_idx, selected_cost, selected_action, selected_gradient, selected_idx, search_cost, search_action, search_gradient, step_direction, search_magnitudes, armijo_threshold_c_1, curvature_threshold_c_2, strong_wolfe, approx_wolfe, n_linesearch, opt_dim, batchsize):
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
