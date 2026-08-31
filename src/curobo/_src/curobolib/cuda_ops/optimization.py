"""Autograd-compatible portable implementations of cuRobo's CUDA optimizer seams."""

from __future__ import annotations

import torch
from torch.autograd import Function

from curobo._src.curobolib.backends import optimization as optimization_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int32_tensors,
    check_uint8_tensors,
)
from curobo._src.util.logging import log_and_raise


def wolfe_line_search(
    iteration_state: OptimizationIterationState,
    line_search_context: LineSearchContext,
    exploration_idx: torch.Tensor,
    selected_idx: torch.Tensor,
    search_cost: torch.Tensor,
    search_action: torch.Tensor,
    search_gradient: torch.Tensor,
    step_direction: torch.Tensor,
    strong_wolfe: bool,
    approx_wolfe: bool,
):
    """Update an upstream-shaped iteration state using the portable line search."""
    optimization_cu.launch_line_search(
        iteration_state.best_cost,
        iteration_state.best_action,
        iteration_state.best_iteration,
        iteration_state.current_iteration,
        iteration_state.converged,
        line_search_context.convergence_iteration,
        line_search_context.cost_delta_threshold,
        line_search_context.cost_relative_threshold,
        iteration_state.exploration_cost,
        iteration_state.exploration_action,
        iteration_state.exploration_gradient,
        exploration_idx.view(-1),
        iteration_state.cost,
        iteration_state.action,
        iteration_state.gradient,
        selected_idx.view(-1),
        search_cost,
        search_action,
        search_gradient,
        step_direction,
        line_search_context.line_search_scale,
        line_search_context.line_search_c_1,
        line_search_context.line_search_c_2,
        strong_wolfe,
        approx_wolfe,
        line_search_context.n_linesearch,
        line_search_context.opt_dim,
        line_search_context.num_problems,
    )
    return iteration_state, exploration_idx, selected_idx


class LBFGScu(Function):
    """Upstream-signature L-BFGS step backed by ordinary PyTorch tensors."""

    @staticmethod
    def forward(
        ctx,
        step_vec,
        rho_buffer,
        y_buffer,
        s_buffer,
        q,
        grad_q,
        x_0,
        grad_0,
        epsilon=0.1,
        stable_mode=False,
        use_shared_buffers=True,
    ):
        m, b, v_dim, _ = y_buffer.shape
        return optimization_cu.launch_lbfgs_step(
            step_vec,
            rho_buffer,
            y_buffer,
            s_buffer,
            q,
            grad_q,
            x_0,
            grad_0,
            epsilon,
            b,
            m,
            v_dim,
            stable_mode,
            use_shared_buffers,
        )[0].view_as(step_vec)

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 11


# Keep the pinned annotation exports while deferring optimizer-package
# initialization until the callable implementations above are available. This
# prevents a clean interpreter from cycling through optim.__init__ -> lbfgs ->
# this partially initialized module before LBFGScu exists.
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
