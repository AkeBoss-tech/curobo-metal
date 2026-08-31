# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from curobo._src.util.tensor_util import shift_buffer
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator(only_valid_for_compile=True)
def jit_lbfgs_compute_step_direction(
    alpha_buffer: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    grad_q: torch.Tensor,
    m: int,
    epsilon: float,
    stable_mode: bool = True,
):
    # Older portable callers stored history as ``[batch, m, dof]`` while the
    # pinned CUDA implementation stores ``[m, batch, dof]``.  Retain that
    # observable CPU/MPS contract without changing the pinned public shape.
    if alpha_buffer.shape[0] < m and s_buffer.shape[1] >= m:
        q = grad_q.clone()
        alphas = []
        for i in range(m):
            alpha = rho_buffer[:, i] * (s_buffer[:, i] * q).sum(-1)
            alphas.append(alpha)
            q = q - alpha[:, None] * y_buffer[:, i]
        sy = (s_buffer[:, 0] * y_buffer[:, 0]).sum(-1)
        yy = y_buffer[:, 0].square().sum(-1)
        gamma = torch.where(
            yy > epsilon,
            sy / yy.clamp_min(epsilon),
            torch.ones_like(yy),
        )
        q = q * gamma[:, None]
        for i in reversed(range(m)):
            beta = rho_buffer[:, i] * (y_buffer[:, i] * q).sum(-1)
            q = q + s_buffer[:, i] * (alphas[i] - beta)[:, None]
        return -q

    grad_q = grad_q.transpose(-1, -2)
    gq = grad_q.detach().clone()
    rho_s = rho_buffer * s_buffer.transpose(-1, -2)
    for i in range(m - 1, -1, -1):
        alpha_buffer[i] = rho_s[i] @ gq
        gq = gq - (alpha_buffer[i] * y_buffer[i])

    numerator = s_buffer[-1].transpose(-1, -2) @ y_buffer[-1]
    var1 = numerator / (y_buffer[-1].transpose(-1, -2) @ y_buffer[-1])
    if stable_mode:
        var1 = torch.nan_to_num(var1, epsilon, epsilon, epsilon)
    gamma = torch.nn.functional.relu(var1)
    r = gamma * gq.clone()
    rho_y = rho_buffer * y_buffer.transpose(-1, -2)
    for i in range(m):
        r = r + (alpha_buffer[i] - (rho_y[i] @ r)) * s_buffer[i]
    return -1.0 * r


@get_torch_jit_decorator(only_valid_for_compile=True)
def jit_lbfgs_update_buffers(
    q,
    grad_q,
    s_buffer,
    y_buffer,
    rho_buffer,
    x_0,
    grad_0,
    stable_mode: bool,
):
    grad_q = grad_q.transpose(-1, -2)
    q = q.unsqueeze(-1)
    y = grad_q - grad_0
    s = q - x_0
    rho = 1 / (y.transpose(-1, -2) @ s)
    if stable_mode:
        rho = torch.nan_to_num(rho, 0.0, 0.0, 0.0)
    s_buffer[0] = s
    s_buffer[:] = torch.roll(s_buffer, -1, dims=0)
    y_buffer[0] = y
    y_buffer[:] = torch.roll(y_buffer, -1, dims=0)
    rho_buffer[0] = rho
    rho_buffer[:] = torch.roll(rho_buffer, -1, dims=0)
    x_0.copy_(q)
    grad_0.copy_(grad_q)
    return s_buffer, y_buffer, rho_buffer, x_0, grad_0


@get_torch_jit_decorator()
def lbfgs_shift_buffers_jit(
    x_0,
    grad_0,
    y_buffer,
    s_buffer,
    shift_steps: int,
    action_dim: int,
):
    shift_d = shift_steps * action_dim
    x_0.copy_(shift_buffer(x_0, shift_d, action_dim, shift_steps))
    grad_0.copy_(shift_buffer(grad_0, shift_d, action_dim, shift_steps))
    y_buffer.copy_(shift_buffer(y_buffer, shift_d, action_dim, shift_steps))
    s_buffer.copy_(shift_buffer(s_buffer, shift_d, action_dim, shift_steps))
    return x_0, grad_0, y_buffer, s_buffer


@get_torch_jit_decorator()
def lbfgs_reset_jit(
    x_0,
    grad_0,
    s_buffer,
    y_buffer,
    rho_buffer,
    alpha_buffer,
    step_q_buffer,
):
    x_0.fill_(0.0)
    grad_0.fill_(0.0)
    step_q_buffer.fill_(0.0)
    s_buffer.fill_(0.0)
    y_buffer.fill_(0.0)
    rho_buffer.fill_(0.0)
    alpha_buffer.fill_(0.0)
    return x_0, grad_0, s_buffer, y_buffer, rho_buffer, alpha_buffer, step_q_buffer


@get_torch_jit_decorator()
def lbfgs_reset_problem_ids_jit(
    x_0,
    grad_0,
    s_buffer,
    y_buffer,
    rho_buffer,
    alpha_buffer,
    step_q_buffer,
    reset_problem_ids,
):
    x_0[reset_problem_ids] = 0.0
    grad_0[reset_problem_ids] = 0.0
    step_q_buffer[reset_problem_ids] = 0.0
    s_buffer[:, reset_problem_ids] = 0.0
    y_buffer[:, reset_problem_ids] = 0.0
    rho_buffer[:, reset_problem_ids] = 0.0
    alpha_buffer[:, reset_problem_ids] = 0.0
    return x_0, grad_0, s_buffer, y_buffer, rho_buffer, alpha_buffer, step_q_buffer
