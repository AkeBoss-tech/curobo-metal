# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Levenberg-Marquardt utilities for SE(3) pose optimization.

The public contracts mirror the pinned cuRobo implementation. The math stays
in ordinary PyTorch so it remains usable on CPU and Metal; raw Warp/CUDA is not
part of this module's observable contract.
"""

from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.util.torch_util import get_profiler_decorator, get_torch_jit_decorator


@get_profiler_decorator("optim_pose_lm/compute_predicted_reduction")
@get_torch_jit_decorator(dynamic=False)
def compute_predicted_reduction(
    delta: torch.Tensor,
    Jtr: torch.Tensor,
    JtJ: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted reduction in the linearized residual cost.

    Besides the upstream single-pose shape, the contractions intentionally
    support leading batch dimensions. This is algebraically identical to the
    upstream ``dot`` expression for a single six-vector.
    """
    term1 = -(delta * Jtr).sum(dim=-1)
    term2 = -0.5 * torch.einsum("...i,...ij,...j->...", delta, JtJ, delta)
    return term1 + term2


@get_profiler_decorator("optim_pose_lm/trust_region_update")
@get_torch_jit_decorator(dynamic=False)
def trust_region_update(
    cand_n_valid: torch.Tensor,
    sum_sq_residuals: torch.Tensor,
    cand_JtJ: torch.Tensor,
    cand_Jtr: torch.Tensor,
    cand_position: torch.Tensor,
    cand_quaternion: torch.Tensor,
    best_error: torch.Tensor,
    best_sum_sq: torch.Tensor,
    best_n_valid: torch.Tensor,
    best_JtJ: torch.Tensor,
    best_Jtr: torch.Tensor,
    best_position: torch.Tensor,
    best_quaternion: torch.Tensor,
    pred_reduction: torch.Tensor,
    lambda_damping: torch.Tensor,
    n_total_valid: torch.Tensor,
    min_valid_ratio: float,
    rho_min: float,
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    inf_tensor: torch.Tensor,
    minimum_valid_count: int = 10,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply the pinned cuRobo LM acceptance and damping update.

    ``min_valid_ratio`` and ``rho_min`` are accepted for upstream compatibility;
    the pinned revision uses the non-negative trust-ratio rule below. Returning
    fresh tensors rather than mutating the ``best_*`` inputs is part of that
    observable lifecycle contract.
    """
    # Retain the signature-level compatibility parameters without changing the
    # pinned acceptance semantics.
    del n_total_valid, min_valid_ratio, rho_min

    has_enough_valid = cand_n_valid > minimum_valid_count
    cand_error = torch.where(
        has_enough_valid,
        torch.sqrt(sum_sq_residuals / (cand_n_valid + 1e-8)),
        inf_tensor,
    )
    actual_reduction = best_sum_sq - sum_sq_residuals
    trust_ratio = actual_reduction / (pred_reduction + 1e-8)
    step_accepted = torch.logical_and(trust_ratio >= 0, has_enough_valid)

    new_lambda = torch.where(
        step_accepted,
        lambda_damping / lambda_factor,
        lambda_damping * lambda_factor,
    )
    new_lambda = torch.clamp(new_lambda, lambda_min, lambda_max)

    new_best_position = torch.where(step_accepted[..., None], cand_position, best_position)
    new_best_quaternion = torch.where(
        step_accepted[..., None], cand_quaternion, best_quaternion
    )
    new_best_error = torch.where(step_accepted, cand_error, best_error)
    new_best_sum_sq = torch.where(step_accepted, sum_sq_residuals, best_sum_sq)
    new_best_n_valid = torch.where(step_accepted, cand_n_valid, best_n_valid)

    accept_mask_2d = step_accepted[..., None, None]
    new_best_JtJ = torch.where(accept_mask_2d, cand_JtJ, best_JtJ)
    accept_mask_1d = step_accepted[..., None]
    new_best_Jtr = torch.where(accept_mask_1d, cand_Jtr, best_Jtr)

    return (
        new_best_position,
        new_best_quaternion,
        new_best_error,
        new_best_sum_sq,
        new_best_n_valid,
        new_best_JtJ,
        new_best_Jtr,
        new_lambda,
    )


@get_profiler_decorator("optim_pose_lm/solve_lm_step")
@get_torch_jit_decorator(dynamic=False)
def solve_lm_step(
    JtJ: torch.Tensor,
    Jtr: torch.Tensor,
    lambda_damping: torch.Tensor,
    eye6: torch.Tensor,
) -> torch.Tensor:
    """Solve ``(JtJ + lambda I) delta = -Jtr`` by Cholesky factorization."""
    A = JtJ + lambda_damping[..., None, None] * eye6
    L, _ = torch.linalg.cholesky_ex(A)
    delta = torch.cholesky_solve((-Jtr).unsqueeze(-1), L).squeeze(-1)
    return delta
