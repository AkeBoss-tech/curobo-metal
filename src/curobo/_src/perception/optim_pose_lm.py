"""Small, differentiable Levenberg-Marquardt primitives."""

import torch


def compute_predicted_reduction(delta, Jtr, JtJ):
    return -(delta * Jtr).sum(-1) - 0.5 * torch.einsum(
        "...i,...ij,...j->...", delta, JtJ, delta
    )


def solve_lm_step(JtJ, Jtr, lambda_damping, eye6):
    damping = lambda_damping
    while damping.ndim < JtJ.ndim:
        damping = damping.unsqueeze(-1)
    return torch.linalg.solve(JtJ + damping * eye6, -Jtr.unsqueeze(-1)).squeeze(-1)


def trust_region_update(
    cand_n_valid, sum_sq_residuals, cand_JtJ, cand_Jtr, cand_position,
    cand_quaternion, best_error, best_sum_sq, best_n_valid, best_JtJ, best_Jtr,
    best_position, best_quaternion, pred_reduction, lambda_damping,
    n_total_valid, min_valid_ratio, rho_min, lambda_factor, lambda_min,
    lambda_max, inf_tensor, minimum_valid_count=10,
):
    valid = cand_n_valid >= torch.maximum(
        torch.as_tensor(minimum_valid_count, device=cand_n_valid.device),
        (n_total_valid * min_valid_ratio).to(cand_n_valid.dtype),
    )
    error = torch.where(valid, sum_sq_residuals / cand_n_valid.clamp_min(1), inf_tensor)
    actual = best_error - error
    rho = actual / pred_reduction.clamp_min(torch.finfo(error.dtype).eps)
    accept = valid & (rho > rho_min) & (error < best_error)
    for candidate, best in (
        (error, best_error), (sum_sq_residuals, best_sum_sq),
        (cand_n_valid, best_n_valid), (cand_JtJ, best_JtJ),
        (cand_Jtr, best_Jtr), (cand_position, best_position),
        (cand_quaternion, best_quaternion),
    ):
        mask = accept
        while mask.ndim < candidate.ndim:
            mask = mask.unsqueeze(-1)
        best.copy_(torch.where(mask, candidate, best))
    lambda_damping.copy_(torch.where(
        accept,
        (lambda_damping / lambda_factor).clamp_min(lambda_min),
        (lambda_damping * lambda_factor).clamp_max(lambda_max),
    ))
    return accept
