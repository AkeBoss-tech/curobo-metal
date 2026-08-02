"""Pinned MPPI surface backed by portable particle optimization."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer


class BaseActionType(Enum):
    REPEAT = "REPEAT"
    NULL = "NULL"
    RANDOM = "RANDOM"


@dataclass
class MPPICfg(PortableOptCfg):
    solver_type: str = "mppi"
    solver_name: str = "mppi"
    gamma: float = 1.0
    sample_mode: Any = "MEAN"
    seed: int = 0
    store_rollouts: bool = False
    null_act_frac: float = 0.0
    init_mean: Optional[torch.Tensor] = None
    init_cov: float = 0.5
    base_action: BaseActionType = BaseActionType.REPEAT
    step_size_mean: float = 0.9
    step_size_cov: float = 0.1
    squash_fn: Any = "CLAMP"
    cov_type: Any = "DIAG_A"
    sample_params: Any = None
    update_cov: bool = True
    random_mean: bool = False
    beta: float = 0.1
    alpha: float = 1.0
    kappa: float = 0.01
    sample_per_problem: bool = True

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if isinstance(self.base_action, str):
            self.base_action = BaseActionType[self.base_action]


class MPPI(PortableOptimizer):
    strategy = "mppi"


def jit_blend_cov(cov_action, cov_update, step_size_cov, kappa):
    """Blend diagonal covariance tensors with a positive numerical floor."""
    value = (1.0 - step_size_cov) * cov_action + step_size_cov * cov_update
    return value.clamp_min(torch.as_tensor(kappa, device=value.device, dtype=value.dtype))


def jit_blend_mean(mean_action, new_mean, step_size_mean):
    return (1.0 - step_size_mean) * mean_action + step_size_mean * new_mean


def jit_calculate_exp_util(beta, total_costs):
    shifted = total_costs - total_costs.amin(dim=-1, keepdim=True)
    return torch.softmax(-shifted / max(float(beta), torch.finfo(total_costs.dtype).eps), dim=-1)


def jit_compute_total_cost(gamma_seq, costs):
    gamma = torch.as_tensor(gamma_seq, device=costs.device, dtype=costs.dtype)
    return (costs * gamma).sum(dim=-1)


def jit_calculate_exp_util_from_costs(costs, gamma_seq, beta):
    return jit_calculate_exp_util(beta, jit_compute_total_cost(gamma_seq, costs))


def jit_diag_a_cov_update(w, actions, mean_action):
    delta = actions - mean_action.unsqueeze(-3)
    while w.ndim < delta.ndim:
        w = w.unsqueeze(-1)
    return (w * delta.square()).sum(dim=-3)


def jit_mean_cov_diag_a(
    costs, actions, gamma_seq, mean_action, cov_action, step_size_mean, step_size_cov, kappa, beta
):
    weights = jit_calculate_exp_util_from_costs(costs, gamma_seq, beta)
    expanded = weights
    while expanded.ndim < actions.ndim:
        expanded = expanded.unsqueeze(-1)
    new_mean = (expanded * actions).sum(dim=-3)
    new_cov = jit_diag_a_cov_update(weights, actions, new_mean)
    return (
        jit_blend_mean(mean_action, new_mean, step_size_mean),
        jit_blend_cov(cov_action, new_cov, step_size_cov, kappa),
    )


__all__ = [
    "BaseActionType", "MPPICfg", "MPPI", "jit_blend_cov", "jit_blend_mean",
    "jit_calculate_exp_util", "jit_calculate_exp_util_from_costs", "jit_compute_total_cost",
    "jit_diag_a_cov_update", "jit_mean_cov_diag_a",
]
