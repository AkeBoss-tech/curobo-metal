"""Pinned evolution-strategies surface."""

from dataclasses import dataclass

import torch

from .mppi import MPPI, MPPICfg


@dataclass
class EvolutionStrategiesCfg(MPPICfg):
    solver_type: str = "es"
    solver_name: str = "es"
    learning_rate: float = 0.1


class EvolutionStrategies(MPPI):
    strategy = "cem"


def calc_exp(total_costs):
    """Stable exponential utilities used by the portable ES update."""
    shifted = total_costs - total_costs.amin(dim=-1, keepdim=True)
    return torch.exp(-shifted).div(torch.exp(-shifted).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(total_costs.dtype).eps))


def compute_es_mean(w, actions, mean_action, full_inv_cov, num_particles, learning_rate):
    del full_inv_cov, num_particles
    while w.ndim < actions.ndim:
        w = w.unsqueeze(-1)
    proposal = (w * actions).sum(dim=-3)
    return mean_action + float(learning_rate) * (proposal - mean_action)


__all__ = ["EvolutionStrategiesCfg", "EvolutionStrategies", "calc_exp", "compute_es_mean"]
