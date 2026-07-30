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


__all__ = ["BaseActionType", "MPPICfg", "MPPI"]
