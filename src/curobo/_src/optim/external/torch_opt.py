"""Portable wrapper for the pinned torch optimizer import path."""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer


@dataclass
class TorchOptCfg(PortableOptCfg):
    solver_type: str = "torch"
    solver_name: str = "torch"
    torch_optim_name: str = "Adam"
    torch_optim_kwargs: dict = field(default_factory=dict)
    torch_optim_class: Optional[Any] = None

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.torch_optim_class is None:
            self.torch_optim_class = getattr(torch.optim, self.torch_optim_name)


class TorchOpt(PortableOptimizer):
    strategy = "lbfgs"


__all__ = ["TorchOpt", "TorchOptCfg"]
