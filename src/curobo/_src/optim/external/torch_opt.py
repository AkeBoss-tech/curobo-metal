"""Portable wrapper for the pinned torch optimizer import path."""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer, _objective


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

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run the requested torch optimizer against a differentiable rollout.

        This is intentionally eager and device-resident; it is an adapter for
        callers who choose a torch optimizer, not a CUDA graph emulation.
        """
        if not self.enabled:
            return seed_action
        objective = _objective(self.rollout_fn)
        value = seed_action.detach().clone().requires_grad_(True)
        cls = self.config.torch_optim_class
        kwargs = dict(self.config.torch_optim_kwargs)
        kwargs.setdefault("lr", self.config.step_scale)
        optimizer = cls([value], **kwargs)
        for _ in range(self.config.num_iters):
            def closure():
                optimizer.zero_grad()
                cost = objective(value).sum()
                cost.backward()
                return cost
            if issubclass(cls, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
        return value if torch.is_grad_enabled() else value.detach()


__all__ = ["TorchOpt", "TorchOptCfg"]
