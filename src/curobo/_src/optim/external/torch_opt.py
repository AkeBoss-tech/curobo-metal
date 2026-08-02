"""Portable adapter for the pinned ``torch.optim`` cuRobo optimizer wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from curobo._src.optim._portable import PortableOptCfg
from curobo._src.optim.external._portable import ExternalOptimizerBase, create_data_dict
from curobo._src.types.device_cfg import DeviceCfg


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
        if self.inner_iters <= 0:
            raise ValueError("inner_iters must be positive")
        if self.torch_optim_class is None:
            try:
                self.torch_optim_class = getattr(torch.optim, self.torch_optim_name)
            except AttributeError as exc:
                raise ValueError(f"unknown torch optimizer: {self.torch_optim_name}") from exc

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        return create_data_dict(cls, data_dict, device_cfg, child_dict)


class TorchOpt(ExternalOptimizerBase):
    """Eager CPU/MPS ``torch.optim`` rollout optimizer.

    CUDA graph capture and the upstream ``torch.compile`` fast-path are
    intentionally absent.  All evaluation and update tensors remain on the
    configured CPU or MPS device.
    """

    def __init__(self, config: TorchOptCfg, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._torch_optimizer = None
        self._current_q = None
        self._init_torch_optimizer()

    def _init_torch_optimizer(self):
        cls = self.config.torch_optim_class
        kwargs = dict(self.config.torch_optim_kwargs)
        # The portable config's step scale is the only available equivalent
        # when callers did not provide torch's canonical ``lr`` argument.
        kwargs.setdefault("lr", self.config.step_scale)
        self._torch_optimizer = cls([self._optimization_variable], **kwargs)

    def _loss_fn(self, action: torch.Tensor):
        action = self._action_view(action)
        cost, _ = self._rollout_values(action)
        return cost

    def _closure(self):
        self._torch_optimizer.zero_grad(set_to_none=True)
        cost = self._loss_fn(self._optimization_variable)
        cost.sum().backward()
        self._track_best(cost)
        return cost.sum()

    def _track_best(self, cost):
        current = self._action_view(self._optimization_variable)
        mask = cost.detach() < self.best_cost
        while mask.ndim < current.ndim:
            mask = mask.unsqueeze(-1)
        self.best_q.copy_(torch.where(mask, current.detach(), self.best_q))
        self.best_cost.copy_(torch.minimum(self.best_cost, cost.detach()))

    def _opt_step(self, iteration_state=None):
        del iteration_state
        cls = self.config.torch_optim_class
        if issubclass(cls, torch.optim.LBFGS):
            self._torch_optimizer.step(self._closure)
            cost = self._loss_fn(self._optimization_variable)
        else:
            self._torch_optimizer.zero_grad(set_to_none=True)
            cost = self._loss_fn(self._optimization_variable)
            cost.sum().backward()
            self._torch_optimizer.step()
            self._track_best(cost)
        action = self._action_view(self._optimization_variable)
        return self._record(action, cost, self.best_q, self.best_cost)

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action

        def run():
            action = self._action_view(seed_action).to(
                device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )
            if self._reset_storage_for_action(action):
                self._init_torch_optimizer()
            with torch.no_grad():
                self._optimization_variable.copy_(action)
                self.best_q.copy_(action)
                initial_cost = self._loss_fn(self._optimization_variable).detach()
                self.best_cost.copy_(initial_cost)
            self._record(action, initial_cost, self.best_q, self.best_cost)
            for _ in range(self.config.num_iters):
                self._opt_step()
            return self.best_q.detach().clone()

        return self._timed(run)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        result = super().reinitialize(action, mask, clear_optimizer_state, reset_num_iters)
        if clear_optimizer_state:
            self._init_torch_optimizer()
        return result

    def update_num_problems(self, num_problems):
        super().update_num_problems(num_problems)
        # ``__init__`` invokes this before the optimizer field exists.
        if hasattr(self, "_torch_optimizer"):
            self._init_torch_optimizer()


__all__ = ["TorchOpt", "TorchOptCfg"]
