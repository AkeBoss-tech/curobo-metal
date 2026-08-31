from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from curobo._src.util.logging import log_warn

if TYPE_CHECKING:
    from curobo._src.cost.cost_base_cfg import BaseCostCfg


class BaseCost:
    """Pinned cuRobo base-cost lifecycle implemented with ordinary tensors."""

    def __init__(self, config: BaseCostCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        self._init_post_config()
        self.weight = self._weight
        self.use_grad_input = config.use_grad_input
        self._batch_size = -1
        self._horizon = -1
        self.batch_size = None
        self.horizon = None
        self._dt = 1

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        if batch_size != self._batch_size or horizon != self._horizon:
            self._batch_size = self.batch_size = batch_size
            self._horizon = self.horizon = horizon
        return True

    def _init_post_config(self):
        self._weight = self.config.weight.clone()
        self.cost_fn = None
        self._cost_enabled = True
        if torch.sum(self._weight) == 0.0:
            self.disable_cost()

    def forward(self, **kwargs) -> Union[Any, torch.Tensor]:
        log_warn("BaseCost forward is not implemented")
        return torch.zeros(
            (self._batch_size, self._horizon, 1),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

    def disable_cost(self):
        if not self._cost_enabled:
            return
        self._weight *= 0.0
        self._cost_enabled = False

    def enable_cost(self):
        if self._cost_enabled:
            return
        self._weight.copy_(self.config.weight)
        self._cost_enabled = bool(torch.sum(self._weight) != 0.0)

    @property
    def enabled(self):
        return self._cost_enabled

    def update_dt(self, dt: Union[float, torch.Tensor]):
        self._dt = dt
        self.dt = dt

    def reset(
        self,
        reset_problem_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        pass


__all__ = ["BaseCost"]
