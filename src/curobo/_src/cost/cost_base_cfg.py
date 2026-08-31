from __future__ import annotations

from dataclasses import dataclass
from typing import List, Type, Union

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class BaseCostCfg:
    weight: Union[torch.Tensor, float, List[float]]
    class_type: Type[BaseCost] = BaseCost
    device_cfg: DeviceCfg = DeviceCfg()
    convert_to_binary: bool = False
    use_grad_input: bool = False

    def __post_init__(self):
        if isinstance(self.weight, (bool, int)):
            log_and_raise(
                "BaseCostCfg: weight must be a tensor, float, or list of floats, got int"
            )
        if isinstance(self.weight, float):
            self.weight = self.device_cfg.to_device([self.weight])
        elif isinstance(self.weight, list):
            self.weight = self.device_cfg.to_device(self.weight)
        elif isinstance(self.weight, torch.Tensor):
            self.weight = self.weight.to(self.device_cfg.device, dtype=self.device_cfg.dtype)

    def clone(self):
        return BaseCostCfg(
            weight=self.weight.clone(),
            class_type=self.class_type,
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            use_grad_input=self.use_grad_input,
        )


__all__ = ["BaseCost", "BaseCostCfg"]
