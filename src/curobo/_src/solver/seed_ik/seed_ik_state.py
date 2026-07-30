from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class SeedIKState:
    success: Optional[torch.Tensor] = None
    improvement: Optional[torch.Tensor] = None
    joint_position: Optional[torch.Tensor] = None
    error_norm: Optional[torch.Tensor] = None
    jTerror: Optional[torch.Tensor] = None
    jacobian: Optional[torch.Tensor] = None
    lambda_damping: Optional[torch.Tensor] = None
    position_errors: Optional[torch.Tensor] = None
    orientation_errors: Optional[torch.Tensor] = None

    def clone(self):
        return type(self)(**{
            item.name: getattr(self, item.name).clone()
            if getattr(self, item.name) is not None else None
            for item in fields(self)
        })

    def copy_(self, other):
        for item in fields(self):
            source = getattr(other, item.name)
            target = getattr(self, item.name)
            if source is None:
                setattr(self, item.name, None)
            elif target is None or target.shape != source.shape:
                setattr(self, item.name, source.clone())
            else:
                target.copy_(source)
        return self


__all__ = ["SeedIKState"]
