"""Device-resident optimizer iteration values."""
from dataclasses import dataclass, fields
from typing import Optional, Union, Any
import torch

@dataclass
class OptimizationIterationState:
    action: torch.Tensor
    cost: Optional[torch.Tensor] = None
    gradient: Optional[torch.Tensor] = None
    exploration_action: Optional[torch.Tensor] = None
    exploration_gradient: Optional[torch.Tensor] = None
    exploration_cost: Optional[torch.Tensor] = None
    step_direction: Optional[torch.Tensor] = None
    best_action: Optional[torch.Tensor] = None
    best_cost: Optional[torch.Tensor] = None
    best_iteration: Optional[torch.Tensor] = None
    current_iteration: Optional[torch.Tensor] = None
    state: Optional[Union[Any, torch.Tensor]] = None
    converged: Optional[torch.Tensor] = None
    jacobian: Optional[torch.Tensor] = None

    @property
    def data_ptr(self):
        return tuple(v.data_ptr() if isinstance(v, torch.Tensor) else None for v in vars(self).values())

    def clone(self):
        return type(self)(**{f.name: (getattr(self, f.name).clone() if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name)) for f in fields(self)})

    def copy_(self, other):
        for f in fields(self):
            source, target = getattr(other, f.name), getattr(self, f.name)
            if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
                target.copy_(source)
            else:
                setattr(self, f.name, source.clone() if isinstance(source, torch.Tensor) else source)
        return self
